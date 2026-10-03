#!/usr/bin/env python3
"""ocr_eval.py — качество распознавания почерка против эталона (data/gold/<работа>/gold_solution_N.json).

    python3 ocr_eval.py                                  # прогон по умолчанию: runs/google_gemini-3.8-flash
    python3 ocr_eval.py --run google_gemini-3.8-flash@другая_метка

Метрики:
  CER            — доля ошибочных символов (расстояние Левенштейна / длина эталона), пробелы не считаются;
  math_edits     — правки эталона, меняющие математический смысл (из разметки эталона);
  pages_with_math_error — доля страниц, где есть хоть одна такая правка;
  ошибки ученика сохранены — сколько неверных равенств из эталона (например 44+44=38) есть в транскрипции
                  в том же виде (главная метрика против «автоисправления»);
  трудные места   — строки эталона, где базовый прогон ошибся по смыслу: сколько из них этот прогон прочитал верно.
CER считается только для страниц с эталоном; для другого прогона сравнивается его transcription.json.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GOLD = ROOT / "data" / "gold"


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(lines):
    return re.sub(r"\s+", "", "\n".join(lines))


def norm_math(lines):
    """Мягкая нормализация записи: \\frac{a}{b} ~ a/b, \\cdot ~ ·, $ и \\left/\\right убраны —
    чтобы модель с другим стилем LaTeX не штрафовалась за стиль."""
    from features import delatex
    t = delatex("\n".join(lines)).replace("·", "*").replace("×", "*").replace("−", "-")
    t = re.sub(r"[()]", "", t)
    return re.sub(r"\s+", "", t)


_ARITH = re.compile(r"(?<![\w.)^√±/*+\-:(])(\d+(?:\.\d+)?)([+\-*:/])(\d+(?:\.\d+)?)=(-?\d+(?:\.\d+)?)(?![\w.^(√])")


def _flat(line):
    from features import delatex
    t = delatex(line).replace("·", "*").replace("×", "*").replace("−", "-").replace("–", "-")
    return re.sub(r"\s+", "", re.sub(r"(\d),(\d)", r"\1.\2", t))


def student_arith_errors(lines):
    """Неверные арифметические равенства вида «a op b = c» в эталоне (например 44+44=38) — ошибки ученика,
    которые распознавание обязано сохранить. Берутся только самостоятельные выражения, не куски формул."""
    out = []
    for line in lines:
        if "~~" in line:
            line = re.sub(r"~~.*?~~", " ", line)  # зачёркнутое не считаем
        for m in _ARITH.finditer(_flat(line)):
            a, op, b, c = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
            val = {"+": a + b, "-": a - b, "*": a * b, ":": a / b if b else None, "/": a / b if b else None}[op]
            if val is not None and abs(val - c) > 1e-6:
                out.append(m.group(0))
    return out


def flag_quality(gold_lines, page):
    """Насколько пометки двойного распознавания (disagreements) ловят ошибки прочтения.
    → (строк эталона с ошибкой, из них попали в пометку, всего пометок, пометок на верно прочитанных строках)."""
    import difflib
    if "disagreements" not in page:
        return None
    hyp = page.get("lines", [])
    hn = [_flat(x) for x in hyp]
    gold_set = {_flat(x) for x in gold_lines}
    flagged = set()
    for d in page["disagreements"]:
        a1, a2 = d.get("a_lines", [0, 0])
        flagged.update(range(a1, max(a2, a1 + 1)))
    err = hit = 0
    for g in gold_lines:
        gn = _flat(g)
        if not gn or gn in hn:
            continue
        err += 1
        best = max(range(len(hn)), key=lambda i: difflib.SequenceMatcher(None, gn, hn[i]).ratio(), default=None)
        if best is not None and difflib.SequenceMatcher(None, gn, hn[best]).ratio() >= 0.5 and best in flagged:
            hit += 1
    false = sum(1 for d in page["disagreements"]
                if all(_flat(x) in gold_set for x in hyp[d["a_lines"][0]:d["a_lines"][1]]) and d["a_lines"][1] > d["a_lines"][0])
    return err, hit, len(page["disagreements"]), false


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="google_gemini-3.8-flash")
    ap.add_argument("--second", action="store_true",
                    help="оценить второе прочтение (pages2/ после --ocr-double; это промпт v2) вместо основного")
    args = ap.parse_args()
    rows, kinds, missing, broken = [], Counter(), 0, 0
    for gf in sorted(GOLD.glob("W*/gold_*.json")):
        work = gf.parent.name
        g = json.load(open(gf, encoding="utf-8"))
        tpath = ROOT / "runs" / args.run / work / "transcription.json"
        if not tpath.exists():
            missing += 1
            continue
        pages = {p.get("file"): p for p in json.load(open(tpath, encoding="utf-8"))}
        p = pages.get(g["file"])
        if args.second and p is not None:
            alt = tpath.parent / "pages2" / (Path(g["file"]).stem + ".json")
            p = json.load(open(alt, encoding="utf-8")) if alt.exists() else None
        if not p:
            missing += 1
            continue
        if p.get("parse_error") or p.get("failed"):
            broken += 1
            continue
        ref, hyp = norm(g["lines"]), norm(p.get("lines", []))
        edits = g.get("edits", [])
        same_run = args.run == "google_gemini-3.8-flash"  # правки размечены относительно этого прогона
        if same_run:
            kinds.update(e.get("kind") for e in edits)
        hyp_n = "\n".join(_flat(x) for x in p.get("lines", []))
        traps = student_arith_errors(g["lines"])
        hard = [_flat(e["gold"]) for e in edits
                if e.get("affects_math") and e.get("gold", "").strip()]
        fq = flag_quality(g["lines"], p)
        rows.append({"work": work, "file": g["file"], "legibility": p.get("legibility"),
                     "err_lines": fq[0] if fq else "", "err_flagged": fq[1] if fq else "",
                     "flags": fq[2] if fq else "", "false_flags": fq[3] if fq else "",
                     "student_errors": len(traps), "student_errors_kept": sum(t in hyp_n for t in traps),
                     "hard_spots": len(hard), "hard_spots_ok": sum(h in hyp_n for h in hard if h),
                     "ref_chars": len(ref), "cer": lev(hyp, ref) / max(len(ref), 1),
                     "cer_norm": lev(norm_math(p.get("lines", [])), norm_math(g["lines"])) / max(len(norm_math(g["lines"])), 1),
                     "edits": len(edits) if same_run else "",
                     "math_edits": sum(bool(e.get("affects_math")) for e in edits) if same_run else ""})
    if not rows:
        raise SystemExit("Нет страниц с эталоном и транскрипцией этого прогона")
    out = ROOT / "reports" / f"ocr_{args.run}{'_second' if args.second else ''}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    chars = sum(r["ref_chars"] for r in rows) or 1
    cer = sum(r["cer"] * r["ref_chars"] for r in rows) / chars
    cer_n = sum(r["cer_norm"] * r["ref_chars"] for r in rows) / chars
    print(f"Прогон {args.run}{' (второе прочтение)' if args.second else ''}: страниц {len(rows)}, символов эталона {chars}")
    print(f"  CER (взвешенный по длине): {100 * cer:.1f}%; без учёта стиля записи: {100 * cer_n:.1f}%")
    if missing or broken:
        print(f"  ВНИМАНИЕ: нет транскрипции для {missing} страниц эталона, сбой разбора на {broken} — "
              f"сравнивай прогоны на одинаковом наборе страниц")
    by_leg = {}
    for r in rows:
        by_leg.setdefault(r["legibility"], []).append(r["cer"])
    for k, v in by_leg.items():
        print(f"  legibility={k}: страниц {len(v)}, средний CER {100 * sum(v) / len(v):.1f}%")
    se, sk = sum(r["student_errors"] for r in rows), sum(r["student_errors_kept"] for r in rows)
    hs, hk = sum(r["hard_spots"] for r in rows), sum(r["hard_spots_ok"] for r in rows)
    print(f"  ошибки ученика в арифметике сохранены: {sk} из {se}")
    dbl = [r for r in rows if r["err_lines"] != ""]  # страницы, где есть второе прочтение
    if dbl:
        el, eh = sum(r["err_lines"] for r in dbl), sum(r["err_flagged"] for r in dbl)
        fl, ff = sum(r["flags"] for r in dbl), sum(r["false_flags"] for r in dbl)
        print(f"  страниц с двойным распознаванием: {len(dbl)}")
        print(f"  двойное распознавание: строк с ошибкой прочтения {el}, из них помечены расхождением {eh} "
              f"({100 * eh / max(el, 1):.0f}%); пометок всего {fl}, из них на верно прочитанных строках {ff}")
    print(f"  трудные места (где базовый прогон исказил математику) прочитаны верно: {hk} из {hs}")
    if rows[0]["edits"] != "":
        me = [r["math_edits"] for r in rows]
        print(f"  правок всего {sum(r['edits'] for r in rows)}, меняют математику {sum(me)}; "
              f"страниц с такой ошибкой {sum(x > 0 for x in me)} из {len(rows)}")
        print("  по типам:", dict(kinds.most_common()))
    print(f"Подробно: {out}")


if __name__ == "__main__":
    main()
