#!/usr/bin/env python3
"""features.py — таблица признаков для модели доверия («можно ли принять вердикт модели без куратора»).

Одна строка = одна размеченная задача в одном прогоне (runs/<модель>[@метка]/<работа>/).
Источники: check.json (+ check_r2.json … при --repeats), transcription.json, usage.csv,
data/labels.csv (цель), data/keys/answer_key.csv (независимый ключ ответов), data/works.csv.

    python3 features.py                 # → data/features.csv
    python3 features.py --out other.csv
    python3 features.py --include-unlabeled --out data/features_all.csv
        # + задачи из списка проверки без метки (начата и брошена / зачёркнута / не решена): label пустой.
        # Нужны, чтобы оценить нагрузку на куратора в режиме «список задач от репетитора» (ml/apply_trust.py).

Цели (не использовать как признаки!):
  label        — разметка: 1 верно, 0 есть ошибка
  model_wrong  — 1, если вердикт модели не совпал с разметкой (главная цель модели доверия)
Служебные колонки (не признаки): run, work, task_no, student_id, task_key, split, human_missed,
  student_answer, model_correct_answer, key_answer. Группы для GroupKFold — student_id или task_key.
"""
import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from check import DATA, RUNS, load_labels, norm_task, task_hit, work_splits

LEG = {"good": 2, "medium": 1, "poor": 0}
ERR_TYPES = ("computational", "conceptual", "notational", "presentation")


# ----------------------------------------------------------------- сравнение ответов

def _brace_arg(s, i):
    """Аргумент LaTeX-команды с позиции i: {…} с вложенностью или один символ. → (текст, новая позиция)."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i < len(s) and s[i] == "{":
        depth, j = 0, i
        while j < len(s):
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            if depth == 0:
                return s[i + 1:j], j + 1
            j += 1
        return s[i + 1:], len(s)
    return (s[i], i + 1) if i < len(s) else ("", i)


def delatex(s):
    r"""LaTeX → простая запись: \frac{a}{b} → (a)/(b) (с вложенностью, \frac12 тоже), \sqrt{x} → √(x),
    \pi → π, \infty → ∞, x^{2} → x^(2), индексы x_1 убираются."""
    s = str(s or "").replace("$", "").replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"(\d)\s*(\\[dt]?frac)", r"\1 \2", s)  # смешанное число 1\frac{3}{5} → «1 3/5»
    out, i = [], 0
    while i < len(s):
        m = re.match(r"\\[dt]?frac", s[i:])
        if m:
            a, i = _brace_arg(s, i + m.end())
            b, i = _brace_arg(s, i)
            out.append(f"({delatex(a)})/({delatex(b)})")
            continue
        m = re.match(r"\\sqrt\s*\[([^\]]*)\]", s[i:])  # корень n-й степени: \sqrt[5]{5} → (5)^(1/(5))
        if m:
            a, i = _brace_arg(s, i + m.end())
            out.append(f"({delatex(a)})^(1/({m.group(1)}))")
            continue
        m = re.match(r"\\sqrt", s[i:])
        if m:
            a, i = _brace_arg(s, i + m.end())
            out.append(f"√({delatex(a)})")
            continue
        if s[i] == "^":
            a, i = _brace_arg(s, i + 1)
            out.append(f"^({delatex(a)})")
            continue
        out.append(s[i])
        i += 1
    s = "".join(out)
    s = re.sub(r"_\{[^{}]*\}|_\w", "", s)  # индексы x_1, x_{1,2}
    rep = {r"\pi": "π", r"\infty": "∞", r"\cup": "∪", r"\cdot": "·", r"\times": "·", r"\leq": "≤",
           r"\geq": "≥", r"\le": "≤", r"\ge": "≥", r"\pm": "±", r"\mp": "±"}
    for k, v in rep.items():
        s = s.replace(k, v)
    s = re.sub(r"\\[a-zA-Z]+", " ", s).replace("\\", " ").replace("{", "").replace("}", "")
    s = re.sub(r"\(([^()]*)\)/\(([^()]*)\)\s*π", r"(\1π)/(\2)", s)  # \frac{3}{4}\pi → 3π/4
    for _ in range(3):  # (19π)/(4) → 19π/4, (√(2))/(2) → √2/2, √(3) → √3
        s = re.sub(r"√\((\d+(?:[.,]\d+)?)\)", r"√\1", s)
        s = re.sub(r"\(([^()]*)\)/\(([^()]*)\)", r"\1/\2", s)
        s = re.sub(r"\^\((-?\d+)\)", r"^\1", s)
    return s


_ATOM = r"(?:(\d+(?:\.\d+)?)?(√(\d+(?:\.\d+)?))?(π)?)"
_NUM = re.compile(r"(-?)inf|([-+]?)" + _ATOM + r"(?:\^(-?\d+))?(?:/" + _ATOM + r")?(π)?")


def _num_tokens(s):
    """Числа из ответа (дроби, корни, π, степени, ±, ∞) как отсортированный список."""
    s = delatex(s).replace("−", "-").replace("–", "-")
    s = re.sub(r"—(?=\d)", "-", s)
    s = re.sub(r"(?<![\d./^])(\d+)\s+(\d+)/(\d+)(?![\d.])",  # смешанное число «1 3/5» → 1.6
               lambda m: repr(int(m.group(1)) + int(m.group(2)) / int(m.group(3))), s)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)  # десятичная запятая (до удаления пробелов: «1, 10» — два числа)
    s = s.replace("·", "*").replace(" ", "").replace("+∞", "inf").replace("∞", "inf")
    if "±" in s:
        return sorted(_num_tokens(s.replace("±", "+", 1)) + _num_tokens(s.replace("±", "-", 1)))

    def atom(a, root, r, pi):
        if not (a or root or pi):
            return None
        v = float(a) if a else 1.0
        if root:
            v *= math.sqrt(float(r))
        if pi:
            v *= math.pi
        return v
    vals = []
    for m in _NUM.finditer(s):
        if not m.group(0) or m.group(0) in "+-":
            continue
        if m.group(0).endswith("inf"):
            vals.append(-math.inf if m.group(1) else math.inf)
            continue
        num = atom(m.group(3), m.group(4), m.group(5), m.group(6))
        if num is None:
            continue
        if m.group(7):
            num **= int(m.group(7))
        den = atom(m.group(8), m.group(9), m.group(10), m.group(11))
        if den:
            num /= den
        if m.group(12):
            num *= math.pi
        vals.append(round(-num if m.group(2) == "-" else num, 4))
    return sorted(vals)


def _brackets(s):
    return "".join(c for c in str(s or "") if c in "[]()")


def _pairs(s):
    return sorted(tuple(_num_tokens_unsorted(g)) for g in re.findall(r"\(([^()]*)\)", delatex(s)))


def _num_tokens_unsorted(g):
    # порядок внутри пары важен: (1; 2) ≠ (2; 1)
    parts = re.split(r"[;]|,(?!\d)", g)
    return [x for p in parts for x in _num_tokens(p)]


def answers_match(a, b, kind=""):
    """1 — совпадают, 0 — нет, None — сравнить нельзя.
    kind — тип ответа из ключа: для expression/proof/text/construction и серий корней сравнение
    числами ненадёжно → None (серии — только буквальное совпадение)."""
    if kind in ("proof", "construction", "text", "expression"):
        return None
    if kind == "series":
        na, nb = (re.sub(r"[\s,;]", "", delatex(x)).replace("n", "k") for x in (a, b))
        return 1 if na == nb else None
    da, db = delatex(a), delatex(b)
    ineq = lambda x: bool(re.search(r"[<>≤≥]", x))
    intv = lambda x: bool(re.search(r"[\[\]]|∞|inf", x))
    if ineq(da) != ineq(db) and (intv(da) or intv(db)):
        return None  # «x ≤ 1» против «(-∞; 1]» числами не сравнить
    if kind == "pair":
        pa, pb = _pairs(a), _pairs(b)
        if pa and pb:
            return int(pa == pb)
    na, nb = _num_tokens(a), _num_tokens(b)
    if not na or not nb:
        return None
    if na != nb and sorted(set(na)) != sorted(set(nb)):
        return 0
    if kind == "interval" and _brackets(da) and _brackets(db):
        return int(_brackets(da) == _brackets(db))
    return 1


# ----------------------------------------------------------------- загрузка

def load_keys():
    path = DATA / "keys" / "answer_key.csv"
    keys = {}
    if path.exists():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            for w in r["works"].split():
                keys[(w, norm_task(r["task_no"]))] = r
    return keys


def load_usage(run_dir):
    """work -> {step: {prompt, completion, reasoning, effort}}"""
    out = {}
    path = run_dir / "usage.csv"
    if not path.exists():
        return out
    for r in csv.DictReader(open(path, encoding="utf-8")):
        d = out.setdefault(r["work"], {}).setdefault(r["step"], {})  # "check", "check:r2", "transcribe:…"
        d.update(prompt=int(r.get("prompt_tokens") or 0), completion=int(r.get("completion_tokens") or 0),
                 reasoning=int(r.get("reasoning_tokens") or 0) if (r.get("reasoning_tokens") or "").isdigit() else None,
                 effort=r.get("effort") or "default")
    return out


def predictions(path):
    res = json.load(open(path, encoding="utf-8")).get("results", [])
    pred = {}
    for r in res:
        k = norm_task(r.get("task_no", ""))
        if k not in pred or r.get("verdict") == "incorrect":
            pred[k] = r
    return pred


def find_near(pred, k):
    """Та же нечёткая сверка номеров, что в отчёте check.py (13 ~ 13а, 5 ~ Б1-5)."""
    near = [p for p in pred if task_hit(p, k) or task_hit(k, p)]
    return pred[near[0]] if len(near) == 1 else None


_LABEL = re.compile(r"(?:№|\bN)\s*«?\s*([0-9]+[а-яa-z]?(?:\.[0-9]+)?)", re.I)
_LINE_LABEL = re.compile(r"^\s*~*\s*(?:№|N)\s*([0-9]+[а-яa-z]?(?:\.[0-9]+)?)", re.I)


def task_segment(page_text, where):
    """Строки решения этой задачи на её странице: от строки с номером («№6», «N10») до следующего номера.
    Номер не нашёлся — вся страница (грубее: там могут быть числа соседних задач)."""
    lines = str(page_text or "").splitlines()
    m = _LABEL.search(str(where or ""))
    if m:
        lab = m.group(1).lower()
        starts = [i for i, ln in enumerate(lines) if (mm := _LINE_LABEL.match(ln)) and mm.group(1).lower() == lab]
        if starts:
            i = starts[0]
            j = next((k for k in range(i + 1, len(lines)) if _LINE_LABEL.match(lines[k])), len(lines))
            return "\n".join(lines[i + 1:j]), True
    return "\n".join(lines), False


def key_in_work(segment, key_answer):
    """Правильное значение (ключ) встречается среди чисел решения ученика. Для ключей 0, 1, 2 не считаем:
    такие числа есть почти в любом решении."""
    kv = _num_tokens(key_answer) if key_answer else []
    if len(kv) != 1 or abs(kv[0]) <= 2 and float(kv[0]).is_integer():
        return None
    def safe(ln):
        try:
            return _num_tokens(ln)
        except (OverflowError, ValueError, ZeroDivisionError):  # степени вроде 10^400 в распознанном тексте
            return []
    nums = [n for ln in str(segment).splitlines() for n in safe(ln)]
    return int(any(abs(n - kv[0]) < 1e-6 for n in nums))


def page_of(where):
    m = re.search(r"(?:стр\w*|page)\s*\.?\s*(\d+)", str(where or ""), re.I)
    return int(m.group(1)) if m else None


def entropy(votes):
    n = sum(votes.values())
    return -sum(c / n * math.log2(c / n) for c in votes.values() if c) if n else None


# ----------------------------------------------------------------- сборка

def build(include_unlabeled=False):
    labels, splits, keys = load_labels(), work_splits(), load_keys()
    students = {r["work_id"]: r.get("student_id", "") for r in csv.DictReader(open(DATA / "works.csv", encoding="utf-8"))}
    task_keys = {}
    if (DATA / "keys" / "answer_key.csv").exists():
        for r in csv.DictReader(open(DATA / "keys" / "answer_key.csv", encoding="utf-8")):
            for w in r["works"].split():
                task_keys[w] = r["task_key"]
    rows = []
    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        usage = load_usage(run_dir)
        for wd in sorted(p for p in run_dir.iterdir() if (p / "check.json").exists()):
            work = wd.name
            main_json = json.load(open(wd / "check.json", encoding="utf-8"))
            if "parse_error" in main_json or main_json.get("truncated"):
                continue
            meta = main_json.get("_meta") or {}
            main = predictions(wd / "check.json")
            reps = []  # повторы — только удачные и с теми же настройками, что основной прогон
            for p in sorted(wd.glob("check_r*.json")):
                if p.stem.endswith("_failed"):
                    continue
                rj = json.load(open(p, encoding="utf-8"))
                rm = rj.get("_meta") or {}
                if "parse_error" in rj or (meta and rm and {k: rm.get(k) for k in ("effort", "images", "prompt")}
                                           != {k: meta.get(k) for k in ("effort", "images", "prompt")}):
                    continue
                reps.append(predictions(p))
            trans = json.load(open(wd / "transcription.json", encoding="utf-8")) if (wd / "transcription.json").exists() else []
            leg = [LEG.get(p.get("legibility"), None) for p in trans]
            leg = [x for x in leg if x is not None]
            text_by_page = {p.get("page"): "\n".join(p.get("lines", [])) for p in trans}
            all_text = "\n".join(text_by_page.values())
            u = usage.get(work, {}).get("check", {})
            n_checked = len(main) or 1
            for k, e in labels.get(work, {}).items():
                if e["label"] not in (0, 1) and not include_unlabeled:
                    continue
                r = main.get(k) or find_near(main, k) or {}
                verdict = r.get("verdict") or "absent"
                pred = {"correct": 1, "incorrect": 0, "not_solved": 0}.get(verdict)  # как в check.py --report
                errs = r.get("errors", []) or []
                et = Counter(x.get("type") for x in errs)
                votes = Counter([verdict] + [(rp.get(k) or find_near(rp, k) or {}).get("verdict", "absent") for rp in reps])
                key = keys.get((work, k), {})
                kind = key.get("answer_type", "")
                pg = page_of(r.get("where_in_solution"))
                ptxt = text_by_page.get(pg, "")
                seg, seg_found = task_segment(ptxt, r.get("where_in_solution"))
                row = {
                    # служебное
                    "run": run_dir.name, "work": work, "task_no": e["task_no"], "student_id": students.get(work, ""),
                    "task_key": task_keys.get(work, work), "split": splits.get(work, "dev"),
                    # цели
                    "label": e["label"] if e["label"] in (0, 1) else "",
                    "model_wrong": None if pred is None or e["label"] not in (0, 1) else int(pred != e["label"]),
                    "human_missed": int(e["human_missed"]),
                    # вердикт модели
                    "verdict": verdict, "pred_error": None if pred is None else 1 - pred,
                    "n_errors": len(errs), **{f"n_err_{t}": et.get(t, 0) for t in ERR_TYPES},
                    "n_err_substantive": et.get("computational", 0) + et.get("conceptual", 0),
                    # проверка на адекватность (промпт проверки v3): подтверждают ли следующие шаги найденную ошибку
                    "n_err_confirmed": sum(x.get("check_next") == "confirmed" for x in errs),
                    "n_err_contradicted": sum(x.get("check_next") in ("contradicted", "nonsense") for x in errs),
                    "len_explanations": sum(len(str(x.get("explanation", ""))) for x in errs),
                    # ключ ответов (независим от разметки)
                    "student_answer": r.get("student_answer", ""), "model_correct_answer": r.get("correct_answer", ""),
                    "key_answer": key.get("answer", ""), "key_type": kind, "key_confidence": key.get("confidence", ""),
                    "key_vs_student": answers_match(r.get("student_answer"), key.get("answer"), kind),
                    "key_vs_model_answer": answers_match(r.get("correct_answer"), key.get("answer"), kind),
                    "model_answer_vs_student": answers_match(r.get("correct_answer"), r.get("student_answer"), kind),
                    # стабильность (повторные прогоны)
                    "n_runs": 1 + len(reps),
                    "frac_incorrect": votes.get("incorrect", 0) / (1 + len(reps)),
                    "vote_entropy": entropy(votes) if reps else None,
                    # «думание» и объём
                    "effort_check": meta.get("effort") or u.get("effort", ""),
                    "check_images": int(bool(meta.get("images"))),
                    "check_with_key": int(bool(meta.get("key"))),
                    "check_reasoning_tokens": u.get("reasoning"), "check_completion_tokens": u.get("completion"),
                    "reasoning_per_task": (u["reasoning"] / n_checked) if u.get("reasoning") is not None else None,
                    "n_tasks_in_check": len(main),
                    # качество распознавания
                    "n_pages": len(trans), "legibility_mean": sum(leg) / len(leg) if leg else None,
                    "legibility_min": min(leg) if leg else None,
                    "unclear_marks_work": all_text.count("[?"), "strikes_work": all_text.count("~~") // 2,
                    "task_page_known": int(pg is not None),
                    # правильное значение есть в решении ученика (правило разметки от 27.09: «нашёл нужное, выписал
                    # другое» — не ошибка); segment_found — решение задачи выделено по номеру, а не вся страница
                    "key_in_work": key_in_work(seg, key.get("answer", "")) if pg else None,
                    "segment_found": int(seg_found),
                    "unclear_marks_page": ptxt.count("[?") if pg else None,
                    # двойное распознавание (--ocr-double): сколько мест, где два прочтения разошлись
                    "ocr_double": int(any("disagreements" in p for p in trans)),
                    "ocr_disagree_work": sum(len(p.get("disagreements", [])) for p in trans)
                    if any("disagreements" in p for p in trans) else None,
                    "ocr_disagree_page": (len(next((p.get("disagreements", []) for p in trans
                                                    if p.get("page") == pg), [])) if pg else None)
                    if any("disagreements" in p for p in trans) else None,
                    "len_student_answer": len(str(r.get("student_answer", ""))),
                }
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DATA / "features.csv"))
    ap.add_argument("--include-unlabeled", action="store_true",
                    help="включить задачи из списка проверки без метки (для оценки нагрузки на куратора)")
    args = ap.parse_args()
    rows = build(args.include_unlabeled)
    if not rows:
        raise SystemExit("Нет прогонов с check.json в runs/")
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    runs = Counter(r["run"] for r in rows)
    print(f"{len(rows)} строк → {args.out}")
    for run, n in runs.items():
        sub = [r for r in rows if r["run"] == run]
        dec = [r for r in sub if r["model_wrong"] is not None]
        km = [r for r in sub if r["key_vs_student"] is not None]
        print(f"  {run}: задач {n}, с вердиктом {len(dec)}, модель ошиблась {sum(r['model_wrong'] for r in dec)}, "
              f"ключ сравним {len(km)}")


if __name__ == "__main__":
    main()
