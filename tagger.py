"""Слой «теггер» (гибрид, EXPERIMENTS.md п. 23): проверка v1 → разметка найденных ею отклонений по ситуациям
rules/policy.csv → политика (policy.py).

Проверка v1 остаётся как есть (она находит ошибки лучше детектора «с нуля»). Теггер — дешёвый текстовый запрос
без картинок и без транскрипции: условие, ключ, ответ ученика и список отклонений от v1. Он не решает задачу и не
меняет вердикт — только относит каждое отклонение к ситуации. Вердикт даёт policy.decide по режиму работы.
Это та же постановка, что офлайн-оценка п. 20 (reports/policy_tags.json, 93,2% на test2), только теги ставит Gemini.

Запрос — на несколько работ, ~10 задач (справочник ситуаций в начале, чтобы провайдер кэшировал общий префикс); только задачи,
где v1 нашёл отклонения. Задача «неверно» без списка отклонений → G.answer без запроса.

  python3 tagger.py --src google_gemini-3.8-flash@low_key_k2 --dry-run      # сколько запросов и прикидка цены
  python3 tagger.py --src google_gemini-3.8-flash@low_key_k2 --max-rub 20   # → runs/<src>_tag/
  python3 tagger.py --src google_gemini-3.8-flash@low_key_k2 --report       # сравнение с v1 и офлайн-тегами
"""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter

import check
import policy

MODEL = "google/gemini-3.8-flash"
PRICE_IN, PRICE_OUT = 94.68, 473.39  # ₽ за 1M токенов (data/models.json)
MAX_TOKENS = 12000  # «думание» Gemini даже на minimal доходит до ~5000 токенов

PROMPT = """Ты размечаешь отклонения в решениях учеников по справочнику ситуаций. Другой проверяющий уже проверил \
решения и перечислил, что не так (errors). Твоя задача — для КАЖДОГО элемента errors выбрать ситуацию из справочника.

СПРАВОЧНИК СИТУАЦИЙ (id — суть):
<<SITUATIONS>>

Если ни одна ситуация не подходит, используй общий код:
- G.calc — арифметическая ошибка / ошибка знака в шаге;
- G.method — неверный факт, формула, метод, неверное утверждение;
- G.cond — ученик неверно понял/переписал условие, решал не ту величину (если нет подходящего 1.x/3.4/12.4);
- G.lost — потеряны корни/серии/случаи/граница (если нет подходящего 6.5/6.9/7.1/7.2);
- G.proof — в доказательстве не выведено требуемое / не обоснован ключевой шаг (если нет подходящего 5.x);
- G.answer — итоговый ответ неверен, конкретный шаг не назван;
- G.none — это не отклонение: проверяющий сам пишет, что всё верно, или замечание без сути;
- other — отклонение есть, но ничего не подходит.

Правила:
- не решай задачу заново и не угадывай истинный вердикт — только классифицируй утверждения проверяющего;
- выбирай по смыслу ситуации; если объяснение говорит, что на решение/ответ не повлияло или дальше исправлено —
  выбирай ситуацию про описку/без влияния, если такая есть;
- если по объяснению похоже, что проверяющий неверно прочитал запись ученика (например, «переписал условие неверно»,
  а по смыслу решение верное) — всё равно размечай по утверждению, но поставь reading_doubt: true.

ЗАДАЧИ (mode: exam — формат экзамена, training — тренировка; для выбора ситуации не важен):
<<TASKS>>

Ответь JSON: {"tasks": [{"id": "…", "tags": [{"i": номер элемента errors с 0, "situation": "1.1 или G.calc …",
"reading_doubt": false}]}]} — по одному объекту на каждую задачу и по одному тегу на каждый элемент errors."""


def situations_text():
    rows = csv.DictReader(open(check.ROOT / "rules" / "policy.csv", encoding="utf-8"))
    return "\n".join(f"{r['id']} — {r['situation']}" for r in rows)


def load_keys():
    keys = {}
    for r in csv.DictReader(open(check.DATA / "keys" / "answer_key.csv", encoding="utf-8")):
        for w in r["works"].split():
            keys[(w, check.norm_task(r["task_no"]))] = r
    return keys


def build_items(src, keys, modes):
    """Задачи работы, которым нужен теггер (формат reports/policy_tag_input.json)."""
    out = {}
    for wd in sorted(p for p in (check.RUNS / src).iterdir() if (p / "check.json").exists()):
        data = json.load(open(wd / "check.json", encoding="utf-8"))
        if "parse_error" in data:
            continue
        conds = {}
        if (wd / "task.json").exists():
            for t in json.load(open(wd / "task.json", encoding="utf-8")).get("tasks", []):
                conds[check.norm_task(t.get("task_no", ""))] = t.get("text", "")
        items = []
        for r in data.get("results", []):
            if not r.get("errors"):
                continue
            k = check.norm_task(r.get("task_no", ""))
            items.append({"id": f"{wd.name}|{r.get('task_no')}", "mode": modes.get(wd.name, "training"),
                          "condition": conds.get(k, "")[:1500], "key": keys.get((wd.name, k), {}).get("answer", ""),
                          "student_answer": r.get("student_answer", ""), "verdict": r.get("verdict"),
                          "errors": [{x: e.get(x) for x in ("type", "step", "explanation")} for e in r["errors"]],
                          "comment": r.get("comment", "")})
        if items:
            out[wd.name] = items
    return out


def prompt_for(items):
    tasks = json.dumps([{x: it[x] for x in ("id", "mode", "condition", "key", "student_answer", "errors", "comment")}
                        for it in items], ensure_ascii=False, indent=0)
    return PROMPT.replace("<<SITUATIONS>>", situations_text()).replace("<<TASKS>>", tasks)


def estimate(p, n_err):
    tin, tout = len(p) / 2.6, 150 * n_err + 1500  # по факту 30.09: ~2,6 символа/токен; «думание» 0–4500, в среднем ~1500
    return (tin * PRICE_IN + tout * PRICE_OUT) / 1e6


def apply(src, tags):
    """v1 + теги → runs/<src>_tag/<работа>/check.json (вердикт по политике, формат v1 — понятен --report)."""
    from features import answers_match
    pol, modes, keys = policy.load_policy(), policy.load_modes(), load_keys()
    dst = check.RUNS / (src + "_tag")
    n = Counter()
    for wd in sorted(p for p in (check.RUNS / src).iterdir() if (p / "check.json").exists()):
        data = json.load(open(wd / "check.json", encoding="utf-8"))
        if "parse_error" in data:
            continue
        mode = modes.get(wd.name, "training")
        out = []
        for r in data.get("results", []):
            tid = f"{wd.name}|{r.get('task_no')}"
            r = dict(r)
            k = keys.get((wd.name, check.norm_task(r.get("task_no", ""))), {})
            if r.get("verdict") in ("correct", "incorrect"):
                tg = tags.get(tid)
                if tg is None and r.get("verdict") == "incorrect":
                    tg = [{"i": -1, "situation": "G.answer"}]
                if tg is not None:
                    m = answers_match(r.get("student_answer", ""), k.get("answer", ""), k.get("answer_type", "")) \
                        if k.get("answer") and r.get("student_answer") else None
                    v, cur, remarks = policy.decide(tg, mode, pol, m,
                                                    proof=k.get("answer_type") in ("proof", "construction"))
                    r.update(verdict="correct" if v == 1 else "incorrect", remarks=remarks, to_curator=cur,
                             tags=tg, verdict_v1=r.get("verdict"))
            r["mode"] = mode
            n[r["verdict"]] += 1
            out.append(r)
        (dst / wd.name).mkdir(parents=True, exist_ok=True)
        for f in wd.iterdir():  # транскрипция, условие, ответы — нужны features.py (разборчивость и т. п.)
            if f.is_file() and f.name != "check.json":
                shutil.copy2(f, dst / wd.name / f.name)
        check.save(dst / wd.name / "check.json", {"results": out, "_meta": dict(data.get("_meta", {}), tagger=MODEL)})
    print(f"{dst.name}: {dict(n)}")


def report(src):
    """Сравнение на одних задачах: v1; v1 + офлайн-теги (Claude, п. 20); v1 + теггер (Gemini). Каждый вариант —
    с doubt=keep (сомнение в прочтении → только к куратору) и doubt=remark (ещё и ошибка → замечание)."""
    from features import answers_match
    labels = check.load_labels()
    base = {(r["work"], r["task_no"]): r for r in check.evaluate(check.RUNS / src, labels) if r["label"] in (0, 1)}
    offline = {x["id"]: x["tags"] for x in json.load(open(check.REPORTS / "policy_tags.json", encoding="utf-8"))}
    tpath = check.RUNS / (src + "_tag") / "tags.json"
    gem = json.load(open(tpath, encoding="utf-8")) if tpath.exists() else {}
    pol, modes, keys = policy.load_policy(), policy.load_modes(), load_keys()

    def verdict(tags, r, w, doubt):
        v1 = r["pred"] if r["pred"] is not None else 0
        if tags is None:
            if r.get("verdict") != "incorrect":
                return v1
            tags = [{"i": -1, "situation": "G.answer"}]
        k = keys.get((w, check.norm_task(r["task_no"])), {})
        m = answers_match(r.get("student_answer") or "", k.get("answer", ""), k.get("answer_type", "")) \
            if k.get("answer") and r.get("student_answer") else None
        return policy.decide(tags, modes.get(w, "training"), pol, m, doubt,
                             proof=k.get("answer_type") in ("proof", "construction"))[0]

    res, diff, n_doubt = Counter(), [], 0
    for (w, t), r in sorted(base.items()):
        tid, lab = f"{w}|{t}", r["label"]
        v1 = r["pred"] if r["pred"] is not None else 0
        row = {"v1": v1}
        for name, src_tags in (("offline", offline), ("tagger", gem)):
            for doubt in ("keep", "remark"):
                row[f"{name}_{doubt}"] = verdict(src_tags.get(tid), r, w, doubt)
        n_doubt += any(x.get("reading_doubt") for x in gem.get(tid, []))
        for k_, v in row.items():
            res[k_] += v == lab
        res["+v1"] += v1 == lab and row["tagger_keep"] != lab
        res["+tag"] += row["tagger_keep"] == lab and v1 != lab
        if row["tagger_keep"] != lab or row["tagger_remark"] != lab:
            diff.append([w, t, lab, v1, row["offline_keep"], row["tagger_keep"], row["tagger_remark"],
                         " ".join(x["situation"] + ("?" if x.get("reading_doubt") else "") for x in gem.get(tid, [])),
                         " ".join(x["situation"] for x in offline.get(tid, []))])
    n = len(base)
    print(f"Задач {n} ({src}); размечено теггером {len(gem)}, из них с reading_doubt {n_doubt}")
    print(f"  v1: {res['v1'] / n:.1%}")
    for name, title in (("offline", "v1 + офлайн-теги (Claude)"), ("tagger", "v1 + теггер (Gemini)")):
        print(f"  {title:27} + политика: {res[name + '_keep'] / n:.1%};  с doubt→замечание: "
              f"{res[name + '_remark'] / n:.1%}")
    print(f"  парно теггер (keep) против v1: +v1 {res['+v1']}, +теггер {res['+tag']}")
    with open(check.REPORTS / "tagger_misses.csv", "w", encoding="utf-8", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["work", "task_no", "label", "v1", "offline", "tagger_keep", "tagger_remark", "tags_gemini",
                     "tags_offline"])
        w_.writerows(diff)
    print("  промахи теггера: reports/tagger_misses.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="google_gemini-3.8-flash@low_key_k2")
    ap.add_argument("--works", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--max-rub", type=float, default=20)
    ap.add_argument("--batch", type=int, default=10, help="задач в одном запросе (по целым работам)")
    a = ap.parse_args()
    if a.report:
        return report(a.src)
    dst = check.RUNS / (a.src + "_tag")
    out = dst / "tags.json"
    done = json.load(open(out, encoding="utf-8")) if out.exists() else {}
    work_items = build_items(a.src, load_keys(), policy.load_modes())
    if a.works:
        work_items = {w: v for w, v in work_items.items() if w in a.works}
    ok = {k for k, v in done.items() if not any(t.get("missing") for t in v)}  # оборванные/пустые — переспросить
    todo = {w: [it for it in v if it["id"] not in ok] for w, v in work_items.items()}
    todo = {w: v for w, v in todo.items() if v}
    batches, cur = [], []
    for w, v in todo.items():  # несколько работ в одном запросе (до ~10 задач): справочник платится реже
        cur += v
        if len(cur) >= a.batch:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)
    est = sum(estimate(prompt_for(b), sum(len(it["errors"]) for it in b)) for b in batches)
    print(f"Работ с отклонениями {len(work_items)}, задач {sum(map(len, work_items.values()))}; "
          f"к разметке {len(todo)} работ → {len(batches)} запросов; прикидка ≈{est:.1f} ₽ (лимит {a.max_rub} ₽)")
    if a.dry_run:
        return
    dst.mkdir(parents=True, exist_ok=True)
    client = check.Client(MODEL, a.max_rub, dst, False, {"tag": "minimal"})
    for items in batches:
        n_err = sum(len(it["errors"]) for it in items)
        w = "+".join(sorted({it["id"].split("|")[0] for it in items}))
        # Без повтора «с удвоенным лимитом»: оборванный ответ оплачен целиком, повтор платится ещё раз
        # (30.09 так сгорело 12,8 ₽ из 21,4). Большой лимит сразу — платим только за реально выданные токены.
        try:
            text, finish = client.chat([{"type": "text", "text": prompt_for(items)}], MAX_TOKENS, w, f"tag:{w}")
        except check.BudgetExceeded as e:
            print(f"\nОстановлено: {e}")
            break
        try:
            res = check.parse_json(text) if finish != "length" else {}
        except (ValueError, json.JSONDecodeError):
            res = {}
        if not res:
            print(f"\n  {w}: ответ оборван/не разобран ({finish}) — задачи останутся «к куратору», повтор при следующем запуске", end="")
        got = {str(t.get("id")): t.get("tags") or [] for t in (res.get("tasks") or []) if isinstance(t, dict)}
        for it in items:
            tg = [t for t in got.get(it["id"], []) if isinstance(t, dict) and t.get("situation")]
            covered = {t.get("i") for t in tg}
            # элемент errors без тега → other (безопасная сторона: ошибка + куратор)
            tg += [{"i": i, "situation": "other", "missing": True} for i in range(len(it["errors"])) if i not in covered]
            done[it["id"]] = tg
        check.save(out, done)
        print(".", end="", flush=True)
    print(f"\nПотрачено {client.spent:.2f} ₽")
    apply(a.src, done)
    print(f"Дальше: python3 tagger.py --src {a.src} --report")


if __name__ == "__main__":
    sys.exit(main())
