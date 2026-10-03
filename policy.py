"""Слой «политика»: факты об отклонениях (id ситуации из rules/policy.csv) + режим задания → вердикт.

Детектор (LLM) сообщает, ЧТО не так в решении, и относит каждое отклонение к ситуации из rules/policy.csv или к
общему коду. Решение «ошибка / замечание / не упоминать» принимает этот код по режиму задания
(data/cards/modes.csv: exam — формат экзамена, training — тренировка/школа).

Общие коды: G.calc, G.method, G.cond, G.lost, G.proof, G.answer — ошибка в любом режиме; G.none — не отклонение;
other — ошибка + к куратору (неизвестная ситуация → безопасная сторона).

Офлайн-оценка (без запросов к модели) по разметке отклонений ответов v1 — reports/policy_tags.json:
  python3 policy.py --eval
Применить к прогону детектора (check.py --check-prompt v4 --tag det):
  python3 policy.py --apply google_gemini-3.8-flash@det [--doubt remark]
  → runs/<прогон>_pol/<работа>/check.json с вердиктами политики; дальше обычный python3 check.py --report
"""
import argparse
import csv
import json
import os
from collections import Counter

import check

GENERIC_ERROR = {"G.calc", "G.method", "G.cond", "G.lost", "G.proof", "G.answer"}
# отклонения, которые при consequence = fixed_later считаются опиской (не ошибкой по сути)
TYPO = {"3.1", "3.2", "3.5", "3.6"}
FIXABLE = {"G.calc", "G.cond", "G.method", "3.3", "3.4", "3.5", "3.6", "8.1", "8.2", "other"}


def load_policy(path=check.ROOT / "rules" / "policy.csv"):
    return {r["id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}


def load_modes(path=check.DATA / "cards" / "modes.csv"):
    return {r["work_id"]: r["mode"] for r in csv.DictReader(open(path, encoding="utf-8"))} if os.path.exists(path) else {}


def outcome(situation, mode, policy):
    """Исход одной ситуации: error | remark | ignore | curator (+ служебные blank / answer_only / as_solved)."""
    if situation in GENERIC_ERROR:
        return "error"
    if situation == "G.none":
        return "ignore"
    if situation == "other" or situation not in policy:
        return "other"
    return policy[situation]["exam" if mode == "exam" else "training"] or "error"


def decide(tags, mode, policy, answer_match=None, doubt="keep", proof=False):
    """tags: [{situation, reading_doubt}] → (verdict 0/1, к куратору?, замечания).
    answer_match — совпал ли ответ с ключом (для answer_only). doubt: keep — сомнение в прочтении не меняет исход,
    remark — отклонение с сомнением в прочтении не считается ошибкой (только к куратору)."""
    err, cur, remarks = False, False, []
    for t in tags:
        sit = t["situation"]
        o = outcome(sit, mode, policy)
        # согласование ситуации и факта consequence (v5): описка, которая испортила дальнейшее, — ошибка (как 3.3);
        # «описка повлекла ошибку» / «условие переписано неверно» без последствий — описка (3.1) + к куратору;
        # «не обоснован ключевой шаг» (5.1) вне задач на доказательство — пропущенные преобразования (4.10)
        cons = t.get("consequence")
        if sit in TYPO and cons == "propagated":
            sit, o = "3.3", outcome("3.3", mode, policy)
        elif sit in ("3.3", "3.4") and cons in ("no_effect", "fixed_later"):
            sit, o, cur = "3.1", outcome("3.1", mode, policy), True
        elif sit == "5.1" and cons is not None and not proof:
            sit, o = "4.10", outcome("4.10", mode, policy)
        # v5: дальше в решении используется верное значение → это описка (или ошибка распознавания), не ошибка по
        # сути: считаем как 3.1 «описка в промежуточной строке, дальше всё верно» и отправляем к куратору
        if t.get("consequence") == "fixed_later" and o in ("error", "other") and t["situation"] in FIXABLE:
            sit, o = "3.1", outcome("3.1", mode, policy)
            cur = True
        if t.get("reading_doubt"):
            cur = True
            if doubt == "remark" and o == "error":
                o = "remark"
        if o == "error":
            err = True
        elif o == "other":
            err, cur = True, True
        elif o == "curator":
            err, cur = True, True
        elif o == "answer_only":
            if answer_match == 0:
                err = True
        elif o == "remark":
            remarks.append(sit)
        # ignore / blank / as_solved — вердикт не меняют
    return (0 if err else 1), cur, remarks


def evaluate():
    from features import answers_match
    policy, modes = load_policy(), load_modes()
    tags = {x["id"]: x["tags"] for x in json.load(open(check.REPORTS / "policy_tags.json", encoding="utf-8"))}
    inp = {x["id"]: x for x in json.load(open(check.REPORTS / "policy_tag_input.json", encoding="utf-8"))}
    labels = check.load_labels()
    rows = []
    for run in ("google_gemini-3.8-flash@low_key", "google_gemini-3.8-flash@low_key_k2"):
        for r in check.evaluate(check.RUNS / run, labels):
            if r["label"] in (0, 1):
                rows.append(dict(r, run=run))
    k2 = {r["work"] for r in rows if r["run"].endswith("k2")}
    rows = [r for r in rows if r["run"].endswith("k2") or r["work"] not in k2]
    res = {"v1": [], "policy": [], "policy_doubt": []}
    by_mode = Counter()
    changed = []
    for r in rows:
        tid = f"{r['work']}|{r['task_no']}"
        mode = modes.get(r["work"], "training")
        v1 = r["pred"] if r["pred"] is not None else 0
        if tid in tags:
            x = inp[tid]
            m = answers_match(x["student_answer"], x["key"]) if x["key"] and x["student_answer"] else None
            p, _, _ = decide(tags[tid], mode, policy, m)
            pd, _, _ = decide(tags[tid], mode, policy, m, doubt="remark")
        else:
            p = pd = v1
        res["v1"].append(v1 == r["label"])
        res["policy"].append(p == r["label"])
        res["policy_doubt"].append(pd == r["label"])
        if p != v1:
            changed.append((tid, mode, r["label"], v1, p))
        by_mode[(mode, "n")] += 1
        by_mode[(mode, "v1")] += v1 == r["label"]
        by_mode[(mode, "pol")] += p == r["label"]
    n = len(rows)
    print(f"Задач: {n} (dev, итоговая система: low_key + k2 для работ test2)")
    for k, v in res.items():
        print(f"  {k:13} совпадение {sum(v) / n:.1%}")
    a = sum(x and not y for x, y in zip(res["v1"], res["policy"]))
    b = sum(y and not x for x, y in zip(res["v1"], res["policy"]))
    print(f"  парно policy против v1: +v1 {a}, +policy {b}")
    for mode in ("exam", "training"):
        m = by_mode[(mode, "n")]
        if m:
            print(f"  {mode:9} задач {m:5}: v1 {by_mode[(mode, 'v1')] / m:.1%} → policy {by_mode[(mode, 'pol')] / m:.1%}")
    with open(check.REPORTS / "policy_changes.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "mode", "label", "v1", "policy", "tags"])
        for tid, mode, lab, v1, p in changed:
            w.writerow([tid, mode, lab, v1, p, " ".join(t["situation"] + ("?" if t.get("reading_doubt") else "")
                                                         for t in tags.get(tid, []))])
    print(f"  изменённых вердиктов: {len(changed)} → reports/policy_changes.csv")


def apply_run(run, doubt="keep"):
    """Прогон детектора v4 → прогон с вердиктами политики (формат check.json v1, понятный --report)."""
    import hashlib
    from features import answers_match
    policy, modes = load_policy(), load_modes()
    keys = {}
    for r in csv.DictReader(open(check.DATA / "keys" / "answer_key.csv", encoding="utf-8")):
        for w in r["works"].split():
            keys[(w, check.norm_task(r["task_no"]))] = r
    src = check.RUNS / run
    dst = check.RUNS / (run + "_pol" + ("_doubt" if doubt == "remark" else ""))
    phash = hashlib.md5(open(check.ROOT / "rules" / "policy.csv", "rb").read()).hexdigest()[:8]
    n = Counter()
    for wd in sorted(p for p in src.iterdir() if (p / "check.json").exists()):
        data = json.load(open(wd / "check.json", encoding="utf-8"))
        if "parse_error" in data:
            continue
        mode = modes.get(wd.name, "training")
        out = []
        for r in data.get("results", []):
            st = r.get("status", "solved")
            devs = r.get("deviations", [])
            k = keys.get((wd.name, check.norm_task(r.get("task_no", ""))), {})
            m = answers_match(r.get("student_answer", ""), k.get("answer", ""), k.get("answer_type", "")) \
                if k.get("answer") and r.get("student_answer") else None
            if st in ("not_solved", "unclear"):
                verdict, cur, remarks = st, True, []
            else:
                v, cur, remarks = decide(devs, mode, policy, m, doubt, proof=k.get("answer_type") in ("proof", "construction"))
                verdict = "correct" if v == 1 else "incorrect"
            errs = [{"step": d.get("step"), "type": d.get("situation"), "explanation": d.get("explanation")}
                    for d in devs if outcome(d.get("situation", "other"), mode, policy) in ("error", "other", "curator")
                    and not (doubt == "remark" and d.get("reading_doubt"))]
            out.append({**{x: r.get(x) for x in ("task_no", "where_in_solution", "student_answer", "correct_answer",
                                                   "comment")},
                        "verdict": verdict, "errors": errs, "remarks": remarks, "to_curator": cur, "mode": mode,
                        "deviations": devs})
            n[verdict] += 1
        (dst / wd.name).mkdir(parents=True, exist_ok=True)
        meta = dict(data.get("_meta", {}), policy=phash, doubt=doubt)
        json.dump({"results": out, "_meta": meta}, open(dst / wd.name / "check.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print(f"{dst.name}: {dict(n)}; дальше: python3 check.py --report")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--apply", metavar="RUN")
    ap.add_argument("--doubt", choices=["keep", "remark"], default="keep")
    a = ap.parse_args()
    if a.eval:
        evaluate()
    elif a.apply:
        apply_run(a.apply, a.doubt)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
