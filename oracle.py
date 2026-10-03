"""Разбор «какой слой сколько стоит» — бесплатно, по уже сохранённым прогонам (без запросов к модели).

Система с ключом на dev: runs/<base> (+ runs/<over> поверх для работ, где он есть — перепрогон с ключом).
1) Слой извлечения ответа кодом (answers.py): покрытие и согласие с ответом, который прочитала LLM.
2) Какая доля ошибок разметки видна по ответу (ответ ≠ ключ), а какая — только в ходе решения (ответ = ключ).
3) Гибрид «ответ сверяет код»: ответ ≠ ключ → ошибка, иначе вердикт LLM — против чистой LLM (парно).
Пишет reports/oracle_rows.csv — все задачи с полями для ручного разбора расхождений по слоям.

  python3 oracle.py --base google_gemini-3.8-flash@low_key --over google_gemini-3.8-flash@low_key_k2
"""
import argparse
import csv
import json
from collections import Counter

import check
from features import answers_match, load_keys

ANS_KINDS = ("number", "interval", "set", "pair", "")


def load_rows(run, labels):
    rows = check.evaluate(check.RUNS / run, labels)
    for r in rows:
        r["run"] = run
    return [r for r in rows if r["label"] in (0, 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="google_gemini-3.8-flash@low_key")
    ap.add_argument("--over", default="google_gemini-3.8-flash@low_key_k2")
    a = ap.parse_args()
    labels, keys = check.load_labels(), load_keys()
    rows = load_rows(a.base, labels)
    if a.over:
        over = load_rows(a.over, labels)
        ow = {r["work"] for r in over}
        rows = [r for r in rows if r["work"] not in ow] + over
    ans_cache = {}
    for r in rows:
        w, k = r["work"], check.norm_task(r["task_no"])
        if (r["run"], w) not in ans_cache:
            p = check.RUNS / r["run"] / w / "answers.json"
            ans_cache[(r["run"], w)] = {check.norm_task(t): v for t, v in json.load(open(p, encoding="utf-8")).items()} \
                if p.exists() else {}
        ca = ans_cache[(r["run"], w)].get(k, {"answer": "", "source": "no_file"})
        key = keys.get((w, k))
        kind = (key or {}).get("answer_type", "")
        r.update(code_answer=ca["answer"], code_source=ca["source"], key=(key or {}).get("answer", ""), kind=kind)
        r["m_llm"] = answers_match(r["student_answer"], r["key"], kind) if key and r["student_answer"] else None
        r["m_code"] = answers_match(ca["answer"], r["key"], kind) if key and ca["answer"] else None
        r["code_vs_llm"] = answers_match(ca["answer"], r["student_answer"], kind) \
            if ca["answer"] and r["student_answer"] else None

    n = len(rows)
    print(f"Задач с разметкой: {n} ({len({r['work'] for r in rows})} работ); с ключом: "
          f"{sum(bool(r['key']) for r in rows)}; ключ сравним числами: {sum(r['kind'] in ANS_KINDS[:-1] for r in rows)}")

    print("\n1. Извлечение ответа кодом")
    src = Counter(r["code_source"] for r in rows)
    for s, c in src.most_common():
        print(f"   {s:12} {c:5}  {c / n:.0%}")
    both = [r for r in rows if r["code_vs_llm"] is not None]
    agree = sum(r["code_vs_llm"] == 1 for r in both)
    print(f"   код и LLM прочитали одинаково: {agree} из {len(both)} сравнимых ({agree / max(len(both), 1):.0%})")
    for s in ("answer_line", "short"):
        b = [r for r in both if r["code_source"] == s]
        print(f"     {s}: {sum(r['code_vs_llm'] == 1 for r in b)} из {len(b)}")

    print("\n2. Видна ли ошибка по ответу (ответ ученика, прочитанный LLM, против ключа)")
    for lab, name in ((0, "ошибка по разметке"), (1, "верно по разметке")):
        g = [r for r in rows if r["label"] == lab]
        c = Counter({1: "ответ = ключ", 0: "ответ ≠ ключ", None: "сравнить нельзя"}[r["m_llm"]] for r in g)
        print(f"   {name:20} {len(g):5}: " + ", ".join(f"{k} {v} ({v / len(g):.0%})" for k, v in c.most_common()))

    print("\n3. Гибрид «ответ сверяет код» против чистой LLM (совпадение с разметкой)")
    def acc(pred):
        return sum(p == r["label"] for p, r in zip(pred, rows)) / n
    llm = [r["pred"] for r in rows]
    for name, m in (("ответ читает LLM", "m_llm"), ("ответ читает код", "m_code")):
        hyb = [0 if r[m] == 0 else r["pred"] for r in rows]
        plus_a = sum(p == r["label"] and h != r["label"] for p, h, r in zip(llm, hyb, rows))
        plus_b = sum(h == r["label"] and p != r["label"] for p, h, r in zip(llm, hyb, rows))
        print(f"   {name:17}: LLM {acc(llm):.1%} → гибрид {acc(hyb):.1%}  (+LLM {plus_a}, +гибрид {plus_b})")
        for r, h in zip(rows, hyb):
            r[f"hyb_{m}"] = h

    out = check.REPORTS / "oracle_rows.csv"
    cols = ["work", "run", "task_no", "label", "pred", "hyb_m_llm", "hyb_m_code", "student_answer", "code_answer",
            "code_source", "key", "kind", "m_llm", "m_code", "label_text", "model_errors"]
    with open(out, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nВсе задачи: {out}")


if __name__ == "__main__":
    main()
