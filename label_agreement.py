#!/usr/bin/env python3
"""label_agreement.py — независимая переразметка вслепую для оценки качества разметки (межэкспертное согласие).

    python3 label_agreement.py --make      # создать reports/blind_relabel.csv (задачи без меток) для разметки вслепую
    python3 label_agreement.py             # после заполнения колонки is_correct: согласие и каппа Коэна с labels.csv

Зачем: все метрики системы считаются против разметки. Если разметка шумная, «ошибки системы» частично — ошибки
разметки. Прежняя оценка (каппа 0,98) сравнивала две разметки одной и той же модели — это верхняя граница.
Честное число даёт человек, который размечает те же задачи заново, не глядя на старые метки и на вердикты модели.
Правила — LABELING_RULES.md; is_correct: 1 — верно, 0 — ошибка, пусто — не решено/брошено.
"""
import argparse
import csv
import random
import sys

import check

OUT = check.REPORTS / "blind_relabel.csv"


def make(n_tasks, seed):
    labels, splits = check.load_labels(), check.work_splits()
    pilot = {"W017", "W022", "W024", "W025", "W035", "W041", "W053", "W062", "W069", "W074"}  # их разбирали чаще всего
    dev = sorted(w for w in labels if splits.get(w, "dev") == "dev" and w not in pilot)
    size = {w: sum(e["label"] in (0, 1) for e in labels[w].values()) for w in dev}
    works = []
    for w in random.Random(seed).sample(dev, len(dev)):  # случайные работы до ~n_tasks задач, без гигантских
        if size[w] <= 25 and sum(size[x] for x in works) < n_tasks:
            works.append(w)
    works.sort()
    rows = [{"work_id": w, "task_no": e["task_no"], "is_correct": "", "comment": ""}
            for w in works for k, e in sorted(labels[w].items(), key=lambda kv: check.natural_key(kv[1]["task_no"]))
            if e["label"] in (0, 1)]
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} задач из работ {', '.join(works)} → {OUT}")
    print("Размечай по фото в data/anonymized/<работа>/solution_*.jpg (условие — task_*), НЕ открывая labels.csv и отчёты.")


def compare():
    labels = check.load_labels()
    pairs = []
    for r in csv.DictReader(open(OUT, encoding="utf-8")):
        if r["is_correct"] in ("0", "1"):
            e = labels.get(r["work_id"], {}).get(check.norm_task(r["task_no"]))
            if e and e["label"] in (0, 1):
                pairs.append((int(r["is_correct"]), e["label"], r))
    if not pairs:
        sys.exit("в reports/blind_relabel.csv нет заполненных is_correct")
    n = len(pairs)
    po = sum(a == b for a, b, _ in pairs) / n
    pa, pb = sum(a for a, _, _ in pairs) / n, sum(b for _, b, _ in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"задач: {n}; согласие {po:.1%}; каппа Коэна {kappa:.3f}")
    for a, b, r in pairs:
        if a != b:
            print(f"  расхождение {r['work_id']} {r['task_no']}: вслепую {a}, в labels.csv {b}  {r['comment']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--tasks", type=int, default=60, help="сколько задач примерно набрать")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    make(a.tasks, a.seed) if a.make else compare()
