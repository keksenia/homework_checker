"""Карточка задания от репетитора → ключ ответов + список заданных задач.

Карточка — CSV в data/cards/ (образец: data/cards/_template.csv, описание: SUBMISSION_GUIDE.md):
  task_no     — номер ровно как в условии («85», «3.4», «13а»);
  works       — работы учеников с этим заданием через пробел;
  mode        — exam (формат экзамена, строго) | training (тренировка, лояльно) | criteria (по критериям ФИПИ);
  answer, answer_type — ответ (можно пусто) и тип: number | interval | set | series | pair | expression | text | proof;
  source      — откуда задача (ФИПИ 85, задачник, вариант); criteria — набор критериев (на будущее);
  comment     — пояснение к ответу.

  python3 card.py data/cards/<файл>.csv            # показать, что изменится
  python3 card.py data/cards/<файл>.csv --write    # записать в data/keys/answer_key.csv и <файл>_scope.csv
Scope-файл подаётся в проверку: python3 check.py … --scope-file data/cards/<файл>_scope.csv
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

from check import DATA, norm_task

MODES = {"exam", "training", "criteria", ""}
TYPES = {"number", "interval", "set", "series", "pair", "expression", "text", "proof", "construction", ""}
KEY = DATA / "keys" / "answer_key.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("card")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    card = Path(a.card)
    rows = list(csv.DictReader(open(card, encoding="utf-8")))
    bad = [f"строка {i + 2}: {msg}" for i, r in enumerate(rows) for msg in (
        [] if r.get("task_no", "").strip() else ["нет task_no"]) + (
        [] if r.get("works", "").strip() else ["нет works"]) + (
        [] if r.get("mode", "").strip() in MODES else [f"mode «{r['mode']}» — нужно exam / training / criteria"]) + (
        [] if r.get("answer_type", "").strip() in TYPES else [f"answer_type «{r['answer_type']}» неизвестен"])]
    if bad:
        sys.exit("Карточка с ошибками:\n  " + "\n  ".join(bad))
    key = list(csv.DictReader(open(KEY, encoding="utf-8")))
    fields = list(key[0].keys())
    index = {(r["task_key"], norm_task(r["task_no"])): i for i, r in enumerate(key)}
    new, upd = 0, 0
    for r in rows:
        works = r["works"].split()
        row = {"task_key": works[0], "task_no": r["task_no"].strip(), "works": " ".join(works),
               "answer": r.get("answer", "").strip(), "answer_type": r.get("answer_type", "").strip(),
               "method": "tutor", "confidence": "high" if r.get("answer", "").strip() else "",
               "comment": "; ".join(x for x in (r.get("source", "").strip(), r.get("comment", "").strip()) if x),
               "mode": r.get("mode", "").strip()}
        i = index.get((works[0], norm_task(row["task_no"])))
        if i is None:
            key.append(row)
            new += 1
        else:
            key[i] = {**key[i], **{k: v for k, v in row.items() if v}}
            upd += 1
    scope = card.with_name(card.stem + "_scope.csv")
    print(f"{card.name}: задач {len(rows)}; в ключе новых строк {new}, обновлённых {upd}; scope → {scope.name}")
    if not a.write:
        print("Это проверка. Записать: добавь --write")
        return
    shutil.copy2(KEY, DATA / "private" / "answer_key_before_card.csv")
    with open(KEY, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(key)
    with open(scope, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["work_id", "task_no"])
        w.writerows([wk, r["task_no"].strip()] for r in rows for wk in r["works"].split())
    print(f"Записано. Копия прежнего ключа: data/private/answer_key_before_card.csv")


if __name__ == "__main__":
    main()
