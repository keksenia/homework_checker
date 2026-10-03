#!/usr/bin/env python3
"""label_sensitivity.py — насколько метрики системы зависят от правок разметки.

    python3 label_sensitivity.py --split dev      # → reports/label_sensitivity_dev.txt
    python3 label_sensitivity.py --split test     # после прогона test (TEST_PROTOCOL.md, анализ чувствительности)

Правки разметки делались там, где ключ или модель спорили с проверяющим, — это сдвигает метрики в пользу системы.
Скрипт считает совпадение и recall итоговой системы по разметке на каждом этапе правок (копии labels_before_*.csv).
"""
import argparse
import csv
import sys

import check

# по времени: аудит по ключу (26.09 утром) → перевод на новые правила (26.09 днём) → разбор промахов с Ксенией (27.09)
# → перепроверка 9 меток test по уточнениям 27.09 (28.09, до запуска test; dev не менялся)
STAGES = [("labels_before_audit.csv", "исходная (до аудита по ключу)"),
          ("labels_before_policy.csv", "после аудита, до новых правил"),
          ("labels_before_error_review.csv", "после новых правил, до разбора"),
          ("labels_before_test_recheck.csv", "после разбора промахов"),
          ("labels.csv", "текущая (+ перепроверка test)")]


def load(path):
    labels = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        e = labels.setdefault(r["work_id"], {}).setdefault(check.norm_task(r["task_no"]),
                                                           {"task_no": r["task_no"], "vals": [], "text": [""]})
        e["vals"].append(r["is_correct"])
    for w in labels.values():
        for e in w.values():
            e["label"] = 0 if "0" in e["vals"] else 1 if "1" in e["vals"] else None
            e["human_missed"] = False
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--run", default="google_gemini-3.8-flash@low_key")
    a = ap.parse_args()
    lines = [f"Чувствительность метрик {a.run} ({a.split}) к правкам разметки"]
    for name, what in STAGES:
        path = check.DATA / name
        if not path.exists():
            continue
        rows = [r for r in check.evaluate(check.RUNS / a.run, load(path)) if r["split"] == a.split and r["label"] in (0, 1)]
        if not rows:
            sys.exit(f"нет строк части {a.split} в прогоне {a.run}")
        err = [r for r in rows if r["label"] == 0]
        lines.append(f"  {what:32s} задач {len(rows)}, ошибок {len(err)}, "
                     f"совпадение {100 * sum(r['pred'] == r['label'] for r in rows) / len(rows):.1f}%, "
                     f"recall {100 * sum(r['pred'] == 0 for r in err) / max(len(err), 1):.1f}%")
    out = check.REPORTS / f"label_sensitivity_{a.split}.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines) + f"\n→ {out}")


if __name__ == "__main__":
    main()
