#!/usr/bin/env python3
"""features_wide.py — «широкая» таблица для модели доверия: одна строка = одна размеченная задача,
рядом голоса всех прогонов (Gemini без ключа, с ключом, другие уровни, DeepSeek, повторы).

    python3 features.py          # сначала обновить data/features.csv (строка = задача × прогон)
    python3 features_wide.py     # → data/features_wide.csv

Колонки:
  служебные (НЕ признаки): work, task_no, student_id, task_key, split, key_answer
  цели (НЕ признаки):      label (1 верно / 0 ошибка), y_error = 1 − label, human_missed,
                           <прогон>__wrong (ошибся ли вердикт этого прогона)
  общие признаки:          тема (exam, exam_task, topic, subtopic, has_figure), key_type, key_confidence,
                           разборчивость и объём распознавания (из основного прогона)
  по прогонам:             <прогон>__pred_error, __n_err_substantive, __key_vs_student, __key_vs_model_answer,
                           __frac_incorrect, __vote_entropy, __reasoning_per_task
  согласие:                n_voters, votes_error_share, voters_disagree, key_vs_student (ответ ученика, как его
                           выписала итоговая система, против ключа), model_vs_key_disagree

Группы для кросс-валидации: task_key (одно задание целиком в одном фолде) или student_id.
Test (split = test) не используем ни для обучения, ни для выбора модели — только в самом конце.
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

PER_RUN = ["pred_error", "n_err_substantive", "n_err_confirmed", "n_err_contradicted", "key_vs_student", "key_vs_model_answer",
           "frac_incorrect", "vote_entropy", "reasoning_per_task", "student_answer",
           "key_in_work", "segment_found"]  # ответ ученика — в прочтении проверки
COMMON = ["legibility_mean", "legibility_min", "n_pages", "unclear_marks_work", "strikes_work",
          "unclear_marks_page", "task_page_known", "key_type", "key_confidence", "key_answer"]
OCR_DOUBLE = ["ocr_disagree_work", "ocr_disagree_page"]  # берутся из любого прогона с двойным распознаванием


def short(run):
    """google_gemini-3.8-flash@low_key → g_low_key; deepseek_deepseek-v4.1-flash@high_key → ds_high_key."""
    base, _, tag = run.partition("@")
    pref = "g" if "gemini" in base else "ds" if "deepseek" in base else re.sub(r"\W+", "_", base)[:12]
    return f"{pref}_{tag or 'base'}"


def lead_num(task_no):
    m = re.search(r"(\d+)", str(task_no).split("-")[-1])
    return int(m.group(1)) if m else None


def load_topics():
    path = DATA / "topics.csv"
    topics = defaultdict(list)
    if path.exists():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            topics[r["task_key"]].append(r)
    return topics


def topic_for(topics, task_key, task_no):
    """Строка тем: префикс пустой — всё задание; 'a..b' — диапазон ведущего номера; число — ровно этот номер;
    иначе task_no начинается с префикса."""
    best = None
    for r in topics.get(task_key, []):
        p = r["task_prefix"]
        if not p:
            ok = True
        elif ".." in p:
            a, b = (int(x) for x in p.split(".."))
            n = lead_num(task_no)
            ok = n is not None and a <= n <= b
        elif p.isdigit():
            ok = lead_num(task_no) == int(p)
        else:
            ok = str(task_no).startswith(p)
        if ok and (best is None or len(p) > len(best["task_prefix"])):
            best = r
    return best or {}


def to_num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default=str(DATA / "features.csv"))
    ap.add_argument("--out", default=str(DATA / "features_wide.csv"))
    ap.add_argument("--runs", default="g_low_key,ds_high_img",
                    help="какие прогоны брать в таблицу (короткие имена через запятую). По умолчанию — только то, что "
                         "работает в итоговой системе: Gemini с ключом и DeepSeek как второй голос. Признаки прогонов, "
                         "которых в системе нет (например, g_low), дали бы расхождение обучения и применения "
                         "(train/serve skew). Пусто — все прогоны с покрытием ≥ --min-coverage (для анализа).")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="при пустом --runs: брать прогоны, покрывающие не меньше этой доли задач")
    ap.add_argument("--independent-key-run", default="g_low_key",
                    help="прогон, из ответа которого берётся «ответ ученика ≠ ключу» (по умолчанию — итоговая система)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.features, encoding="utf-8")))
    topics = load_topics()
    by_task = defaultdict(dict)
    for r in rows:
        by_task[(r["work"], r["task_no"])][short(r["run"])] = r
    n_tasks = len(by_task)
    coverage = Counter(run for runs in by_task.values() for run in runs)
    if args.runs:
        runs = [r.strip() for r in args.runs.split(",") if r.strip()]
        missing = [r for r in runs if r not in coverage]
        if missing:
            raise SystemExit(f"нет прогонов {missing}; есть: {sorted(coverage)}")
    else:
        runs = sorted(r for r, c in coverage.items() if c >= args.min_coverage * n_tasks)
    dropped = sorted(set(coverage) - set(runs))

    out, problems = [], []
    for (work, task_no), rr in sorted(by_task.items()):
        any_r = next(iter(rr.values()))
        labels = {r["label"] for r in rr.values()}
        if len(labels) > 1:
            problems.append(f"{work} {task_no}: разные label в прогонах {labels}")
        primary = rr.get(args.independent_key_run) or any_r  # общие признаки (почерк, страницы) — из итоговой системы
        t = topic_for(topics, any_r["task_key"], task_no)
        row = {"work": work, "task_no": task_no, "student_id": any_r["student_id"], "task_key": any_r["task_key"],
               "split": any_r["split"], "label": any_r["label"], "y_error": 1 - int(any_r["label"]) if any_r["label"] in ("0", "1") else "",
               "human_missed": any_r["human_missed"],
               "exam": t.get("exam", ""), "exam_task": t.get("exam_task", ""), "topic": t.get("topic", ""),
               "subtopic": t.get("subtopic", ""), "has_figure": t.get("has_figure", "")}
        for c in COMMON:
            row[c] = primary.get(c, "")
        dbl = next((r for r in rr.values() if r.get("ocr_double") == "1"), None)
        for c in OCR_DOUBLE:
            row[c] = dbl.get(c, "") if dbl else ""
        votes = []
        for run in runs:
            r = rr.get(run)
            for c in PER_RUN:
                row[f"{run}__{c}"] = r.get(c, "") if r else ""
            row[f"{run}__wrong"] = r.get("model_wrong", "") if r else ""
            if r and r.get("pred_error") in ("0", "1"):
                votes.append(int(r["pred_error"]))
        ind = rr.get(args.independent_key_run)
        kvs = ind.get("key_vs_student", "") if ind else ""
        row["n_voters"] = len(votes)
        row["votes_error_share"] = sum(votes) / len(votes) if votes else ""
        row["voters_disagree"] = int(0 < sum(votes) < len(votes)) if votes else ""
        row["key_vs_student"] = kvs
        pe = ind.get("pred_error") if ind else ""
        row["model_vs_key_disagree"] = (int((pe == "1") == (kvs == "1")) if pe in ("0", "1") and kvs in ("0", "1")
                                        else "")
        out.append(row)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"{len(out)} задач → {args.out}")
    print("прогоны в таблице:", ", ".join(f"{r} ({coverage[r]})" for r in runs))
    if dropped:
        why = "не входят в итоговую систему, см. --runs" if args.runs else "мало задач, см. --min-coverage"
        print(f"не вошли ({why}):", ", ".join(f"{r} ({coverage[r]})" for r in dropped))
    sp = Counter(r["split"] for r in out)
    print("по частям:", dict(sp), "| ошибок в разметке:", sum(r["y_error"] for r in out if r["y_error"] != ""))
    no_topic = sum(1 for r in out if not r["topic"])
    if no_topic:
        print(f"ВНИМАНИЕ: без темы {no_topic} задач — проверь data/topics.csv")
    if problems:
        print("ВНИМАНИЕ, несогласованная разметка между прогонами:", *problems[:10], sep="\n  ")


if __name__ == "__main__":
    main()
