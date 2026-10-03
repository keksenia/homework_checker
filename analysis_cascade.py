#!/usr/bin/env python3
"""analysis_cascade.py — бесплатная симуляция каскада на dev по готовым прогонам (data/features_wide.csv).
Основная проверка — Gemini с ключом (g_low_key); второй голос — DeepSeek с фото без ключа (ds_high_img);
сигнал ключа — совпал ли ответ ученика с эталоном (key_vs_student).
Политики: какие задачи оставить «на автомате», а какие отдать куратору (считаем, что куратор решает верно)
или решить вторым голосом. → reports/cascade.txt"""
import csv
import random

rows = [r for r in csv.DictReader(open("data/features_wide.csv", encoding="utf-8")) if r["split"] == "dev"]
for r in rows:
    r["y"] = int(r["y_error"])
    r["g"] = int(r["g_low_key__pred_error"]) if r["g_low_key__pred_error"] in ("0", "1") else None
    r["d"] = int(r["ds_high_img__pred_error"]) if r["ds_high_img__pred_error"] in ("0", "1") else None
    r["k"] = r["key_vs_student"]  # '1' ответ = ключу, '0' ≠, '' — не сравнить


def evaluate(name, route):
    """route(r) → 'auto' (вердикт Gemini), 'ds' (вердикт DeepSeek), 'human' (куратор, верно)."""
    res = []
    for r in rows:
        how = route(r)
        pred = r["y"] if how == "human" or r["g"] is None else r["d"] if how == "ds" and r["d"] is not None else r["g"]
        res.append((r, how, pred))
    n = len(res)
    human = sum(h == "human" for _, h, _ in res)
    wrong = sum(p != r["y"] for r, _, p in res)
    missed = sum(p == 0 and r["y"] == 1 for r, _, p in res)
    false = sum(p == 1 and r["y"] == 0 for r, _, p in res)
    errs = sum(r["y"] for r, _, _ in res)
    by_work = {}
    for r, h, p in res:
        by_work.setdefault(r["work"], []).append(int(p != r["y"]))
    ws = list(by_work.values()); rnd = random.Random(0); bs = []
    for _ in range(2000):
        pick = [ws[rnd.randrange(len(ws))] for _ in ws]
        bs.append(sum(map(sum, pick)) / sum(map(len, pick)))
    bs.sort()
    line = (f"{name:58s} куратору {100*human/n:5.1f}%  итоговое совпадение {100*(1-wrong/n):5.1f}% "
            f"[{100*(1-bs[1949]):.1f}–{100*(1-bs[50]):.1f}]  пропущено ошибок {missed}/{errs}  ложных тревог {false}")
    print(line)
    return line


disagree = lambda r: r["d"] is not None and r["g"] != r["d"]
key_conflict = lambda r: (r["g"] == 1 and r["k"] == "1") or (r["g"] == 0 and r["k"] == "0")
out = [
    evaluate("0. только Gemini с ключом", lambda r: "auto"),
    evaluate("1. Gemini ≠ DeepSeek → куратору", lambda r: "human" if disagree(r) else "auto"),
    evaluate("2. вердикт Gemini спорит с ключом → куратору", lambda r: "human" if key_conflict(r) else "auto"),
    evaluate("3. (1) или (2) → куратору", lambda r: "human" if disagree(r) or key_conflict(r) else "auto"),
    evaluate("4. спор с ключом → решает DeepSeek (без человека)", lambda r: "ds" if key_conflict(r) else "auto"),
    evaluate("5. спор с ключом И Gemini ≠ DeepSeek → куратору", lambda r: "human" if key_conflict(r) and disagree(r) else "auto"),
    evaluate("6. все «ошибки» Gemini → куратору", lambda r: "human" if r["g"] == 1 else "auto"),
]
open("reports/cascade.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
