#!/usr/bin/env python3
"""compare_tutor_keys.py — сверка нашего ключа (data/keys/answer_key.csv) с ключами репетитора (data/tutor_keys/).

    python3 compare_tutor_keys.py        # → reports/tutor_key_compare.csv + сводка

Ключи репетитора — ответы задачников Е. А. Ширяевой (PDF с текстовым слоем) и скриншоты ответов.
Сопоставление «домашка → раздел задачника» задано в MAPPING ниже (проверено по условиям домашек).
Ответы с дробями и корнями в текстовом слое PDF рассыпаются по строкам — такие помечаются
«нужна сверка по PDF» (garbled) и не входят в автоматическую статистику, пока не сверены глазами
(ручная сверка — в MANUAL).
"""
import csv
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import check
from features import answers_match

TK = check.DATA / "tutor_keys"
BOOKS = {"oge26": "shiryaeva_oge_2026_20-25_answers.pdf", "oge25": "shiryaeva_oge_2025_20-25_answers.pdf",
         "ege26": "shiryaeva_ege_prof_2026_01-12_answers.pdf"}

# task_key → (книга, раздел, правило разбора номера задачи); правило возвращает (блок, задание, пункт) или None
def _blk(no):  # «Б1-15» → (1, None, 15)
    m = re.fullmatch(r"Б(\d)-(\d+)", no)
    return (int(m.group(1)), None, int(m.group(2))) if m else None


def _plain(no):  # «15» → (1, None, 15)
    return (1, None, int(no)) if no.isdigit() else None


def _dot(no):  # «4.4» → задание 4, пункт 4 (блок 1)
    m = re.fullmatch(r"(\d+)\.(\d+)", no)
    return (1, int(m.group(1)), int(m.group(2))) if m else None


def _big(no):  # T25: номера ≥ 97 — из задачника (21, блок 1); 1–11 — вариант с сайта, ключа репетитора нет
    return (1, None, int(no)) if no.isdigit() and int(no) >= 97 else None


MAPPING = {
    # T11 (графики) — задания на соответствие из другого раздела задачника, ответов к нему нет
    "T1": ("ege26", "01", _plain),
    "T16": ("oge26", "20", _dot), "T17": ("oge26", "20", _dot), "T18": ("oge26", "20", _dot),
    "T19": ("oge26", "23", _plain), "T20": ("oge26", "23", _plain),
    "T24": ("oge26", "21", _blk), "T25": ("oge26", "21", _big),
}
# ключи со скриншотов (набраны вручную): task_key → {номер задачи: ответ}
SCREENSHOTS = {
    "M5": ("screenshot_2.png", {"1": "(1/8; 1/2) ∪ (8; 32)", "2": "(1/81; 1)", "3": "(0; 1) ∪ {8} ∪ (64; +∞)"}),
}
# ответы, сверенные глазами по PDF там, где текстовый слой испорчен (дроби, корни): (книга, раздел, блок, задание, пункт)
# (сверено по картинке страниц PDF 27.09.2026)
MANUAL = {
    ("oge26", "20", 1, 5, 3): "-1/3; 1", ("oge26", "20", 1, 5, 5): "-1/5; 1/2", ("oge26", "20", 1, 5, 11): "2; 13/4",
    ("oge26", "20", 1, 5, 13): "3/2; 15/7",
    ("oge26", "20", 1, 6, 2): "2-√3; 2+√3", ("oge26", "20", 1, 6, 4): "4-√7; 4+√7", ("oge26", "20", 1, 6, 6): "3-√5; 3+√5",
    ("oge26", "20", 1, 8, 1): "(4/3; 0); (1; -1)", ("oge26", "20", 1, 8, 4): "(11/5; 0); (1; -6)",
    ("oge26", "20", 1, 8, 8): "(5/4; 0); (2; 6)", ("oge26", "20", 1, 8, 11): "(-1; 2); (1; 2)",
    ("oge26", "20", 1, 8, 12): "(-1; 7); (1; 7)",
    ("oge26", "20", 1, 9, 1): "(2; -3); (2; 3)", ("oge26", "20", 1, 9, 4): "(3; -4); (3; 4)",
    ("oge26", "20", 1, 9, 7): "(2; -2); (2; 2)", ("oge26", "20", 1, 9, 8): "(3; -3); (3; 3)",
    ("oge26", "20", 1, 9, 12): "(3; 7)",
    ("oge26", "23", 1, None, 2): "50", ("oge26", "23", 1, None, 24): "14,4", ("oge26", "23", 1, None, 26): "16,8",
    ("oge26", "23", 1, None, 27): "120/13", ("oge26", "23", 1, None, 32): "240/13",
    ("oge26", "23", 1, None, 54): "8", ("oge26", "23", 1, None, 56): "14", ("oge26", "23", 1, None, 59): "11√3",
    ("oge26", "23", 1, None, 62): "20√6", ("oge26", "23", 1, None, 64): "18√3",
}


def parse_book(pdf):
    """→ {(раздел, блок, задание, пункт): (ответ, испорчен?)}."""
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
    lines = text.splitlines()
    out, sec, blk, zad = {}, None, None, None
    item_re = re.compile(r"(?:(?<=\s)|^)(\d{1,3})\)")
    for i, line in enumerate(lines):
        m = re.match(r"^\s*(\d{2})\.\s+[А-ЯA-Z]", line)
        if m:
            sec, blk, zad = m.group(1), None, None
            continue
        m = re.search(r"Блок\s+(\d)", line)
        if m and ")" not in line:
            blk, zad = int(m.group(1)), None
            continue
        m = re.match(r"^\s*Задание\s+(\d+)\.", line)
        if m:
            zad = int(m.group(1))
            continue
        marks = list(item_re.finditer(line))
        if not marks or sec is None:
            continue
        near = [lines[j] for j in (i - 1, i + 1) if 0 <= j < len(lines)]
        garbled_line = any(x.strip() and not item_re.search(x) and not re.search(r"Блок|Задание|^\s*[IVX]+\)", x)
                           and not re.match(r"^\s*(\d{2})\.\s", x) and "Ширяева" not in x for x in near)
        for k, mk in enumerate(marks):
            end = marks[k + 1].start() if k + 1 < len(marks) else len(line)
            ans = line[mk.end():end].strip().rstrip(";").strip()
            out[(sec, blk or 1, zad, int(mk.group(1)))] = (ans, garbled_line or ans == "")
    return out


def norm(a):
    a = a.replace("–", "-").replace("−", "-").replace("º", "").replace("°", "")
    a = re.sub(r"\s+и\s+", "; ", a)
    return a.strip()


def main():
    books = {b: parse_book(TK / f) for b, f in BOOKS.items() if (TK / f).exists()}
    wmap = {r["folder"]: r["work_id"] for r in csv.DictReader(open(check.DATA / "private" / "work_map.csv", encoding="utf-8"))}
    tk_works, splits = defaultdict(set), check.work_splits()
    for r in csv.DictReader(open(check.DATA / "splits_new.csv", encoding="utf-8")):
        if r["folder"] in wmap:
            tk_works[r["task_key"]].add(wmap[r["folder"]])
    key = list(csv.DictReader(open(check.DATA / "keys" / "answer_key.csv", encoding="utf-8")))
    rows = []
    for tkey in sorted(set(MAPPING) | set(SCREENSHOTS)):
        ws = tk_works.get(tkey, set())
        split = "/".join(sorted({splits.get(w, "dev") for w in ws}))
        for r in [r for r in key if set(r["works"].split()) & ws]:
            src, tutor, garbled = "", None, False
            if tkey in SCREENSHOTS:
                shot, ans = SCREENSHOTS[tkey]
                if r["task_no"] in ans:
                    src, tutor = shot, ans[r["task_no"]]
            else:
                book, sec, rule = MAPPING[tkey]
                pos = rule(r["task_no"])
                if pos:
                    k = (sec, pos[0], pos[1], pos[2])
                    src = f"{BOOKS[book]} {sec} Б{pos[0]}" + (f" зад.{pos[1]}" if pos[1] else "") + f" №{pos[2]}"
                    if (book,) + k in MANUAL:
                        tutor = MANUAL[(book,) + k]
                    elif k in books.get(book, {}):
                        tutor, garbled = books[book][k]
            if tutor is None:
                continue
            same = re.sub(r"\s+", "", norm(r["answer"])) == re.sub(r"\s+", "", norm(tutor))
            m = 1 if same else answers_match(r["answer"], norm(tutor), r["answer_type"]) if not garbled else None
            status = ("нужна сверка по PDF" if garbled else "совпало" if m == 1 else "РАСХОЖДЕНИЕ" if m == 0
                      else "не сравнить автоматически")
            rows.append({"task_key": tkey, "split": split, "task_no": r["task_no"], "our_answer": r["answer"],
                         "tutor_answer": tutor, "status": status, "source": src, "method": r["method"],
                         "comment": r["comment"][:80]})
    out = check.REPORTS / "tutor_key_compare.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print("Всего сопоставлено задач:", len(rows), dict(Counter(r["status"] for r in rows)))
    for tkey in sorted({r["task_key"] for r in rows}):
        c = Counter(r["status"] for r in rows if r["task_key"] == tkey)
        print(f"  {tkey}: {dict(c)}")
    for r in rows:
        if r["status"] != "совпало":
            print(f"  [{r['status']}] {r['task_key']} {r['task_no']}: наш «{r['our_answer']}» — репетитор «{r['tutor_answer']}» ({r['source']})")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
