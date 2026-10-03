"""Слой «перечитать ответ» — эксперимент (EXPERIMENTS.md, п. 19).

Когда прочитанный ответ ученика ≠ ключу, это либо настоящая ошибка, либо модель неверно прочитала почерк
(131 → 731, 2,5 → 225, «(2» → «[2»). Здесь модель ещё раз смотрит на фото страницы крупно и переписывает
ТОЛЬКО итоговый ответ задачи. Ключ ей не показываем, чтобы не подсказать ожидаемое.

Выборка (из reports/oracle_rows.csv, строит oracle.py):
  A — верно по разметке, но прочитанный ответ ≠ ключ (кандидаты в ошибки чтения) — все;
  B — ошибка по разметке и ответ ≠ ключ (настоящие ошибки) — случайные 60, seed 29.
Хорошо: в A перечитанный ответ = ключ; в B перечитанный ответ по-прежнему ≠ ключ.

  python3 reread.py --dry-run          # выборка и прикидка цены, без запросов
  python3 reread.py --max-rub 40       # запросы (результат в runs/google_gemini-3.8-flash@reread/)
  python3 reread.py --report           # итоги
"""
import argparse
import csv
import json
import random
import sys

from PIL import Image

import check
from features import answers_match, find_near, page_of, predictions

MODEL = "google/gemini-3.8-flash"
RUN = check.RUNS / "google_gemini-3.8-flash@reread"
PROMPT = """На фото — страница рукописного решения ученика по математике.
Найди задачу {task} (ученик мог подписать её так: {where}).
Перепиши ДОСЛОВНО итоговый ответ ученика к этой задаче: строку «Ответ»/«Отв», а если её нет — последний
результат решения этой задачи.
Правила:
- не решай задачу и не исправляй ученика — нужен ровно тот ответ, что написан, даже если он неверный;
- зачёркнутое не бери; если ответ исправлен — бери последний незачёркнутый вариант;
- пометки проверяющего (красная/цветная ручка) — не ответ ученика;
- внимательно различай похожие символы: 1/7, 3/8, 5/6, 0/6, запятую и точку, круглые и квадратные скобки,
  знак минус; если символ читается неоднозначно — укажи варианты в alternatives.
Ответь JSON: {{"found": true/false, "answer": "ответ как написан", "alternatives": ["…"],
"confidence": "high|medium|low"}}"""


def sample():
    rows = list(csv.DictReader(open(check.REPORTS / "oracle_rows.csv", encoding="utf-8")))
    a = [r for r in rows if r["label"] == "1" and r["m_llm"] == "0"]
    b = [r for r in rows if r["label"] == "0" and r["m_llm"] == "0"]
    random.Random(29).shuffle(b)
    return [dict(r, group="A") for r in a] + [dict(r, group="B") for r in b[:60]]


def locate(r):
    """Страницы, где искать задачу: по полю where_in_solution из проверки; не нашлось — все (не больше 3)."""
    d = check.RUNS / r["run"] / r["work"]
    pred = predictions(d / "check.json")
    k = check.norm_task(r["task_no"])
    p = pred.get(k) or find_near(pred, k) or {}
    where = p.get("where_in_solution", "")
    pages = json.load(open(d / "transcription.json", encoding="utf-8"))
    files = {pg.get("page"): pg.get("file") for pg in pages}
    n = page_of(where)
    chosen = [files[n]] if n in files else [pg.get("file") for pg in pages][:3]
    return where, [check.ANON / r["work"] / f for f in chosen if f]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--max-rub", type=float, default=40)
    a = ap.parse_args()
    items = sample()
    out = RUN / "reread.json"
    done = json.load(open(out, encoding="utf-8")) if out.exists() else {}
    if a.report:
        return report(items, done)
    calls = sum(len(locate(r)[1]) for r in items)
    print(f"Выборка: A {sum(r['group'] == 'A' for r in items)}, B {sum(r['group'] == 'B' for r in items)}; "
          f"запросов (страниц) {calls}, уже сделано {len(done)}; прикидка ≈{0.3 * calls:.0f} ₽")
    if a.dry_run:
        return
    RUN.mkdir(parents=True, exist_ok=True)
    client = check.Client(MODEL, a.max_rub, RUN, False, {"check": "minimal"})
    client.effort = {"reread": "minimal"}
    for r in items:
        key = f"{r['work']}|{r['task_no']}"
        if key in done:
            continue
        where, paths = locate(r)
        answers = []
        try:
            for pth in paths:
                content = [{"type": "text", "text": PROMPT.format(task=r["task_no"], where=where or "номер не известен")},
                           {"type": "image_url", "image_url": {"url": check.image_data_url(Image.open(pth), 2400)}}]
                res = check.ask_json(client, content, 700, r["work"], f"reread:{r['task_no']}:{pth.name}")
                answers.append({"file": pth.name, **{k: res.get(k) for k in ("found", "answer", "alternatives", "confidence")}})
                if res.get("found"):
                    break
        except check.BudgetExceeded as e:
            print(f"\nОстановлено: {e}")
            break
        done[key] = {"group": r["group"], "pages": answers}
        check.save(out, done)
        print(".", end="", flush=True)
    print(f"\nПотрачено {client.spent:.1f} ₽. Итоги: python3 reread.py --report")


def report(items, done):
    stats = {"A": [0, 0, 0, 0], "B": [0, 0, 0, 0]}  # [всего, = ключу, = ключу и high, не нашёл]
    lines = []
    for r in items:
        d = done.get(f"{r['work']}|{r['task_no']}")
        if not d:
            continue
        found = [p for p in d["pages"] if p.get("found")]
        ans = found[0] if found else {}
        m = answers_match(ans.get("answer") or "", r["key"], r["kind"]) if ans.get("answer") else None
        alt = any(answers_match(x or "", r["key"], r["kind"]) == 1 for x in (ans.get("alternatives") or []))
        s = stats[r["group"]]
        s[0] += 1
        s[1] += m == 1
        s[2] += m == 1 and ans.get("confidence") == "high"
        s[3] += not found
        lines.append([r["group"], r["work"], r["task_no"], r["student_answer"], ans.get("answer", ""),
                      "; ".join(ans.get("alternatives") or []), ans.get("confidence", ""), r["key"], m, int(alt)])
    for g, name, good in (("A", "A (верно, ответ прочитан ≠ ключу)", "перечитан = ключу — исправлено"),
                          ("B", "B (настоящие ошибки)", "перечитан = ключу — ВРЕД (ошибка стала бы «верно»)")):
        s = stats[g]
        if s[0]:
            print(f"{name}: {s[0]} задач; {good}: {s[1]} ({s[1] / s[0]:.0%}), из них с high {s[2]}; не нашёл задачу {s[3]}")
    with open(check.REPORTS / "reread.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "work", "task_no", "read_before", "reread", "alternatives", "confidence", "key", "match", "alt_match"])
        w.writerows(lines)
    print("Подробно: reports/reread.csv")


if __name__ == "__main__":
    sys.exit(main())
