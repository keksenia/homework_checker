"""Слой «извлечение ответа» — только код, без запросов к модели (бесплатно).

Берёт готовую транскрипцию (runs/<прогон>/<работа>/transcription.json), делит строки на блоки по номерам задач
(«1)», «№5», «13а)», «Задание 3», «а)» внутри задачи) и достаёт из блока итоговый ответ:
  answer_line — строка «Ответ: …» (надёжно);
  short       — короткий блок «5) 0,33» (задачи с кратким ответом);
  none        — не нашёл → нужен LLM (или куратор).
Пишет runs/<прогон>/<работа>/answers.json. Проверка качества — python3 oracle.py.

  python3 answers.py --run google_gemini-3.8-flash@low_key_k2
"""
import argparse
import json
import re

from check import RUNS, load_labels, norm_task, task_hit

LABEL = re.compile(r"^\s*~*\s*(?:(?:№|N|задание|задача)\s*)?(\d+(?:\.\d+)*)\s*([а-еa-e])?\s*[).:]\s*(.*)$", re.I)
LABEL_NO = re.compile(r"^\s*~*\s*(?:№|N|задание|задача)\s*(\d+(?:\.\d+)*)\s*([а-еa-e])?\b\s*(.*)$", re.I)
SUB = re.compile(r"^\s*([а-е])\s*\)\s*(.*)$", re.I)
ANSWER = re.compile(r"(?:^|[\s(])отв(?:ет)?(?:\s*[:.\-—]\s*|\s+(?=[-−\d(\[x{∅\\]))(.*)$", re.I)
STRUCK = re.compile(r"~~.*?~~")


def clean(line):
    return STRUCK.sub("", str(line)).strip()


ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
SECTION = re.compile(r"^\s*\(?\s*([ivx]+)\s*\)?\s*$", re.I)          # «(I)», «II»
SECTION_TASK = re.compile(r"^\s*(\d+)\s+(\d+)\s*\)\s*(.*)$")           # «2 1) …» — раздел 2, задача 1


def blocks(pages):
    """[(номера-кандидаты, [строки])] в порядке страниц. Номер задачи внутри раздела даёт два кандидата:
    «5» и «4.5» (раздел 4). Строки до первого номера — блок без номера."""
    out, cur, parent, section, base = [], [[], []], None, None, []
    for p in sorted(pages, key=lambda p: p.get("page", 0)):
        for raw in p.get("lines", []):
            line = clean(raw)
            if not line:
                continue
            sec, st = SECTION.match(line), SECTION_TASK.match(line)
            m = LABEL.match(line) or LABEL_NO.match(line)
            s = SUB.match(line)
            if sec and sec.group(1).lower() in ROMAN:
                section = str(ROMAN[sec.group(1).lower()])
                continue
            if st:
                section, parent = st.group(1), st.group(2)
                base = [parent, f"{section}.{parent}"]
                out.append(cur)
                cur = [list(base), [st.group(3)] if st.group(3) else []]
            elif m and not re.fullmatch(r"\d+(?:[.,]\d+)?", line):  # «81.» — не номер, а число
                out.append(cur)
                parent = m.group(1)
                sub = (m.group(2) or "").lower()
                base = [parent] + ([f"{section}.{parent}"] if section and "." not in parent else [])
                cur = [[b + sub for b in base], [m.group(3)] if m.group(3) else []]
            elif s and parent:
                out.append(cur)
                cur = [[b + s.group(1).lower() for b in base], [s.group(2)] if s.group(2) else []]
            else:
                cur[1].append(line)
    out.append(cur)
    return [b for b in out if b[0] or b[1]]


def tidy(a):
    a = re.sub(r"\\(?:underline|boxed|text|mathbf)\{([^{}]*)\}", r"\1", a)
    a = re.split(r"\\qquad|\\quad|\|", a)[0]
    a = a.replace("^\\circ", "").replace("°", "").strip().rstrip(".").strip()
    return a


def answer_of(lines):
    for i in range(len(lines) - 1, -1, -1):
        ms = list(ANSWER.finditer(lines[i]))
        if ms:
            a = ms[-1].group(1).strip() or (lines[i + 1].strip() if i + 1 < len(lines) else "")
            if a:
                return tidy(a), "answer_line"
    content = [l for l in lines if l.strip()]
    if 1 <= len(content) <= 2 and len(content[-1]) <= 40 and "|" not in content[-1]:
        return tidy(content[-1].split("=")[-1]), "short"
    return "", "none"


def extract(pages, scope):
    bl = blocks(pages)
    res = {}
    for s in scope:
        k = norm_task(s)
        hit = [b for b in bl if k in {norm_task(c) for c in b[0]}]
        if not hit:  # задача «13» размечена целиком, а у ученика пункты «13а», «13б» — склеиваем
            hit = [b for b in bl if any(norm_task(c).startswith(k) and not norm_task(c)[len(k):][:1].isdigit()
                                        and norm_task(c) != k for c in b[0])]
        if len(hit) != 1 and not (hit and all(norm_task(c) != k for b in hit for c in b[0])):
            res[s] = {"answer": "", "source": "no_block" if not hit else "ambiguous"}
            continue
        a, src = answer_of([l for b in hit for l in b[1]])
        res[s] = {"answer": a, "source": src}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    labels = load_labels()
    run = RUNS / a.run
    n = 0
    for w, tasks in labels.items():
        tj = run / w / "transcription.json"
        if not tj.exists():
            continue
        scope = [e["task_no"] for e in tasks.values() if e["label"] in (0, 1)]
        res = extract(json.load(open(tj, encoding="utf-8")), scope)
        json.dump(res, open(run / w / "answers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        n += 1
    print(f"answers.json записан для {n} работ в {run}")


if __name__ == "__main__":
    main()
