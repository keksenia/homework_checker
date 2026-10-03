#!/usr/bin/env python3
"""HomeworkChecker — прототип проверки рукописных домашних работ по математике.

Шаги для каждой работы (результаты кэшируются в runs/<модель>/<работа>/):
  1. transcribe — каждая страница решения → дословная транскрипция (LaTeX, JSON);
  2. task       — условие (фото или PDF) → список задач с описанием рисунков;
  3. check      — условие + транскрипция → вердикт по каждой задаче.
Отчёт (--report) сравнивает вердикты с data/labels.csv.

Модели видят только solution и task — никогда markup/feedback.

Примеры:
  python3 check.py --model google/gemini-3.8-flash --works W003
  python3 check.py --model google/gemini-3.8-flash --works all --max-rub 300
  python3 check.py --report
  python3 check.py --works W003 W017 --effort-check low --tag low --steps check   # A/B по «думанию» на готовых транскрипциях

Экономия: транскрипция и условие идут с минимальным «думанием» (--effort-ocr minimal);
условие одного и того же файла распознаётся один раз на модель (кэш runs/<модель>/_tasks/);
в проверку отправляются только нужные задачи условия, а не весь задачник.
Ключ и адрес API берутся из .env (LLM_API_KEY, LLM_BASE_URL).
"""
import argparse
import base64
import csv
import hashlib
import http.client
import socket
import shutil
import io
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("Нужен Pillow: pip3 install pillow")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ANON = DATA / "anonymized"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"

# ----------------------------------------------------------------- промпты

NUMBERING = (
    "вариант контрольной и номер: «Вариант 1, №2, пункт 1)» → В1-2.1, «Вариант 2, №4» → В2-4; "
    "блок задачника: «Блок 2, задание 5» → Б2-5; подпункт k) задания n → n.k (например 4.3); "
    "буквенные подпункты — 13а, 13б, Б1-13А; номера из задачника с точкой (25.1, 14.6) — как есть, "
    "пункты а)/б) к ним — 25.1а, 25.1б"
)

TRANSCRIBE_PROMPT = """Ты — точный транскрибатор рукописных решений по математике. Перед тобой одна страница тетради ученика.
Перепиши ДОСЛОВНО всё, что написано, строка за строкой; формулы — в LaTeX.
Правила:
1. Ничего не исправляй. Ошибки ученика (арифметика, знаки, скобки, описки, неверные ответы) сохраняй в точности: если написано 6·7=56 — пиши 6·7=56.
2. Не дописывай пропущенное и не додумывай по смыслу. Неразборчивое — [?], два варианта прочтения — [?a|b].
3. Зачёркнутое пиши как ~~...~~.
4. Номера задач и пунктов — как написаны в тетради («3)», «№4», «(14)»), не перенумеровывай.
5. Рисунки (числовая прямая со знаками и стрелками, графики, чертежи, векторы) опиши в поле drawings: что изображено, подписи, знаки на промежутках.
Ответ — только JSON:
{"lines": ["..."], "drawings": ["..."], "task_labels": ["номера задач на странице, как написаны"], "legibility": "good|medium|poor"}"""

TRANSCRIBE_PROMPT_V2 = """Ты — сканер рукописных решений по математике. Перед тобой одна страница тетради ученика.
Перепиши страницу строка за строкой; формулы — в LaTeX (\\frac, \\sqrt, ^, системы — \\begin{cases}), как обычно.
Твоя задача — зафиксировать, что НАПИСАНО, а не что ДОЛЖНО быть написано. В тетрадях много ошибок учеников, и сохранить их в точности — самое важное: по твоему тексту потом ищут эти ошибки.
Правила:
1. Каждое число, знак, скобку и символ переписывай ровно так, как на фото, даже если выражение неверно. Написано 44+44=38 — пиши 44+44=38. Написано x<0, хотя по смыслу должно быть x≤0, — пиши x<0. Написано (−∞; 4), хотя из решения следует 6, — пиши (−∞; 4). НИКОГДА не пересчитывай, не сверяй арифметику и не «достраивай» по смыслу.
2. Похожие символы (1 и 7, 4 и 9, 3 и 8, 0 и 6, 5 и 3, < и ≤, ( и [, − и =, знак № и цифра 1) определяй по начертанию, а не по математике. Если не уверена(н) — пиши [?a|b] с двумя вариантами; совсем неразборчиво — [?].
3. ~~зачёркнутое~~ — только если символ реально перечёркнут линией. Подчёркивание, дуги, стрелки, рамки — не зачёркивание. Если новая цифра написана поверх старой, пиши новую и добавь в corrections: «строка N: 8 исправлено на 5».
4. Номера задач переписывай как есть («№14», «N14» — это номер 14, буква N, а не цифра 1). Не перенумеровывай.
5. Пометки другим цветом (красные, розовые, фиолетовые — это проверяющий) не переписывай; пиши только то, что написал ученик.
6. Рисунки (числовая прямая со знаками, графики, чертежи) опиши в drawings: что изображено, подписи, знаки на промежутках.
Ответ — только JSON:
{"lines": ["..."], "drawings": ["..."], "task_labels": ["номера задач на странице, как написаны"], "corrections": ["..."], "uncertain": ["места, где прочтение сомнительно"], "legibility": "good|medium|poor"}"""

# v3 = v1 (формат, который лучше всего работает на проверке) + правила против «автоисправления»,
# справочник рукописных начертаний (собран по реальным ошибкам распознавания из reports/error_analysis_low_key.csv)
# и правила для зачёркиваний и надписей над выражениями
TRANSCRIBE_PROMPT_V3 = TRANSCRIBE_PROMPT.replace(
    "Ответ — только JSON:",
    """ГЛАВНОЕ: ты фиксируешь, что НАПИСАНО, а не что ДОЛЖНО быть. По твоему тексту потом ищут ошибки ученика, поэтому
любое «исправление» — это пропущенная ошибка. Не пересчитывай арифметику и не сверяй со смыслом: 35 + 6 = 31 пиши как
35 + 6 = 31; x < 0 пиши как x < 0, даже если в ответе стоит ] ; знак минуса ставь, только если он виден.
Каждую цифру определяй по начертанию, а не по тому, какое число «подходит». Рукописные цифры в этих тетрадях:
- 7 часто пишут с горизонтальной перекладиной посередине; такая 7 — не 1, не 4 и не «13»;
- 1 бывает с длинным наклонным «флажком» сверху и похожа на 7 без перекладины — нет перекладины, значит 1;
- 3 бывает с плоским верхом (как З или z) и похожа на 7 — у 3 внизу округлая петля;
- 2 бывает с длинным хвостом внизу — не 1; 9 с маленькой петлёй — не 3; 8 — не 3; 4 открытая сверху — не 9;
- длинная косая запятая перед цифрой («6,2») — это запятая, а не цифра 1.
Если две цифры одинаково вероятны — пиши [?a|b], а не угадывай по смыслу.
Мелкие числа, надписанные над скобками или выражением (промежуточные результаты), пиши отдельно в квадратных скобках
после выражения: (8·4) + (−9·3) [над скобками: 32, −27] — это не степени.
Зачёркнутое ~~...~~ — только то, что реально перечёркнуто. Если ученик зачеркнул число и написал рядом или сверху
другое, пиши оба: x = ~~5~~ 3 — новое значение НЕ зачёркнуто. Не распространяй зачёркивание на соседние строки.
Ответ — только JSON:""")
OCR_PROMPTS = {"v1": TRANSCRIBE_PROMPT, "v2": TRANSCRIBE_PROMPT_V2, "v3": TRANSCRIBE_PROMPT_V3}

TASK_PROMPT = """Перед тобой условие домашнего задания по математике (страницы задачника или фото листа). Перепиши все задачи дословно, формулы — в LaTeX.
- Номер задачи сохраняй полностью, в такой нумерации: {numbering}.
- Если к задаче есть рисунок, опиши его так, чтобы задачу можно было решить без картинки: координаты начала и конца векторов на сетке, ключевые точки графика (пересечения с осями, экстремумы, промежутки возрастания/убывания), для касательной — две точки с целыми координатами, все данные чертежа.
- Опечатки и задвоения в условии не исправляй, переписывай как есть.
{extra}
Ответ — только JSON: {{"tasks": [{{"task_no": "...", "text": "...", "figure": "описание рисунка или пустая строка"}}]}}"""

CHECK_PROMPT = """Ты — опытный проверяющий домашних работ по математике (подготовка к ОГЭ/ЕГЭ). Тебе даны условие (транскрипция) и дословная транскрипция решения ученика по страницам: формулы в LaTeX, зачёркнутое в ~~...~~, неразборчивое [?].

Для каждой задачи:
1. Найди её решение в транскрипции ПО СОДЕРЖАНИЮ, а не по номеру: ученики часто подписывают задачи иначе, чем в условии (пункт 4.3 может быть подписан «3)»).
2. Реши задачу сам и получи верный ответ.
3. Сверь, ЧТО СПРАШИВАЮТ в условии, с тем, что нашёл ученик (точка максимума или значение, наибольшее на отрезке, целые решения или промежуток и т. п.).
4. Пройди решение ученика по шагам и найди все ошибки.

Вердикт (ошибка — это только ошибка в рассуждениях или вычислениях):
- correct — рассуждения и вычисления верны. Тоже correct (с замечанием типа presentation в errors), если нужная величина в решении найдена верно, но в ответ выписано другое число (данное, промежуточное, результат лишнего безошибочного шага); не сделан последний тривиальный шаг (досчитать сумму, извлечь корень из уже найденного квадрата, 180° − найденный угол); описка при переписывании ответа или строки, исправленная дальше. Мелочи записи (скобка у бесконечности) и пропущенные очевидные обоснования — тоже correct с замечанием notational/presentation.
- incorrect — есть ошибка по сути хотя бы в одном шаге, даже если итоговый ответ совпал с верным: неверный факт, формула или метод; арифметическая ошибка или ошибка знака; неверное утверждение в решении (даже если не повлияло на ответ); потеряны корни, серии, случаи, граница промежутка; найдена не та величина из-за непонимания условия и нужная не получена; в доказательстве не выведено требуемое; найдены не все объекты, которые требовалось найти (пары, корни) — перечисли, каких не хватает. Округление, меняющее ответ, — ошибка.
- not_solved — решения нет, записано только условие, решение начато и брошено или зачёркнуто.
- unclear — по транскрипции нельзя судить (неразборчиво).
Неточности и опечатки в самом условии не считай ошибкой ученика — отметь их в comment.

Типы ошибок: computational (арифметика, знак при вычислении или переносе), conceptual (неверный метод, формула, понимание), notational (запись, не влияющая на решение), presentation (пропущены обоснования, оформление).
task_no пиши в нумерации условия: {numbering}.
Не выдумывай ошибок: верный шаг не отмечай. Объяснения — кратко, по-русски, для ученика: где ошибка, как правильно, верный ответ.
{scope}
Ответ — только JSON:
{{"results": [{{"task_no": "...", "where_in_solution": "страница и подпись у ученика", "student_answer": "...", "correct_answer": "...", "verdict": "correct|incorrect|not_solved|unclear", "errors": [{{"step": "цитата шага", "type": "computational|conceptual|notational|presentation", "explanation": "..."}}], "comment": "..."}}]}}"""

CHECK_EXAMPLES_V2 = """

КАК НЕ ПРИДИРАТЬСЯ (правила куратора, соблюдай строго). Ошибка — только ошибка в рассуждениях или вычислениях. Примеры:
- Ученик нашёл нужный угол 31°, а в строку «Ответ» выписал данный угол 14° → correct (в errors: presentation, «в ответ выписано не то число»).
- Получено d² = 196, в ответе 196 (корень не извлечён) → correct, presentation. Записано ∠ABC = 51 + 42 без суммы → correct.
- Верное решение, верный ответ, но в ответе лишняя величина (выписаны обе высоты, спрашивали большую) → correct.
- Промежуточная строка с опиской, а следующая строка верна (−9x < 45 при 45 − 9x < 0, затем x > 5) → correct, notational.
- Точка экстремума найдена верно и она единственная на отрезке, концы не проверены → correct, presentation.
- Итоговое выражение верное, но не доупрощено → correct, presentation. Неточности в самом условии — не ошибка ученика.
- Ошибка знака, потерянный корень, неверная формула, неверное утверждение (даже не повлиявшее на ответ) → incorrect.
Перед вердиктом incorrect назови конкретный шаг и почему он неверен ПО СУТИ; если претензия только к оформлению или к последнему тривиальному шагу — ставь correct.
Если сомнение связано с распознаванием (в транскрипции [?], странные символы, число не следует из предыдущей строки, строка помечена ~~зачёркнутой~~, хотя решение явно продолжается из неё) — не делай вывода об ошибке ученика: verdict unclear и опиши сомнение в comment."""

# v3 = v1 + «проверка на адекватность» каждой найденной ошибки по следующим шагам + уточнённые правила куратора
CHECK_RULES_V3 = """
Уточнения куратора:
- Верный ответ не отменяет ошибок в ходе решения (неверно записанные данные при совпавшем ответе — incorrect).
- «Неверно выписанный ответ» — не ошибка, только если выписанное число есть в решении (данное, промежуточное) или это
  явная описка при переписывании; значение, которого нет нигде в решении, — ошибка.
- Описка в строке, после которой решение и ответ верные, или описка, которую ученик сам исправил дальше, — correct,
  но опиши её в comment как замечание для ученика («в строке … описка: …»).
- В доказательстве не обоснованы ключевые (неочевидные) шаги — incorrect, но в comment напиши «ход мыслей верный,
  не хватает обоснования: …». Пропущенные очевидные обоснования — correct.
- Рисунки — обоснование: отмеченные на прямой концы отрезка и точка между ними = проверка концов.
- В короткой задаче (одно-два действия) краткая запись данных в удобной форме — не ошибка.
- Если ученик решил другую задачу из того же условия (перепутал номер), проверяй по той задаче, которую он решил,
  и напиши в comment «перепутана задача, проверено по №…».
- Если итоговый ответ зачёркнут, поищи рядом (сверху, справа, ниже) исправленное значение, прежде чем ставить not_solved.

ПРОВЕРКА НА АДЕКВАТНОСТЬ (обязательно перед вердиктом incorrect). Транскрипция делается по фото и может содержать
ошибки распознавания. Для КАЖДОЙ найденной ошибки посмотри 1–2 следующих шага ученика:
- дальнейшие вычисления продолжают неверное значение (ошибка «протекает» дальше) — ошибка ученика подтверждена;
- дальнейшие шаги согласуются с ВЕРНЫМ значением (например, в строке 38t = 99,8, а дальше t = 2,6, что верно для 98,8) —
  скорее всего, это ошибка распознавания, а не ученика: НЕ считай её ошибкой;
- запись бессмысленна для ученика этого уровня (странные степени, символы, числа ниоткуда) — это распознавание, не ученик;
- ошибка в последней строке или в ответе, следующих шагов нет — оцени по предыдущим шагам, откуда могло взяться число.
Если все претензии оказались «похоже на распознавание» — verdict unclear, в comment «возможно, ошибка распознавания: …».
Для каждой ошибки заполни поле check_next: confirmed (следующие шаги продолжают ошибку), contradicted (следующие шаги
согласуются с верным значением), last_step (дальше шагов нет) или nonsense (запись бессмысленна)."""
CHECK_PROMPT_V3 = CHECK_PROMPT.replace("\nТипы ошибок:", CHECK_RULES_V3 + "\n\nТипы ошибок:").replace(
    '"explanation": "..."}}]', '"explanation": "...", "check_next": "confirmed|contradicted|last_step|nonsense"}}]')
assert CHECK_PROMPT_V3.count("check_next") == 2 and "АДЕКВАТНОСТЬ" in CHECK_PROMPT_V3


def _situations_text():
    """Справочник ситуаций для детектора (v4) из rules/policy.csv — только id и суть, без исходов (режим детектору
    не нужен: ошибка / замечание решает слой политики, policy.py)."""
    path = ROOT / "rules" / "policy.csv"
    if not path.exists():
        return "(справочник не найден)"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return "\n".join(f"{r['id']} — {r['situation']}" for r in rows).replace("{", "{{").replace("}", "}}")


CHECK_PROMPT_V4 = """Ты — внимательный проверяющий домашних работ по математике (подготовка к ОГЭ/ЕГЭ). Тебе даны условие \
(транскрипция) и дословная транскрипция решения ученика по страницам: формулы в LaTeX, зачёркнутое в ~~...~~, неразборчивое [?].

Твоя задача — НЕ выносить итоговый вердикт, а точно описать факты: какие отклонения есть в решении каждой задачи и к
какой ситуации из справочника относится каждое. Считать ли отклонение ошибкой, решит отдельная политика по режиму
задания (экзамен / тренировка), поэтому не смягчай и не усиливай: перечисляй и мелочи записи (политика их может
проигнорировать), но не выдумывай отклонений — верный шаг не отмечай.

Для каждой задачи:
1. Найди её решение ПО СОДЕРЖАНИЮ, а не по номеру (ученики часто подписывают задачи иначе, чем в условии).
2. Реши задачу сам; если даны эталонные ответы — они верные.
3. Выпиши итоговый ответ ученика дословно (student_answer).
4. Пройди решение по шагам и перечисли ВСЕ отклонения: step — цитата шага; situation — id ситуации из справочника
   или общий код; explanation — кратко по-русски для ученика: что не так и как правильно; reading_doubt — true, если
   отклонение может быть ошибкой распознавания, а не ученика (запись бессмысленна для ученика этого уровня, дальше
   шаги согласуются с верным значением, похожие символы 1/7, 3/8, 5/6, ( и [).
Общие коды (если ни одна ситуация справочника не подходит): G.calc — арифметическая ошибка или ошибка знака;
G.method — неверный факт, формула, метод, неверное утверждение; G.cond — неверно понято или переписано условие;
G.lost — потеряны корни, серии, случаи, граница; G.proof — в доказательстве не выведено требуемое или не обоснован
ключевой шаг; G.answer — итоговый ответ неверен, а конкретный шаг назвать нельзя; other — отклонение есть, но ничего
не подходит (опиши в explanation).
status: solved — решение есть; not_solved — решения нет, только условие, начато и брошено или зачёркнуто;
unclear — по транскрипции нельзя судить. Неточности в самом условии — только в comment.

СПРАВОЧНИК СИТУАЦИЙ:
""" + _situations_text() + """

task_no пиши в нумерации условия: {numbering}.
{scope}
Ответ — только JSON:
{{"results": [{{"task_no": "...", "where_in_solution": "страница и подпись у ученика", "student_answer": "...", \
"correct_answer": "...", "status": "solved|not_solved|unclear", "deviations": [{{"step": "цитата шага", \
"situation": "1.1|3.1|G.calc|…", "explanation": "...", "reading_doubt": false}}], "comment": "..."}}]}}"""


# v5 (30.09, после пробы v4): модель плохо оценивает собственную неуверенность (reading_doubt в v4 — 0 раз из 90),
# поэтому вместо оценки просим наблюдаемый факт — что происходит со значением ДАЛЬШЕ в решении. Вывод «описка или
# ошибка», «ученик или распознавание» делает код (policy.py, правило consequence).
CHECK_PROMPT_V5 = CHECK_PROMPT_V4.replace(
    """   шаги согласуются с верным значением, похожие символы 1/7, 3/8, 5/6, ( и [).""",
    """   шаги согласуются с верным значением, похожие символы 1/7, 3/8, 5/6, ( и [);
   consequence — ОБЯЗАТЕЛЬНО, это факт, а не оценка: что происходит дальше в решении с тем местом, где отклонение:
     fixed_later — в следующих строках ученик пользуется ВЕРНЫМ значением/выражением (например, в строке записано
       240 200, а дальше все вычисления и ответ соответствуют 270 200 из условия; записано −10/11 = −1,1, а дальше
       используется −1,1);
     propagated — неверное значение идёт дальше и портит последующие вычисления или ответ;
     no_effect — дальше это место не используется, на ход решения и ответ не влияет;
     final — отклонение в последней строке или в самом ответе, дальше шагов нет.
   Перед тем как выбрать consequence, найди в следующих строках, какое значение реально использовано.""").replace(
    '"explanation": "...", "reading_doubt": false}}]',
    '"explanation": "...", "reading_doubt": false, "consequence": "fixed_later|propagated|no_effect|final"}}]')
assert CHECK_PROMPT_V5.count("consequence") >= 3

CHECK_PROMPTS = {"v1": CHECK_PROMPT, "v2": CHECK_PROMPT + CHECK_EXAMPLES_V2, "v3": CHECK_PROMPT_V3, "v4": CHECK_PROMPT_V4, "v5": CHECK_PROMPT_V5}

# ----------------------------------------------------------------- утилиты


def load_env():
    env = {}
    path = ROOT / ".env"
    if not path.exists():
        sys.exit("Нет файла .env с LLM_API_KEY и LLM_BASE_URL")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def slug(model):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


LOOKALIKE = str.maketrans("ABCEHKMOPTXaceopxyb", "АВСЕНКМОРТХасеорхуб")


TASK_WORD = re.compile(r"^\s*(задание|задача|упражнение)\s*", re.I)


def norm_task(t):
    t = TASK_WORD.sub("", str(t))  # «Задание 3» → 3 (так и в правилах разметки)
    t = t.strip().replace("№", "").replace(" ", "").replace(")", "").replace("(", "")
    t = t.replace("—", "-").replace("–", "-").rstrip(".")
    return t.translate(LOOKALIKE).lower()


def _close_json(text):
    """Чинит оборванный или «замусоренный» в конце JSON: обрезает всё после последнего завершённого значения
    внутри верхнего уровня и дописывает недостающие закрывающие скобки. Строки и экранирование учитываются."""
    stack, in_str, esc, last_ok = [], False, False, None
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            last_ok = (i, list(stack))
            if not stack:
                return text[:i + 1]
    if last_ok is None:
        raise ValueError("не удалось восстановить JSON")
    i, rest = last_ok
    return text[:i + 1] + "".join(reversed(rest))


def parse_json(text):
    """Достаёт JSON из ответа модели (с ```json ... ``` или без). Устойчив к типичным сбоям:
    ответ-массив вместо объекта ({"results": [...]}), мусор после JSON, оборванный конец (дописываем скобки).
    Если пришлось чинить, в результате будет поле _repaired."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("в ответе нет JSON")
    text = text[min(starts):]
    repaired = False
    try:
        res, _ = json.JSONDecoder().raw_decode(text)  # мусор после завершённого JSON просто игнорируется
    except json.JSONDecodeError:
        res = json.loads(_close_json(text))
        repaired = True
    if isinstance(res, list):
        res = {"results": res}
    if repaired:
        res["_repaired"] = True
    return res


def reparse_failed(run_dir):
    """Повторный разбор сохранённых сырых ответов (*_failed.json) без новых запросов к API."""
    fixed = []
    for f in sorted(Path(run_dir).glob("W*/*_failed.json")):
        old = json.load(open(f, encoding="utf-8"))
        if "raw" not in old:
            continue
        try:
            res = parse_json(old["raw"])
        except (ValueError, json.JSONDecodeError):
            continue
        if "_meta" in old:
            res["_meta"] = old["_meta"]
        target = f.with_name(f.name.replace("_failed", ""))
        if not target.exists():
            save(target, res)
            fixed.append(f"{f.parent.name}/{target.name}" + (" (починен конец ответа)" if res.get("_repaired") else ""))
    return fixed


def image_data_url(img, max_side):
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def work_files(work):
    """Файлы работы по ролям из data/files.csv."""
    files = {"solution": [], "task": []}
    for r in csv.DictReader(open(DATA / "files.csv", encoding="utf-8")):
        if r["work_id"] == work and r["role"] in files:
            files[r["role"]].append(ANON / work / r["file"])
    for k in files:
        files[k].sort(key=lambda p: natural_key(p.name))
    return files


def pdf_pages(path, dpi):
    try:
        import pymupdf as fitz
    except ImportError:
        sys.exit("Для условий в PDF нужен PyMuPDF: pip3 install pymupdf")
    doc = fitz.open(path)
    out = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        out.append((img, page.get_text()))
    return out

# ----------------------------------------------------------------- API


class BudgetExceeded(Exception):
    pass


class Client:
    def __init__(self, model, max_rub, run_dir, dry_run=False, effort=None):
        env = load_env()
        self.url = env["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
        self.key = env["LLM_API_KEY"]
        self.model = model
        self.max_rub = max_rub
        self.spent = 0.0
        self.lock = threading.Lock()
        self.run_dir = run_dir
        self.dry_run = dry_run
        self.effort = effort or {}
        self.price_in, self.price_out, self.json_mode, self.reasoning_ok = self._model_info()

    def _model_info(self):
        path = DATA / "models.json"
        if path.exists():
            for m in json.load(open(path, encoding="utf-8")):
                if m["id"] == self.model:
                    p = m.get("pricing", {})
                    if "image" not in m.get("input_modalities", ["image"]):
                        print(f"ВНИМАНИЕ: {self.model} не принимает изображения")
                    sp = m.get("supported_parameters", [])
                    return (p.get("input_rub_per_million", 0), p.get("output_rub_per_million", 0),
                            "response_format" in sp, "reasoning" in sp or "reasoning_effort" in sp)
        print(f"ВНИМАНИЕ: {self.model} нет в data/models.json, стоимость не посчитается")
        return 0, 0, False, False

    def chat(self, content, max_tokens, work, step):
        with self.lock:
            if self.spent >= self.max_rub:
                raise BudgetExceeded(f"достигнут лимит {self.max_rub} ₽")
        body = {"model": self.model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": content}]}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        kind = step.split(":")[0]
        effort = self.effort.get({"transcribe2": "transcribe"}.get(kind, kind))  # второе прочтение = как первое
        if effort and effort != "default" and self.reasoning_ok:
            body["reasoning"] = {"effort": effort}
        if self.dry_run:
            n_img = sum(1 for c in content if c.get("type") == "image_url")
            n_chars = sum(len(c.get("text", "")) for c in content)
            print(f"  [dry-run] {work} {step}: картинок {n_img}, символов текста {n_chars}")
            return '{"dry_run": true}', "stop"
        data, body = self._post(body)  # body — то, что реально ушло (для честного лога)
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        finish = choice.get("finish_reason") or ""
        usage = data.get("usage") or {}
        if not usage and not getattr(self, "_warned_usage", False):
            self._warned_usage = True
            print("\n  ВНИМАНИЕ: API не вернул usage — стоимость и лимит бюджета не считаются")
        pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", "")
        rub = (pt * self.price_in + ct * self.price_out) / 1e6
        with self.lock:
            self.spent += rub
            log = self.run_dir / "usage.csv"
            new = not log.exists()
            with open(log, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "work", "step", "prompt_tokens", "completion_tokens", "rub", "finish",
                                "reasoning_tokens", "effort"])
                w.writerow([datetime.now().isoformat(timespec="seconds"), work, step, pt, ct, f"{rub:.3f}", finish,
                            rt, body.get("reasoning", {}).get("effort", "")])
        return text, finish

    def _post(self, body):
        """Возвращает (ответ, фактически отправленное тело). Повторяет при 429/5xx и пустом ответе;
        после таймаута чтения повторяет только один раз (долгий ответ мог быть уже оплачен)."""
        timeouts = 0
        for attempt in range(4):
            req = urllib.request.Request(
                self.url, data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    data = json.load(r)
                if not data.get("choices"):
                    raise ValueError(f"ответ без choices: {str(data)[:300]}")
                return data, body
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:500]
                if e.code == 400 and "response_format" in body and re.search(r"response_format|json", msg, re.I):
                    body = {k: v for k, v in body.items() if k != "response_format"}
                    continue
                if e.code == 400 and "reasoning" in body and re.search(r"reasoning|effort|thinking", msg, re.I):
                    # не отключаем молча: иначе прогон с меткой «high» окажется прогоном без настройки
                    raise RuntimeError(f"модель не приняла reasoning={body['reasoning']}: {msg[:300]} — "
                                       f"запусти с другим --effort-* или default")
                if e.code in (408, 429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(5 * 3 ** attempt)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {msg}")
            except (TimeoutError, socket.timeout) as e:
                timeouts += 1
                if timeouts <= 1:
                    time.sleep(5)
                    continue
                raise RuntimeError(f"таймаут ответа: {e}")
            except (urllib.error.URLError, ValueError, http.client.HTTPException, ConnectionError) as e:
                if attempt < 3:
                    time.sleep(5 * 3 ** attempt)
                    continue
                raise RuntimeError(f"Сеть/ответ: {e}")
        raise RuntimeError("не удалось получить ответ")

# ----------------------------------------------------------------- шаги


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ask_json(client, content, max_tokens, work, step):
    """Запрос с разбором JSON. Если ответ оборвался по лимиту токенов (модель долго «думала»),
    повторяем один раз с удвоенным лимитом."""
    doubled = False
    for attempt in range(3):
        text, finish = client.chat(content, max_tokens, work, step)
        if finish == "length" and not doubled:
            print(f"\n  {work} {step}: ответ оборван по лимиту {max_tokens} токенов, повторяю с лимитом {2 * max_tokens}",
                  end="")
            max_tokens *= 2
            doubled = True
            continue
        if (finish == "error" or not text.strip()) and attempt < 2:
            print(f"\n  {work} {step}: пустой ответ/ошибка провайдера, повторяю", end="")
            time.sleep(10)
            continue
        break
    try:
        res = parse_json(text)
    except (ValueError, json.JSONDecodeError) as e:
        res = {"parse_error": str(e), "raw": text}
    if finish == "length":
        res["truncated"] = True
    return res


def _canon(line):
    """Запись без стиля: LaTeX-окружения, \\text, $, пробелы, фигурные скобки и знаки умножения не важны —
    сравниваем только содержание (цифры, буквы, знаки, скобки)."""
    from features import delatex  # ленивый импорт: features сам импортирует check
    t = str(line).replace("$", "")
    t = re.sub(r"\\(begin|end)\{[a-z*]+\}(\{[a-z]+\})?", " ", t)
    t = re.sub(r"\\text\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\(overset|underset)\{[^{}]*\}", "", t)
    t = t.replace("\\\\", " ").replace("\\cdot", "*").replace("·", "*").replace("×", "*").replace("−", "-")
    t = delatex(t)
    t = re.sub(r"[\s{}*]", "", t).lower()
    return t.lstrip("[")  # «[» в начале строки — скобка совокупности, стиль записи


def line_disagreements(a_lines, b_lines, limit=40):
    """Места, где два независимых распознавания страницы разошлись по содержанию (стиль записи и разбиение
    на строки не учитываются). a — строки основного варианта, b — соответствующий кусок альтернативного."""
    import difflib
    ca = [_canon(x) for x in a_lines]
    starts, pos = [], 0
    for c in ca:
        starts.append(pos)
        pos += len(c) + 1
    A = "|".join(ca)
    cb = [_canon(x) for x in b_lines]
    bstarts, pos = [], 0
    for c in cb:
        bstarts.append(pos)
        pos += len(c) + 1
    B = "|".join(cb)

    def lines_of(starts, texts, i1, i2):
        hit = [k for k, st in enumerate(starts) if st <= max(i1, i2 - 1) and st + len(texts[k]) >= i1]
        if not hit and texts:
            hit = [max(0, min(len(texts) - 1, next((k for k, st in enumerate(starts) if st > i1), len(texts)) - 1))]
        return hit
    spans = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if tag == "equal" or (not A[i1:i2].strip("|") and not B[j1:j2].strip("|")):
            continue  # совпадает или разошлись только границы строк
        la, lb = lines_of(starts, ca, i1, i2), lines_of(bstarts, cb, j1, j2)
        if la:
            spans.append((min(la), max(la) + 1, set(lb)))
    out = []
    for a1, a2, lb in sorted(spans, key=lambda x: x[:2]):
        if out and a1 <= out[-1]["a_lines"][1]:
            out[-1]["a_lines"][1] = max(out[-1]["a_lines"][1], a2)
            out[-1]["_b"] |= lb
        else:
            out.append({"a_lines": [a1, a2], "_b": set(lb)})
    for d in out:
        d["a"] = " | ".join(a_lines[d["a_lines"][0]:d["a_lines"][1]])
        d["b"] = " | ".join(b_lines[k] for k in sorted(d.pop("_b")))
    return out[:limit]


def second_reading(client, work, out, pages, result, force, prompt_version):
    """Двойное распознавание: второе прочтение каждой страницы другим промптом и расхождения с первым.
    Первое прочтение берётся готовым (не перезапрашивается)."""
    alt_version = "v2" if prompt_version == "v1" else "v1"
    ph = hashlib.md5(OCR_PROMPTS[alt_version].encode()).hexdigest()[:8]
    page_dir = out / "pages2"
    by_file = {p["file"]: p for p in result}

    def one(path):
        pc = page_dir / f"{path.stem}.json"
        if pc.exists() and not force:
            cached = json.load(open(pc, encoding="utf-8"))
            if cached.get("_prompt", ph) == ph:  # старые файлы без отпечатка промпта тоже берём
                return path.name, cached
        content = [{"type": "text", "text": OCR_PROMPTS[alt_version]},
                   {"type": "image_url", "image_url": {"url": image_data_url(Image.open(path), 2000)}}]
        try:
            res = ask_json(client, content, 12000, work, f"transcribe2:{path.name}")
        except BudgetExceeded:
            raise
        except Exception as e:
            return path.name, {"failed": str(e)}
        res["_prompt"] = ph
        if not client.dry_run and "parse_error" not in res and not res.get("truncated"):
            save(pc, res)
        return path.name, res

    with ThreadPoolExecutor(4) as ex:
        alts = dict(ex.map(one, pages))
    failed = [f for f, r in alts.items() if r.get("failed")]
    if failed:
        raise RuntimeError(f"второе прочтение не удалось для {', '.join(failed)} (остальное сохранено, перезапусти)")
    for f, alt in alts.items():
        p = by_file.get(f)
        if p is not None:
            p["disagreements"] = line_disagreements(p.get("lines", []), alt.get("lines", []))
            p["alt_legibility"] = alt.get("legibility")
    return result


def step_transcribe(client, work, out, force, prompt_version="v1", double=False):
    pages = work_files(work)["solution"]
    target = out / "transcription.json"
    meta = {"prompt": prompt_version, "effort": client.effort.get("transcribe", "default")}
    meta_path = out / "transcription_meta.json"
    prompt = OCR_PROMPTS[prompt_version]

    def check_meta(old, what):
        # старые файлы без метаданных считаем сделанными промптом v1
        if (old or {"prompt": "v1"}).get("prompt") != meta["prompt"]:
            raise RuntimeError(f"{what} распознан промптом {(old or {}).get('prompt', 'v1')}, а запрошен "
                               f"{meta['prompt']}: используй --tag (и --force), чтобы не смешивать")
    if target.exists() and not force:
        old_meta = json.load(open(meta_path, encoding="utf-8")) if meta_path.exists() else None
        check_meta(old_meta, "transcription.json")
        result = json.load(open(target, encoding="utf-8"))
        if double:  # второе прочтение из кэша (бесплатно) или дозапрос; расхождения пересчитываются всегда
            result = second_reading(client, work, out, pages, result, force, prompt_version)
            save(target, result)
            save(meta_path, {**(old_meta or meta), "double": True})
        return result

    page_dir = out / "pages"  # постраничный кэш: сбой одной страницы не выбрасывает уже оплаченные

    def one(i_path):
        i, path = i_path
        pc = page_dir / f"{path.stem}.json"
        if pc.exists() and not force:
            cached = json.load(open(pc, encoding="utf-8"))
            check_meta(cached.get("_meta"), pc.name)
            return cached
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": image_data_url(Image.open(path), 2000)}}]
        try:
            res = ask_json(client, content, 12000, work, f"transcribe:{path.name}")
        except BudgetExceeded:
            raise
        except Exception as e:
            return {"page": i, "file": path.name, "failed": str(e)}
        res["page"] = i
        res["file"] = path.name
        res["_meta"] = meta
        if not client.dry_run and "parse_error" not in res and not res.get("truncated"):
            save(pc, res)  # битые страницы не кэшируем — перезапросятся при следующем запуске
        return res

    with ThreadPoolExecutor(4) as ex:
        result = list(ex.map(one, enumerate(pages, 1)))
    failed = [p["file"] for p in result if p.get("failed")]
    if failed:
        reason = next(p["failed"] for p in result if p.get("failed"))
        raise RuntimeError(f"не распознаны страницы {', '.join(failed)} (остальные сохранены, перезапусти работу). "
                           f"Причина: {reason[:300]}")
    if double:
        result = second_reading(client, work, out, pages, result, force, prompt_version)
    save(target, result)
    save(meta_path, {**meta, "double": bool(double)})
    # копия для чтения глазами: сверка с фото на over-correction
    md = [f"# {work} — транскрипция ({client.model})\n"]
    for p in result:
        md.append(f"\n## Страница {p['page']} ({p['file']})\n")
        if p.get("parse_error") or p.get("truncated"):
            md.append(f"**Проблема: {p.get('parse_error') or 'ответ оборван по лимиту токенов'}**\n")
        md += [f"    {line}" for line in p.get("lines", [])] or [p.get("raw", "")]
        for d in p.get("drawings", []):
            md.append(f"\n*Рисунок:* {d}")
        for d in p.get("disagreements", []):
            md.append(f"\n*Расхождение прочтений:* «{d['a']}» vs «{d['b']}»")
    (out / "transcription.md").write_text("\n".join(md), encoding="utf-8")
    return result


TASK_REFRESHED = set()  # файлы условий, уже пересчитанные в этом запуске (для --force)


def pdf_needed_pages(texts, scope):
    """Номера страниц PDF, где в текстовом слое встречаются номера нужных задач («17. …», «25.1. …»).
    None — выбрать не получилось (нет scope/текстового слоя или какой-то номер не найден) → все страницы."""
    if not scope or not any(t.strip() for t in texts):
        return None
    pats = []
    for s in scope:
        base = re.sub(r"[^\d.]+$", "", norm_task(s).split("-")[-1])  # б1-13а → 13, 25.1а → 25.1
        if not re.match(r"^\d", base):
            return None
        cands = {base} | ({base.split(".")[0]} if "." in base else set())
        pats.append([re.compile(rf"(?:^|\s){re.escape(c)}\.{'?' if '.' in c else ''}\s") for c in cands])
    hit = set()
    for ps in pats:
        pages = [i for i, t in enumerate(texts) if any(p.search(t) for p in ps)]
        if not pages:
            return None
        hit.update(pages)
    return sorted(hit)


_BLOCK_PREFIX = re.compile(r"^\s*[БбBb]\d+\s*-")


def pdf_blocks(texts):
    """Заголовки «Блок N» в текстовом слое: для каждой страницы — [(позиция, номер блока)] и блок на её начало.
    None, если в задачнике меньше двух блоков (тогда префиксы не нужны)."""
    heads = [[(m.start(), int(m.group(1))) for m in re.finditer(r"Блок\s*(\d+)", t)] for t in texts]
    if len({b for h in heads for _, b in h}) < 2:
        return None
    start, cur = [], None
    for h in heads:
        start.append(cur)
        if h:
            cur = h[-1][1]
    first = next(b for h in heads for _, b in h)
    return heads, start, first


def with_block(task_no, text, heads, start_block, first_block):
    """Добавляет к номеру задачи блок задачника (Б2-101), если модель его не указала: блок определяется по тому,
    какой заголовок «Блок N» стоит на странице выше этой задачи. Нужен, чтобы №101 блока 1 не путался с №101 блока 2."""
    no = str(task_no)
    if _BLOCK_PREFIX.match(no):
        return no
    m = re.search(r"\d+", no)
    pos = 0
    if m:
        hit = re.search(rf"(?:^|\s){m.group(0)}\.\s", text)
        pos = hit.start() if hit else 0
    blk = start_block
    for p, b in heads:
        if p <= pos:
            blk = b
    return f"Б{blk or first_block}-{no}"


def task_file(client, work, path, force, scope=None, focus=None):
    """Распознаёт один файл условия. Кэш — по содержимому файла и тексту промпта (для PDF — постранично),
    поэтому задачник, выданный нескольким ученикам, распознаётся один раз. Из PDF берутся только
    страницы, где есть номера нужных задач."""
    fkey = hashlib.md5(path.read_bytes() + (TASK_PROMPT + NUMBERING).encode()).hexdigest()[:16]
    cdir = client.run_dir / "_tasks"
    whole = cdir / f"{fkey}.json"
    if whole.exists() and not (force and whole.name not in TASK_REFRESHED) and not client.dry_run:
        return json.load(open(whole, encoding="utf-8"))
    if path.suffix.lower() != ".pdf" and focus:
        fc = cdir / f"{fkey}_focus{hashlib.md5(' '.join(sorted(focus)).encode()).hexdigest()[:6]}.json"
        if fc.exists() and not client.dry_run:
            return json.load(open(fc, encoding="utf-8"))
        extra = ("ОБЯЗАТЕЛЬНО найди и перепиши задачи с номерами: " + ", ".join(focus) +
                 " (в прошлый раз они потерялись). Остальные можно не переписывать.")
        content = [{"type": "text", "text": TASK_PROMPT.format(numbering=NUMBERING, extra=extra)},
                   {"type": "image_url", "image_url": {"url": image_data_url(Image.open(path), 2000)}}]
        res = ask_json(client, content, 12000, work, f"task:{path.name}:focus")
        result = {"file": path.name, "tasks": res.get("tasks", []), "errors": [res] if "parse_error" in res else []}
        if not client.dry_run and not result["errors"]:
            save(fc, result)
        return result
    if path.suffix.lower() != ".pdf":
        content = [{"type": "text", "text": TASK_PROMPT.format(numbering=NUMBERING, extra="")},
                   {"type": "image_url", "image_url": {"url": image_data_url(Image.open(path), 2000)}}]
        res = ask_json(client, content, 12000, work, f"task:{path.name}")
        result = {"file": path.name, "tasks": res.get("tasks", []), "errors": [res] if "parse_error" in res else []}
        if not client.dry_run and result["tasks"] and not result["errors"]:
            save(whole, result)
        TASK_REFRESHED.add(whole.name)
        return result
    pages = pdf_pages(path, 150)
    texts = [t for _, t in pages]
    blocks = pdf_blocks(texts)
    need = pdf_needed_pages(texts, focus or scope)
    tasks, errors = [], []
    ftag = "_focus" + hashlib.md5(" ".join(sorted(focus)).encode()).hexdigest()[:6] if focus else ""
    for i in (need if need is not None else range(len(pages))):
        pc = cdir / f"{fkey}_p{i + 1}{ftag}.json"
        if pc.exists() and not (force and pc.name not in TASK_REFRESHED) and not client.dry_run:
            res = json.load(open(pc, encoding="utf-8"))
        else:
            img, text = pages[i]
            nxt = pages[i + 1][1][:800] if i + 1 < len(pages) else ""
            extra = (f"Это страница {i + 1} из {len(pages)}. Перепиши задачи, которые начинаются на этой странице; "
                     f"если задача продолжается на следующей — допиши её по тексту начала следующей страницы.\n"
                     f"Текстовый слой этой страницы (формулы в нём могут быть искажены, сверяй с картинкой):\n{text}\n"
                     f"Начало следующей страницы:\n{nxt}")
            if focus:
                extra += ("\nОБЯЗАТЕЛЬНО найди на этой странице и перепиши задачи с номерами: " + ", ".join(focus) +
                          " (в прошлый раз они потерялись). Остальные задачи можно не переписывать.")
            content = [{"type": "text", "text": TASK_PROMPT.format(numbering=NUMBERING, extra=extra)},
                       {"type": "image_url", "image_url": {"url": image_data_url(img, 1800)}}]
            res = ask_json(client, content, 12000, work, f"task:{path.name}:{i + 1}{ftag}")
            if not client.dry_run and "parse_error" not in res:
                save(pc, res)
            TASK_REFRESHED.add(pc.name)
        page_tasks = res.get("tasks", [])
        if blocks:
            heads, start, first = blocks
            page_tasks = [{**t, "task_no_raw": t.get("task_no", ""),
                           "task_no": with_block(t.get("task_no", ""), texts[i], heads[i], start[i], first)}
                          for t in page_tasks]
        tasks += page_tasks
        errors += [res] if "parse_error" in res else []
    return {"file": path.name, "tasks": tasks, "errors": errors}


def missing_tasks(tasks, scope):
    """Номера из scope, для которых в распознанном условии нет подходящей задачи."""
    have = [norm_task(t.get("task_no", "")) for t in tasks]
    return [s for s in (scope or []) if not any(task_hit(h, norm_task(s)) or task_hit(norm_task(s), h) for h in have)]


def step_task(client, work, out, force, scope=None, rebuild=False):
    """Условие работы. rebuild=True — пересобрать task.json из кэша страниц (бесплатно) и дозапросить
    только потерянные задачи."""
    target = out / "task.json"
    if target.exists() and not force and not rebuild:
        cached = json.load(open(target, encoding="utf-8"))
        if cached.get("tasks"):
            return cached
    if not scope and (out / "transcription.json").exists():  # без разметки — номера, подписанные учеником
        trans = json.load(open(out / "transcription.json", encoding="utf-8"))
        scope = [x for p in trans for x in p.get("task_labels", [])]
    files = work_files(work)["task"]
    tasks, errors = [], []
    for path in files:
        res = task_file(client, work, path, force, scope)
        tasks += res.get("tasks", [])
        errors += res.get("errors", [])
    missing = missing_tasks(tasks, scope)
    if missing and not client.dry_run and len(missing) <= 40:
        print(f"\n  {work}: в условии не нашлись {', '.join(missing[:12])}{'…' if len(missing) > 12 else ''} — дозапрашиваю",
              end="")
        for path in files:
            res = task_file(client, work, path, False, scope, focus=missing)
            known = {norm_task(t.get("task_no", "")) for t in tasks}
            tasks += [t for t in res.get("tasks", []) if norm_task(t.get("task_no", "")) not in known]
        missing = missing_tasks(tasks, scope)
    if not tasks and not client.dry_run:
        raise RuntimeError("условие не распознано (пустой ответ) — запусти эту работу ещё раз")
    result = {"tasks": tasks, "errors": errors, "missing": missing}
    if target.exists():  # условие изменилось → старые проверки этой работы стоит пересчитать
        old = json.load(open(target, encoding="utf-8")).get("tasks", [])
        sig = lambda ts: sorted((str(t.get("task_no")), str(t.get("text", ""))[:80]) for t in ts)
        if sig(old) != sig(tasks) and not client.dry_run:
            REPORTS.mkdir(exist_ok=True)
            path = REPORTS / "changed_tasks.txt"
            done = set(path.read_text().split()) if path.exists() else set()
            if work not in done:
                with open(path, "a") as f:
                    f.write(work + "\n")
            print(f"\n  {work}: условие изменилось (записано в reports/changed_tasks.txt)", end="")
    if missing:
        print(f"\n  ВНИМАНИЕ {work}: в условии так и не нашлись {', '.join(missing[:12])}", end="")
    save(target, result)
    return result


def task_hit(n, k):
    """Нечёткое совпадение нормализованных номеров: 13 ~ 13а, 5 ~ Б1-5, 15-б2-8 ~ б2-8."""
    return (k == n or k.endswith("-" + n) or n.endswith("-" + k)
            or (k.startswith(n) and len(k) > len(n) and not k[len(n)].isdigit()))


def select_tasks(tasks, scope):
    """Оставляет из условия только задачи из scope (весь задачник в проверку не шлём — это лишние
    входные токены). Если хоть один номер не сопоставился — отдаём всё условие (иначе модель
    проверяла бы эту задачу вслепую)."""
    if not scope:
        return tasks
    keys = {norm_task(s) for s in scope}

    def score(n, k):
        # точное совпадение лучше всего; номер без блока («101») по умолчанию — из блока 1 (Б1-101)
        if n == k:
            return 3
        if task_hit(n, k) or task_hit(k, n):
            return 2 if (n.startswith("б1-") and not k.startswith("б")) else 1
        return 0
    chosen = set()
    for k in keys:
        sc = [(score(norm_task(t.get("task_no", "")), k), i) for i, t in enumerate(tasks)]
        best = max((x for x, _ in sc), default=0)
        if best == 0:
            return tasks  # номер не сопоставился — отдаём всё условие, чтобы не проверять вслепую
        chosen.update(i for x, i in sc if x == best)
    return [t for i, t in enumerate(tasks) if i in chosen]


def load_key(work):
    """Эталонные ответы к задачам работы из data/keys/answer_key.csv (решены независимо от разметки)."""
    path = DATA / "keys" / "answer_key.csv"
    if not path.exists():
        return {}
    return {r["task_no"]: r for r in csv.DictReader(open(path, encoding="utf-8")) if work in r["works"].split()}


def step_check(client, work, out, force, scope, repeat=1, images=False, with_key=False, prompt_version="v1"):
    """repeat=1 → check.json, repeat=r>1 → check_r{r}.json (повторные прогоны для оценки уверенности:
    та же задача, тот же вход — насколько стабилен вердикт). images=True — вместе с транскрипцией
    модель видит фото страниц и может перепроверить сомнительные места."""
    target = out / ("check.json" if repeat == 1 else f"check_r{repeat}.json")
    meta = {"model": client.model, "effort": client.effort.get("check", "default"), "images": bool(images),
            "key": bool(with_key),
            "prompt": hashlib.md5((CHECK_PROMPTS[prompt_version] + NUMBERING).encode()).hexdigest()[:8]}
    if target.exists() and not force:
        cached = json.load(open(target, encoding="utf-8"))
        old = cached.get("_meta")
        if old and {k: old.get(k, False) for k in ("effort", "images", "prompt", "key")} != \
                {k: meta[k] for k in ("effort", "images", "prompt", "key")}:
            raise RuntimeError(f"{target.name} посчитан с другими настройками {old}; для новых настроек "
                               f"используй --tag (или --force, чтобы перезаписать)")
        return cached
    if not (out / "task.json").exists() or not (out / "transcription.json").exists():
        raise RuntimeError("нет распознанного решения или условия — сначала запусти "
                           "--steps transcribe,task (без --tag), потом проверку")
    task = json.load(open(out / "task.json", encoding="utf-8"))
    if not task.get("tasks"):
        raise RuntimeError("нет распознанного условия — проверку не запускаю")
    trans = json.load(open(out / "transcription.json", encoding="utf-8"))
    pages = [{k: p.get(k) for k in ("page", "lines", "drawings", "task_labels", "corrections", "uncertain", "disagreements", "raw") if p.get(k)} for p in trans]
    if scope:
        scope_text = ("Проверь ровно эти задачи: " + ", ".join(scope) +
                      ". Если какой-то из них нет в решении — verdict not_solved. Другие задачи не включай.")
    else:
        scope_text = "Проверь все задачи, решение которых есть в работе; задачи без решения не включай."
    prompt = CHECK_PROMPTS[prompt_version].format(numbering=NUMBERING, scope=scope_text)
    content = [{"type": "text", "text": prompt +
                "\n\nУСЛОВИЕ:\n" + json.dumps(select_tasks(task["tasks"], scope), ensure_ascii=False) +
                "\n\nРЕШЕНИЕ УЧЕНИКА (транскрипция по страницам):\n" + json.dumps(pages, ensure_ascii=False)}]
    if any(p.get("disagreements") for p in pages):
        content[0]["text"] += ("\n\nПоле disagreements — места, где два независимых распознавания страницы разошлись "
                               "(a — вариант в lines, b — альтернативный). Там прочтение ненадёжно: не делай вывода "
                               "об ошибке ученика, если он зависит только от такого места — ставь unclear или "
                               "выбери прочтение, согласованное с соседними строками.")
    if with_key:
        key = load_key(work)
        rows = [f"{t}: {r['answer']}" + (f" ({r['comment'][:120]})" if r["answer_type"] in ("proof", "construction") else "")
                for t, r in key.items() if not scope or norm_task(t) in {norm_task(x) for x in scope}]
        if rows:
            content[0]["text"] += ("\n\nЭТАЛОННЫЕ ОТВЕТЫ (верные ответы, решены заранее; сверяй с ними итоговый ответ "
                                   "ученика, но ход решения всё равно проверяй сам — верный ответ не гарантирует "
                                   "верного решения, а «—» означает задачу на доказательство):\n" + "\n".join(rows))
    if images:
        content[0]["text"] += ("\n\nНиже — фото этих же страниц в том же порядке. Транскрипция может содержать ошибки "
                               "распознавания: если число или знак в транскрипции вызывает сомнение, сверься с фото "
                               "и опирайся на фото.")
        content += [{"type": "image_url", "image_url": {"url": image_data_url(Image.open(p), 1600)}}
                    for p in work_files(work)["solution"]]
    result = ask_json(client, content, 32000, work, "check" if repeat == 1 else f"check:r{repeat}")
    result["_meta"] = meta
    if "parse_error" in result or result.get("truncated"):
        save(target.with_name(target.stem + "_failed.json"), result)  # не кэшируем: перезапросится
        return result
    save(target, result)
    return result

# ----------------------------------------------------------------- сравнение с разметкой


def work_splits():
    """work_id -> dev/test. Разбиение хранится по имени папки (data/splits_new.csv),
    номер работы берётся из data/private/work_map.csv. Работы без записи — dev."""
    path, wmap = DATA / "splits_new.csv", DATA / "private" / "work_map.csv"
    if not path.exists() or not wmap.exists():
        sys.exit("Нет data/splits_new.csv или data/private/work_map.csv — без них тестовые работы смешаются с dev")
    folder2id = {r["folder"]: r["work_id"] for r in csv.DictReader(open(wmap, encoding="utf-8"))}
    return {folder2id[r["folder"]]: r["split"] for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["folder"] in folder2id}


def load_scope_file(path):
    """CSV work_id,task_no → {work_id: [task_no, …]} (порядок сохраняется, повторы убираются)."""
    scope = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        lst = scope.setdefault(r["work_id"].strip(), [])
        t = r["task_no"].strip()
        if t and t not in lst:
            lst.append(t)
    return scope


def load_labels():
    labels = {}
    for r in csv.DictReader(open(DATA / "labels.csv", encoding="utf-8")):
        e = labels.setdefault(r["work_id"], {}).setdefault(
            norm_task(r["task_no"]), {"task_no": r["task_no"], "vals": [], "text": []})
        e["vals"].append(r["is_correct"])
        e["text"].append("; ".join(x for x in (r["explanation"], r["notes"]) if x))
    for work in labels.values():
        for e in work.values():
            e["label"] = 0 if "0" in e["vals"] else 1 if "1" in e["vals"] else None
            e["human_missed"] = e["label"] == 0 and any("поставил +" in t for t in e["text"])
    return labels


def evaluate(model_dir, labels):
    splits = work_splits()
    rows = []
    for work_dir in sorted(p for p in model_dir.iterdir() if (p / "check.json").exists()):
        work = work_dir.name
        data = json.load(open(work_dir / "check.json", encoding="utf-8"))
        if "parse_error" in data or data.get("truncated"):
            print(f"ВНИМАНИЕ: {model_dir.name}/{work}: ответ не разобран — работа не входит в метрики, перезапусти "
                  f"или python3 check.py --reparse --tag …", file=sys.stderr)
            continue
        res = data.get("results", [])
        pred = {}
        for r in res:
            k = norm_task(r.get("task_no", ""))
            if k not in pred or r.get("verdict") == "incorrect":
                pred[k] = r
        for k, e in labels.get(work, {}).items():
            r = pred.pop(k, None)
            if r is None:  # модель подписала задачу чуть иначе (13 вместо 13а, 5 вместо Б1-5)
                near = [p for p in pred if task_hit(p, k) or task_hit(k, p)]
                if len(near) == 1:
                    r = pred.pop(near[0])
            v = (r or {}).get("verdict")
            rows.append({"work": work, "split": splits.get(work, "dev"), "task_no": e["task_no"], "label": e["label"],
                         # not_solved на размеченной задаче = «требует внимания» → считаем тревогой
                         "pred": {"correct": 1, "incorrect": 0, "not_solved": 0}.get(v), "verdict": v or "нет в ответе",
                         "human_missed": e["human_missed"], "label_text": " | ".join(t for t in e["text"] if t),
                         "model_errors": " | ".join(f"[{x.get('type')}] {x.get('explanation')}"
                                                    for x in (r or {}).get("errors", [])),
                         "student_answer": (r or {}).get("student_answer", ""),
                         "correct_answer": (r or {}).get("correct_answer", "")})
        for k, r in pred.items():  # задачи, которых нет в разметке
            rows.append({"work": work, "split": splits.get(work, "dev"), "task_no": r.get("task_no"), "label": None, "pred": None,
                         "verdict": f"лишняя: {r.get('verdict')}", "human_missed": False, "label_text": "",
                         "model_errors": "", "student_answer": r.get("student_answer", ""),
                         "correct_answer": r.get("correct_answer", "")})
    return rows


def _counts(rows):
    """Счётчики для метрик: (tp, ошибок, fp, верных, совпало, всего размечено).
    Задача без вердикта (модель её не нашла, ответ не разобран) считается несовпадением: в знаменателе все размеченные
    задачи, иначе выпавшие задачи незаметно улучшали бы совпадение."""
    lab = [r for r in rows if r["label"] in (0, 1)]
    err = [r for r in lab if r["label"] == 0]
    ok = [r for r in lab if r["label"] == 1]
    return (sum(r["pred"] == 0 for r in err), len(err), sum(r["pred"] == 0 for r in ok), len(ok),
            sum(r["pred"] == r["label"] for r in lab), len(lab))


def _rates(c):
    tp, ne, fp, nok, agree, dec = c
    return (tp / ne if ne else None, fp / nok if nok else None,
            tp / (tp + fp) if tp + fp else None, agree / dec if dec else None)


def bootstrap_ci(rows, n_boot=1000, seed=0):
    """95%-интервалы для (recall, ложные тревоги, precision, совпадение) — bootstrap по РАБОТАМ
    (задачи одной работы зависимы: один почерк, одна транскрипция), фиксированный seed."""
    import random
    by_work = {}
    for r in rows:
        by_work.setdefault(r["work"], []).append(r)
    per = [_counts(v) for v in by_work.values()]
    if len(per) < 3:
        return None
    rnd = random.Random(seed)
    samples = [[], [], [], []]
    for _ in range(n_boot):
        pick = [per[rnd.randrange(len(per))] for _ in per]
        for i, v in enumerate(_rates(tuple(sum(x[j] for x in pick) for j in range(6)))):
            if v is not None:
                samples[i].append(v)
    out = []
    for s in samples:
        s.sort()
        out.append((s[int(0.025 * len(s))], s[min(int(0.975 * len(s)), len(s) - 1)]) if len(s) >= 50 else None)
    return out


def metrics(rows, ci=True):
    lab = [r for r in rows if r["label"] in (0, 1)]
    err = [r for r in lab if r["label"] == 0]
    tp, ne, fp, nok, agree, dec = _counts(rows)
    undecided = sum(r["pred"] not in (0, 1) for r in lab)
    hm = [r for r in err if r["human_missed"]]
    cis = bootstrap_ci(rows) if ci else None

    def pct(a, b, i):
        if not b:
            return "—"
        s = f"{100 * a / b:.0f}%"
        if cis and cis[i]:
            s += f"  [{100 * cis[i][0]:.0f}–{100 * cis[i][1]:.0f}]"
        return s
    return {"задач с разметкой": len(lab), "ошибок в разметке": ne,
            "найдено ошибок (recall)": pct(tp, ne, 0),
            "ложные тревоги (от верных)": pct(fp, nok, 1),
            "точность тревог (precision)": pct(tp, tp + fp, 2),
            "совпадение вердиктов": pct(agree, dec, 3),
            "без вердикта (считается несовпадением)": f"{100 * undecided / len(lab):.0f}%" if lab else "—",
            "пропуски проверяющего пойманы": f"{sum(r['pred'] == 0 for r in hm)} из {len(hm)}"}


def paired_compare(rows_a, rows_b, n_boot=2000, seed=0):
    """Парное сравнение двух прогонов на ОДНИХ И ТЕХ ЖЕ размеченных задачах:
    точный тест Мак-Немара (по задачам, где прав ровно один) и bootstrap-ДИ разницы совпадения по работам."""
    import random
    from math import comb
    key = lambda r: (r["work"], norm_task(r["task_no"]))
    # все размеченные задачи, есть в обоих прогонах; задача без вердикта — несовпадение (как в metrics)
    A = {key(r): r for r in rows_a if r["label"] in (0, 1)}
    B = {key(r): r for r in rows_b if r["label"] in (0, 1)}
    common = [k for k in A if k in B]
    if not common:
        return None
    ok = lambda r: int(r["pred"] == r["label"])
    only_a = sum(ok(A[k]) and not ok(B[k]) for k in common)
    only_b = sum(ok(B[k]) and not ok(A[k]) for k in common)
    n, m = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0
    by_work = {}
    for k in common:
        d = by_work.setdefault(k[0], [0, 0])
        d[0] += ok(B[k]) - ok(A[k])
        d[1] += 1
    works = list(by_work.values())
    rnd = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        pick = [works[rnd.randrange(len(works))] for _ in works]
        diffs.append(sum(x[0] for x in pick) / sum(x[1] for x in pick))
    diffs.sort()
    return {"работ": len(by_work), "задач": len(common),
            "совпадение A": sum(ok(A[k]) for k in common) / len(common),
            "совпадение B": sum(ok(B[k]) for k in common) / len(common),
            "прав только A": only_a, "прав только B": only_b, "p (Мак-Немар)": p,
            "ДИ разницы B−A": (diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)])}


def reference_run():
    """Эталонный прогон для парных сравнений — тот, что задаёт набор final в config.json."""
    try:
        cfg = json.load(open(ROOT / "config.json", encoding="utf-8")).get("final", {})
    except (OSError, ValueError):
        return None
    tag = cfg.get("tag")
    return f"{slug(cfg.get('model', 'google/gemini-3.8-flash'))}@{slug(tag)}" if tag else None


def paired_section(rows_by_dir):
    ref = reference_run()
    if not ref or ref not in rows_by_dir:
        return []
    dev = lambda rows: [r for r in rows if r["split"] == "dev"]
    out = []
    for name, rows in sorted(rows_by_dir.items()):
        if name == ref or "@" not in name:  # базовые прогоны без метки — старые пилоты, не сравниваем
            continue
        c = paired_compare(dev(rows_by_dir[ref]), dev(rows))
        if not c or c["задач"] < 20:
            continue
        lo, hi = c["ДИ разницы B−A"]
        out.append({"A (эталон)": ref, "B": name, "работ": c["работ"], "задач": c["задач"],
                    "совпадение A": f"{100 * c['совпадение A']:.1f}%", "совпадение B": f"{100 * c['совпадение B']:.1f}%",
                    "прав только A": c["прав только A"], "прав только B": c["прав только B"],
                    "p (Мак-Немар)": f"{c['p (Мак-Немар)']:.2f}", "ДИ разницы B−A": f"[{100 * lo:+.1f}; {100 * hi:+.1f}] п.п."})
    if out:
        with open(REPORTS / "paired.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
    return out


def report():
    labels = load_labels()
    REPORTS.mkdir(exist_ok=True)
    summary = []
    model_dirs = sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("_")) if RUNS.exists() else []
    ov_path = ROOT / TEST_OVERLAP  # задачи test, которые те же самые, что в dev (другие ученики)
    overlap = {(r["work_id"], norm_task(r["task_no"])) for r in csv.DictReader(open(ov_path, encoding="utf-8"))} \
        if ov_path.exists() else set()
    rows_by_dir = {}
    for model_dir in model_dirs:
        rows = rows_by_dir[model_dir.name] = evaluate(model_dir, labels)
        if not rows:
            continue
        cols = ["work", "split", "task_no", "label", "pred", "verdict", "human_missed", "label_text",
                "model_errors", "student_answer", "correct_answer"]
        with open(REPORTS / f"{model_dir.name}_tasks.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        with open(REPORTS / f"{model_dir.name}_diff.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(r for r in rows if r["label"] is not None and r["pred"] != r["label"])
        rub = 0.0
        if (model_dir / "usage.csv").exists():
            rub = sum(float(r["rub"]) for r in csv.DictReader(open(model_dir / "usage.csv", encoding="utf-8")))
        for part in sorted({r["split"] for r in rows}):
            sub = [r for r in rows if r["split"] == part]
            summary.append({"модель": model_dir.name, "часть": part, "работ": len({r["work"] for r in sub}),
                            **metrics(sub), "потрачено ₽ (всего)": f"{rub:.1f}"})
            if part == "test" and overlap:  # честная оценка: без задач, которые встречались в dev
                clean = [r for r in sub if (r["work"], norm_task(r["task_no"])) not in overlap]
                summary.append({"модель": model_dir.name, "часть": "test без повторов из dev",
                                "работ": len({r["work"] for r in clean}), **metrics(clean), "потрачено ₽ (всего)": ""})
    # A/B-сравнение честно только на одинаковом наборе работ: для прогонов одной модели
    # (имя до «@») добавляем строки на пересечении работ
    groups = {}
    for model_dir in model_dirs:
        works = {p.name for p in model_dir.iterdir() if (p / "check.json").exists()}
        if works and "@" in model_dir.name:  # сравниваем варианты (--tag); основной прогон — отдельно
            groups.setdefault(model_dir.name.split("@")[0], []).append((model_dir, works))
    for base, members in groups.items():
        if len(members) < 2:
            continue
        common = set.intersection(*(w for _, w in members))
        for model_dir, _ in members:
            rows = [r for r in rows_by_dir[model_dir.name] if r["work"] in common]
            for part in sorted({r["split"] for r in rows}):
                sub = [r for r in rows if r["split"] == part]
                summary.append({"модель": model_dir.name, "часть": f"{part}, общие работы",
                                "работ": len({r["work"] for r in sub}), **metrics(sub), "потрачено ₽ (всего)": ""})
    if not summary:
        sys.exit("Нет прогонов в runs/")
    paired = paired_section(rows_by_dir)
    with open(REPORTS / "summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    for m in summary:
        print("\n" + "\n".join(f"{k:32s} {v}" for k, v in m.items()))
    if paired:
        ref = paired[0]["A (эталон)"]
        print(f"\nПарное сравнение с {ref} на общих задачах dev "
              f"(B лучше, если разница > 0; значимо, если ДИ не содержит 0 и p < 0.05). Эталон — итоговая система с ключом, "
              f"поэтому строки прогонов без ключа смешивают влияние ключа и настроек (уровни думания — EXPERIMENTS.md, п. 1):")
        print(f"  {'прогон B':44s} {'работ':>5s} {'задач':>5s} {'A':>6s} {'B':>6s} {'+A':>4s} {'+B':>4s} {'p':>5s}  ДИ B−A")
        for x in paired:
            print(f"  {x['B']:44s} {x['работ']:5d} {x['задач']:5d} {x['совпадение A']:6s} {x['совпадение B']:6s} "
                  f"{x['прав только A']:4d} {x['прав только B']:4d} {x['p (Мак-Немар)']:5s}  {x['ДИ разницы B−A']}")
    print("\nВ квадратных скобках — 95% bootstrap-интервал по работам.")
    print(f"Подробно: {REPORTS}/ (summary.csv, paired.csv, <модель>_diff.csv — расхождения с разметкой)")

# ----------------------------------------------------------------- main


def parse_args(argv=None):
    """Разбор флагов с учётом --preset (набор из config.json задаёт значения по умолчанию)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemini-3.8-flash", help="id модели (по умолчанию google/gemini-3.8-flash)")
    ap.add_argument("--works", nargs="+", default=[], help="W003 W004 … или all")
    ap.add_argument("--steps", default="transcribe,task,check")
    ap.add_argument("--max-rub", type=float, default=150, help="лимит расходов на запуск, ₽")
    ap.add_argument("--no-scope", action="store_true", help="не передавать список задач из разметки")
    ap.add_argument("--scope-file", default="",
                    help="CSV work_id,task_no — список заданных задач (от репетитора) вместо разметки. Режим работы "
                         "на новых неделях: задачи из списка без решения получат not_solved и уйдут куратору. "
                         "Запускать с отдельным --tag, чтобы не смешать кэш с оценочными прогонами")
    ap.add_argument("--force", action="store_true", help="пересчитать, даже если есть кэш")
    ap.add_argument("--dry-run", action="store_true", help="собрать запросы, но не отправлять")
    ap.add_argument("--report", action="store_true", help="сравнить все прогоны с labels.csv")
    ap.add_argument("--effort-ocr", default="minimal", choices=["minimal", "low", "medium", "high", "default"],
                    help="«думание» модели при транскрипции решения и условия (по умолчанию minimal)")
    ap.add_argument("--effort-check", default="default", choices=["minimal", "low", "medium", "high", "default"],
                    help="«думание» при проверке (default — как решит модель)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="сколько раз прогнать проверку (для оценки уверенности; файлы check_r2.json, …)")
    ap.add_argument("--check-images", action="store_true",
                    help="на шаге проверки дать модели и фото страниц (используй с --tag)")
    ap.add_argument("--with-key", action="store_true",
                    help="дать модели эталонные ответы из data/keys/answer_key.csv (используй с --tag)")
    ap.add_argument("--check-prompt", default="v1", choices=["v1", "v2", "v3", "v4", "v5"],
                    help="версия промпта проверки (v2 — с примерами «что не считать ошибкой»; v4 — детектор для policy.py; используй с --tag)")
    ap.add_argument("--ocr-double", action="store_true",
                    help="двойное распознавание: второе прочтение другим промптом, расхождения помечаются (с --tag)")
    ap.add_argument("--refresh-tasks", action="store_true",
                    help="пересобрать task.json из кэша страниц (номера блоков) и дозапросить потерянные задачи")
    ap.add_argument("--ocr-prompt", default="v1", choices=["v1", "v2", "v3", "v4", "v5"],
                    help="версия промпта распознавания (v2 — строже против «исправления» ошибок ученика; с --tag и --force)")
    ap.add_argument("--ocr-from", default="",
                    help="имя папки в runs/, откуда брать транскрипции и условия (например google_gemini-3.8-flash)")
    ap.add_argument("--tag", default="", help="метка прогона: результаты в runs/<модель>@<метка>/, "
                    "готовые транскрипции и условия берутся из основного прогона этой модели")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"],
                    help="какую часть данных брать (по умолчанию dev; test — только для финального прогона)")
    ap.add_argument("--reparse", action="store_true",
                    help="заново разобрать сохранённые неразобранные ответы (*_failed.json) прогона без запросов к API")
    ap.add_argument("--preset", default="", help="набор настроек из config.json (например final)")
    pre, _ = ap.parse_known_args(argv)
    if pre.preset:
        cfg = json.load(open(ROOT / "config.json", encoding="utf-8"))
        if pre.preset not in cfg:
            sys.exit(f"В config.json нет набора «{pre.preset}»")
        # набор задаёт значения по умолчанию; явно указанные в командной строке флаги важнее
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg[pre.preset].items() if not k.startswith("_")})
    return ap, ap.parse_args(argv)


# настройки, которые определяют результат прогона; для test они должны совпасть с заморозкой
RUN_SETTINGS = ["model", "effort_ocr", "effort_check", "with_key", "check_images", "check_prompt", "ocr_prompt",
                "ocr_double", "ocr_from", "repeats", "tag", "no_scope", "scope_file"]
# файлы, от которых зависит результат test; их хэши фиксирует freeze_test.py
# текущий раунд test. Первый раунд (TEST_PROTOCOL.md, reports/test_lock.json, test_runs.log, data/test_overlap.csv)
# использован и сохранён как история; его работы теперь в dev.
# Второй раунд (TEST_PROTOCOL_2.md, reports/test2_lock.json, test2_runs.log) тоже использован; его работы в dev.
TEST_PROTOCOL = "TEST_PROTOCOL_3.md"
TEST_OVERLAP = "data/test3_overlap.csv"  # задачи test, совпадающие с dev (для test3 должно быть пусто)
TEST_LOCK = REPORTS / "test3_lock.json"
TEST_RUNS_LOG = "test3_runs.log"
FROZEN_FILES = ["check.py", "config.json", TEST_PROTOCOL, "data/labels.csv", "data/keys/answer_key.csv",
                "data/splits_new.csv", "data/works.csv", TEST_OVERLAP, "data/private/work_map.csv",
                # итоговая система test3: проверка v1 → теггер ситуаций → политика по режиму (EXPERIMENTS.md, п. 23)
                "tagger.py", "policy.py", "rules/policy.csv", "data/cards/modes.csv"]


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_test_lock(args):
    """Test запускается только на замороженной конфигурации (см. TEST_PROTOCOL, freeze_test.py)."""
    if not TEST_LOCK.exists():
        sys.exit(f"Test ещё не заморожен. Сначала заполни {TEST_PROTOCOL} и запусти: python3 freeze_test.py")
    lock = json.load(open(TEST_LOCK, encoding="utf-8"))
    changed = [f for f, h in lock["files"].items() if not (ROOT / f).exists() or sha256(ROOT / f) != h]
    if changed:
        sys.exit("После заморозки изменились файлы: " + ", ".join(changed) +
                 "\nВерни их или оформи отклонение: python3 freeze_test.py --amend \"причина\"")
    diff = {k: (lock["settings"].get(k), getattr(args, k)) for k in RUN_SETTINGS
            if lock["settings"].get(k) != getattr(args, k)}
    if diff:
        sys.exit("Настройки не совпадают с замороженными (заморожено → сейчас): " +
                 "; ".join(f"{k}: {a} → {b}" for k, (a, b) in diff.items()) +
                 f"\nЗапускай test командой из {TEST_PROTOCOL}: python3 check.py --preset final --split test --works all")
    with open(REPORTS / TEST_RUNS_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{' '.join(sys.argv[1:])}\n")
    print(f"Заморозка test проверена ({lock['frozen_at']}): файлы и настройки совпадают. Запуск записан в reports/{TEST_RUNS_LOG}")


def main(argv=None):
    ap, args = parse_args(argv)
    if args.preset:
        print(f"Набор настроек «{args.preset}» из config.json")

    if args.report:
        report()
        return
    if args.reparse:
        run = RUNS / (slug(args.model) + (f"@{slug(args.tag)}" if args.tag else ""))
        fixed = reparse_failed(run)
        print(f"{run.name}: разобрано заново {len(fixed)}" + ("".join(f"\n  {x}" for x in fixed)))
        return
    if not args.works:
        ap.error("нужны --works (или --report)")
    if args.scope_file and args.split != "dev":
        sys.exit("--scope-file — режим работы на новых неделях; оценка на test идёт по протоколу (scope из разметки)")

    works_rows = list(csv.DictReader(open(DATA / "works.csv", encoding="utf-8")))
    splits = work_splits()
    allowed = [r["work_id"] for r in works_rows if r.get("redacted") == "1"
               and (args.split == "all" or splits.get(r["work_id"], "dev") == args.split)]
    if args.split != "dev":
        if "check" not in args.steps.split(","):
            # подготовка до заморозки: только распознавание решения и условия, без вердиктов — чтобы до заморозки
            # проверить, что номера задач разметки нашлись в условии (leakage_audit.py, п. 6)
            print(f"Подготовка части '{args.split}': только {args.steps}, проверки (вердиктов) не будет")
        else:
            print(f"ВНИМАНИЕ: прогон на части '{args.split}'. Тест запускается один раз, после фиксации промптов.")
            if not args.dry_run:
                verify_test_lock(args)
    works = allowed if args.works == ["all"] else args.works
    works = [w for w in works if w in allowed] or sys.exit("Нет работ с redacted = 1 среди выбранных")
    labels = load_labels()
    file_scope = load_scope_file(args.scope_file) if args.scope_file else {}
    # :batch — та же модель за полцены (ответ может идти дольше); транскрипции берём из основного прогона
    own_dir = RUNS / ("_dryrun" if args.dry_run else "") / slug(args.model.replace(":batch", ""))
    suffix = ("@batch" if args.model.endswith(":batch") else "") + (f"@{slug(args.tag)}" if args.tag else "")
    run_dir = own_dir.parent / (own_dir.name + suffix)
    # --ocr-from: проверка другой моделью по уже готовым транскрипциям (например, Gemini читает, DeepSeek проверяет)
    base_dir = own_dir.parent / args.ocr_from if args.ocr_from else own_dir
    if args.ocr_from and not base_dir.exists():
        sys.exit(f"Нет прогона runs/{args.ocr_from}")
    run_dir.mkdir(parents=True, exist_ok=True)
    effort = {"transcribe": args.effort_ocr, "task": args.effort_ocr, "check": args.effort_check}
    client = Client(args.model, args.max_rub, run_dir, args.dry_run, effort)
    print(f"Модель {args.model}, думание: транскрипция/условие — {args.effort_ocr}, проверка — {args.effort_check}"
          f"{'' if client.reasoning_ok else ' (модель не поддерживает настройку, игнорирую)'}")
    steps = args.steps.split(",")

    for work in works:
        out = run_dir / work
        if run_dir != base_dir:  # берём готовые транскрипцию и условие из основного прогона
            for name in ("transcription.json", "transcription.md", "transcription_meta.json", "task.json"):
                src, dst = base_dir / work / name, out / name
                if src.exists() and (not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime):
                    out.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
        if args.scope_file:
            scope = file_scope.get(work)
            if not scope:
                print(f"{work}: нет в {args.scope_file} — пропускаю")
                continue
        else:
            scope = None if args.no_scope else [e["task_no"] for e in labels.get(work, {}).values()] or None
        print(f"{work}: ", end="", flush=True)
        try:
            if "transcribe" in steps:
                step_transcribe(client, work, out, args.force or args.dry_run, args.ocr_prompt, args.ocr_double)
                print("транскрипция ✓ ", end="", flush=True)
            if "task" in steps:
                step_task(client, work, out, args.force or args.dry_run, scope, rebuild=args.refresh_tasks)
                print("условие ✓ ", end="", flush=True)
            if "check" in steps and not args.dry_run:
                for r in range(2, args.repeats + 1):
                    step_check(client, work, out, args.force, scope, r, args.check_images, args.with_key,
                               args.check_prompt)
                res = step_check(client, work, out, args.force, scope, 1, args.check_images, args.with_key,
                                 args.check_prompt)
                bad = "parse_error" in res or res.get("truncated")
                print(f"проверка {'✗ (ответ не разобран)' if bad else '✓'} ({len(res.get('results', []))} задач)", end="")
                pages = json.load(open(out / "transcription.json", encoding="utf-8"))
                broken = [p["file"] for p in pages if p.get("parse_error") or p.get("truncated")]
                if broken:
                    print(f" | проблемные страницы: {', '.join(broken)}", end="")
            print(f"  | потрачено {client.spent:.1f} ₽")
        except BudgetExceeded as e:
            print(f"\nОстановлено: {e}")
            break
        except Exception as e:
            print(f"\n  ошибка: {e}")
    print(f"\nИтого за запуск: {client.spent:.1f} ₽. Результаты: {run_dir}")
    if not args.dry_run:
        print("Сравнение с разметкой: python3 check.py --report")


if __name__ == "__main__":
    main()
