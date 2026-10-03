# Сессия: AI-проверка №13, 15, 16 ЕГЭ

_Экспорт 30.09.2026. Ранняя часть диалога сохранилась только в виде сводки (контекст был сжат); ниже — сводка и всё, что было после неё. Вызовы инструментов показаны одной строкой._

## Сводка ранней части диалога

This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

Summary:
1. Primary Request and Intent:
   Ksenia (HSE "Экономика и анализ данных", 2nd year; address as «ты», Russian; solve fully without holding back) wants a sellable product: AI checking of handwritten solutions to EGE profile part-2 tasks №13, 15, 16, scored 0/1/2 by FIPI criteria and aimed at tutors and online schools. She will build it from scratch; the old HomeworkChecker project is context and a source of ideas only.

   Evolution of her requests:
   - Market and pain validation.
   - 3 tasks vs. all tasks.
   - Context design and sales format.
   - Auto-keys and pre-solving client task bases.
   - Competitor analysis, including hands-on Sokrat. This was abandoned: "ладно давай без сократа".
   - Updating the market doc and planning next steps.
   - Reusing the old project's code and prompts in a professional plan.
   - Fine-tuning a handwriting model ("почему мы сразу снова гемини решили брать?… может взять модель которую можно дообучать по почерку").
   - Legal side of using others' works.
   - Open datasets. School Notebooks and FERMAT were inspected.
   - Two-stage idea: a best-in-class reader, then a cheaper grader; training on FERMAT.

   Final agreed step: build a gold set of ~25 verbatim-transcribed pages to compare readers (Yandex Vision, an open VLM, Gemini as ceiling) before any fine-tuning. She answered "да" to "Начинаю отбирать и расшифровывать страницы?"

2. Key Technical Concepts:
   - FIPI criteria for №13, 15, 16:
     - 13: 2 points = both parts justified. 1 point = part a) justified, or computational error with a correct sequence of all steps.
     - 15: 1 point = answer differs by boundary points, or computational error.
     - 16: 1 point = correct model.
   - Checkpoints answered by the LLM, score computed by code.
   - Pipeline layers: key (offline, code-verified via sympy/numeric), reading, segmentation (task → pages), answers compared by code, criteria questions (LLM), score plus trust model (logreg + LTT) with routing to a human.
   - Baselines:
     - B0: rule "answer = key → 2, else 0".
     - B1: one LLM call with photo, condition, key and criteria.
   - Metrics:
     - Exact score agreement.
     - Overestimation rate (the key constraint).
     - Underestimation rate.
     - Weighted kappa.
     - Share routed to a human.
     - Per-layer metrics.
     - Bootstrap over students; paired comparisons.
     - Test protocol frozen in advance.
   - Splits by student. External test on FIPI examples. Final test on new tutors' students.
   - Reading "as is": clean transcription plus a separate list of doubts. Only doubts in critical places matter:
     - 13: roots, interval, selection.
     - 15: brackets and boundary points, ≤ vs <.
     - 16: model numbers.
     - Everywhere: the answer.

     A critical doubt triggers a high-res re-read of the crop, then routing to a human. Neighbouring lines inform routing only and never overwrite the transcription.
   - Reader metrics: answer-read accuracy and "autocorrect rate" of student errors. CER is secondary.
   - Two-stage architecture: reader (best off-the-shelf or fine-tuned open VLM such as Qwen-VL/InternVL, LoRA in a Russian cloud) → grader (cheaper model with text, key, criteria questions, and image crops for drawings like the number circle or interval method).
   - Staged fine-tune: FERMAT (pairs image → pert_a = verbatim targets with errors) → School Notebooks (Cyrillic words, teacher_comment class) → own Russian 13/15/16 set.
   - "Russian FERMAT": adults handwrite solutions with injected errors from the Школково catalog and the FIPI typical-error list. The score is known by construction.
   - Gemini: ceiling/benchmark only, never in the product. Russian candidates: GigaChat, YandexGPT, Yandex Vision OCR (models `handwritten` for ru/en and `math-markdown` for LaTeX).
   - Legal:
     - 152-FZ: operator/processor ("поручение", art. 6 part 3); separate consent documents since 01.09.2025; parents consent for minors; data localization; cross-border transfer.
     - Fines since 30.05.2025.
     - AI law signed 26.07.2026, in force 01.09.2026. Copyright training exception from 01.03.2027, possibly only for sovereign/national models.
     - Preferred data flow: school anonymizes (no names, no EXIF, random IDs).
     - No-transfer audit variant: school runs the check itself and shares only aggregates.

3. Files and Code Sections:
   - `/root/.claude/uploads/.../Рынок AI-проверки ДЗ по фото ОГЭ ЕГЭ анализ.md` — market analysis (the uploaded version of the Claude Doc 772e95ab).
   - `PROJECT_CONTEXT.md` — old project context.
     - Constraints to keep verbatim: "Ключ API лежит только в `.env`… Никогда не выводить и не вставлять ключи в чат. Код «[скрыто]», который Ксения однажды вставила, не использовать и не повторять." "Email Ксении (ekseniya14@gmail.com) не отправлять сторонним сервисам." "В подключённых папках ничего не удалять без разрешения — переносить в `_to_delete/`." "Не запускать git-команды внутри VM."
     - Old results: dev 96.4% vs test1 90.4% (96.5% at 14.4% routed to curator); test2 with key 92.3%; OCR ≈24 of 42 misses; detector v4 + policy 90.6% vs v1 94.0%.
   - `~/homework_checker/check.py` (1,449 lines) — reviewed prompts TRANSCRIBE_PROMPT v1/v2/v3 (glyph notes 1/7, 3/8…), TASK_PROMPT, CHECK_PROMPT v1–v4, NUMBERING. Also policy.py, rules/policy.csv (112 situations), card.py, config.json (final: effort_ocr minimal, effort_check low, with_key, check_prompt v1, tag low_key, max_rub 160), EXPERIMENTS.md §21–22.
   - `~/Documents/ege_checker_works/` — 339 submissions from 6 students (Иван 104, Игорь 86, Матвей 54, Докка 39, Павел 28, Дмитрий 28); files решениеN.jpg, условиеN.jpg/pdf, фидбекN.jpg, комментарий.txt, задание.txt.
   - Created `ege_checker_works/_разметка_13_15_16/матвей_типы_дз.csv` — 54 rows: folder, type, has_expert_scores, note. Counts: ? 13, 15: 12, 13: 11, 16: 9, variant 5, other 3, 16-old 1. Only 04_05_2026 has explicit expert scores: 13 = 1, 15 = 2, 16 = 2.
   - Created `ege_checker_works/_разметка_13_15_16/правила_школково_13_15_16.csv` — 37 situations. Columns: id, task, situation, outcome, score_effect, source_text_short, checked_vs_fipi. Outcomes: comment 14, score_0 11, no_penalty 4, judgement 4, definition 1, part_b_0 1, set_1 1, max_1 1.
   - `_разметка_13_15_16/источники/Как проверять пробники ЕГЭ (Школково).xlsx` — the original.
   - `ege_checker_works/_фипи_материалы/` — created but empty; the FIPI PDF download failed.
   - `ege_checker_works/_to_delete/_sheets` and `_to_delete/_sheets2` — preview contact sheets (scratch).
   - Claude Docs:
     - Market doc 772e95ab-a954-47f4-8a98-bc7b41fc886a. Added section "Конкуренты изнутри: как они проверяют вторую часть" (table: Сократ AI, Обучай, Urokly, AI-Exam, Репет, Алиса).
     - Plan doc "Проверка №13, 15, 16: план проекта", id 6f744ac9-1e9f-4f34-9534-dff882974604, body node 836d582b-3480, session prefix m98g3v93gfr, last rev 13. Sections:
       - Что берём из старого проекта
       - Что делаем иначе
       - Конвейер проверки одной задачи
       - Данные и разметка (+ Школково paragraph)
       - Метрики и протокол оценки
       - Материалы ФИПИ
       - Порядок работ (7 stages)
       - Модель: что дообучать и когда (+ приёмы «как есть», revised doubts bullet)
       - Открытые вопросы (checklist)
   - Scratchpad `/tmp/claude-0/-home-claude/7128b226-6592-54f6-8690-159e4729ce79/scratchpad/`:
     - `zoom.py`:
       ```python
       import sys
       from PIL import Image, ImageOps
       p,out=sys.argv[1],sys.argv[2]
       x0,y0,x1,y1=map(float,sys.argv[3:7]); s=float(sys.argv[7]) if len(sys.argv)>7 else 2
       im=ImageOps.exif_transpose(Image.open(p)).convert('RGB')
       W,H=im.size
       c=im.crop((int(x0*W),int(y0*H),int(x1*W),int(y1*H)))
       c=c.resize((int(c.size[0]*s),int(c.size[1]*s)),Image.LANCZOS)
       c.save(out,quality=92); print(W,H,c.size)
       ```
     - `crop.py` — similar, without scaling.
     - `gold/P01.json` … `gold/P11.json` — gold transcriptions. Schema: page_id, file, student, task_types, task_labels, legibility, lines (verbatim LaTeX, crossed-out as ~~…~~), drawings, other_marks, answers, student_errors_to_preserve, doubts [{line, read, alt, critical, note}].
       - P01 Матвей 16_02 r1 (15): errors −6/x; answer (−∞;3)∪(2;3]. Critical doubts on missing minus signs.
       - P02 Матвей 16_02 r2 (15): ≤ flipped to ≥; doubt x(x+11)−(x−11) vs (x+11).
       - P03 Матвей 08_12 r1 (13): lost minus sign → cos x = √2/2; б) π, 2π, 7π/4.
       - P04 Матвей 04_05 r1 (16 table, 21/11 S).
       - P05 Матвей 04_05 r2: 16 end S=3; crossed-out 15; 13 with wrong б) 17π/6, 13π/3, 23π/6.
       - P06 Матвей 04_05 r4 (15): extra −1 in a line; answer (−∞;−1−log₃5]∪[1;+∞).
       - P07 Игорь 20_04 r2 (16): r = 7%.
       - P08 Игорь 20_04 r4 (16): S = 36 mln; doubt 8 vs 6.
       - P09 Игорь 20_04 r5 (16): differentiated payments used for an equal-payments task; 268400 rubles.
       - P10 Игорь 11_05 r3 (16): answer "7%" is a unit error.
       - P11 Дмитрий 20_01 r1 (13): −2 sin x²; series πn, 5π/4, 7π/4; doubt 5 vs 6.
   - Staged gold source images are under `/mnt/user-data/uploads/ege_checker_works/<student>/<date>_<student>/решениеN.jpg`.

4. Errors and fixes:
   - Chrome batch actions failed ("Couldn't determine which page") → re-ran tabs_context and smaller batches.
   - React number inputs didn't register with triple_click+type → fixed by click, cmd+a, Backspace, type, Tab.
   - Sokrat form reset when a click landed while the page re-rendered; the title got a stray "ммм" → re-filled via JS checks, then cmd+a retype.
   - FIPI PDFs: 403 from the proxy in both the cloud and Mac device_bash → used WebFetch for content; she needs to download in the browser (or permit a Chrome download).
   - HF API: 403 via curl → used Chrome JS fetch of resolve URLs. FERMAT is gated and needs her HF login and acceptance.
   - The built-in browser has no file upload → Sokrat hands-on test abandoned by the user.
   - CSV filename had a Latin "o" and unquoted commas → rewrote with Python csv and a correct Cyrillic name.
   - User feedback incorporated:
     - Mandatory tutor confirmation isn't "automatic" → clarified it is done at assignment time and only on failed verification.
     - Gemini shouldn't be the product model → Gemini is the ceiling only; fine-tuning of the reader added.
     - [?a|b] inline markers would be noisy → clean text plus a separate doubts list, filtered by critical places.

5. Problem Solving:
   - Solved:
     - Competitor analysis (Sokrat: part 2 is a draft only, no math FIPI criteria presets, key field plus "Сократ решит сам").
     - Legal overview.
     - Open-data survey (School Notebooks RU has virtually no math; FERMAT recipe fits; MathWriting is NC).
     - Architecture decision: two-stage reader and grader; gold set first.
   - Ongoing: the gold transcription set for the reader bake-off.

6. All user messages:
   - "хочу сделать решение которое можно будет продавать, вот уже анализ рынка / например думаю можно углубиться в самые популярные номера 2 части егэ (13 15 16) и научить модельки идеально их проверять но я не знаю нужно ли кому такое будет и достаточно ли силная боль"
   - "ну вот мне сначала нужно качественно обучить модель. это лучше сделать на этих трех заданиях или попробовать все сразу"
   - "я вот планирую загрузить в модель все критерии фипи, всю информацию и все подсказки по чтению критерий и короче весь весь весь контекст который найду, но как лучше это в целом будет оформить подумай? если я попробую предлагать репетиторам и маленьким онлайн школам, то в каком формате это лучше будет делать?"
   - "ну у каждого репетитора же свои задачи, я не смогу свой банк и ключи всем продвигать"
   - "…ну это не так круто как то что модель все за тебя делает / вообще кстати … можно заранее ведь договвариватьсч о том что они свою базу заданий скинут … / а ну вот да / насчет ключей только не совсем поняла что за автоключ"
   - "…ну нет ты же говоришь что репетитор будет подтверждать сам, как я тогда могу претендовать на автоматическую проверку, если некоторые из учеников не получат задания сразу же а будут ждать пока репетитор окнет"
   - PROJECT_CONTEXT.md upload: "это вот немного контекста … сейчас же у меня немного другая цель - именно специализирвоаться на конкретных заданиях и попытаьься продать свое решение / изучи и скажи кратко что думаешь"
   - "ненене с тобой я бы вообще другую модель сделала и вообще как бы все с нуля более тщательно и более глубоко все анализируя / контекст тебе просто для понимания и каких то идей"
   - "насчет отправки данных - на фотках персональных данных нет так что все хорошо с этим / так окей ладно сейчас мне еще хочется проанализировать конкурентов - ты можешь сам через хром потыкаться куда то? / и видишь кстати папки с работами?"
   - "запроси доступ к папкам / в сократа вошла, протести его и остальное все что в твоих силах сделай"
   - "вошла"
   - "ладно давай без сократа"
   - "ну да допиши / и подумай тщательно что можно дальше делать / может работы разметить, выделить задания / тебе же видны они?"
   - "что такое бейзлайн? это первый шаг в котором мы будем использовать модель, верно?"
   - "я могу ксттаи еще данных накопать, надо ведь?"
   - "[quote about FIPI] давай / я конечно щас поспрашиваю, но я не уверена что быстро наберутся, а начать все делать хочется уже сейчас / может взять какие то промптики и фишщки из моего старого проекта и просто подстраивать их под новый формат? запроси доступ ко всему что нужно … проанализируй там код и тд и предложи новые пути для именно нашего проекта, как будет шграмотнее и профессиональнее всего поступить"
   - Mid-turn: "ну стой почему мы сразу снова гемини решили брать? … а сейчас должно получиться продающее, которое должно максимально четко и точно все проверять, а значит и. максимално четко распознавать / может быть в этот раз стоит взять какую то модель которую можно дообучать по почерку? и в целом можно дообучать, чтобы она была максимально гибкой четкой и послушной"
   - Pasted HTR tips text + "вот просто полезная инфа"
   - "Температура 0 … ххотя блин это для русского полезно а для математики мб опасно ну слишком много шумных символов будет"
   - xlsx upload: "кстати вот ребята из школково таким файликом поделились / и скоро я смогу получить много работ еще"
   - "расскажи о законной стороне использования чужих работ"
   - "эх нет они не согласятся. есть ли модет открытые данные?"
   - "[quote about audit] ну так я буду делать сто проц но сейчас же нам именно для обучения нужны данные / [quote] Могу посмотреть School Notebooks и FERMAT… посмотри"
   - "так стоп ну смотри / во первых нет ли уже качественных обученных моделей которые прям хорошо распознают цифры и почерк? … во вторых ну мы же получается можем обучить модель на фермате, и обучить хорошо, и опять же поверх нее для именно проверки по критериям использовать другую моелль"
   - "[quote about gold set] ну давай но ты точно уверен что это целесообразно? может уже сразу начать обучать на фермате? или стоит попробовать сторонние читатели"
   - "да"

7. Pending Tasks:
   - Task #9 (in progress): transcribe the remaining gold pages verbatim, P12 onward:
     - Дмитрий: 20_01_2026 r3 (current), 20_01_2026 r4, 09_01_2026 r2, 09_01_2026 r6
     - Докка: 20_01_2026 r2, 30_11_2025 r4, 30_11_2025 r5, 10_12_2025 r9
     - Иван: 31_01_2026 r1, 18_02_2026 r4, 19_01_2026 r3, 19_01_2026 r4
     - Павел: 02_12_2025 r5, 28_04_2026 r1, 28_04_2026 r4

     All are already staged under /mnt/user-data/uploads/ege_checker_works/.
   - Task #10: build a review HTML for Ksenia (photo alongside transcription, doubts highlighted, downscaled images). Commit it plus the gold JSONs into `ege_checker_works/_разметка_13_15_16/ocr_gold/`.
   - Then propose pre-registered reader thresholds (e.g., ≥98% answers read, ≤2% autocorrect) and the bake-off: Yandex Vision `handwritten` / `math-markdown`, an open VLM, Gemini as ceiling. This needs her Yandex AI Studio key.
   - Her parallel to-dos, already suggested:
     - Accept FERMAT terms on HF.
     - Get a Yandex AI Studio key.
     - Look at GPU rental.
     - Download FIPI methodical PDFs (2023–2026) into `_фипи_материалы` via the browser.
   - Leftovers the user may clean up: the Sokrat test student and homeworks "Тест A/B"; the built-in browser is logged in as the student.

8. Current Work:
   Transcribing gold pages. P01–P11 are written to scratchpad/gold/. I had just viewed Дмитрий/20_01_2026_Дмитрий/решение3.jpg (P12) and was about to write P12.json. Content read:
   - sin2x + 2sin(−x) + cos(−x) − 1 = 0
   - sin2x + 2 ~~?~~ −sinx + cosx − 1~~?~~ = 0 (heavily overwritten symbols after "2" and after "−1")
   - sin2x − 2sinx + cosx = 1
   - 2sinx·cosx − 2sinx + cosx = 1
   - 2sinx·cosx − 2sinx + cosx − 1 = 0
   - (cosx − 1)·(2sinx + 1) = 0
   - cosx − 1 = 0
   - 2sinx + 1 = 0
   - x = 7π/6 + 2πn, n∈Z
   - x = 11π/6 + 2πn, n∈Z

   Student error to preserve: the cos x = 1 series (x = 2πn) is lost in the final series. The condition is likely Матвей's 08_12 "Задание 3: sin2x+2sin(−x)+cos(−x)−1=0, б) [2π; 7π/2]". The factoring is correct.

9. Optional Next Step:
   Continue exactly where I left off. Write gold/P12.json for Дмитрий 20_01 r3 with the lines above, then transcribe the remaining pages (Дмитрий 20_01 r4 and 09_01 r2, r6; Докка; Иван; Павел), zooming with zoom.py where needed. After that, build the review HTML and commit everything to `_разметка_13_15_16/ocr_gold/`.

   The user's last instruction was "да" in reply to my question "Начинаю отбирать и расшифровывать страницы?" The agreed plan was: "Эталон (я): 25 страниц — по 5 от пяти учеников, поровну 13/15/16… Сверка (ты): проверяешь только мои помеченные места и ответы".

## Ксения

`/export session.md`

## Ксения

Continue from where you left off.

> 🔧 `Bash` — Check gold files and staged images

> 🔧 `Bash` — List staged files per folder

> 🔧 `Bash` — Save P12 and prepare next page view

> 🔧 `Bash` — Inspect session transcript structure

> 🔧 `Bash` — Look for earlier transcript files

> 🔧 `Bash` — Preview transcript message entries

> 🔧 `Bash` — Preview transcript message entries

> 🔧 `Bash` — Build session.md export from transcript
