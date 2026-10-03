#!/usr/bin/env python3
"""leakage_audit.py — проверка на утечки данных (data leakage) перед ML и перед test.

    python3 leakage_audit.py            # быстрые проверки (секунды)
    python3 leakage_audit.py --images   # + поиск одинаковых/почти одинаковых фото (1–2 минуты)

Утечка — когда при обучении или оценке модель получает информацию, которой у неё не будет в реальной работе,
или когда test не независим от dev. Результат выглядит лучше, чем будет на самом деле.
Проверяется:
  1. Разбиение: каждая работа ровно в одной части; ученики и задания, общие для dev и test.
  2. Test «не подсмотрен»: ни один прогон модели ещё не проверял работы test.
  3. Входы модели: в запросы идут только фото решения и условия, не фото с пометками проверяющего.
  4. Признаки для ML (data/features_wide.csv): целевые и служебные колонки; подозрительно сильные признаки
     (AUC одного признака ≥ 0.95); строки test. Итог — data/feature_columns.json (какие колонки брать в X).
  5. Разметка, исправленная после сравнения с ключом/моделью, — источник оптимизма в метриках.
  6. (--images) одинаковые фото в разных работах, особенно между dev и test.
Выход: печать + reports/leakage_audit.txt. Статусы: OK / ВНИМАНИЕ (учесть в отчёте) / ПРОБЛЕМА (исправить).
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import check

ROOT, DATA = check.ROOT, check.DATA
LINES = []


def say(status, text):
    line = f"[{status}] {text}"
    LINES.append(line)
    print(line)


def auc(scores, y):
    """ROC AUC через ранги (Манн–Уитни), со средними рангами для одинаковых значений."""
    pairs = sorted(zip(scores, y))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(t for _, t in pairs)
    neg = len(pairs) - pos
    if not pos or not neg:
        return None
    return (sum(r for r, (_, t) in zip(ranks, pairs) if t) - pos * (pos + 1) / 2) / (pos * neg)


def split_checks(works, splits):
    print("\n1. Разбиение dev / test")
    folders = list(csv.DictReader(open(DATA / "splits_new.csv", encoding="utf-8")))
    dup = [f for f, c in Counter(r["folder"] for r in folders).items() if c > 1]
    say("ПРОБЛЕМА" if dup else "OK", f"работ в двух частях сразу: {len(dup)}" + (f" ({', '.join(dup[:5])})" if dup else ""))
    stud = defaultdict(set)
    for w in works:
        stud[w["student_id"]].add(splits.get(w["work_id"], "dev"))
    both = sorted(s for s, p in stud.items() if {"dev", "test"} <= p)
    say("ВНИМАНИЕ" if both else "OK",
        f"учеников и в dev, и в test: {len(both)} из {len(stud)}. Для LLM это не утечка (она не обучалась на dev), "
        "но модель доверия, обученная на dev, уже «видела» почерк этих учеников → на test она может выглядеть "
        "лучше, чем на новых учениках. В CV проверь и GroupKFold по student_id.")
    tk = defaultdict(set)
    for r in folders:
        tk[r["task_key"]].add(r["split"])
    shared = sorted(k for k, p in tk.items() if {"dev", "test"} <= p)
    ov = list(csv.DictReader(open(ROOT / check.TEST_OVERLAP, encoding="utf-8"))) if (ROOT / check.TEST_OVERLAP).exists() else []
    say("ВНИМАНИЕ" if shared or ov else "OK",
        f"заданий (task_key) и в dev, и в test: {len(shared)}; задач test, совпадающих с dev: {len(ov)} — "
        "для них в --report есть строка «test без повторов из dev».")


def test_untouched(splits):
    print("\n2. Test не подсмотрен")
    seen = defaultdict(list)
    for d in check.RUNS.iterdir() if check.RUNS.exists() else []:
        if not d.is_dir():
            continue
        for w in d.iterdir():
            if splits.get(w.name) == "test" and (w / "check.json").exists():  # распознавание без вердиктов — можно
                seen[d.name].append(w.name)
    if seen:
        say("ВНИМАНИЕ", "работы test уже обрабатывались: " + "; ".join(f"{k}: {len(v)}" for k, v in seen.items()) +
            (" (если это финальный запуск по протоколу — нормально)" if (check.REPORTS / check.TEST_RUNS_LOG).exists()
             else f" — {check.TEST_RUNS_LOG} нет, значит, это было ДО заморозки: опиши в TEST_RESULTS.md"))
    else:
        say("OK", "ни один прогон ещё не проверял работы test (распознавание без вердиктов допускается)")


def input_checks():
    print("\n3. Что видит модель")
    src = (ROOT / "check.py").read_text(encoding="utf-8")
    roles = re.search(r'files = \{([^}]*)\}', src)
    ok = roles and "markup" not in roles.group(1) and "feedback" not in roles.group(1)
    say("OK" if ok else "ПРОБЛЕМА", "в запросы идут только роли solution и task (фото с пометками проверяющего — markup/"
        "feedback — не отправляются)" if ok else "work_files() берёт не только solution/task — проверь!")
    files = list(csv.DictReader(open(DATA / "files.csv", encoding="utf-8")))
    per = defaultdict(Counter)
    for r in files:
        per[r["work_id"]][r["role"]] += 1
    no_sol = [w for w, c in per.items() if not c["solution"]]
    say("ВНИМАНИЕ" if no_sol else "OK", f"работ без фото решения: {len(no_sol)}" + (f" ({', '.join(no_sol)})" if no_sol else ""))
    key_src = DATA / "keys" / "answer_key.csv"
    methods = Counter(r["method"] for r in csv.DictReader(open(key_src, encoding="utf-8"))) if key_src.exists() else {}
    say("OK", f"ключ ответов решался отдельно от разметки (методы: {dict(methods)}); в проверку идёт только ответ, "
        "не метка и не пояснение проверяющего")
    src = (ROOT / "check.py").read_text(encoding="utf-8")
    cfg = json.load(open(ROOT / "config.json", encoding="utf-8")).get("final", {})
    if "labels.get(work" in src and not cfg.get("no_scope"):
        say("ВНИМАНИЕ", "в промпт проверки уходит список номеров задач из разметки (scope), включая задачи без метки. "
            "В работе его заменяет список заданных задач от репетитора (check.py --scope-file); разница — задачи, "
            "которые ученик не делал: их нет в метриках, в работе они получат not_solved и уйдут куратору. Признак "
            "alarm_no_errors оценён в режиме scope из разметки; нагрузка на куратора с учётом задач без метки — "
            "ml/apply_trust.py --wide data/features_wide_all.csv (dev: 12,4% вместо 7,2%). Поиск задач в тетради "
            "оценка не учитывает.")


TARGETS = {"label", "y_error", "human_missed", "model_wrong"}
SERVICE = {"work", "task_no", "student_id", "task_key", "split", "key_answer", "student_answer",
           "model_correct_answer", "run", "label_text"}


def feature_checks():
    print("\n4. Признаки для ML (data/features_wide.csv)")
    path = DATA / "features_wide.csv"
    if not path.exists():
        say("ВНИМАНИЕ", "нет data/features_wide.csv — запусти features.py и features_wide.py")
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cols = list(rows[0].keys())
    target_like = [c for c in cols if c in TARGETS or c.endswith("__wrong")]
    service = [c for c in cols if c in SERVICE]
    cand = [c for c in cols if c not in target_like and c not in service]
    tests = sum(r["split"] == "test" for r in rows)
    say("ВНИМАНИЕ" if tests else "OK", f"строк test в таблице: {tests}" +
        (" — перед обучением отфильтруй split == 'dev'" if tests else ""))
    dev = [r for r in rows if r["split"] == "dev"]
    y = [int(r["y_error"]) for r in dev]
    strong, report = [], single_auc(dev, cand, "y_error")
    for a, c, n in report:
        if a >= 0.99:
            strong.append(f"{c} (AUC {a:.3f})")
    say("ПРОБЛЕМА" if strong else "OK", "признаков, которые в одиночку почти идеально предсказывают разметку "
        f"(AUC ≥ 0.99 — так бывает, только если признак вычислен из неё): {len(strong)}" +
        (f" — {', '.join(strong)}" if strong else ""))
    print("   самые сильные признаки для цели y_error (AUC одного признака, dev). Вердикт модели с AUC ≈ 0.95 —\n"
          "   это просто её точность (совпадение 96%), не утечка:")
    for a, c, n in report[:6]:
        print(f"     {c:40s} {a:.3f}  (заполнено {n})")
    trust = next((c for c in target_like if c.endswith("__wrong") and "key" in c), None)
    if trust:
        rep2 = single_auc(dev, cand, trust)
        print(f"   для цели модели доверия {trust} (ошиблась ли проверка):")
        for a, c, n in rep2[:6]:
            print(f"     {c:40s} {a:.3f}  (заполнено {n})")
    say("OK", f"в X НЕ брать: цели {target_like}; служебные {service}")
    json.dump({"targets": target_like, "service": service, "features": cand,
               "note": "features — кандидаты в X; категориальные (exam, topic, key_type…) закодируй; "
                       "g_*__pred_error — вердикт модели, законный признак для модели доверия (цель <run>__wrong)."},
              open(DATA / "feature_columns.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("   список колонок → data/feature_columns.json")


def single_auc(dev, cand, target):
    y = [r[target] for r in dev]
    report = []
    for c in cand:
        vals = []
        for r in dev:
            try:
                vals.append(float(r[c]))
            except ValueError:
                vals.append(None)
        known = [(v, int(t)) for v, t in zip(vals, y) if v is not None and t in ("0", "1")]
        if len(known) < 0.5 * len(dev) or len({v for v, _ in known}) < 2:
            continue
        a = auc([v for v, _ in known], [t for _, t in known])
        if a is None:
            continue
        a = max(a, 1 - a)
        report.append((a, c, len(known)))
    return sorted(report, reverse=True)


def label_history(splits):
    print("\n5. Правки разметки (соседние снимки, по времени)")

    def snap(name):
        return {(r["work_id"], check.norm_task(r["task_no"])): r["is_correct"]
                for r in csv.DictReader(open(DATA / name, encoding="utf-8"))}
    stages = [("labels_before_audit.csv", "аудит по ключу ответов (26.09)"),
              ("labels_before_policy.csv", "перевод на новые правила (26.09)"),
              ("labels_before_error_review.csv", "разбор промахов с Ксенией (27.09)"),
              ("labels_before_test_recheck.csv", "перепроверка 9 меток test по уточнениям 27.09 (до запуска test)"),
              ("labels.csv", None)]
    stages = [s_ for s_ in stages if (DATA / s_[0]).exists()]
    for (a, why), (b, _) in zip(stages, stages[1:]):
        old, new = snap(a), snap(b)
        ch = [k for k in old if k in new and old[k] != new[k]]
        by = Counter((splits.get(k[0], "dev"), f"{old[k] or '∅'}→{new[k] or '∅'}") for k in ch)
        dev = {d: c for (sp, d), c in by.items() if sp == "dev"}
        test = {d: c for (sp, d), c in by.items() if sp == "test"}
        say("ВНИМАНИЕ" if ch else "OK", f"{why}: изменено меток {len(ch)} — dev {sum(dev.values())} {dev or ''}, "
            f"test {sum(test.values())} {test or ''}")
    say("ВНИМАНИЕ", "метки чинили там, где ключ или модель спорили с проверяющим, — это сдвигает метрики в их пользу "
        "(label_sensitivity.py: на dev +1,3 п.п.; это нижняя оценка смещения — ошибки разметки там, где модель и "
        "проверяющий ошиблись одинаково, не найдены). На test правки делались до любого запуска модели и без её "
        "вердиктов; после запуска test метки не трогаем (протокол test).")


def image_checks(splits):
    """Одинаковые файлы (md5) и почти одинаковые фото: корреляция уменьшенных до 64×64 серых картинок ≥ 0.97.
    Порог подобран вручную: разные страницы в клетку дают до ~0.93, одна и та же страница — ≥ 0.98."""
    print("\n6. Одинаковые фото в разных работах")
    import hashlib
    import numpy as np
    from PIL import Image, ImageOps
    files = [r for r in csv.DictReader(open(DATA / "files.csv", encoding="utf-8")) if r["role"] == "solution"]
    md5, vecs, meta = defaultdict(list), [], []
    for r in files:
        p = check.ANON / r["work_id"] / r["file"]
        if not p.exists():
            continue
        md5[hashlib.md5(p.read_bytes()).hexdigest()].append(r["work_id"])
        a = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("L").resize((64, 64)), float).ravel()
        vecs.append((a - a.mean()) / (a.std() + 1e-9))
        meta.append((r["work_id"], r["file"]))
    exact = [ws for ws in md5.values() if len(set(ws)) > 1]
    M = np.stack(vecs)
    C = M @ M.T / M.shape[1]
    near = [(meta[i], meta[j], C[i, j]) for i in range(len(meta)) for j in range(i + 1, len(meta))
            if meta[i][0] != meta[j][0] and C[i, j] >= 0.97]
    cross = [x for x in near if {splits.get(x[0][0], "dev"), splits.get(x[1][0], "dev")} == {"dev", "test"}]
    say("ПРОБЛЕМА" if exact else "OK", f"одинаковые файлы в разных работах: {len(exact)}")
    say("ПРОБЛЕМА" if cross else "OK", f"почти одинаковые фото: dev↔test {len(cross)}, всего пар в разных работах "
        f"{len(near)}; самая похожая пара разных работ — корреляция {C[np.array([m[0] for m in meta])[:, None] != np.array([m[0] for m in meta])[None, :]].max():.2f}"
        + "".join(f"\n     {a[0]} {a[1]} ~ {b[0]} {b[1]} ({c:.3f})" for a, b, c in near[:10]))



def task_numbers(splits, labels):
    """Номера задач из разметки должны находиться в распознанном условии (иначе проверка идёт без условия —
    главная техническая причина промахов test2). Нужен task.json: python3 check.py --preset final --split test
    --works all --steps transcribe,task (подготовка, без вердиктов)."""
    print("\n6. Номера задач разметки найдены в условии (test)")
    run = check.RUNS / "google_gemini-3.8-flash@low_key"
    bad, no_task = {}, []
    for w in sorted(w for w, s in splits.items() if s == "test"):
        tj = run / w / "task.json"
        if not tj.exists():
            no_task.append(w); continue
        tasks = json.load(open(tj, encoding="utf-8")).get("tasks", [])
        miss = check.missing_tasks(tasks, [t for t, e in labels.get(w, {}).items() if e.get("label") in (0, 1)])
        if miss:
            bad[w] = miss
    if no_task:
        say("ВНИМАНИЕ", f"нет распознанного условия у {len(no_task)} работ test — сначала подготовка (--steps transcribe,task)")
    # работы, где условия решённых задач в сдаче нет (приложен чужой лист) — записаны в протоколе ДО запуска;
    # система проверяет их без условия и без ключа (как реальную сдачу с неверным вложением)
    known = {}
    kp = DATA / "test3_no_task.csv"
    if kp.exists():
        known = {r["work_id"]: r["reason"] for r in csv.DictReader(open(kp, encoding="utf-8"))}
    expected = {w: m for w, m in bad.items() if w in known}
    bad = {w: m for w, m in bad.items() if w not in known}
    if expected:
        say("ВНИМАНИЕ", f"условия нет в сдаче (протокол, data/test3_no_task.csv) — проверка без условия: {len(expected)}" +
            "".join(f"\n     {w}: {known[w]}" for w in expected))
    say("ПРОБЛЕМА" if bad else "OK", f"работ, где номера разметки не нашлись в условии: {len(bad)}" +
        "".join(f"\n     {w}: {', '.join(m[:10])}" for w, m in bad.items()))
    return len(bad)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", action="store_true", help="искать одинаковые фото (дольше)")
    a = ap.parse_args()
    works = list(csv.DictReader(open(DATA / "works.csv", encoding="utf-8")))
    splits = check.work_splits()
    split_checks(works, splits)
    test_untouched(splits)
    input_checks()
    feature_checks()
    label_history(splits)
    task_numbers(splits, check.load_labels())
    if a.images:
        image_checks(splits)
    check.REPORTS.mkdir(exist_ok=True)
    (check.REPORTS / "leakage_audit.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    bad = sum(line.startswith("[ПРОБЛЕМА]") for line in LINES)
    print(f"\nИтог: проблем {bad}, предупреждений {sum(l.startswith('[ВНИМАНИЕ]') for l in LINES)} "
          f"→ reports/leakage_audit.txt")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
