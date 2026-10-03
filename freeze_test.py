#!/usr/bin/env python3
"""freeze_test.py — заморозка перед единственным запуском test текущего раунда (check.TEST_PROTOCOL, сейчас TEST_PROTOCOL_3.md).

    python3 freeze_test.py                          # заморозить: хэши файлов + настройки набора final + git-тег
    python3 freeze_test.py --extra other.file       # добавить в заморозку ещё файлы (модель доверия — всегда)
    python3 freeze_test.py --amend "причина"        # отклонение от протокола ПОСЛЕ начала test (записывается в лог)

Что делает:
  1) проверяет, что в протоколе заполнены все поля «___» и что число задач и ошибок test в labels.csv
     совпадает с п. 2 протокола;
  2) считает SHA-256 файлов, от которых зависит результат (код, config.json, разметка, ключ, разбиение, протокол);
  3) сохраняет их и итоговые настройки `--preset final --split test` в check.TEST_LOCK (reports/test3_lock.json);
  4) если есть git — делает коммит и тег before-test3 (при --amend — before-test3-amend-N).
check.py с --split test сверяет файлы и настройки с заморозкой и без неё не запустится.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime

import check


# модель доверия и код признаков: без них apply_trust.py не запустится на test
TRUST_FILES = ["ml/trust_model_v5_curpol.pkl", "ml/trust_threshold_v5_curpol.json", "ml/trust_features.py",
               "ml/apply_trust.py", "features.py", "features_wide.py"]


def test_counts():
    """Число размеченных задач test и задач с ошибкой в текущем labels.csv."""
    labels, splits = check.load_labels(), check.work_splits()
    test = [e for w, t in labels.items() if splits.get(w) == "test" for e in t.values() if e["label"] in (0, 1)]
    return {"tasks": len(test), "errors": sum(e["label"] == 0 for e in test)}


def git(*a):
    return subprocess.run(["git", *a], cwd=check.ROOT, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="доп. файлы в заморозку; модель доверия и код признаков (TRUST_FILES) добавляются всегда")
    ap.add_argument("--amend", default="", help="причина отклонения, если test уже запускался")
    ap.add_argument("--no-git", action="store_true", help="не делать git-коммит и тег")
    a = ap.parse_args()

    name = check.TEST_PROTOCOL
    proto = check.ROOT / name
    if not proto.exists():
        sys.exit(f"Нет {name}")
    todo = [ln.strip() for ln in proto.read_text(encoding="utf-8").splitlines() if "___" in ln]
    if todo:
        sys.exit(f"В {name} не заполнено:\n  " + "\n  ".join(todo))
    counts = test_counts()
    m = re.search(r"(\d+) размеченных задач, из них (\d+) с ошибкой", proto.read_text(encoding="utf-8"))
    if not m or (int(m.group(1)), int(m.group(2))) != (counts["tasks"], counts["errors"]):
        sys.exit(f"Разметка test в labels.csv (задач {counts['tasks']}, с ошибкой {counts['errors']}) не совпадает с "
                 f"{name}, п. 2 ({m.group(1) + ' / ' + m.group(2) if m else 'строка не найдена'}). "
                 f"Сначала разберись, какой файл устарел")

    runs_log = check.REPORTS / check.TEST_RUNS_LOG
    started = runs_log.exists() and runs_log.read_text(encoding="utf-8").strip()
    old = json.load(open(check.TEST_LOCK, encoding="utf-8")) if check.TEST_LOCK.exists() else None
    if started and not a.amend:
        sys.exit(f"Test уже запускался (reports/{check.TEST_RUNS_LOG}). Перезаморозка — только как отклонение от протокола:\n"
                 "  python3 freeze_test.py --amend \"что и почему меняем\"")

    extra = list(dict.fromkeys(TRUST_FILES + (old or {}).get("extra", []) + a.extra))
    files = list(dict.fromkeys(check.FROZEN_FILES + extra))
    missing = [f for f in files if not (check.ROOT / f).exists()]
    if missing:
        sys.exit("Нет файлов: " + ", ".join(missing))
    _, args = check.parse_args(["--preset", "final", "--split", "test", "--works", "all"])
    lock = {"frozen_at": datetime.now().isoformat(timespec="seconds"),
            "command": "python3 check.py --preset final --split test --works all",
            "settings": {k: getattr(args, k) for k in check.RUN_SETTINGS},
            "files": {f: check.sha256(check.ROOT / f) for f in files},
            "extra": extra, "test_counts": counts}

    if a.amend:
        changed = [f for f, h in lock["files"].items() if not old or old["files"].get(f) != h]
        with open(check.REPORTS / check.TEST_RUNS_LOG.replace("runs", "deviations"), "a", encoding="utf-8") as f:
            f.write(f"{lock['frozen_at']}\tпричина: {a.amend}\tизменены: {', '.join(changed) or '—'}\n")
        lock["amendments"] = (old or {}).get("amendments", []) + [{"at": lock["frozen_at"], "reason": a.amend,
                                                                   "changed": changed}]

    check.REPORTS.mkdir(exist_ok=True)
    save = lambda: json.dump(lock, open(check.TEST_LOCK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    save()  # сам файл заморозки тоже попадает в коммит
    tag = None
    if not a.no_git and shutil.which("git") and git("rev-parse", "--is-inside-work-tree").returncode == 0:
        base = "before-" + check.TEST_LOCK.stem.replace("_lock", "")  # before-test2
        tag = base if not a.amend else f"{base}-amend-{len(lock['amendments'])}"
        if git("rev-parse", "-q", "--verify", f"refs/tags/{tag}").returncode == 0 and not started:
            git("tag", "-d", tag)  # перезаморозка до запуска test — тег переносим
        git("add", "-A")
        c = git("commit", "-q", "--allow-empty", "-m", f"Заморозка перед test ({tag})" + (f": {a.amend}" if a.amend else ""))
        t = git("tag", tag)
        if c.returncode or t.returncode:
            print("git: не получилось сделать коммит/тег —", (c.stderr or t.stderr).strip())
            tag = None
        else:
            lock["git_commit"] = git("rev-parse", "HEAD").stdout.strip()
            lock["git_tag"] = tag
    else:
        print("git не найден или папка не репозиторий — заморозка только по хэшам")

    save()
    print(f"Заморожено {lock['frozen_at']}: {len(files)} файлов → reports/{check.TEST_LOCK.name}" + (f", git-тег {tag}" if tag else ""))
    print("Настройки test:", ", ".join(f"{k}={v}" for k, v in lock["settings"].items()))
    print(f"Команда запуска: {lock['command']}")
    print(f"Разметка test: задач {counts['tasks']}, с ошибкой {counts['errors']} — совпадает с {name}, п. 2")


if __name__ == "__main__":
    main()
