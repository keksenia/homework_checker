"""
organize.py — превращает сырые папки с домашками в аккуратный анонимизированный датасет.

Вход:
    data/raw/2026-09-15_Маша/      # одна папка на домашку: ДАТА_ИМЯ[_N]
        IMG_1234.jpg, IMG_1235.jpg, условие.png, ...
    сбор_домашек.xlsx               # таблица сбора (или .csv с теми же колонками)

Запуск:
    pip install pandas openpyxl pillow            # + pillow-heif, если есть фото с iPhone (.heic)
    python organize.py --raw data/raw --table сбор_домашек.xlsx --out data

Выход (в --out):
    anonymized/W001/01.jpg ...      # копии без EXIF (гео, модель телефона, время), с правильным поворотом
    works.csv                       # одна строка на работу: work_id, student_id, дата, класс, тема, ...
    files.csv                       # файлы работ с ролями (solution/task/markup/feedback); файлы с именами
                                    # решение_N / условие / разметка_N / фидбек получают роль автоматически
    labels.csv                      # пустой шаблон разметки по задачам (создаётся один раз, не перезаписывается)
    private/                        # ТОЛЬКО ЛОКАЛЬНО: соответствие имён и id, исходные имена файлов
    .gitignore                      # закрывает raw/ и private/ от git

Скрипт идемпотентный: id учеников и работ стабильны между запусками, уже обработанные
папки пропускаются (флаг --force пересобирает их заново). Метаданные из таблицы
обновляются при каждом запуске, а колонка redacted в works.csv сохраняется.

ВАЖНО: имена на самих фото скрипт не видит. После запуска закрась их в anonymized/
и поставь redacted = 1 в works.csv. Работы с redacted = 0 нельзя отправлять во внешние API.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_OK = True
except ImportError:
    HEIF_OK = False

# Имя папки: ДАТА_ИМЯ[_N]. Дата в формате ГГГГ-ММ-ДД или ДД_ММ_ГГГГ (разделители _ . -)
FOLDER_RE_ISO = re.compile(r"^(\d{4})[-_.](\d{2})[-_.](\d{2})_(.+?)(?:_(\d+))?$")
FOLDER_RE_DMY = re.compile(r"^(\d{1,2})[-_.](\d{1,2})[-_.](\d{4})_(.+?)(?:_(\d+))?$")


def parse_folder(name: str) -> tuple[str, str] | None:
    """Возвращает (дата в ISO, имя ученика) или None."""
    name = name.strip()
    m = FOLDER_RE_ISO.match(name)
    if m:
        y, mo, d, student = m.group(1), m.group(2), m.group(3), m.group(4)
    else:
        m = FOLDER_RE_DMY.match(name)
        if not m:
            return None
        d, mo, y, student = m.group(1), m.group(2), m.group(3), m.group(4)
    try:
        date = pd.Timestamp(year=int(y), month=int(mo), day=int(d)).strftime("%Y-%m-%d")
    except ValueError:
        return None
    return date, student.strip()
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
HEIF_EXT = {".heic", ".heif"}
TEXT_EXT = {".txt", ".md"}
SKIP_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}

# Роль файла определяется по началу его имени, дальше может идти что угодно:
#   «решениеал1.jpg» -> solution_1.jpg, «фидбекм2.docx» -> feedback_2.docx, «условие.txt» -> task.txt
# Номер берётся из цифр в конце имени. Имена учеников после переименования не остаются.
# «Фидбек»/«комментарий» в виде картинки считается разметкой (markup), в виде текста/docx — фидбеком.
ROLE_PREFIXES = [
    ("решени", "solution"),
    ("услови", "task"), ("задани", "task"),
    ("разметк", "markup"), ("исправлени", "markup"), ("проверк", "markup"), ("пометк", "markup"),
    ("фидбек", "feedback"), ("комментари", "feedback"), ("коммент", "feedback"), ("отзыв", "feedback"),
]

TABLE_COLS = {
    "Папка": "folder",
    "Класс": "grade",
    "Тема": "topic",
    "Формат": "source_format",
    "Есть мои исправления": "has_corrections",
    "Есть фидбек ученику": "has_feedback",
    "Комментарий": "comment",
}
WORKS_COLS = ["work_id", "student_id", "date", "grade", "topic", "source_format",
              "n_files", "has_corrections", "has_feedback", "comment", "redacted"]
MANUAL_COLS = ["redacted"]  # колонки, которые ты правишь руками в works.csv
FILES_COLS = ["work_id", "file", "role"]
LABELS_COLS = ["work_id", "task_no", "topic", "is_correct", "error_step", "error_type",
               "is_consequence", "explanation", "checker", "notes"]

warnings: list[str] = []


def warn(msg: str) -> None:
    warnings.append(msg)


# ---------- стабильные id ----------

def load_map(path: Path, key: str, val: str) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    return dict(zip(df[key], df[val]))


def next_id(existing: dict[str, str], prefix: str) -> str:
    nums = [int(v[len(prefix):]) for v in existing.values() if v.startswith(prefix)]
    return f"{prefix}{(max(nums) + 1) if nums else 1:03d}"


def norm_name(name: str) -> str:
    return " ".join(name.split()).casefold()


# ---------- таблица сбора ----------

def read_table(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=list(TABLE_COLS.values()))
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        df = pd.read_excel(path, sheet_name="Сбор", dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    df = df.rename(columns=TABLE_COLS)
    df = df[[c for c in TABLE_COLS.values() if c in df.columns]]
    df = df.dropna(subset=["folder"])
    df["folder"] = df["folder"].str.strip()
    df = df[~df["folder"].str.upper().str.startswith("ПРИМЕР")]
    for col in ("has_corrections", "has_feedback"):
        if col in df:
            df[col] = df[col].str.strip().str.lower().map({"да": 1, "нет": 0}).astype("Int64")
    dups = df["folder"][df["folder"].duplicated()].unique()
    for d in dups:
        warn(f"В таблице папка {d!r} встречается несколько раз — беру первую строку")
    return df.drop_duplicates("folder").set_index("folder")


# ---------- обработка файлов ----------

def strip_image(src: Path, dst_stem: Path) -> Path:
    """Сохраняет картинку без метаданных. Поворот из EXIF применяется к пикселям,
    иначе после удаления EXIF фото с телефона окажутся повёрнутыми."""
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        ext = src.suffix.lower()
        if ext in HEIF_EXT or ext in {".bmp", ".tif", ".tiff"}:
            ext = ".jpg"
        if ext in {".jpg", ".jpeg"}:
            ext = ".jpg"
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            dst = dst_stem.with_suffix(ext)
            im.save(dst, "JPEG", quality=95, subsampling=0)
        elif ext == ".png":
            dst = dst_stem.with_suffix(ext)
            im.save(dst, "PNG")  # без pnginfo текстовые чанки не переносятся
        else:  # .webp
            dst = dst_stem.with_suffix(ext)
            im.save(dst, "WEBP", quality=95)
    return dst


def role_stem(src: Path, used: set[str]) -> tuple[str | None, str | None]:
    """Если имя файла начинается со слова роли, возвращает (новое имя без расширения, роль)."""
    stem = src.stem.strip().lower()
    role = next((r for prefix, r in ROLE_PREFIXES if stem.startswith(prefix)), None)
    if role is None:
        return None, None
    if role == "feedback" and src.suffix.lower() in IMAGE_EXT | HEIF_EXT:
        role = "markup"  # «фидбек» на фото — это размеченная страница, а не текстовый комментарий
    digits = re.search(r"(\d+)\s*$", stem)
    base = role + (f"_{int(digits.group(1))}" if digits else "")
    new, k = base, 2
    while new + src.suffix.lower() in used or (src.suffix.lower() in HEIF_EXT | {".bmp", ".tif", ".tiff"}
                                                and new + ".jpg" in used):
        new, k = f"{base}_{k}", k + 1
    return new, role


def process_work(folder: Path, work_id: str, out_dir: Path) -> list[tuple[str, str, str | None]]:
    """Возвращает список (новое имя, исходное имя, роль)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    files = sorted(p for p in folder.rglob("*")
                   if p.is_file() and not p.name.startswith(".")
                   and p.name.lower() not in SKIP_NAMES)
    result = []
    used: set[str] = set()
    for src in files:
        named, role = role_stem(src, used)
        if role is None:
            warn(f"{work_id}: роль не распознана по имени {str(src.relative_to(folder))!r} "
                 f"— проставь её в files.csv руками")
        stem = out_dir / (named or f"{len(result) + 1:02d}")
        ext = src.suffix.lower()
        rel = str(src.relative_to(folder))
        try:
            if ext in IMAGE_EXT or ext in HEIF_EXT:
                if ext in HEIF_EXT and not HEIF_OK:
                    warn(f"{work_id}: {rel} — формат HEIC, поставь pillow-heif; файл пропущен")
                    continue
                dst = strip_image(src, stem)
            else:
                dst = stem.with_suffix(ext)
                shutil.copy2(src, dst)
                if ext not in TEXT_EXT:
                    warn(f"{work_id}: {rel} — не картинка и не текст, скопирован как есть "
                         f"(метаданные внутри не чищены, проверь вручную)")
        except Exception as e:  # битый файл не должен ронять весь прогон
            warn(f"{work_id}: {rel} — не удалось обработать ({e}); пропущен")
            continue
        result.append((dst.name, rel, role))
        used.add(dst.name)
    if not result:
        warn(f"{work_id}: в папке нет подходящих файлов")
    return result


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--table", type=Path, default=None, help="сбор_домашек.xlsx или .csv")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--force", action="store_true", help="пересобрать уже обработанные работы")
    args = ap.parse_args()

    if not args.raw.is_dir():
        sys.exit(f"Нет папки {args.raw}")

    out, private = args.out, args.out / "private"
    anon = out / "anonymized"
    private.mkdir(parents=True, exist_ok=True)
    anon.mkdir(parents=True, exist_ok=True)

    student_map_p = private / "student_map.csv"
    work_map_p = private / "work_map.csv"
    file_map_p = private / "file_map.csv"

    students = load_map(student_map_p, "name_key", "student_id")
    student_display = load_map(student_map_p, "name_key", "name") if student_map_p.exists() else {}
    works = load_map(work_map_p, "folder", "work_id")
    file_map = pd.read_csv(file_map_p, dtype=str) if file_map_p.exists() else pd.DataFrame(
        columns=["work_id", "file", "original"])

    table = read_table(args.table)
    old_works = (pd.read_csv(out / "works.csv", dtype=str).set_index("work_id")
                 if (out / "works.csv").exists() else pd.DataFrame())
    old_files = (pd.read_csv(out / "files.csv", dtype=str)
                 if (out / "files.csv").exists() else pd.DataFrame(columns=FILES_COLS))

    folders = sorted(p for p in args.raw.iterdir() if p.is_dir() and not p.name.startswith("."))
    work_rows, n_new, n_skipped = [], 0, 0
    auto_roles: dict[tuple[str, str], str] = {}
    seen_folders = set()

    for folder in folders:
        parsed = parse_folder(folder.name)
        if parsed is None:
            warn(f"Папка {folder.name!r}: не удалось разобрать дату и имя "
                 f"(ожидается ДД_ММ_ГГГГ_Имя или ГГГГ-ММ-ДД_Имя) — пропущена")
            continue
        date, name = parsed
        seen_folders.add(folder.name)

        key = norm_name(name)
        if key not in students:
            students[key] = next_id(students, "S")
            student_display[key] = name
        sid = students[key]

        if folder.name not in works:
            works[folder.name] = next_id(works, "W")
        wid = works[folder.name]

        reprocessed = not ((anon / wid).exists() and not args.force)
        if not reprocessed:
            n_skipped += 1
        else:
            pairs = process_work(folder, wid, anon / wid)
            file_map = file_map[file_map["work_id"] != wid]
            file_map = pd.concat([file_map, pd.DataFrame(
                [{"work_id": wid, "file": f, "original": o} for f, o, _ in pairs])], ignore_index=True)
            auto_roles.update({(wid, f): r for f, _, r in pairs if r})
            n_new += 1

        n_files = len(file_map[file_map["work_id"] == wid])
        meta = table.loc[folder.name] if folder.name in table.index else None
        if meta is None and args.table is not None:
            warn(f"{wid} ({folder.name}): нет строки в таблице сбора — метаданные пустые")
        row = {
            "work_id": wid, "student_id": sid, "date": date,
            "grade": None if meta is None else meta.get("grade"),
            "topic": None if meta is None else meta.get("topic"),
            "source_format": None if meta is None else meta.get("source_format"),
            "n_files": n_files,
            "has_corrections": None if meta is None else meta.get("has_corrections"),
            "has_feedback": None if meta is None else meta.get("has_feedback"),
            "comment": None if meta is None else meta.get("comment"),
            "redacted": 0,
        }
        # пересобранная работа снова содержит незакрашенные фото — redacted не переносим
        if wid in old_works.index and not reprocessed:
            for col in MANUAL_COLS:
                v = old_works.at[wid, col] if col in old_works.columns else None
                if pd.notna(v):
                    row[col] = v
        work_rows.append(row)

    for f in table.index:
        if f not in seen_folders:
            warn(f"В таблице есть {f!r}, но такой папки нет в {args.raw}")

    # --- запись ---
    works_df = pd.DataFrame(work_rows, columns=WORKS_COLS).sort_values("work_id")
    for col in ("has_corrections", "has_feedback", "redacted", "n_files"):
        works_df[col] = pd.to_numeric(works_df[col], errors="coerce").astype("Int64")
    works_df.to_csv(out / "works.csv", index=False)

    # files.csv: сохраняем уже проставленные role, если файл не поменялся
    new_files = file_map[["work_id", "file"]].copy()
    new_files = new_files.merge(old_files[["work_id", "file", "role"]], on=["work_id", "file"], how="left")
    rebuilt = set(works_df["work_id"]) if args.force else set()
    new_files.loc[new_files["work_id"].isin(rebuilt), "role"] = None
    for (wid, f), r in auto_roles.items():  # роль из имени файла; ручную не перетираем
        mask = (new_files["work_id"] == wid) & (new_files["file"] == f)
        new_files.loc[mask & new_files["role"].isna(), "role"] = r
    new_files[FILES_COLS].sort_values(["work_id", "file"]).to_csv(out / "files.csv", index=False)

    if not (out / "labels.csv").exists():
        pd.DataFrame(columns=LABELS_COLS).to_csv(out / "labels.csv", index=False)

    pd.DataFrame({"name_key": list(students), "name": [student_display.get(k, k) for k in students],
                  "student_id": list(students.values())}).to_csv(student_map_p, index=False)
    pd.DataFrame({"folder": list(works), "work_id": list(works.values())}).to_csv(work_map_p, index=False)
    file_map.to_csv(file_map_p, index=False)

    gi = out / ".gitignore"
    if not gi.exists():
        gi.write_text("raw/\nprivate/\n", encoding="utf-8")

    # --- отчёт ---
    not_redacted = int((works_df["redacted"].astype(str) != "1").sum())
    print(f"Обработано новых работ: {n_new}, пропущено уже готовых: {n_skipped}")
    print(f"Всего работ: {len(works_df)}, учеников: {works_df['student_id'].nunique()}")
    if warnings:
        print(f"\nПредупреждения ({len(warnings)}):")
        for w in warnings:
            print("  •", w)
    print(f"\nНе закрашены имена на фото: {not_redacted} работ(ы). "
          f"Закрась их в {anon}/ и поставь redacted = 1 в works.csv.")


if __name__ == "__main__":
    main()
