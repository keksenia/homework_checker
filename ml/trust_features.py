"""trust_features.py — признаки модели доверия. ЕДИНСТВЕННОЕ место, где они считаются: и ноутбук (обучение),
и apply_trust.py (применение к test) импортируют отсюда, чтобы обучение и применение не разошлись.

Вход — строки data/features_wide.csv (одна строка = задача). Признаки финальной модели берутся только из прогона
итоговой системы g_low_key (Gemini с эталонными ответами); ds_high_img (DeepSeek) — только для сравнения в ноутбуке.
"""
import re

import numpy as np
import pandas as pd

MAIN, SECOND = "g_low_key", "ds_high_img"
CANDIDATES = [
    f"{MAIN}__pred_error",          # проверка сказала «ошибка» или «не решено» (1) либо «верно» (0)
    "key_conflict",                 # вердикт спорит с ключом: «ошибка», хотя ответ = ключу, или «верно», хотя ответ ≠ ключу
    "legibility_mean",              # разборчивость страниц работы по оценке распознавания: 0 — плохо, 1 — средне, 2 — хорошо
    "alarm_no_errors",              # тревога без единой названной ошибки по сути (в основном «не решено»)
    f"{MAIN}__n_err_substantive",   # сколько ошибок по сути назвала проверка (первая версия признака, см. ноутбук)
    "second_disagree",              # DeepSeek не согласен с вердиктом Gemini (нужен отдельный прогон DeepSeek)
    # v3 (после разбора промахов test, 30.09): бесплатные признаки из ответа ученика и ключа, без новых запросов
    "near_key",                     # «ошибка», а ответ отличается от ключа одной цифрой или тривиальным шагом
                                    # (1/x, x², √x, 180 − x, 90 − x, …): вероятно неверное прочтение или «последний шаг»
    "key_missing",                  # ответ нельзя сверить с ключом (доказательство, график, нет ключа)
    # v4 (30.09): near_key разделён на два случая + «правильное значение есть в решении» (правило разметки от 27.09)
    "near_digit",                   # «ошибка», ответ отличается от ключа одной цифрой (прочтение или описка)
    "near_step",                    # «ошибка», ответ = тривиальный шаг от ключа (1/x, x², 180 − x, …)
    "key_in_work",                  # «ошибка», но правильное значение встречается в решении ученика
]
# финальная модель (ноутбук, раздел 3): без DeepSeek — второй голос не дал выигрыша, а стоит отдельного прогона;
# число ошибок заменено индикатором «тревога без названных ошибок» — линейный счётчик давал артефакт (раздел 3)
FEATURES_V2 = [f"{MAIN}__pred_error", "key_conflict", "legibility_mean", "alarm_no_errors"]  # модель на test 29.09
# v3 (ноутбук trust_model_v3.ipynb): обучена на dev + старом test, два новых признака без новых запросов
FEATURES = FEATURES_V2 + ["near_key", "key_missing"]


def _num(s):
    s = str(s).replace(",", ".").replace("−", "-").replace(" ", "").strip()
    return float(s) if re.fullmatch(r"-?\d+(\.\d+)?", s) else None


def _digits(s):
    return re.sub(r"[^0-9]", "", str(s)) if isinstance(s, str) or s == s else ""


def _lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def near_digit(student, key) -> bool:
    """Ответ отличается от ключа одной цифрой (заменена, лишняя или пропущена)."""
    a, b = _digits(student), _digits(key)
    return bool(a and b and a != b and _lev(a, b) <= 1)


def near_step(student, key) -> bool:
    """Ответ = тривиальное преобразование ключа: 1/x, x², √x, 180 − x, 90 − x, 360 − x, −x, 2x, x/2."""
    s, k = _num(student), _num(key)
    if s is None or k is None or s == k:
        return False
    cands = [k * k, 180 - k, 90 - k, 360 - k, -k, 2 * k, k / 2]
    if k:
        cands.append(1 / k)
    if k > 0:
        cands.append(k ** 0.5)
    return any(abs(s - c) < 1e-6 for c in cands)


def near_key(student, key) -> bool:
    """Ответ «рядом с ключом»: одна цифра или тривиальный шаг (признак v3)."""
    return near_digit(student, key) or near_step(student, key)


def build(wide: pd.DataFrame) -> pd.DataFrame:
    """Возвращает таблицу с колонками CANDIDATES (индекс как у wide). Пропуски остаются NaN — их заполняет модель."""
    num = lambda c: pd.to_numeric(wide[c], errors="coerce") if c in wide else pd.Series(np.nan, index=wide.index)
    g, k, d = num(f"{MAIN}__pred_error"), num(f"{MAIN}__key_vs_student"), num(f"{SECOND}__pred_error")
    ne = num(f"{MAIN}__n_err_substantive")
    out = pd.DataFrame(index=wide.index)
    out[f"{MAIN}__pred_error"] = g
    out["key_conflict"] = (((g == 1) & (k == 1)) | ((g == 0) & (k == 0))).astype(float)
    out["legibility_mean"] = num("legibility_mean")
    out["alarm_no_errors"] = ((g == 1) & (ne.fillna(0) == 0)).astype(float)
    out[f"{MAIN}__n_err_substantive"] = ne
    out["second_disagree"] = ((d != g) & d.notna() & g.notna()).astype(float)
    sa = wide[f"{MAIN}__student_answer"] if f"{MAIN}__student_answer" in wide else pd.Series("", index=wide.index)
    ka = wide["key_answer"] if "key_answer" in wide else pd.Series("", index=wide.index)
    nk = [near_key(a, b) for a, b in zip(sa.fillna(""), ka.fillna(""))]
    out["near_key"] = ((g == 1) & pd.Series(nk, index=wide.index)).astype(float)
    out["key_missing"] = k.isna().astype(float)
    nd = pd.Series([near_digit(a, b) for a, b in zip(sa.fillna(""), ka.fillna(""))], index=wide.index)
    ns = pd.Series([near_step(a, b) for a, b in zip(sa.fillna(""), ka.fillna(""))], index=wide.index)
    out["near_digit"] = ((g == 1) & nd).astype(float)
    out["near_step"] = ((g == 1) & ns & ~nd).astype(float)
    out["key_in_work"] = ((g == 1) & (num(f"{MAIN}__key_in_work") == 1)).astype(float)
    return out[CANDIDATES]


def predict(models, X: pd.DataFrame) -> np.ndarray:
    """Финальный предиктор — среднее по моделям всех фолдов кросс-валидации (cross-fitting): это ровно те модели,
    на выходах которых выбран порог, поэтому порог переносится на новые данные без сдвига шкалы вероятностей."""
    return np.mean([m.predict_proba(X[FEATURES])[:, 1] for m in models], axis=0)
