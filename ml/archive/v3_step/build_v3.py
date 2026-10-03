import nbformat as nbf
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Модель доверия v3: обучение на dev + старом test

**Зачем.** Единственный запуск test (TEST_RESULTS.md) показал, что модель доверия v2 не поймала 21 ошибку проверки
из 57. 13 из них — ответы «рядом с ключом»: отличаются от ключа одной цифрой или тривиальным шагом (неверное
прочтение цифры или «последний шаг»). Признак «спор с ключом» таких случаев не видит.

**Что меняется.**
1. Данные для разработки = dev + старый test: 78 работ, 1 479 задач, 89 ошибок проверки вместо 32. Старый test
   использован, поэтому честную оценку даст только **новый test** на новых работах (TEST_PROTOCOL_2.md).
2. Два новых бесплатных признака (без новых запросов к модели): `near_key` — «ошибка», а ответ рядом с ключом;
   `key_missing` — ответ не с чем сверить (доказательство, график).

**Правило выбора, записанное до обучения:** v3 (v2 + два признака) берём, только если она не хуже v2 при бюджете
куратора 10% **и** в общей кросс-валидации, **и** при переносе «обучили на dev → проверили на старом test».
Иначе оставляем v2, переобученную на всех данных.

**Оптимизм, который здесь не убрать:** признак `near_key` придуман после просмотра ошибок старого test, поэтому любые
цифры на этих данных завышены. Проверка — только новый test.
""")

code(r"""
import json, pickle
import numpy as np, pandas as pd, matplotlib.pyplot as plt, sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta, binom
import trust_features as tf

wide = pd.read_csv("../data/features_wide.csv")
df = wide[wide[f"{tf.MAIN}__wrong"].notna()].reset_index(drop=True)
y = df[f"{tf.MAIN}__wrong"].astype(int)
F = tf.build(df)
print(f"задач {len(df)}, работ {df.work.nunique()}, заданий {df.task_key.nunique()}, ошибок проверки {y.sum()}")
shift = pd.DataFrame({"ошибок проверки": y, "разборчивость": F.legibility_mean, "near_key": F.near_key,
                      "key_missing": F.key_missing, "«ошибка»": F[f"{tf.MAIN}__pred_error"]}).groupby(df.split).mean()
shift.insert(0, "задач", df.groupby("split").size())
shift.round(3)
""")

md(r"""
Сдвиг между частями виден сразу: на старом test почерк хуже, в 4 раза чаще нечего сверять с ключом, а ошибок проверки
вдвое больше. Модель, обученная только на dev, такого почти не видела.
""")

code(r"""
print("ошибка проверки при near_key = 1 / 0:")
print(pd.crosstab([df.split, F.near_key], y).rename(columns={0: "проверка права", 1: "ошиблась"}).to_string())
""")

md(r"""
На dev `near_key` почти не информативен (ошибка проверки в 10% таких задач), на старом test — в 45%. Поэтому на одном
dev его бы не нашли. Вопрос раздела 3 — даёт ли он выигрыш, если учиться на обеих частях.

## 1. Инструменты (те же, что в ноутбуке v2)
""")

code(r"""
BUDGETS = [0.05, 0.10, 0.15, 0.20]

def routed_accuracy(y_true, score, budget, seed=1):
    y_true = np.asarray(y_true)
    order = np.lexsort((np.random.default_rng(seed).random(len(score)), -np.asarray(score, float)))
    return 1 - y_true[order[int(round(budget * len(y_true))):]].sum() / len(y_true)

def report(scores, y_true):
    rows = []
    for name, s in scores.items():
        row = {"": name, "ROC-AUC": roc_auc_score(y_true, s), "PR-AUC": average_precision_score(y_true, s)}
        row.update({f"@{int(b*100)}%": routed_accuracy(y_true, s, b) for b in BUDGETS})
        rows.append(row)
    return pd.DataFrame(rows).set_index("").round(3)

def logreg(cols):
    return lambda: make_pipeline(ColumnTransformer([("x", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), cols)]),
                                 LogisticRegression(C=1.0, max_iter=2000))

def oof(make, X, y_true, groups, repeats=20, keep=False, seed0=0):
    y_true = pd.Series(np.asarray(y_true)); out, models = np.zeros(len(X)), []
    for seed in range(seed0, seed0 + repeats):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y_true, groups):
            m = make().fit(X.iloc[tr], y_true.iloc[tr]); out[te] += m.predict_proba(X.iloc[te])[:, 1]
            if keep: models.append(m)
    return (out / repeats, models) if keep else out / repeats

def paired_bootstrap(a, b, y_true, works, budget, n_boot=2000, seed=0):
    rnd, yv = np.random.default_rng(seed), np.asarray(y_true)
    idx = {w: np.where(works == w)[0] for w in np.unique(works)}; keys = list(idx); d = []
    for _ in range(n_boot):
        s = np.concatenate([idx[w] for w in rnd.choice(keys, len(keys))])
        d.append(routed_accuracy(yv[s], a[s], budget) - routed_accuracy(yv[s], b[s], budget))
    return np.mean(d), *np.percentile(d, [2.5, 97.5])

V2 = [f"{tf.MAIN}__pred_error", "key_conflict", "legibility_mean", "alarm_no_errors"]
V3 = V2 + ["near_key", "key_missing"]
MODELS = {"логрег v2": logreg(V2), "логрег v3": logreg(V3), "v3 без key_missing": logreg(V2 + ["near_key"])}
""")

md(r"""
## 2. Перенос: обучили на dev → проверили на старом test

Ровно та ситуация, которая провалилась: модель видит только dev и применяется к новым работам. Для v2 это
повторение того, что было на test (с ансамблем фолдов), для v3 — оценка нового признака (оптимистичная, см. выше).
""")

code(r"""
dev, tst = (df.split == "dev").values, (df.split == "test").values
res = {}
for name, make in MODELS.items():
    _, ms = oof(make, F[dev].reset_index(drop=True), y[dev], df.task_key[dev].reset_index(drop=True), repeats=20, keep=True)
    res[name] = np.mean([m.predict_proba(F[tst])[:, 1] for m in ms], axis=0)
res["все «ошибки» проверки"] = F[f"{tf.MAIN}__pred_error"][tst].values
print("старый test, модели обучены только на dev:")
report(res, y[tst])
""")

code(r"""
for b in (0.10, 0.15):
    m_, lo, hi = paired_bootstrap(res["логрег v3"], res["логрег v2"], y[tst].values, df.work[tst].values, b)
    print(f"@{int(b*100)}%  v3 − v2 на старом test: {100*m_:+.2f} п.п. [{100*lo:+.2f}; {100*hi:+.2f}]")
""")

md(r"""
## 3. Общая кросс-валидация на dev + старом test

Группа = задание (task_key), 5 фолдов × 20 повторов; оценка — на всех 1 479 задачах.
""")

code(r"""
groups = df.task_key
OOF = {name: oof(make, F, y, groups) for name, make in MODELS.items()}
kc, pe = F.key_conflict, F[f"{tf.MAIN}__pred_error"]
BASE = {"все «ошибки» проверки": pe.values,
        "правило v2 (4·max(kc, ane) + 2·pe − leg/3)": (4 * np.maximum(kc, F.alarm_no_errors) + 2 * pe - F.legibility_mean.fillna(F.legibility_mean.median()) / 3).values}
report({**BASE, **OOF}, y)
""")

code(r"""
works = df.work.values
for b in (0.05, 0.10, 0.15):
    for other in ["логрег v2", "v3 без key_missing"]:
        m_, lo, hi = paired_bootstrap(OOF["логрег v3"], OOF[other], y.values, works, b)
        print(f"@{int(b*100):2d}%  v3 − {other:20s} {100*m_:+.2f} п.п. [{100*lo:+.2f}; {100*hi:+.2f}]")
""")

code(r"""
d_oof = {"группы = задания": OOF["логрег v3"], "группы = ученики": oof(MODELS["логрег v3"], F, y, df.student_id)}
print("устойчивость к группировке (v3):"); report(d_oof, y)
""")

md(r"""
**Решение по правилу выше** — в выводе следующей ячейки (считается по цифрам, а не выбирается глазами).
""")

code(r"""
cv_ok = routed_accuracy(y, OOF["логрег v3"], 0.10) >= routed_accuracy(y, OOF["логрег v2"], 0.10)
tr_ok = routed_accuracy(y[tst], res["логрег v3"], 0.10) >= routed_accuracy(y[tst], res["логрег v2"], 0.10)
FINAL = "логрег v3" if (cv_ok and tr_ok) else "логрег v2"
FEATS = V3 if FINAL == "логрег v3" else V2
print(f"общая CV: v3 {'не хуже' if cv_ok else 'хуже'} v2; перенос dev → test: v3 {'не хуже' if tr_ok else 'хуже'} v2 → финальная: {FINAL}")
m = MODELS[FINAL]().fit(F, y); lr, sc = m[-1], m[0].named_transformers_["x"][-1]
pd.DataFrame({"стандартизованный": lr.coef_[0], "на единицу признака": lr.coef_[0] / sc.scale_}, index=FEATS).round(3)
""")

md(r"""
## 4. Калибровка и порог (Learn then Test, α = 2%, δ = 0,1 — как раньше)
""")

code(r"""
def ece(y_true, p, n_bins=10):
    y_true, p = np.asarray(y_true), np.asarray(p)
    return sum(len(b) / len(p) * abs(y_true[b].mean() - p[b].mean()) for b in np.array_split(np.argsort(p), n_bins))

def cp_upper(k, n, delta):
    return 1.0 if k == n else beta.ppf(1 - delta, k + 1, n - k)

def ltt_threshold(p, y_true, alpha, delta=0.1):
    best, y_true = None, np.asarray(y_true)
    for t in np.sort(np.unique(p)):
        if cp_upper(int(y_true[p <= t].sum()), len(p), delta) <= alpha: best = t
        else: break
    return best

def point(p, y_true, t):
    y_true, auto = np.asarray(y_true), p <= t
    return {"к куратору": 1 - auto.mean(), "пропущено ошибок проверки": int(y_true[auto].sum()),
            "итоговое совпадение": 1 - y_true[auto].sum() / len(y_true)}

ALPHA, DELTA = 0.02, 0.1
p_main = OOF[FINAL]
print(f"доля ошибок {y.mean():.3f}; средняя вероятность {p_main.mean():.3f}; ECE {ece(y, p_main):.3f}; Brier {brier_score_loss(y, p_main):.4f}")
rows = [{"α": a, "δ": d, "порог": ltt_threshold(p_main, y, a, d), **point(p_main, y, ltt_threshold(p_main, y, a, d))}
        for a in (0.015, 0.02, 0.025) for d in (0.05, 0.1)]
pd.DataFrame(rows).round(4)
""")

code(r"""
t_star = ltt_threshold(p_main, y, ALPHA, DELTA)
pt = point(p_main, y, t_star)
print(f"порог {t_star:.4f}: {pt}; верхняя граница риска {cp_upper(pt['пропущено ошибок проверки'], len(y), DELTA):.4f}")
for sp in ("dev", "test"):
    mk = (df.split == sp).values
    print(f"  {sp:4s}: {point(p_main[mk], y[mk], t_star)}")
rnd, idx = np.random.default_rng(0), {w: np.where(works == w)[0] for w in np.unique(works)}
st = pd.DataFrame([point(p_main[s], y.values[s], t_star) for s in
                   (np.concatenate([idx[w] for w in rnd.choice(list(idx), len(idx))]) for _ in range(2000))])
for c in ["к куратору", "итоговое совпадение"]:
    print(f"  {c}: 95% ДИ по работам [{st[c].quantile(.025):.3f}; {st[c].quantile(.975):.3f}]")
""")

md(r"""
### Вложенная CV полного конвейера (обучение + порог; выбор v2/v3 внутри фолда)
""")

code(r"""
outer, chosen = [], []
for rep in range(10):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=1000 + rep).split(df, y, groups):
        Ftr, ytr, gtr = F.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True), groups.iloc[tr].reset_index(drop=True)
        inner = {}
        for name in ("логрег v2", "логрег v3"):
            p_in, ms = oof(MODELS[name], Ftr, ytr, gtr, repeats=5, keep=True)
            inner[name] = (routed_accuracy(ytr, p_in, 0.10), p_in, ms)
        name = max(inner, key=lambda k: inner[k][0]); chosen.append(name)
        _, p_in, ms = inner[name]
        t = ltt_threshold(p_in, ytr.values, ALPHA, DELTA)
        auto = np.mean([m_.predict_proba(F.iloc[te])[:, 1] for m_ in ms], axis=0) <= t
        outer.append({"риск": y.values[te][auto].sum() / len(te), "к куратору": 1 - auto.mean()})
outer = pd.DataFrame(outer)
print(f"v3 выбрана в {chosen.count('логрег v3')} из {len(chosen)} фолдов; риск {outer['риск'].mean():.4f} "
      f"(превышает α в {(outer['риск'] > ALPHA).mean():.0%} фолдов), к куратору {outer['к куратору'].mean():.3f} "
      f"[{outer['к куратору'].quantile(.025):.3f}; {outer['к куратору'].quantile(.975):.3f}]")
n_fold = len(df) / 5
print(f"ожидаемая доля фолдов > α от одного шума: {binom.sf(int(ALPHA * n_fold), int(n_fold), outer['риск'].mean()):.0%}")
""")

md(r"""
## 5. Нагрузка на куратора с задачами без метки и финальная модель
""")

code(r"""
p_check, fold_models = oof(MODELS[FINAL], F, y, groups, keep=True)
assert np.allclose(p_check, p_main)
wall = pd.read_csv("../data/features_wide_all.csv")
un = wall[wall[f"{tf.MAIN}__wrong"].isna()].reset_index(drop=True)
Fu = tf.build(un)
to_un = (np.mean([m_.predict_proba(Fu)[:, 1] for m_ in fold_models], axis=0) > t_star) | Fu[f"{tf.MAIN}__pred_error"].isna()
lab_cur = int((p_main > t_star).sum())
print(f"без метки: {len(un)} задач, к куратору {int(to_un.sum())}; всего к куратору ({lab_cur} + {int(to_un.sum())}) / "
      f"{len(df) + len(un)} = {(lab_cur + int(to_un.sum())) / (len(df) + len(un)):.1%}")
pickle.dump({"models": fold_models, "features": FEATS, "system": "google_gemini-3.8-flash@low_key",
             "trained_on": "dev + test (2026-09-29)", "sklearn": sklearn.__version__}, open("trust_model.pkl", "wb"))
json.dump({"alpha": ALPHA, "delta": DELTA, "threshold": float(t_star), "model": FINAL, "trained_on": "dev + test",
           "cv": {k: float(v) for k, v in pt.items()}}, open("trust_threshold.json", "w"), ensure_ascii=False, indent=1)
print(f"сохранено: {len(fold_models)} моделей ({FINAL}), порог {t_star:.4f} → ml/trust_model.pkl, ml/trust_threshold.json")
""")

md(r"""
## Итоги

Цифры этого ноутбука — оценки на данных, которые уже использовались для разработки (включая старый test, по ошибкам
которого придуман `near_key`). Они показывают, **что** выбрано и **почему**, но не качество на новых работах.
Качество покажет новый test (TEST_PROTOCOL_2.md): новые работы, разметка без вердиктов модели, заморозка до запуска.
""")

nb = nbf.v4.new_notebook(); nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "trust_model_v3.ipynb")
