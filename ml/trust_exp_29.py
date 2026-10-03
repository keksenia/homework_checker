import sys, warnings; sys.path.insert(0, 'ml'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, trust_features as tf
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

w = pd.read_csv('data/features_wide.csv')
df = w[w[tf.MAIN + '__wrong'].notna()].reset_index(drop=True)
F = tf.build(df); y = df[tf.MAIN + '__wrong'].astype(int).values
V3 = tf.FEATURES
F['votes_error_share'] = df['votes_error_share']; F['voters_disagree'] = df['voters_disagree']
F['ds_pred_error'] = df['ds_high_img__pred_error']
SO = V3 + ['voters_disagree', 'ds_pred_error']
MONO = {f'{tf.MAIN}__pred_error': 1, 'key_conflict': 1, 'legibility_mean': -1, 'alarm_no_errors': 1,
        'near_key': 1, 'key_missing': 1, 'voters_disagree': 1, 'ds_pred_error': 0}

def lr(cols, C=1.0):
    return lambda: make_pipeline(ColumnTransformer([('x', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), cols)]),
                                 LogisticRegression(C=C, max_iter=2000))
def hgb(cols, mono=True, depth=2, it=100, lr_=0.05):
    cst = [MONO.get(c, 0) for c in cols] if mono else None
    return lambda: make_pipeline(ColumnTransformer([('x', 'passthrough', cols)]),
        HistGradientBoostingClassifier(max_depth=depth, max_iter=it, learning_rate=lr_, min_samples_leaf=20,
                                       l2_regularization=1.0, monotonic_cst=cst, random_state=0))

def oof(make, X, yy, groups, repeats=10, k=5):
    out = np.zeros(len(X)); tr_auc = []
    for seed in range(repeats):
        for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, yy, groups):
            m = make().fit(X.iloc[tr], yy[tr]); out[te] += m.predict_proba(X.iloc[te])[:, 1]
            tr_auc.append(roc_auc_score(yy[tr], m.predict_proba(X.iloc[tr])[:, 1]))
    return out / repeats, float(np.mean(tr_auc))

def routed(yy, s, b):
    o = np.argsort(-s, kind='stable'); return 1 - yy[o[int(round(b * len(yy))):]].sum() / len(yy)
def ltt(p, yy, a=0.02, d=0.1):
    best = None
    for t in np.sort(np.unique(p)):
        k = int(yy[p <= t].sum()); ub = 1.0 if k == len(p) else beta.ppf(1 - d, k + 1, len(p) - k)
        if ub <= a: best = t
        else: break
    return best
def row(name, p, yy, trauc):
    t = ltt(p, yy); load = float((p > t).mean()) if t is not None else 1.0
    return {'модель': name, 'AUC train': round(trauc, 3), 'AUC CV': round(roc_auc_score(yy, p), 3),
            'PR-AUC': round(average_precision_score(yy, p), 3), 'итог @10%': round(routed(yy, p, .10), 4),
            'итог @15%': round(routed(yy, p, .15), 4), 'к куратору (LTT 2%)': round(load, 3)}

def run(mask, label, groups_col, extra=None):
    X = F[mask].reset_index(drop=True); yy = y[mask]; g = df.loc[mask, groups_col].values
    print(f'\n=== {label}: задач {len(yy)}, ошибок проверки {yy.sum()}, групп ({groups_col}) {len(set(g))} ===')
    rows = []
    for name, make in (extra or {}).items():
        p, ta = oof(make, X, yy, g); rows.append(row(name, p, yy, ta))
    print(pd.DataFrame(rows).to_string(index=False))

base = {'логрег v3 (6 призн.)': lr(V3), 'бустинг монотонный': hgb(V3, True), 'бустинг без ограничений': hgb(V3, False),
        'бустинг глубокий (depth 4, 300 деревьев)': hgb(V3, False, 4, 300, 0.1)}
dev = (df.split == 'dev').values; alld = np.ones(len(df), bool)
run(dev, 'dev (с ключом), группы — задания', 'task_key', base)
run(dev, 'dev (с ключом), группы — ученики', 'student_id', base)
run(alld, 'dev + test2, группы — ученики', 'student_id', base)
so = dev & df['ds_high_img__pred_error'].notna().values
run(so, 'dev, где есть второе мнение: без него / с ним', 'student_id',
    {'логрег v3': lr(V3), 'логрег v3 + второе мнение': lr(SO), 'бустинг монот. v3': hgb(V3), 'бустинг монот. + второе мнение': hgb(SO)})

# кривая обучения: как AUC на кросс-валидации растёт с объёмом данных (dev, группы — ученики)
print('\n=== кривая обучения (dev, ученики): AUC train / AUC CV при доле обучающих данных ===')
X = F[dev].reset_index(drop=True); yy = y[dev]; g = df.loc[dev, 'student_id'].values
for name, make in [('логрег v3', lr(V3)), ('бустинг глубокий', hgb(V3, False, 4, 300, 0.1))]:
    res = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        tra, cva = [], []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, yy, g):
                tr = rng.choice(tr, int(len(tr) * frac), replace=False)
                if yy[tr].sum() < 3: continue
                m = make().fit(X.iloc[tr], yy[tr])
                tra.append(roc_auc_score(yy[tr], m.predict_proba(X.iloc[tr])[:, 1]))
                cva.append(roc_auc_score(yy[te], m.predict_proba(X.iloc[te])[:, 1]) if len(set(yy[te])) > 1 else np.nan)
        res.append(f'{int(frac*100)}%: {np.mean(tra):.3f}/{np.nanmean(cva):.3f}')
    print(f'{name:18s}', '  '.join(res))
print(f'\nсобытий (ошибок проверки) на признак: dev {y[dev].sum()}/{len(V3)} = {y[dev].sum()/len(V3):.0f}')
