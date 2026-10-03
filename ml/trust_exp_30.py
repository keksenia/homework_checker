"""Модель доверия на текущей системе (30.09): разметка после режимов и политики, test2 проверен с ключом (k2).

Системы: cur — проверка v1 (low_key; для работ test2 — low_key_k2); curpol — та же проверка + слой политики
по офлайн-разметке отклонений (reports/policy_tags.json, сомнение в прочтении → замечание).
Признаки v3 (6 шт.), логистическая регрессия, StratifiedGroupKFold (группы — ученики и задания), 10 повторов,
порог Learn-then-Test (α = 2%, δ = 0,1). Данные: data/features_wide_cur.csv (features.py + features_wide.py).
Сохраняет ml/trust_model_v5_<система>.pkl (кросс-фит модели) и ml/trust_threshold_v5_<система>.json —
старые trust_model.pkl / trust_threshold.json не трогает.

  python3 ml/trust_exp_30.py > reports/trust_exp_30.txt
"""
import json, pickle, sys, warnings
sys.path.insert(0, 'ml'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sklearn
import trust_features as tf
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

W = pd.read_csv('data/features_wide_cur.csv')
V3 = tf.FEATURES


def lr():
    return make_pipeline(ColumnTransformer([('x', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), V3)]),
                         LogisticRegression(max_iter=2000))


def ltt(p, yy, a=0.02, d=0.1):
    best = None
    for t in np.sort(np.unique(p)):
        k = int(yy[p <= t].sum()); ub = 1.0 if k == len(p) else beta.ppf(1 - d, k + 1, len(p) - k)
        if ub <= a: best = t
        else: break
    return best


def routed(yy, s, b):
    o = np.argsort(-s, kind='stable'); return 1 - yy[o[int(round(b * len(yy))):]].sum() / len(yy)


def system(name):
    df = W.rename(columns=lambda c: c.replace(f'g_final_{name}__', f'{tf.MAIN}__'))
    df = df[df[f'{tf.MAIN}__wrong'].notna()].reset_index(drop=True)
    y = df[f'{tf.MAIN}__wrong'].astype(int).values
    X = tf.build(df)
    print(f'\n=== система {name}: задач {len(y)}, работ {df.work.nunique()}, ошибок проверки {y.sum()} '
          f'(совпадение проверки с разметкой {1 - y.mean():.1%}) ===')
    for gcol in ('student_id', 'task_key'):
        g = df[gcol].values; oof = np.zeros(len(y)); tr = []
        models = []
        for seed in range(10):
            for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
                m = lr().fit(X.iloc[a], y[a]); oof[b] += m.predict_proba(X.iloc[b])[:, 1] / 10
                tr.append(roc_auc_score(y[a], m.predict_proba(X.iloc[a])[:, 1])); models.append(m)
        t = ltt(oof, y); load = float((oof > t).mean()) if t is not None else 1.0
        miss = int(y[oof <= t].sum()) if t is not None else 0
        print(f'  группы {gcol:10s}: AUC train {np.mean(tr):.3f}, AUC CV {roc_auc_score(y, oof):.3f}, '
              f'PR-AUC {average_precision_score(y, oof):.3f}; порог LTT {t:.4f}: к куратору {load:.1%}, '
              f'непойманных {miss} ({miss / len(y):.2%}), итог {1 - miss / len(y):.1%}; '
              f'итог при 10% к куратору {routed(y, oof, .10):.1%}, при 15% {routed(y, oof, .15):.1%}')
        if gcol == 'student_id':
            pickle.dump(models, open(f'ml/trust_model_v5_{name}.pkl', 'wb'))
            json.dump({'threshold': float(t), 'alpha': 0.02, 'delta': 0.1, 'features': V3, 'groups': gcol,
                       'curator_share_oof': load, 'sklearn': sklearn.__version__, 'system': name,
                       'note': 'кросс-фит: 50 моделей, предсказание — среднее (tf.predict)'},
                      open(f'ml/trust_threshold_v5_{name}.json', 'w'), ensure_ascii=False, indent=1)
    by = pd.DataFrame({'exam': df['work'].map(lambda w: w), 'wrong': y})
    return y


print('sklearn', sklearn.__version__)
for s in ('cur', 'curpol'):
    system(s)
