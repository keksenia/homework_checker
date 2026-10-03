#!/usr/bin/env python3
"""apply_trust.py — применение замороженной модели доверия к размеченным задачам (test).

    python3 ml/apply_trust.py --split test        # после прогона итоговой системы на test и features.py/features_wide.py

    python3 features.py --include-unlabeled --out data/features_all.csv
    python3 features_wide.py --features data/features_all.csv --out data/features_wide_all.csv --runs g_low_key
    python3 ml/apply_trust.py --split dev --wide data/features_wide_all.csv
        # + задачи из списка проверки без метки («не решено», брошено): нагрузка на куратора в режиме
        #   «список задач от репетитора» (check.py --scope-file). Метрики — только по размеченным задачам.

Берёт data/features_wide.csv, считает признаки тем же кодом, что при обучении (ml/trust_features.py),
предсказывает вероятность «проверка ошиблась» средним 100 моделей фолдов и применяет порог из
ml/trust_threshold.json: выше порога — куратору. Если есть разметка, считает итоговые метрики с 95%-интервалами
(bootstrap по работам). → reports/trust_<split>.csv и сводка на экран.
Внимание: на dev это оценка на обучающих данных (оптимистична); честные цифры dev — в ноутбуке (out-of-fold).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import trust_features as tf  # noqa: E402


def metrics(d):
    n = len(d)
    wrong = d["system_wrong"].astype(int)
    auto = ~d["to_curator"]
    return {"задач": n, "к куратору": d["to_curator"].mean(), "совпадение проверки без куратора": 1 - wrong.mean(),
            "итоговое совпадение (куратор решает верно)": 1 - (wrong & auto).sum() / n,
            "ошибок проверки": int(wrong.sum()), "из них поймано куратором": int((wrong & ~auto).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--wide", default=str(ROOT / "data" / "features_wide.csv"))
    ap.add_argument("--model", default="trust_model.pkl",
                    help="файл модели в ml/ (test3: trust_model_v5_curpol.pkl — список моделей кросс-фита)")
    ap.add_argument("--threshold", default="trust_threshold.json", help="файл порога в ml/ (test3: trust_threshold_v5_curpol.json)")
    ap.add_argument("--run", default="",
                    help="короткое имя прогона итоговой системы в таблице признаков (test3: g_low_key_tag); его колонки "
                         "переименовываются в колонки, на которых обучена модель (g_low_key__…)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="считать, даже если задач test меньше, чем в замороженной разметке (сообщить об этом!)")
    a = ap.parse_args()

    lock = ROOT / "reports" / "test3_lock.json"  # = check.TEST_LOCK (текущий раунд test)
    if a.split == "test":  # на test применяем только замороженные модель, порог и код признаков
        if not lock.exists():
            sys.exit(f"test не заморожен ({lock.name}) — см. TEST_PROTOCOL_3.md")
        import hashlib
        frozen = json.load(open(lock, encoding="utf-8"))["files"]
        need = [f"ml/{a.model}", f"ml/{a.threshold}", "ml/trust_features.py", "ml/apply_trust.py",
                "features.py", "features_wide.py"]
        bad = [f for f in need if f not in frozen or hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != frozen[f]]
        if bad:
            sys.exit(f"файлы модели доверия не заморожены или изменены после заморозки: {bad}")
        expected = json.load(open(lock, encoding="utf-8")).get("test_counts")
    bundle = pickle.load(open(HERE / a.model, "rb"))
    if isinstance(bundle, list):  # v5 (ml/trust_exp_30.py): просто список моделей кросс-фита
        meta = json.load(open(HERE / a.threshold, encoding="utf-8"))
        bundle = {"models": bundle, "features": meta["features"], "sklearn": meta.get("sklearn")}
    import sklearn
    if bundle.get("sklearn") and bundle["sklearn"] != sklearn.__version__:
        print(f"ВНИМАНИЕ: модель сохранена в scikit-learn {bundle['sklearn']}, сейчас {sklearn.__version__} — "
              f"перезапусти ноутбук на этой машине, чтобы пересохранить модель")
    thr = json.load(open(HERE / a.threshold, encoding="utf-8"))
    thr.setdefault("model", a.model)
    if bundle["features"] != tf.FEATURES:
        sys.exit(f"признаки модели {bundle['features']} не совпадают с trust_features.FEATURES {tf.FEATURES}")
    wide = pd.read_csv(a.wide)
    if a.run and a.run != tf.MAIN:
        wide = wide.drop(columns=[c for c in wide.columns if c.startswith(f"{tf.MAIN}__")])
        wide = wide.rename(columns=lambda c: c.replace(f"{a.run}__", f"{tf.MAIN}__"))
    d = wide[wide["split"] == a.split].reset_index(drop=True)
    if d.empty:
        sys.exit(f"в {a.wide} нет строк части {a.split}: сначала прогон системы, затем features.py и features_wide.py")
    if a.split == "test" and expected:
        lab = d[d["label"].notna()]
        got = {"tasks": len(lab), "errors": int((lab["label"] == 0).sum())}
        if got != expected and not a.allow_incomplete:
            sys.exit(f"в таблице признаков {got}, а в замороженной разметке test {expected}: часть работ выпала "
                     f"(неразобранный ответ?). Почини (check.py --reparse) или запусти с --allow-incomplete и "
                     f"сообщи расхождение в TEST_RESULTS.md")
    missing = d[f"{tf.MAIN}__pred_error"].isna()
    if missing.any():
        print(f"ВНИМАНИЕ: у {missing.sum()} задач нет вердикта системы — они уходят куратору")
    X = tf.build(d)
    p = tf.predict(bundle["models"], X)
    out = d[["work", "task_no", "split", "label"]].copy()
    out["p_system_wrong"] = p
    out["to_curator"] = (p > thr["threshold"]) | missing
    out["system_pred_error"] = X[f"{tf.MAIN}__pred_error"]
    # задача без вердикта системы — несовпадение (как в check.py --report); она всё равно уходит куратору
    out["system_wrong"] = (d[f"{tf.MAIN}__wrong"].fillna(1).astype(bool) | missing) if f"{tf.MAIN}__wrong" in d else missing
    tag = "" if Path(a.wide).name == "features_wide.csv" else "_" + Path(a.wide).stem.replace("features_wide_", "")
    path = ROOT / "reports" / f"trust_{a.split}{tag}.csv"
    out.to_csv(path, index=False)
    print(f"порог {thr['threshold']:.4f} (α = {thr['alpha']}, δ = {thr['delta']}); модель: {thr['model']}")
    unl = out["label"].isna()
    if unl.any():
        print(f"  задач без метки (не решено / брошено): {unl.sum()}, из них к куратору {out.loc[unl, 'to_curator'].sum()}; "
              f"к куратору всего с их учётом: {out['to_curator'].mean():.1%} из {len(out)} "
              f"(по размеченным: {out.loc[~unl, 'to_curator'].mean():.1%})")
    lab = out[~unl].reset_index(drop=True)
    if len(lab):
        m = metrics(lab)
        rnd, works = np.random.default_rng(0), lab["work"].values
        idx = {w: np.where(works == w)[0] for w in np.unique(works)}
        boot = [metrics(lab.iloc[np.concatenate([idx[w] for w in rnd.choice(list(idx), len(idx))])]) for _ in range(2000)]
        for k, v in m.items():
            if isinstance(v, float):
                lo, hi = np.percentile([b[k] for b in boot], [2.5, 97.5])
                print(f"  {k}: {v:.3f}  [{lo:.3f}; {hi:.3f}]")
            else:
                print(f"  {k}: {v}")
        from scipy.stats import beta
        k = int((lab["system_wrong"] & ~lab["to_curator"]).sum())
        ub = 1.0 if k == len(lab) else beta.ppf(1 - thr["delta"], k + 1, len(lab) - k)
        print(f"  (ошибок проверки должно быть столько же, сколько несовпадений в строке {a.split} у check.py --report)")
        print(f"  H4: доля задач с непойманной ошибкой проверки {k / len(lab):.4f}; верхняя {1 - thr['delta']:.0%}-граница "
              f"Клоппера–Пирсона {ub:.4f} (α = {thr['alpha']})")
        print("  сравни с гипотезами H1–H4 из TEST_PROTOCOL_3.md и запиши в TEST_RESULTS_3.md")
    else:
        print(f"  к куратору: {out['to_curator'].mean():.1%} задач (разметки нет — метрики не считаются)")
    print(f"→ {path}")


if __name__ == "__main__":
    main()
