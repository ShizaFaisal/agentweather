"""Training pipeline (daily + on drift).

For every target: train naive / Ridge / RandomForest / SARIMAX / LSTM on a time-based
split (70% train, 15% val, 15% test, 24h purge between train and val), compare all with
the naive 24h-ago baseline, and register the winner only if it beats the model that is
currently deployed (re-scored on the same validation window).
"""
import logging, time, uuid
import numpy as np, pandas as pd
from config import *
import store, models as M

log = logging.getLogger("train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
STRIDE = 3   # every 3rd row is scored - keeps SARIMAX's rolling re-forecast affordable


def split(df):
    lab = np.where(df[[f"y_{t}" for t in TARGETS]].notna().all(axis=1))[0]
    n = len(lab)
    a, b = int(.70 * n), int(.85 * n)
    return lab[:a][:-HORIZON], lab[a:b][::STRIDE], lab[b:][::STRIDE], lab[:a]


def ev(art, df, pos, t):
    return M.score(df[f"y_{t}"].to_numpy()[pos], M.predict_rows(art, df, pos, t))


def shap_rows(art, df, va, t, now):
    try:
        import shap
        X = df[art["feat_cols"]].iloc[va][-200:]
        sv = shap.TreeExplainer(art["model"]).shap_values(X.to_numpy())
        imp = pd.Series(np.abs(sv).mean(0), index=art["feat_cols"]).nlargest(15)
        return [dict(id=f"{t}_{f}", ts=now, target=t, feature=f, mean_abs_shap=float(v)) for f, v in imp.items()]
    except Exception as e:
        log.warning("SHAP skipped for %s: %s", t, e)
        return []


def main():
    feats = store.read("weather_features")
    if len(feats) < 24 * 20:
        raise SystemExit(f"Only {len(feats)} feature rows - run `python ingest.py --backfill 90` first")
    df = M.prep(feats)
    tr, va, te, tr_full = split(df)
    train_end = df.index[tr[-1]]
    now = pd.Timestamp.utcnow().tz_localize(None).floor("s")
    log.info("rows: train=%d val=%d test=%d (train_end=%s)", len(tr), len(va), len(te), train_end)

    lstm = None
    try:
        lstm = M.train_lstm(df, tr, va)
    except Exception as e:
        log.warning("LSTM unavailable: %s", e)

    metric_rows, shap_out, deployed_rows = [], [], []
    for t in TARGETS:
        cands = {"naive": M.naive_art()}
        for kind in ("ridge", "rf"):
            cands[kind] = M.train_sk(df, tr, t, kind)
        try:
            cands["sarimax"] = M.train_sarimax(df, tr, t)
        except Exception as e:
            log.warning("SARIMAX failed for %s: %s", t, e)
        if lstm:
            cands["lstm"] = lstm

        results = {}
        for kind, art in cands.items():
            t0 = time.time()
            try:
                v, te_s = ev(art, df, va, t), ev(art, df, te, t)
            except Exception as e:
                log.warning("%s/%s failed: %s", t, kind, e)
                continue
            results[kind] = (v, te_s)
            for data_split_name, m in (("val", v), ("test", te_s)):
                metric_rows.append(dict(id=str(uuid.uuid4()), ts=now, target=t, kind=kind, data_split=data_split_name,
                                        rmse=m["rmse"], mae=m["mae"], r2=m["r2"], n=len(va if data_split_name == "val" else te),
                                        deployed=False, train_end=train_end, baseline_rmse=np.nan))
            log.info("%-14s %-8s val RMSE=%.4f  test RMSE=%.4f  (%.1fs)", t, kind, v["rmse"], te_s["rmse"], time.time() - t0)

        best = min(results, key=lambda k: results[k][0]["rmse"])
        name = f"aw_{t}"
        cur, cur_ver = store.load_model(name)
        cur_val = None
        if cur is not None:
            try:
                cur_val = ev(cur, df, va, t)
            except Exception as e:
                log.warning("could not re-score deployed %s: %s", name, e)
        promote = cur_val is None or results[best][0]["rmse"] < cur_val["rmse"]
        if promote:
            ver = store.save_model(name, {**cands[best], "trained_at": str(now), "train_end": str(train_end)},
                                   {"rmse": results[best][0]["rmse"], "mae": results[best][0]["mae"], "r2": results[best][0]["r2"]})
            dep_kind, dep_val = best, results[best][0]
            log.info(">> %s: promoted %s as v%s", name, best, ver)
        else:
            dep_kind, dep_val = cur["kind"], cur_val
            log.info(">> %s: kept deployed %s v%s (new %s not better)", name, cur["kind"], cur_ver, best)
        deployed_rows.append(dict(id=str(uuid.uuid4()), ts=now, target=t, kind=dep_kind, data_split="deployed_val",
                                  rmse=dep_val["rmse"], mae=dep_val["mae"], r2=dep_val["r2"], n=len(va), deployed=True,
                                  train_end=train_end, baseline_rmse=results["naive"][0]["rmse"] if "naive" in results else np.nan))
        shap_out += shap_rows(cands["rf"], df, va, t, now)

    store.upsert("model_metrics", pd.DataFrame(metric_rows + deployed_rows), ["id"])
    store.upsert("shap_importance", pd.DataFrame(shap_out), ["id"])
    log.info("training finished")


if __name__ == "__main__":
    main()
