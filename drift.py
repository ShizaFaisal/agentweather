"""Drift detection: data drift (KS test) and model drift (error ratio)."""
import os, uuid
import numpy as np, pandas as pd, requests
from scipy.stats import ks_2samp
from config import *


def data_drift(feats: pd.DataFrame, train_end) -> list:
    f = feats.sort_values("ts")
    live = f[f["ts"] > f["ts"].max() - pd.Timedelta(days=7)]
    ref = f[f["ts"] <= (train_end if train_end is not None else live["ts"].min())]
    if len(ref) < 100 or len(live) < 48:
        return []
    out = []
    for c in DRIFT_COLS:
        ks = ks_2samp(ref[c].dropna(), live[c].dropna())
        out.append(dict(metric=c, drift_type="data", score=float(ks.statistic), threshold=DATA_DRIFT_KS,
                        drifted=bool(ks.statistic > DATA_DRIFT_KS and ks.pvalue < 0.01)))
    return out


def model_drift(ledger: pd.DataFrame, feats: pd.DataFrame, metrics: pd.DataFrame) -> list:
    if ledger.empty or metrics.empty:
        return []
    ref = metrics[metrics["data_split"] == "deployed_val"].sort_values("ts").groupby("target").last()
    led = ledger.sort_values("made_at").drop_duplicates(["metric", "target_time"], keep="last")
    actual = feats.set_index("ts")
    out = []
    for m, g in led.groupby("metric"):
        if m not in ref.index or m not in actual.columns:
            continue
        g = g[g["target_time"].isin(actual.index)].tail(48)
        if len(g) < 12:
            continue
        err = np.abs(g["forecast"].to_numpy() - actual.loc[g["target_time"], m].to_numpy()).mean()
        ratio = err / max(float(ref.loc[m, "mae"]), 1e-6)
        out.append(dict(metric=m, drift_type="model", score=float(ratio), threshold=MODEL_DRIFT_RATIO,
                        drifted=bool(ratio > MODEL_DRIFT_RATIO)))
    return out


def notify_team(msg: str) -> None:
    url = os.getenv("SLACK_WEBHOOK_URL")
    print("[DRIFT ALERT]", msg)
    if url:
        try:
            requests.post(url, json={"text": msg}, timeout=10)
        except Exception as e:
            print("notify failed:", e)


def trigger_training() -> bool:
    repo, tok = os.getenv("GITHUB_REPOSITORY"), os.getenv("GITHUB_TOKEN")
    if not (repo and tok):
        print("[drift] GITHUB_REPOSITORY/GITHUB_TOKEN not set - cannot dispatch training")
        return False
    r = requests.post(f"https://api.github.com/repos/{repo}/dispatches", timeout=15,
                      headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"},
                      json={"event_type": "drift-detected"})
    return r.status_code == 204


def run(feats, ledger, metrics, drift_log, now) -> list:
    train_end = None
    if not metrics.empty and (metrics["data_split"] == "deployed_val").any():
        train_end = pd.to_datetime(metrics[metrics["data_split"] == "deployed_val"]["train_end"]).max()
    checks = data_drift(feats, train_end) + model_drift(ledger, feats, metrics)
    drifted = [c for c in checks if c["drifted"]]
    action_for_drifted = "none"
    if drifted:
        recent = pd.DataFrame() if drift_log.empty else drift_log[drift_log["action"] == "trigger_training"]
        cooling = (not recent.empty) and (now - pd.to_datetime(recent["ts"]).max()) < pd.Timedelta(hours=DRIFT_COOLDOWN_H)
        notify_team(f"AgentWeather drift in {', '.join(sorted({d['drift_type'] + ':' + d['metric'] for d in drifted}))}")
        action_for_drifted = "cooldown_alert_only" if cooling else ("trigger_training" if trigger_training() else "alert_team_only")
    for c in checks:
        c.update(id=str(uuid.uuid4()), ts=now, action=action_for_drifted if c["drifted"] else "none")
    return checks
