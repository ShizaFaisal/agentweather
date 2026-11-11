"""LangGraph agent layer: Forecast -> Alert -> Drift -> (Weather Report).

All agents are nodes of ONE graph sharing ONE state. The state (alert keys seen,
last report time) is persisted in the `agent_state` feature group between runs.
Every LLM call is logged (prompt, response, latency, status) in `agent_call_log`;
rows whose status != "ok" are the failure log.
"""
import json, logging, os, time, uuid
from typing import TypedDict
import numpy as np, pandas as pd, requests
from config import *
import store, drift, models as M

log = logging.getLogger("agents")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class State(TypedDict, total=False):
    now: pd.Timestamp
    forecasts: list      # ledger rows (24 target hours x 9 signals)
    hazards: list
    alerts: list
    new_alert: bool
    report: str
    drift_events: list
    prev: dict           # persisted state from the previous run
    llm_log: list
    force_report: bool


# ------------------------------------------------------------------- LLM
def llm(state, agent, system, prompt, validate=None, json_mode=False):
    """Groq chat completion with model fallback chain. Returns text or None (caller uses template)."""
    key = os.getenv("GROQ_API_KEY")
    for model in GROQ_MODELS:
        t0, status, text = time.time(), "ok", ""
        try:
            if not key:
                raise RuntimeError("GROQ_API_KEY not set")
            body = {"model": model, "temperature": 0.3,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=40,
                              headers={"Authorization": f"Bearer {key}"}, json=body)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            if validate:
                validate(text)
        except Exception as e:
            status = f"error: {type(e).__name__}: {str(e)[:160]}"
        state["llm_log"].append(dict(id=str(uuid.uuid4()), ts=pd.Timestamp.utcnow().tz_localize(None), agent=agent,
                                     model=model, prompt=prompt[:3500], response=text[:3500],
                                     latency_ms=int((time.time() - t0) * 1000), status=status))
        if status == "ok":
            return text
        if not key:
            break          # no point trying the second model without a key
    return None


# ------------------------------------------------------------------ nodes
def forecast_node(s: State) -> State:
    df = M.prep(store.read("weather_features"))
    tail = df.tail(24 * 21)
    pos = np.arange(len(tail) - 24, len(tail))          # newest 24 rows have no target -> forecast them
    rows = []
    for t in TARGETS:
        art, ver = store.load_model(f"aw_{t}")
        if art is None:
            log.warning("no registered model for %s - using naive baseline", t)
            art, ver = M.naive_art(), 0
        try:
            pred = M.predict_rows(art, tail, pos, t)
        except Exception as e:
            log.warning("%s/%s failed (%s) -> naive fallback", t, art["kind"], e)
            art, ver = M.naive_art(), 0
            pred = M.predict_rows(art, tail, pos, t)
        for p, ts in zip(pred, tail.index[pos]):
            tt = ts + pd.Timedelta(hours=HORIZON)
            rows.append(dict(id=f"{t}_{s['now']:%Y%m%d%H}_{tt:%Y%m%d%H}", ts=s["now"], made_at=s["now"], target_time=tt,
                             metric=t, forecast=float(p), model_kind=art["kind"], model_version=int(ver)))
    store.upsert("forecast_ledger", pd.DataFrame(rows), ["id"])
    return {"forecasts": rows}


def _daily(forecasts):
    f = pd.DataFrame(forecasts).pivot(index="target_time", columns="metric", values="forecast")
    f["wind_dir"] = (np.degrees(np.arctan2(f["wind_sin"], f["wind_cos"])) + 360) % 360
    return f


def alert_node(s: State) -> State:
    f = _daily(s["forecasts"])
    best = {}
    for metric, kind, thr, level, label in ALERT_RULES:
        col = f[metric]
        hit = col.max() >= thr if kind == "max" else col.min() <= thr
        if hit and LEVEL_ORDER[level] > LEVEL_ORDER.get(best.get(metric, {}).get("level"), 0):
            when = col.idxmax() if kind == "max" else col.idxmin()
            best[metric] = dict(metric=metric, level=level, label=label, threshold=thr,
                                value=float(col.max() if kind == "max" else col.min()), peak_time=str(when))
    hazards = list(best.values())
    alerts, ts = [], s["now"]
    for h in hazards:
        prompt = (f"City: {CITY}. Hazard: {h['label']} ({h['level']}). Forecast {h['metric']} = {h['value']:.1f} "
                  f"around {h['peak_time']} UTC (rule threshold {h['threshold']}). Write short, calm, actionable advice. "
                  'Reply ONLY as JSON: {"public": "...", "sensitive_groups": "...", "outdoor_workers": "..."}')

        def ok(txt):
            j = json.loads(txt)
            assert all(k in j for k in ("public", "sensitive_groups", "outdoor_workers"))

        txt = llm(s, "alert", "You write public-safety weather alerts.", prompt, validate=ok, json_mode=True)
        src = "llm"
        if txt:
            msgs = json.loads(txt)
        else:
            src = "template"
            base = f"{h['label']}: {h['metric']} forecast {h['value']:.1f} around {h['peak_time']} UTC."
            msgs = {"public": base + " Plan outdoor activity accordingly.",
                    "sensitive_groups": base + " Children, elderly and people with heart/lung/asthma conditions should limit exposure.",
                    "outdoor_workers": base + " Schedule breaks, hydrate, use protective gear and adjust hours if possible."}
        alerts.append(dict(id=f"{ts:%Y%m%d%H}_{h['metric']}", ts=ts, metric=h["metric"], level=h["level"], label=h["label"],
                           forecast_value=h["value"], peak_time=h["peak_time"], source=src, **msgs))
    keys = {f"{a['metric']}:{a['level']}" for a in alerts}
    new = bool(keys - set(s["prev"].get("alert_keys", [])))
    store.upsert("alerts", pd.DataFrame(alerts), ["id"])
    return {"hazards": hazards, "alerts": alerts, "new_alert": new}


def drift_node(s: State) -> State:
    feats = store.read("weather_features")
    events = drift.run(feats, store.read("forecast_ledger"), store.read("model_metrics"),
                       store.read("drift_log"), s["now"])
    store.upsert("drift_log", pd.DataFrame(events), ["id"])
    return {"drift_events": events}


def _needs_report(s: State) -> str:
    due = s["now"].hour == REPORT_HOUR_UTC and s["prev"].get("last_report_day") != str(s["now"].date())
    return "report" if (due or s.get("new_alert") or s.get("force_report")) else "end"


def report_node(s: State) -> State:
    f = _daily(s["forecasts"])
    summ = (f"temp {f.temperature.min():.0f}-{f.temperature.max():.0f}C, AQI peak {f.aqi.max():.0f}, rain chance up to "
            f"{100 * f.rain_flag.max():.0f}% ({f.precipitation.sum():.1f} mm total), humidity {f.humidity.mean():.0f}%, "
            f"UV peak {f.uv_index.max():.1f}, wind avg {f.wind_speed.mean():.0f} km/h from {f.wind_dir.mean():.0f} deg")
    alerts = "; ".join(f"{a['label']} ({a['level']})" for a in s["alerts"]) or "none"
    prompt = (f"Write a 120-180 word news-style weather report for {CITY} covering the next 24 hours. "
              f"Forecast summary: {summ}. Active alerts: {alerts}. Plain text, no markdown, no invented numbers.")
    txt = llm(s, "weather_report", "You are a concise broadcast weather journalist.", prompt)
    if not txt:
        txt = (f"{CITY} weather outlook for the next 24 hours: temperatures between {f.temperature.min():.0f} and "
               f"{f.temperature.max():.0f}°C with humidity near {f.humidity.mean():.0f}%. Air quality peaks at an AQI of "
               f"{f.aqi.max():.0f}. Chance of rain reaches {100 * f.rain_flag.max():.0f}% with about {f.precipitation.sum():.1f} mm "
               f"expected. UV index tops out at {f.uv_index.max():.1f}; winds average {f.wind_speed.mean():.0f} km/h. "
               f"Active alerts: {alerts}. (Automated template report - LLM unavailable.)")
    store.upsert("weather_reports", pd.DataFrame([dict(id=f"{s['now']:%Y%m%d%H}", ts=s["now"], report=txt,
                                                        alerts=alerts, reason="alert" if s.get("new_alert") else "daily")]), ["id"])
    return {"report": txt}


# ------------------------------------------------------------------ graph
def build_graph():
    from langgraph.graph import StateGraph, END
    g = StateGraph(State)
    g.add_node("forecast", forecast_node)
    g.add_node("alert", alert_node)
    g.add_node("drift", drift_node)
    g.add_node("report", report_node)
    g.set_entry_point("forecast")
    g.add_edge("forecast", "alert")
    g.add_edge("alert", "drift")
    g.add_conditional_edges("drift", _needs_report, {"report": "report", "end": END})
    g.add_edge("report", END)
    return g.compile()


def main():
    now = pd.Timestamp.utcnow().tz_localize(None).floor("h")
    st = store.read("agent_state")
    prev = {}
    if not st.empty:
        r = st.sort_values("ts").iloc[-1]
        prev = dict(alert_keys=json.loads(r["alert_keys"]), last_report_day=r["last_report_day"])
    state: State = dict(now=now, prev=prev, llm_log=[], force_report=bool(os.getenv("FORCE_REPORT")))
    out = build_graph().invoke(state)
    keys = sorted({f"{a['metric']}:{a['level']}" for a in out.get("alerts", [])})
    last_day = str(now.date()) if out.get("report") else prev.get("last_report_day", "")
    store.upsert("agent_state", pd.DataFrame([dict(id=f"{now:%Y%m%d%H}", ts=now, alert_keys=json.dumps(keys),
                                                    last_report_day=last_day, n_alerts=len(keys))]), ["id"])
    store.upsert("agent_call_log", pd.DataFrame(out["llm_log"]), ["id"])
    log.info("run done: %d alerts, report=%s, drift events=%d, llm calls=%d (%d failed)",
             len(out.get("alerts", [])), bool(out.get("report")),
             sum(e["drifted"] for e in out.get("drift_events", [])), len(out["llm_log"]),
             sum(c["status"] != "ok" for c in out["llm_log"]))


if __name__ == "__main__":
    main()
