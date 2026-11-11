# AgentWeather

24-hour forecasts of AQI, temperature, precipitation (+ rain probability), humidity, UV index and wind
(speed + direction) on a serverless stack: **GitHub Actions** (compute) + **Hopsworks** (Feature Store +
Model Registry) + **Streamlit Cloud** (dashboard). A **LangGraph** agent layer (Forecast → Alert → Drift →
Weather Report) sits on top, and every agent decision is logged.

## Architecture
```
GitHub Actions (hourly)                      GitHub Actions (daily + on drift)
ingest.py -> Feature Store                   train.py -> Model Registry
   |                                            ^ (repository_dispatch "drift-detected")
agents.py  (one LangGraph graph, shared state)  |
  Forecast -> Alert -> Drift ---------------------+
                          \-> Weather Report (daily / on new alert)
Feature Store groups: weather_features, raw_observations, forecast_ledger, alerts, weather_reports,
                      agent_call_log, drift_log, agent_state, model_metrics, shap_importance
```
Data: Open-Meteo Weather + Air-Quality APIs (free, no key). UV is reported by both → air-quality endpoint has
priority, weather endpoint fills gaps. City is set with `CITY/LAT/LON` env vars (config.py).

## Setup
1. Create a free [Hopsworks](https://app.hopsworks.ai) project → API key. Free [Groq](https://console.groq.com) key.
2. Push this repo to GitHub → *Settings → Secrets → Actions*: `HOPSWORKS_API_KEY`, `GROQ_API_KEY`,
   optional `SLACK_WEBHOOK_URL`. (Optional repo variables/env: `CITY`, `LAT`, `LON`.)
3. **Backfill + first training (run once locally):**
   ```bash
   pip install -r requirements.txt
   export HOPSWORKS_API_KEY=...
   python ingest.py --backfill 90     # Open-Meteo allows up to 92 days
   python train.py                    # trains 5 candidates per target, registers winners, computes SHAP
   python agents.py                   # first forecast (set FORCE_REPORT=1 for a first report)
   python eda.py                      # EDA figures -> reports/eda/
   ```
4. Enable the workflows (`hourly.yml`, `train.yml`) - they take over from here.
5. **Dashboard:** share.streamlit.io → new app → repo, main file `dashboard/app.py`; add secret
   `HOPSWORKS_API_KEY = "..."`.

No key? Everything falls back to local files in `./data` and `./models`, so you can run the full pipeline offline.

## Design decisions (short)
- **Targets:** value 24 h ahead (`y_<metric>` = `shift(-24)`). The newest 24 rows have no target; the Forecast Agent
  predicts those 24 rows, giving the full next-24-hour curve.
- **Models per target:** naive 24h-ago baseline, Ridge, RandomForest (classical), SARIMAX(1,0,1)(1,0,0,24)
  (statistical), one multi-output LSTM (deep). Time split 70/15/15 with 24 h purge; best on validation wins and is
  promoted only if it beats the *currently deployed* model re-scored on the same validation window.
- **Wind direction** is modelled as sin/cos (circular) and recombined with atan2. **Rain probability** is the
  clipped output of a regressor trained on the 0/1 rain flag (≥ 0.1 mm).
- **Imputation:** ≤6 h gaps interpolated (precipitation → 0), then ffill ≤24 h, then hour-of-day median.
- **Drift:** data drift = KS statistic live (7 d) vs training window > 0.30; model drift = recent 48 h MAE /
  validation MAE > 1.5. Drift → notify team + `repository_dispatch` → training workflow (12 h cooldown).
- **LLM fallbacks:** Groq 70B → Groq 8B → deterministic template. Invalid JSON counts as a failed call.
- **Explainability:** SHAP (TreeExplainer on the Random Forest) stored in the Feature Store and shown in the dashboard.
