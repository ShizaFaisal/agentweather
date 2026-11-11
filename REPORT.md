# AgentWeather - Project Report
*(Fill the numbers in `<>` from the dashboard / Hopsworks after a few days of running.)*

## 1. Data sources
Open-Meteo Weather API (temperature, humidity, precipitation, wind, UV) and Open-Meteo Air-Quality API
(US AQI, PM2.5, PM10, UV). Fusion: outer join on UTC hour; UV from the air-quality endpoint with
weather endpoint as fallback. Backfill: <N> days (<start> → <end>), <rows> hourly rows.

## 2. Imputation
Strategy and order: see README. Share of imputed cells per variable: <table from raw_observations>.
Longest gap observed: <h> hours.

## 3. EDA findings
<per metric: distribution/skew, diurnal pattern, seasonality, ADF stationarity p-value, key correlations>
Figures: `reports/eda/*.png` and the dashboard EDA tab.

## 4. Models
| Target | Naive RMSE | Ridge | RF | SARIMAX | LSTM | Deployed | Test RMSE / MAE / R² |
|---|---|---|---|---|---|---|---|
| aqi | | | | | | | |
(…one row per target; from `model_metrics`)
Why these models, what won where and why: <discussion>. SHAP findings: <top drivers per target>.

## 5. Issues and fixes
- Open-Meteo archive limit (92 days) → backfill window and short seasonal coverage.
- Hopsworks inserts are asynchronous jobs → logs are written with `wait_for_job=False`, features with `True`.
- Precipitation/rain are zero-inflated → rain modelled as 0/1 target, probability = clipped regression output.
- <your own issues>

## 6. Agent performance
- **Alert quality:** <alerts raised / hazards that actually occurred (compare ledger vs observations) / false alarms>.
- **Drift detection:** <events, true vs false positives, training runs triggered, error before/after retrain>.
- **LLM calls:** <total, failure rate, p50/p95 latency, % template fallbacks> (dashboard → Agent activity).
