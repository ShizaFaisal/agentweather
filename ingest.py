"""Feature pipeline (runs hourly, also used for backfill).

    python ingest.py                 # hourly run: last 4 days, upsert the newest 48h
    python ingest.py --backfill 90   # backfill up to 92 days (Open-Meteo limit)

Sources (both free, no key):
  1. Open-Meteo Weather API       -> temperature, humidity, precipitation, wind, UV
  2. Open-Meteo Air-Quality API   -> US AQI, PM2.5, PM10, UV
Fusion: time-aligned outer join on the UTC hour. For UV (reported by both) the
air-quality endpoint has priority and the weather endpoint fills its gaps.
"""
import argparse, logging, time
import numpy as np, pandas as pd, requests
from config import *
import store

log = logging.getLogger("ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _get(url, params, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("GET %s failed (%s) attempt %d/%d", url.split("/")[2], e, i + 1, tries)
            time.sleep(2 ** i)
    return None


def fetch_weather(past_days: int) -> pd.DataFrame:
    j = _get("https://api.open-meteo.com/v1/forecast", dict(
        latitude=LAT, longitude=LON, past_days=past_days, forecast_days=1, timezone="UTC",
        hourly="temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,uv_index"))
    if not j:
        return pd.DataFrame()
    d = pd.DataFrame(j["hourly"])
    d["ts"] = pd.to_datetime(d.pop("time"))
    return d.rename(columns={"temperature_2m": "temperature", "relative_humidity_2m": "humidity",
                             "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_dir",
                             "uv_index": "uv_wx"}).set_index("ts")


def fetch_air(past_days: int) -> pd.DataFrame:
    j = _get("https://air-quality-api.open-meteo.com/v1/air-quality", dict(
        latitude=LAT, longitude=LON, past_days=min(past_days, 92), forecast_days=1, timezone="UTC",
        hourly="us_aqi,pm2_5,pm10,uv_index"))
    if not j:
        return pd.DataFrame()
    d = pd.DataFrame(j["hourly"])
    d["ts"] = pd.to_datetime(d.pop("time"))
    return d.rename(columns={"us_aqi": "aqi", "uv_index": "uv_aq"}).set_index("ts")


def fuse(wx: pd.DataFrame, aq: pd.DataFrame) -> pd.DataFrame:
    if wx.empty and aq.empty:
        raise RuntimeError("Both data sources are down - nothing to ingest")
    d = pd.concat([x for x in (wx, aq) if not x.empty], axis=1)
    for c in ("uv_aq", "uv_wx"):
        if c not in d:
            d[c] = np.nan
    d["uv_index"] = d["uv_aq"].combine_first(d["uv_wx"])          # source priority
    for c in RAW_COLS:
        if c not in d:
            d[c] = np.nan                                         # a whole source was down
    now = pd.Timestamp.utcnow().tz_localize(None).floor("h")
    d = d[d.index <= now]                                         # observations only
    return d[RAW_COLS].asfreq("h")


def impute(raw: pd.DataFrame) -> pd.DataFrame:
    """Documented strategy (in order):
    1. gaps <=6h: linear interpolation (precipitation: filled with 0 = no rain)
    2. longer gaps: forward-fill up to 24h
    3. still missing: median for that hour of day over the fetched window
    4. last resort: back-fill
    `n_imputed` records how many raw columns were imputed in each row."""
    d = raw.copy()
    n_imp = d.isna().sum(axis=1)
    for c in RAW_COLS:
        if c == "precipitation":
            d[c] = d[c].fillna(0.0)
        else:
            d[c] = d[c].interpolate(limit=6, limit_area="inside").ffill(limit=24)
            d[c] = d[c].fillna(d[c].groupby(d.index.hour).transform("median"))
    d[RAW_COLS] = d[RAW_COLS].bfill()
    d["n_imputed"] = n_imp.astype(float)
    r = np.radians(d["wind_dir"])
    d["wind_sin"], d["wind_cos"] = np.sin(r), np.cos(r)
    d["rain_flag"] = (d["precipitation"] >= RAIN_THRESHOLD_MM).astype(float)
    return d


def make_features(d: pd.DataFrame) -> pd.DataFrame:
    base = TARGETS
    f = d[base + ["pm2_5", "pm10", "n_imputed"]].copy()
    idx = f.index
    f["hour"], f["dayofweek"], f["month"] = idx.hour, idx.dayofweek, idx.month
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * idx.hour / 24), np.cos(2 * np.pi * idx.hour / 24)
    for c in base:
        f[f"{c}_d1"] = f[c].diff(1)                      # rate of change
        f[f"{c}_d3"] = f[c].diff(3)
        f[f"{c}_ma6"] = f[c].rolling(6, min_periods=1).mean()
        f[f"{c}_ma24"] = f[c].rolling(24, min_periods=1).mean()
        f[f"{c}_lag24"] = f[c].shift(24)
    for c in base:
        f[f"y_{c}"] = f[c].shift(-HORIZON)               # target = value 24h ahead
    f = f.iloc[24:].reset_index().rename(columns={"index": "ts"})   # drop warm-up rows
    f.columns = ["ts"] + list(f.columns[1:])
    return f


def main(backfill: int = 0):
    past = backfill if backfill else 4
    raw = fuse(fetch_weather(past), fetch_air(past))
    log.info("fetched %d hourly rows, %d missing cells", len(raw), int(raw.isna().sum().sum()))
    clean = impute(raw)
    feats = make_features(clean)
    if not backfill:
        feats = feats[feats["ts"] >= feats["ts"].max() - pd.Timedelta(hours=48)]
    raw_out = raw.reset_index().rename(columns={"index": "ts"})
    raw_out = raw_out[raw_out["ts"].isin(feats["ts"])] if not backfill else raw_out
    store.upsert("raw_observations", raw_out, ["ts"])
    store.upsert("weather_features", feats, ["ts"], wait=True)
    log.info("stored %d feature rows (latest %s)", len(feats), feats["ts"].max())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=0, help="days of history to backfill (max 92)")
    main(ap.parse_args().backfill)
