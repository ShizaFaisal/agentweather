"""Offline EDA: saves figures + stats to reports/eda/. Run after backfill."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import store
from config import *

OUT = pathlib.Path("reports/eda"); OUT.mkdir(parents=True, exist_ok=True)
METRICS = ["aqi", "temperature", "precipitation", "humidity", "uv_index", "wind_speed"]


def main():
    d = store.read("weather_features").sort_values("ts").set_index("ts")
    raw = store.read("raw_observations")
    rows = []
    for m in METRICS:
        fig, ax = plt.subplots(2, 2, figsize=(12, 7))
        d[m].plot(ax=ax[0, 0], title=f"{m} - hourly series")
        d[m].plot.hist(bins=40, ax=ax[0, 1], title="distribution")
        d.groupby("hour")[m].mean().plot(ax=ax[1, 0], title="mean by hour of day (UTC)")
        pd.plotting.autocorrelation_plot(d[m].dropna().iloc[:24 * 14], ax=ax[1, 1])
        ax[1, 1].set_title("autocorrelation (first 14 days)")
        fig.tight_layout(); fig.savefig(OUT / f"{m}.png", dpi=110); plt.close(fig)
        s = d[m].describe().to_dict()
        s.update(metric=m, skew=d[m].skew(), pct_missing_raw=float(raw[m].isna().mean() * 100) if m in raw else None)
        try:
            from statsmodels.tsa.stattools import adfuller
            s["adf_pvalue"] = adfuller(d[m].dropna())[1]
        except Exception:
            s["adf_pvalue"] = None
        rows.append(s)
    pd.DataFrame(rows).set_index("metric").round(3).to_csv(OUT / "summary_stats.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    c = d[METRICS + ["pm2_5", "pm10"]].corr()
    im = ax.imshow(c, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=60); ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(OUT / "correlation.png", dpi=110)
    print("EDA saved to", OUT)


if __name__ == "__main__":
    main()
