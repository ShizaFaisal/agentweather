"""Model zoo: naive baseline + 3 families (statistical, classical ML, deep learning)."""
import warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import *

warnings.filterwarnings("ignore")
LSTM_WINDOW = 24
SARIMAX_HISTORY = 24 * 14


def feature_cols(df):
    return [c for c in df.columns if c != "ts" and not c.startswith("y_")]


def prep(feats: pd.DataFrame) -> pd.DataFrame:
    """Sort, regularise to an hourly index, fill small feature gaps (never the targets)."""
    d = feats.drop_duplicates("ts").sort_values("ts").set_index("ts").asfreq("h")
    fc = feature_cols(d.reset_index())
    d[fc] = d[fc].interpolate(limit=6).ffill().bfill()
    return d


def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return {"rmse": float(np.sqrt(np.mean((y - p) ** 2))), "mae": float(np.mean(np.abs(y - p))),
            "r2": float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")}


def clip(t, p):
    lo, hi = {"rain_flag": (0, 1), "humidity": (0, 100), "wind_sin": (-1, 1), "wind_cos": (-1, 1)}.get(
        t, (0, None) if t in ("aqi", "precipitation", "uv_index", "wind_speed") else (None, None))
    return np.clip(p, lo, hi)


def _windows(Xn, pos):
    """Stack the LSTM_WINDOW hours ending at each position (front-padded at the start)."""
    return np.stack([np.vstack([np.repeat(Xn[:1], max(0, LSTM_WINDOW - 1 - p), 0),
                                Xn[max(0, p - LSTM_WINDOW + 1): p + 1]]) for p in pos])


# ------------------------------------------------------------------ trainers
def naive_art():
    return {"kind": "naive"}


def train_sk(df, tr, t, kind):
    fc = feature_cols(df.reset_index())
    m = (make_pipeline(StandardScaler(), Ridge(alpha=10.0)) if kind == "ridge" else
         RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=0))
    m.fit(df[fc].iloc[tr].to_numpy(), df[f"y_{t}"].iloc[tr].to_numpy())
    return {"kind": kind, "model": m, "feat_cols": fc}


def train_sarimax(df, tr, t):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = df[t].to_numpy()[: tr[-1] + 1][-24 * 60:]
    order, seas = (1, 0, 1), (1, 0, 0, 24)
    res = SARIMAX(y, order=order, seasonal_order=seas, trend="c").fit(disp=False, maxiter=60)
    return {"kind": "sarimax", "params": res.params, "order": order, "seasonal": seas}


def train_lstm(df, tr, va, epochs=30):
    import torch, torch.nn as nn
    torch.manual_seed(0)
    fc = feature_cols(df.reset_index())
    X = df[fc].to_numpy(float)
    Y = df[[f"y_{t}" for t in TARGETS]].to_numpy(float)
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-6
    ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-6

    def windows(pos):
        return torch.tensor(_windows((X - xm) / xs, pos), dtype=torch.float32)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.l = nn.LSTM(len(fc), 32, batch_first=True)
            s.o = nn.Linear(32, len(TARGETS))
        def forward(s, x):
            return s.o(s.l(x)[0][:, -1])

    net = Net()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    xt, yt = windows(tr), torch.tensor((Y[tr] - ym) / ys, dtype=torch.float32)
    xv, yv = windows(va), torch.tensor((Y[va] - ym) / ys, dtype=torch.float32)
    best, best_state = 1e9, None
    for _ in range(epochs):
        net.train()
        for i in torch.randperm(len(xt)).split(64):
            opt.zero_grad()
            nn.functional.mse_loss(net(xt[i]), yt[i]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(net(xv), yv).item()
        if v < best:
            best, best_state = v, {k: x.detach().numpy().copy() for k, x in net.state_dict().items()}
    return {"kind": "lstm", "state": best_state, "feat_cols": fc, "xm": xm, "xs": xs, "ym": ym, "ys": ys}


# --------------------------------------------------------------- prediction
def _sarimax_predict(art, y, pos):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    out = []
    for i in pos:
        h = y[max(0, i + 1 - SARIMAX_HISTORY): i + 1]
        res = SARIMAX(h, order=art["order"], seasonal_order=art["seasonal"], trend="c").smooth(art["params"])
        out.append(float(np.asarray(res.forecast(HORIZON))[-1]))
    return np.array(out)


def _lstm_predict(art, df, pos, t):
    import torch, torch.nn as nn
    fc = art["feat_cols"]
    Xn = (df[fc].to_numpy(float) - art["xm"]) / art["xs"]
    xb = torch.tensor(_windows(Xn, pos), dtype=torch.float32)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.l = nn.LSTM(len(fc), 32, batch_first=True)
            s.o = nn.Linear(32, len(TARGETS))
        def forward(s, x):
            return s.o(s.l(x)[0][:, -1])

    net = Net()
    net.load_state_dict({k: torch.tensor(v) for k, v in art["state"].items()})
    net.eval()
    with torch.no_grad():
        o = net(xb).numpy()
    j = TARGETS.index(t)
    return o[:, j] * art["ys"][j] + art["ym"][j]


def predict_rows(art, df, pos, t):
    """Prediction of target `t` (24h after each row in `pos`). df = prepped frame."""
    k = art["kind"]
    if k == "naive":
        p = df[t].to_numpy()[pos]                      # value 24h ago == value now for the t+24 target
    elif k in ("ridge", "rf"):
        p = art["model"].predict(df[art["feat_cols"]].to_numpy()[pos])
    elif k == "sarimax":
        p = _sarimax_predict(art, df[t].to_numpy(), pos)
    elif k == "lstm":
        p = _lstm_predict(art, df, pos, t)
    else:
        raise ValueError(k)
    return clip(t, np.asarray(p, float))
