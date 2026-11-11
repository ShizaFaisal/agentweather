"""Storage layer: Hopsworks (Feature Store + Model Registry) with a local fallback.

If HOPSWORKS_API_KEY is set everything goes to Hopsworks; otherwise it falls back to
local pickles in ./data and ./models so the whole project can be run/tested offline.
"""
import json, os, pathlib, joblib
import pandas as pd

DATA = pathlib.Path(os.getenv("LOCAL_STORE", "data"))
MODELS = pathlib.Path(os.getenv("LOCAL_MODELS", "models"))
_project = None


def use_hops() -> bool:
    return bool(os.getenv("HOPSWORKS_API_KEY"))


def _proj():
    global _project
    if _project is None:
        import hopsworks
        _project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
    return _project


def _naive(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]) and getattr(df[c].dt, "tz", None) is not None:
            df[c] = df[c].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


# ---------------------------------------------------------------- feature store
def read(name: str) -> pd.DataFrame:
    try:
        if use_hops():
            fg = _proj().get_feature_store().get_feature_group(name, version=1)
            return _naive(fg.read()) if fg is not None else pd.DataFrame()
        p = DATA / f"{name}.pkl"
        return pd.read_pickle(p) if p.exists() else pd.DataFrame()
    except Exception as e:  # group does not exist yet
        print(f"[store] read({name}) -> empty ({type(e).__name__}: {str(e)[:80]})")
        return pd.DataFrame()


def upsert(name: str, df: pd.DataFrame, pk: list, wait: bool = False) -> None:
    """Insert rows; rows with an existing primary key are overwritten."""
    if df is None or df.empty:
        return
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if use_hops():
        fg = _proj().get_feature_store().get_or_create_feature_group(
            name=name, version=1, primary_key=pk,
            event_time="ts" if "ts" in df.columns else None, description=name)
        fg.insert(df, write_options={"wait_for_job": wait})
    else:
        DATA.mkdir(exist_ok=True)
        old = read(name)
        new = pd.concat([old, df], ignore_index=True).drop_duplicates(pk, keep="last")
        new.to_pickle(DATA / f"{name}.pkl")


# --------------------------------------------------------------- model registry
def load_model(name: str):
    """Return (artifact, version) of the latest registered model, or (None, None)."""
    try:
        if use_hops():
            mr = _proj().get_model_registry()
            models = mr.get_models(name)
            if not models:
                return None, None
            m = max(models, key=lambda x: x.version)
            return joblib.load(pathlib.Path(m.download()) / "model.joblib"), m.version
        d = MODELS / name
        if not (d / "model.joblib").exists():
            return None, None
        return joblib.load(d / "model.joblib"), json.loads((d / "meta.json").read_text())["version"]
    except Exception as e:
        print(f"[store] load_model({name}) failed: {type(e).__name__}: {str(e)[:100]}")
        return None, None


def save_model(name: str, artifact: dict, metrics: dict) -> int:
    d = MODELS / name
    d.mkdir(parents=True, exist_ok=True)
    version = 1
    meta = d / "meta.json"
    if meta.exists():
        version = json.loads(meta.read_text())["version"] + 1
    joblib.dump(artifact, d / "model.joblib")
    meta.write_text(json.dumps({"version": version, "metrics": metrics}))
    if use_hops():
        mr = _proj().get_model_registry()
        m = mr.python.create_model(name, metrics={k: float(v) for k, v in metrics.items()},
                                   description=f"{artifact['kind']} model for {name}")
        m.save(str(d))
        version = m.version
    return version
