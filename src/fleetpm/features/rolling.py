import numpy as np
import pandas as pd


def _engine(X: pd.DataFrame, W: int, stats):
    r = X.rolling(W, min_periods=1)
    f = {"mean": r.mean(), "std": r.std().fillna(0.0), "min": r.min(), "max": r.max()}
    if "slope" in stats:  # closed-form OLS slope over trailing window, exact
        t = pd.Series(np.arange(len(X), dtype=float), index=X.index)
        mp = 3
        n = t.rolling(W, min_periods=mp).count().to_numpy()[:, None]
        St = t.rolling(W, min_periods=mp).sum().to_numpy()[:, None]
        Stt = (t * t).rolling(W, min_periods=mp).sum().to_numpy()[:, None]
        Sx = X.rolling(W, min_periods=mp).sum().to_numpy()
        Sxt = X.mul(t, axis=0).rolling(W, min_periods=mp).sum().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            sl = (n * Sxt - St * Sx) / (n * Stt - St ** 2)
        f["slope"] = pd.DataFrame(sl, index=X.index, columns=X.columns).fillna(0.0)
    if "kurt" in stats:
        k = X.rolling(W, min_periods=4).kurt()
        f["kurt"] = k.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.concat([f[s].add_suffix(f"__{s}") for s in stats], axis=1)


def build_features(Z: pd.DataFrame, sensors, W, stats):
    """Z sorted by (engine_id, cycle). Trailing windows only (causal). Row order preserved."""
    parts = []
    for eid, g in Z.groupby("engine_id", sort=True):
        F = _engine(g[sensors].reset_index(drop=True), W, stats)
        F.insert(0, "cycle", g["cycle"].to_numpy())
        F.insert(0, "engine_id", eid)
        parts.append(F)
    return pd.concat(parts, ignore_index=True)