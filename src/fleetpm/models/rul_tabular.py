import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

TREE = ("extratrees", "xgb")


def make(name, r, seed):
    if name == "ridge":
        return Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=r["ridge"]["alpha"]))])
    if name == "extratrees":
        return ExtraTreesRegressor(**r["extratrees"], random_state=seed, n_jobs=-1)
    if name == "xgb":
        return XGBRegressor(**r["xgb"], random_state=seed, n_jobs=-1)
    raise ValueError(name)


def weights(y, r):
    return np.where(y < r["low_rul_threshold"], r["low_rul_weight"], 1.0)


def fit(m, name, X, y, w):
    if name == "ridge":
        m.fit(X, y, ridge__sample_weight=w)
    else:
        m.fit(X, y, sample_weight=w)
    return m


def predict(m, X):
    return np.clip(m.predict(X), 0, None)


class EngineBootstrap:
    """Bootstrap over ENGINES (not rows). Spread = model variance only (heuristic, not a CI)."""

    def __init__(self, name, r, n, seed):
        self.name, self.r, self.n, self.seed = name, r, n, seed

    def fit(self, X, y, eng):
        ids, start, cnt = np.unique(eng, return_index=True, return_counts=True)
        rng = np.random.default_rng(self.seed)
        self.members = []
        for i in range(self.n):
            pick = rng.integers(0, len(ids), len(ids))
            idx = np.concatenate([np.arange(start[j], start[j] + cnt[j]) for j in pick])
            m = make(self.name, self.r, self.seed + i + 1)
            self.members.append(fit(m, self.name, X[idx], y[idx], weights(y[idx], self.r)))
        return self

    def predict(self, X):
        P = np.stack([predict(m, X) for m in self.members])
        return P.mean(0), P.std(0)