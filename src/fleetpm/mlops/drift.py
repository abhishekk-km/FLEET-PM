import numpy as np


def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-6, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def should_retrain(cur_rmse, base_rmse, days_since, rmse_factor, days):
    return (cur_rmse > rmse_factor * base_rmse) or (days_since >= days)