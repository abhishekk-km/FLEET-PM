import numpy as np


def nasa_score(y, p):
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def summarize(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "mae": float(np.mean(np.abs(p - y))),
            "nasa": nasa_score(y, p),
            "nasa_mean": nasa_score(y, p) / len(y)}


def _bounds(eng):
    _, start, cnt = np.unique(eng, return_index=True, return_counts=True)
    return start, cnt


def last_rows(eng):
    s, c = _bounds(eng)
    return s + c - 1


def cut_rows(eng, min_len, rng):
    """One random truncation row per engine (rows must be contiguous, sorted by cycle)."""
    s, c = _bounds(eng)
    lo = np.minimum(min_len, c)
    return s + rng.integers(lo, c + 1) - 1


def cut_eval(y, p, eng, min_len, reps, seed):
    rng = np.random.default_rng(seed)
    out = [summarize(y[r], p[r]) for r in (cut_rows(eng, min_len, rng) for _ in range(reps))]
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}