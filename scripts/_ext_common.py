import _bootstrap
import json
import warnings
from datetime import datetime, timezone
import numpy as np
from fleetpm.config import art, ROOT
from fleetpm.pipeline import load_prepared
from fleetpm.models.metrics import last_rows

warnings.simplefilter("ignore", RuntimeWarning)
BANDS = [30, 60]
MIN_CYCLE = 30
PRED_FILE = "rul_boost_preds_gru.npz"
INF_Q = 1e9


def load_oof(cfg, fd):
    P = load_prepared(cfg, fd)
    z = np.load(art(cfg, "models", fd, PRED_FILE))
    F, T = P["Ftr"], P["Fte"]
    oof, tst = np.asarray(z["oof_xgb"], float), np.asarray(z["test_xgb"], float)
    if len(oof) != len(F) or len(tst) != len(T):
        raise ValueError(f"{fd}: prediction/feature length mismatch oof {len(oof)} vs {len(F)}, test {len(tst)} vs {len(T)}")
    if not (np.isfinite(oof).all() and np.isfinite(tst).all()):
        raise ValueError(f"{fd}: non-finite predictions in {PRED_FILE}")
    eng = F.engine_id.to_numpy()
    if (np.diff(eng) < 0).any():
        raise ValueError(f"{fd}: training rows not sorted by engine")
    te = T.engine_id.to_numpy()
    L = last_rows(te)
    raw_t = np.asarray(P["rul"], float)
    if len(L) != len(raw_t) or not (te[L] == np.arange(1, len(L) + 1)).all():
        raise ValueError(f"{fd}: test engine order does not match official RUL vector")
    cap = float(P["cap"])
    return dict(P=P, fd=fd, cap=cap, eng=eng, cyc=F.cycle.to_numpy(),
                y=F.rul_cap.to_numpy(float), raw=F.rul.to_numpy(float),
                oof=np.clip(oof, 0, None), pred_t=np.clip(tst[L], 0, None),
                true_raw=raw_t, true_cap=np.minimum(raw_t, cap))


def fin_q(s, alpha):
    s = np.sort(np.asarray(s, float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[k - 1]) if 1 <= k <= n else float("inf")


def conf_qfun(alpha):
    return lambda s: fin_q(s, alpha)


def quant_qfun(q):
    return lambda s: float(np.quantile(s, q))


def band_q(pred, s, qfun, min_n=20):
    """Per predicted-RUL-band quantile of s=pred-true, clipped at >=0 (conservative). Small bands use the marginal value."""
    band = np.digitize(pred, BANDS)
    qm = min(max(qfun(s), 0.0), INF_Q)
    out = []
    for b in range(len(BANDS) + 1):
        m = band == b
        out.append(min(max(qfun(s[m]), 0.0), INF_Q) if m.sum() >= min_n else qm)
    return out


def apply_q(pred, qs):
    return np.clip(pred - np.asarray(qs)[np.digitize(pred, BANDS)], 0, None)


def pct(x):
    return np.nanpercentile(x, [2.5, 97.5]).tolist()


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_merge(path, key, obj):
    d = {}
    if path.exists():
        d = json.loads(path.read_text())
    if isinstance(d.get(key), dict) and isinstance(obj, dict):
        d[key].update(obj)
    else:
        d[key] = obj
    path.write_text(json.dumps(d, indent=1, default=_jd))


def log(title, lines):
    import re
    p = ROOT / "docs" / "SYSTEM_LOG.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    txt = p.read_text() if p.exists() else ""
    parts = [x for x in re.split(r"(?m)^(?=## )", txt) if not x.startswith(f"## {title} (")]
    body = "".join(parts).rstrip("\n") + "\n"
    body += f"\n## {title} ({stamp})\nAuto-generated from script output. Audit pending until reviewed.\n"
    body += "".join(f"- {ln}\n" for ln in lines)
    p.write_text(body)
