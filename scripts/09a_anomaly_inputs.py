import _bootstrap

import time

import joblib
import numpy as np
import torch

from fleetpm.config import art, load_config
from fleetpm.models.anomaly import persistent
from fleetpm.pipeline import feat_cols, load_prepared

TOL_ROW = 1e-6            # a row "differs" if its recomputed score moves by more than this
MAX_FRAC_DIFF = 0.01      # fail if more than 1% of rows differ
MAX_FRAC_DECISION = 0.001  # fail if more than 0.1% of rows change side of any severity cut


def check(fd, cfg):
    t0 = time.time()
    P = load_prepared(cfg, fd)
    Ft = P["Fte"]
    A = joblib.load(art(cfg, "models", fd, "anomaly_ifl.joblib"))   # trusted artifact from 02_train_anomaly.py
    det, ifl = A["det"], A["if_l"]
    Xa = Ft[feat_cols(Ft)].to_numpy(np.float32)
    score = -ifl.decision_function(np.c_[det.latent(Xa), det.errors(Xa)])
    stored = np.load(art(cfg, "models", fd, "anomaly_scores.npz"))["S_IFl"]
    if stored.shape != score.shape:
        raise ValueError(f"{fd}: stored anomaly scores {stored.shape} vs recomputed {score.shape}")
    eng, cyc = Ft.engine_id.to_numpy(), Ft.cycle.to_numpy()
    cut, W, K = A["cut"], int(A["window"]), int(A["persist_k"])
    d = np.abs(stored - score)
    n_diff = int((d > TOL_ROW).sum())
    dec = {k: float(((stored > c) != (score > c)).mean()) for k, c in cut.items()}
    p_new = persistent((score > cut["medium"]) & (cyc >= W), eng, K)
    p_old = persistent((stored > cut["medium"]) & (cyc >= W), eng, K)
    n_pflip = int((p_new != p_old).sum())
    w = int(d.argmax())
    ok = n_diff / len(d) <= MAX_FRAC_DIFF and max(dec.values()) <= MAX_FRAC_DECISION
    print(f"{fd}: rows {len(d)} | max|diff| {d.max():.2e} (engine {int(eng[w])}, cycle {int(cyc[w])}) | p99.9 {np.percentile(d, 99.9):.2e} "
          f"| rows differing >{TOL_ROW:g}: {n_diff} ({n_diff / len(d):.2%}) | decision flips by cut "
          f"{ {k: round(v, 5) for k, v in dec.items()} } | persistent-flag flips {n_pflip} | {'PASS' if ok else 'FAIL'} | {time.time() - t0:.0f}s", flush=True)
    if ok:
        np.savez_compressed(art(cfg, "triage_inputs", f"{fd}.npz"), score=score, pers=p_new, window=W, persist_k=K,
                            cut_medium=cut["medium"], cut_high=cut["high"], cut_critical=cut["critical"],
                            max_abs_diff=float(d.max()), n_rows_differing=n_diff)
    return ok


def main():
    cfg = load_config()
    torch.set_num_threads(2)
    failed = [fd for fd in cfg["data"]["subsets"] if not check(fd, cfg)]
    if failed:
        raise SystemExit(f"FAILED anomaly artifact check for {failed}; paste this output before proceeding")
    print("saved artifacts/triage_inputs/FD00X.npz for all subsets")


if __name__ == "__main__":
    main()
