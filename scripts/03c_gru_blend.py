import _bootstrap

import argparse
import time

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from fleetpm.config import art, load_config
from fleetpm.features.sequences import build_sequences
from fleetpm.models.metrics import cut_eval, last_rows, summarize
from fleetpm.models.rul_seq import SeqRegressor
from fleetpm.pipeline import load_prepared

G_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]


def log(msg):
    print(msg, flush=True)


def boot_diff(y_raw, pa, pb, cap, n=1000, seed=0):
    yc = np.minimum(y_raw, cap)
    idx = np.random.default_rng(seed).integers(0, len(pa), (n, len(pa)))
    d = np.sqrt(((pa[idx] - yc[idx]) ** 2).mean(1)) - np.sqrt(((pb[idx] - yc[idx]) ** 2).mean(1))
    return [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]


def timed_fit(tag, cfg, cap, seed, S, y, eng):
    t = time.time()
    m = SeqRegressor("gru", cfg, cap, seed).fit(S, y, eng)
    log(f"  {tag} seed {seed}: fit {time.time() - t:.0f}s")
    return m


def main():
    ap = argparse.ArgumentParser(description="Evaluate a GRU/XGB blend in its own process.")
    ap.add_argument("--fd", nargs="*")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=0, help="override config epochs (timing test: 2)")
    ap.add_argument("--tag", default="_gru")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = load_config()
    if args.epochs:
        cfg["rul"]["seq"]["epochs"] = args.epochs
    ev, seed = cfg["eval"], cfg["seed"]

    for fd in args.fd or cfg["data"]["subsets"]:
        log(f"\n{'=' * 50}\n{fd}: GRU blend evaluation (epochs={cfg['rul']['seq']['epochs']}, threads={args.threads})\n{'=' * 50}")
        P = load_prepared(cfg, fd)
        cap, F, Ft = P["cap"], P["Ftr"], P["Fte"]
        y, eng, et = F["rul_cap"].to_numpy(float), F.engine_id.to_numpy(), Ft.engine_id.to_numpy()
        last, y_raw = last_rows(et), np.asarray(P["rul"], float)
        z = np.load(art(cfg, "models", fd, f"rul_boost_preds{args.tag}.npz"))
        oof_x, test_x = z["oof_xgb"], z["test_xgb"]
        if len(oof_x) != len(y) or len(test_x) != len(et):
            raise ValueError(f"{fd}: saved XGB predictions do not match current data")
        tr, va = next(iter(GroupKFold(n_splits=ev["cv_folds"]).split(np.zeros(len(y)), y, eng)))
        W, sensors = cfg["features"]["window"], P["pre"].sensors
        S, St = build_sequences(P["Ztr"], sensors, W), build_sequences(P["Zte"], sensors, W)
        if len(S) != len(y) or len(St) != len(et):
            raise ValueError(f"{fd}: sequence rows misaligned")

        pv, pt = [], []
        for s in range(args.seeds):
            pv.append(timed_fit("fold0", cfg, cap, seed + s, S[tr], y[tr], eng[tr]).predict(S[va]))
            pt.append(timed_fit("full ", cfg, cap, seed + s, S, y, eng).predict(St))
        pv, pt = np.mean(pv, axis=0), np.mean(pt, axis=0)

        sc = lambda p: cut_eval(y[va], p, eng[va], ev["min_cut_len"], ev["n_cut_reps"], seed)["rmse"]
        gb = {g: sc((1 - g) * oof_x[va] + g * pv) for g in G_GRID}
        g = min(gb, key=gb.get)
        log(f"[{fd}] fold-0 val engines {len(np.unique(eng[va]))} | cv_rmse xgb={gb[0.0]:.3f} gru={sc(pv):.3f} "
            f"best blend g={g}: {gb[g]:.3f} | adopt (gain>=0.3): {gb[0.0] - gb[g] >= 0.3}")
        pb = (1 - g) * test_x + g * pt
        for nm, p in (("xgb", test_x), ("gru", pt), ("blend", pb)):
            c = summarize(np.minimum(y_raw, cap), p[last])
            log(f"[{fd}] test cap {nm:<5} rmse={c['rmse']:.3f} mae={c['mae']:.3f} nasa={c['nasa']:.0f}")
        log(f"[{fd}] paired bootstrap RMSE(blend)-RMSE(xgb) 95% CI: {boot_diff(y_raw, pb[last], test_x[last], cap)} "
            "(report only, not used to select)")


if __name__ == "__main__":
    main()