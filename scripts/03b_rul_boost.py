import _bootstrap

import argparse
import json
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from fleetpm.config import art, load_config
from fleetpm.features.rolling import build_features
from fleetpm.mlops import tracking as mt
from fleetpm.models import rul_tabular as T
from fleetpm.features.sequences import build_sequences
from fleetpm.models.metrics import cut_eval, last_rows, summarize
from fleetpm.models.rul_seq import SeqRegressor
from fleetpm.pipeline import feat_cols, load_prepared

SPACE = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.08],
    "min_child_weight": [5, 10, 20, 40],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.3, 0.5, 0.8],
    "reg_lambda": [1.0, 5.0, 20.0],
    "n_estimators": [300, 500, 800],
}
K_GRID = [1, 3, 5, 8, 12]
W_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]


def log(msg):
    print(msg, flush=True)


def design(P, split, windows):
    """Anomaly-compatible features plus short-window mean/slope (causal, built from smoothed Z)."""
    F, Z = P["F" + split], P["Z" + split]
    parts = [F[feat_cols(F)].reset_index(drop=True)]
    for w in windows:
        S = build_features(Z, P["pre"].sensors, w, ["mean", "slope"])
        if not (np.array_equal(S["engine_id"], F["engine_id"]) and np.array_equal(S["cycle"], F["cycle"])):
            raise ValueError(f"short-window rows misaligned with {split} features")
        parts.append(S.drop(columns=["engine_id", "cycle"]).add_suffix(f"_w{w}"))
    return pd.concat(parts, axis=1)


def causal_smooth(pred, eng, k, cap):
    """Mean of pred(t-j) - j over the last k cycles of the same engine; saturated predictions are not shifted."""
    if k <= 1:
        return pred.copy()
    g = pd.Series(pred).groupby(eng)
    num, cnt = np.zeros(len(pred)), np.zeros(len(pred))
    for j in range(k):
        v = pred if j == 0 else g.shift(j).to_numpy()
        adj = np.where(v >= 0.98 * cap, v, v - j)
        ok = np.isfinite(adj)
        num[ok] += adj[ok]
        cnt[ok] += 1
    return np.clip(num / cnt, 0, None)


def score(y, p, eng, ev, seed):
    return cut_eval(y, p, eng, ev["min_cut_len"], ev["n_cut_reps"], seed)


def cv_oof(name, rc, X, y, eng, splits, seed):
    oof = np.full(len(y), np.nan)
    for tr, va in splits:
        m = T.fit(T.make(name, rc, seed), name, X[tr], y[tr], T.weights(y[tr], rc))
        oof[va] = T.predict(m, X[va])
    return oof


def tune_xgb(r, X, y, eng, splits, ev, seed, n_iter):
    rng = np.random.default_rng(seed)
    trials = [{}] + [{k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()} for _ in range(n_iter)]
    base_s, best = None, None
    for i, prm in enumerate(trials):
        rc = {**r, "xgb": {**r["xgb"], **prm}}
        oof = cv_oof("xgb", rc, X, y, eng, splits, seed)
        s = score(y, oof, eng, ev, seed)["rmse"]
        base_s = s if i == 0 else base_s
        log(f"  xgb trial {i}/{n_iter} cv_rmse={s:.3f} {prm if prm else '(config default)'}")
        if best is None or s < best[0]:
            best = (s, prm, oof)
    return base_s, best[0], best[1], best[2]


def best_k(oof, y, eng, cap, ev, seed):
    by_k = {k: score(y, causal_smooth(oof, eng, k, cap), eng, ev, seed)["rmse"] for k in K_GRID}
    return min(by_k, key=by_k.get), by_k


def seg(y_raw, p, cap):
    out = {}
    for nm, m in (("le30", y_raw <= 30), ("30_60", (y_raw > 30) & (y_raw <= 60)),
                  ("60_cap", (y_raw > 60) & (y_raw <= cap)), ("gt_cap", y_raw > cap)):
        if m.any():
            e = p[m] - y_raw[m]
            out[nm] = {"n": int(m.sum()), "rmse": round(float(np.sqrt((e ** 2).mean())), 2),
                       "bias": round(float(e.mean()), 2)}
    return out


def test_block(p_all, last, y_raw, cap):
    p = p_all[last]
    return {"cap": summarize(np.minimum(y_raw, cap), p), "raw": summarize(y_raw, p), "seg": seg(y_raw, p, cap)}


G_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]


def boot_ci(y_raw, p, cap, n=1000, seed=0):
    """Engine-level bootstrap 95% CI of capped-label RMSE."""
    yc = np.minimum(y_raw, cap)
    idx = np.random.default_rng(seed).integers(0, len(p), (n, len(p)))
    r = np.sqrt(((p[idx] - yc[idx]) ** 2).mean(1))
    return [round(float(np.percentile(r, 2.5)), 2), round(float(np.percentile(r, 97.5)), 2)]


def boot_diff(y_raw, pa, pb, cap, n=1000, seed=0):
    """Paired engine bootstrap 95% CI of RMSE(a) - RMSE(b)."""
    yc = np.minimum(y_raw, cap)
    idx = np.random.default_rng(seed).integers(0, len(pa), (n, len(pa)))
    d = np.sqrt(((pa[idx] - yc[idx]) ** 2).mean(1)) - np.sqrt(((pb[idx] - yc[idx]) ** 2).mean(1))
    return [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]


def gru_block(P, cfg, y, eng, splits, cap, seed, n_seeds):
    """GRU seed-ensemble: fold-0 validation predictions and full-fit predictions for every test row."""
    W, sensors = cfg["features"]["window"], P["pre"].sensors
    S, St = build_sequences(P["Ztr"], sensors, W), build_sequences(P["Zte"], sensors, W)
    if len(S) != len(y):
        raise ValueError("sequence rows misaligned with training rows")
    tr, va = splits[0]
    pv = np.mean([SeqRegressor("gru", cfg, cap, seed + s).fit(S[tr], y[tr], eng[tr]).predict(S[va])
                  for s in range(n_seeds)], axis=0)
    pt = np.mean([SeqRegressor("gru", cfg, cap, seed + s).fit(S, y, eng).predict(St)
                  for s in range(n_seeds)], axis=0)
    return va, pv, pt


def run(fd, cfg, args):
    t0 = time.time()
    r, ev, seed = cfg["rul"], cfg["eval"], cfg["seed"]
    log(f"\n{'=' * 60}\n{fd}: boosted RUL study (windows={args.windows}, tune_iter={args.tune_iter})\n{'=' * 60}")
    P = load_prepared(cfg, fd)
    cap, F, Ft = P["cap"], P["Ftr"], P["Fte"]
    Xdf, Xtdf = design(P, "tr", args.windows), design(P, "te", args.windows)
    if list(Xdf.columns) != list(Xtdf.columns):
        raise ValueError("train/test feature order mismatch")
    cols, X, Xt = list(Xdf.columns), Xdf.to_numpy(float), Xtdf.to_numpy(float)
    if not (np.isfinite(X).all() and np.isfinite(Xt).all()):
        raise ValueError(f"{fd}: non-finite features")
    y, eng, et = F["rul_cap"].to_numpy(float), F.engine_id.to_numpy(), Ft.engine_id.to_numpy()
    last, y_raw = last_rows(et), np.asarray(P["rul"], float)
    if len(last) != len(y_raw):
        raise ValueError(f"{fd}: test engines {len(last)} != RUL labels {len(y_raw)}")
    splits = list(GroupKFold(n_splits=ev["cv_folds"]).split(X, y, eng))
    log(f"[{fd}] train rows {len(X)} | test rows {len(Xt)} | features {len(cols)} | cap {cap}")

    base_s, best_s, px, oof_x = tune_xgb(r, X, y, eng, splits, ev, seed, args.tune_iter)
    log(f"[{fd}] xgb cv_rmse default={base_s:.3f} tuned={best_s:.3f} (tuned value is selection-biased)")
    oof_e = cv_oof("extratrees", r, X, y, eng, splits, seed)
    wb = {w: score(y, w * oof_x + (1 - w) * oof_e, eng, ev, seed)["rmse"] for w in W_GRID}
    w = min(wb, key=wb.get)

    rx = {**r, "xgb": {**r["xgb"], **px}}
    mx = T.fit(T.make("xgb", rx, seed), "xgb", X, y, T.weights(y, rx))
    me = T.fit(T.make("extratrees", r, seed), "extratrees", X, y, T.weights(y, r))
    px_t, pe_t = T.predict(mx, Xt), T.predict(me, Xt)
    oofs = {"xgb": oof_x, "extratrees": oof_e, "blend": w * oof_x + (1 - w) * oof_e}
    pts = {"xgb": px_t, "extratrees": pe_t, "blend": w * px_t + (1 - w) * pe_t}

    np.savez_compressed(art(cfg, "models", fd, f"rul_boost_preds{args.tag}.npz"), oof_xgb=oof_x, test_xgb=px_t)
    cands = {}
    for nm in oofs:
        k, by_k = best_k(oofs[nm], y, eng, cap, ev, seed)
        cands[nm] = {"k": k, "cv_rmse_by_k": by_k,
                     "cv": score(y, causal_smooth(oofs[nm], eng, k, cap), eng, ev, seed),
                     "test": test_block(causal_smooth(pts[nm], et, k, cap), last, y_raw, cap),
                     "test_unsmoothed": test_block(pts[nm], last, y_raw, cap)}
    sel = min(cands, key=lambda n: cands[n]["cv"]["rmse"])

    log(f"\n[{fd}] blend w_xgb={w} | selected by CV only: {sel} (k={cands[sel]['k']})")
    log(f"{'cand':<11}{'k':>3}{'cv_rmse':>9}{'cv_nasa':>9} | {'tcap_rmse':>9}{'tcap_nasa':>10}{'traw_rmse':>10}{'traw_nasa':>11} | tcap_rmse_k1")
    for nm, c in cands.items():
        t, u = c["test"], c["test_unsmoothed"]
        log(f"{nm:<11}{c['k']:>3}{c['cv']['rmse']:>9.3f}{c['cv']['nasa']:>9.1f} | {t['cap']['rmse']:>9.3f}{t['cap']['nasa']:>10.1f}"
            f"{t['raw']['rmse']:>10.3f}{t['raw']['nasa']:>11.1f} | {u['cap']['rmse']:.3f}")
    log(f"[{fd}] selected test segments (n, rmse, bias vs raw RUL): {json.dumps(cands[sel]['test']['seg'])}")
    kk = {k: round(v, 3) for k, v in cands["xgb"]["cv_rmse_by_k"].items()}
    log(f"[{fd}] xgb cv_rmse by smoothing k: {kk}")
    ci = {nm: boot_ci(y_raw, pts[nm][last], cap) for nm in pts}
    log(f"[{fd}] test capped RMSE 95% engine-bootstrap CI: {ci}")
    if args.gru_seeds > 0:
        va, pv, pg_all = gru_block(P, cfg, y, eng, splits, cap, seed, args.gru_seeds)
        s0x = score(y[va], oofs["xgb"][va], eng[va], ev, seed)["rmse"]
        s0g = score(y[va], pv, eng[va], ev, seed)["rmse"]
        gb = {g: score(y[va], (1 - g) * oofs["xgb"][va] + g * pv, eng[va], ev, seed)["rmse"] for g in G_GRID}
        g = min(gb, key=gb.get)
        pbl = (1 - g) * pts["xgb"] + g * pg_all
        log(f"[{fd}] GRU eval-only | fold-0 val engines {len(np.unique(eng[va]))} | cv_rmse xgb={s0x:.3f} "
            f"gru={s0g:.3f} blend(g={g})={gb[g]:.3f} | adopt rule: fold-0 gain >= 0.3 -> {s0x - gb[g] >= 0.3}")
        tx = test_block(pts["xgb"], last, y_raw, cap)["cap"]
        tg = test_block(pg_all, last, y_raw, cap)["cap"]
        tb = test_block(pbl, last, y_raw, cap)["cap"]
        log(f"[{fd}] test cap rmse/nasa: xgb={tx['rmse']:.3f}/{tx['nasa']:.0f} gru={tg['rmse']:.3f}/{tg['nasa']:.0f} "
            f"blend={tb['rmse']:.3f}/{tb['nasa']:.0f}")
        log(f"[{fd}] paired bootstrap RMSE diff blend minus xgb, 95% CI: "
            f"{boot_diff(y_raw, pbl[last], pts['xgb'][last], cap)} (negative favours blend; report only, not used to select)")

    weights = {"xgb": {"xgb": 1.0, "extratrees": 0.0}, "extratrees": {"xgb": 0.0, "extratrees": 1.0},
               "blend": {"xgb": w, "extratrees": 1 - w}}[sel]
    bundle = {"cols": cols, "windows": args.windows, "models": {"xgb": mx, "extratrees": me},
              "weights": weights, "smooth_k": cands[sel]["k"], "selected": sel, "cap": cap,
              "xgb_params": px, "version": f"{fd}-boost-{sel}-s{seed}"}
    bp = art(cfg, "models", fd, f"rul_bundle_v2{args.tag}.joblib")
    joblib.dump(bundle, bp.with_suffix(".tmp"))
    bp.with_suffix(".tmp").replace(bp)
    out = {"fd": fd, "cap": cap, "n_features": len(cols), "windows": args.windows, "xgb_params": px,
           "xgb_cv_rmse_default": base_s, "xgb_cv_rmse_tuned": best_s, "blend_cv_rmse_by_w": wb,
           "selected": sel, "candidates": cands}
    jp = art(cfg, "models", fd, f"rul_boost{args.tag}.json")
    jp.with_suffix(".tmp").write_text(json.dumps(out, indent=2))
    jp.with_suffix(".tmp").replace(jp)

    flat = {}
    for nm, c in cands.items():
        flat.update({f"{nm}_cv_rmse": c["cv"]["rmse"], f"{nm}_cv_nasa": c["cv"]["nasa"],
                     f"{nm}_test_cap_rmse": c["test"]["cap"]["rmse"], f"{nm}_test_cap_nasa": c["test"]["cap"]["nasa"],
                     f"{nm}_test_raw_rmse": c["test"]["raw"]["rmse"], f"{nm}_test_raw_nasa": c["test"]["raw"]["nasa"]})
    with mlflow.start_run(run_name=f"{fd}_rul_boost"):
        mlflow.log_params({"fd": fd, "cap": cap, "windows": str(args.windows), "n_features": len(cols),
                           "tune_iter": args.tune_iter, "selected": sel, "smooth_k": cands[sel]["k"],
                           "blend_w_xgb": w, **{f"xgb_{k}": v for k, v in px.items()}})
        mlflow.log_metrics(flat)
    log(f"[{fd}] saved {bp.name}, {jp.name} in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description="Tuned, blended, causally smoothed tabular RUL study.")
    ap.add_argument("--fd", nargs="*")
    ap.add_argument("--windows", nargs="*", type=int, default=[10], help="extra short windows; pass none to disable")
    ap.add_argument("--tune-iter", type=int, default=12)
    ap.add_argument("--gru-seeds", type=int, default=0, help="0 disables the GRU blend evaluation")
    ap.add_argument("--tag", default="", help="suffix for output files so experiments do not overwrite artifacts")
    args = ap.parse_args()
    cfg = load_config()
    mt.setup(cfg)
    for fd in args.fd or cfg["data"]["subsets"]:
        run(fd, cfg, args)


if __name__ == "__main__":
    main()