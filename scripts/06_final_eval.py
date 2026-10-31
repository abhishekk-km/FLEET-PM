import _bootstrap

import json

import joblib
import numpy as np
import pandas as pd

from fleetpm.config import art, load_config
from fleetpm.features.rolling import build_features
from fleetpm.models import rul_tabular as T
from fleetpm.models.metrics import last_rows, summarize
from fleetpm.pipeline import feat_cols, load_prepared

TAG = "_gru"


def design(P, split, windows):
    F, Z = P["F" + split], P["Z" + split]
    parts = [F[feat_cols(F)].reset_index(drop=True)]
    for w in windows:
        S = build_features(Z, P["pre"].sensors, w, ["mean", "slope"])
        if not (np.array_equal(S["engine_id"], F["engine_id"]) and np.array_equal(S["cycle"], F["cycle"])):
            raise ValueError(f"short-window rows misaligned with {split} features")
        parts.append(S.drop(columns=["engine_id", "cycle"]).add_suffix(f"_w{w}"))
    return pd.concat(parts, axis=1)


def bands(y_raw, pred, cap):
    out = {}
    for nm, m in (("le30", y_raw <= 30), ("30_60", (y_raw > 30) & (y_raw <= 60)),
                  ("60_cap", (y_raw > 60) & (y_raw <= cap)), ("gt_cap", y_raw > cap)):
        if m.any():
            e = pred[m] - y_raw[m]
            out[nm] = {"n": int(m.sum()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean())}
    return out


def main():
    cfg = load_config()
    summary = {}
    for fd in cfg["data"]["subsets"]:
        bp = art(cfg, "models", fd, f"rul_bundle_v2{TAG}.joblib")
        if not bp.exists():
            raise FileNotFoundError(f"{bp.name} missing; run 03b_rul_boost.py --tag {TAG} first")
        B = joblib.load(bp)  # trusted artifact produced by 03b in this repository
        P = load_prepared(cfg, fd)
        cap = P["cap"]
        Xdf = design(P, "te", B["windows"])
        if list(Xdf.columns) != B["cols"]:
            raise ValueError(f"{fd}: inference feature order differs from the bundle")
        X = Xdf.to_numpy(float)
        if not np.isfinite(X).all():
            raise ValueError(f"{fd}: non-finite test features")
        et = P["Fte"].engine_id.to_numpy()
        last = last_rows(et)
        pred = T.predict(B["models"]["xgb"], X[last])      # final model: XGBoost only, no smoothing
        if not np.isfinite(pred).all():
            raise ValueError(f"{fd}: non-finite predictions")
        y_raw = np.asarray(P["rul"], float)
        y_cap = np.minimum(y_raw, cap)
        df = pd.DataFrame({"engine_id": et[last], "last_cycle": P["Fte"].cycle.to_numpy()[last],
                           "rul_true_raw": y_raw, "rul_true_cap": y_cap, "rul_pred": pred,
                           "error_vs_cap": pred - y_cap})
        df.to_csv(art(cfg, "predictions", f"rul_final_{fd}.csv"), index=False)

        e = df["error_vs_cap"].to_numpy()
        idx = np.random.default_rng(0).integers(0, len(e), (1000, len(e)))
        ci = np.percentile(np.sqrt((e[idx] ** 2).mean(1)), [2.5, 97.5]).round(2).tolist()
        js = json.loads(art(cfg, "models", fd, f"rul_boost{TAG}.json").read_text())
        ref = js["candidates"]["xgb"]["test_unsmoothed"]["cap"]["rmse"]
        sc = summarize(y_cap, pred)
        summary[fd] = {"version": B["version"], "n_test_engines": int(len(df)), "windows": B["windows"], "cap": {**sc, "ci": ci},
                       "raw": summarize(y_raw, pred), "bands": bands(y_raw, pred, cap),
                       "cv_rmse_k1": float(js["candidates"]["xgb"]["cv_rmse_by_k"]["1"]),
                       "repro_diff_vs_training_run": float(sc["rmse"] - ref)}
        print(f"{fd}: n={len(df)} cap RMSE={sc['rmse']:.3f} CI={ci} NASA={sc['nasa']:.0f} | raw RMSE={summary[fd]['raw']['rmse']:.3f} "
              f"| repro diff vs 03b run={summary[fd]['repro_diff_vs_training_run']:+.2e}", flush=True)
    art(cfg, "final_summary.json").write_text(json.dumps(summary, indent=2))
    print("saved final_summary.json and artifacts/predictions/rul_final_FD00X.csv")


if __name__ == "__main__":
    main()