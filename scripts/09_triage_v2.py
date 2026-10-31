import _bootstrap

import inspect
import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from fleetpm.agent.triage import TriageAgent
from fleetpm.config import art, load_config
from fleetpm.features.rolling import build_features
from fleetpm.models import rul_tabular as T
from fleetpm.models.metrics import last_rows, summarize
from fleetpm.pipeline import feat_cols, load_prepared

TAG = "_gru"
LVL = {"REVIEW": -1, "LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def design(P, split, windows):
    F, Z = P["F" + split], P["Z" + split]
    parts = [F[feat_cols(F)].reset_index(drop=True)]
    for w in windows:
        S = build_features(Z, P["pre"].sensors, w, ["mean", "slope"])
        if not (np.array_equal(S["engine_id"], F["engine_id"]) and np.array_equal(S["cycle"], F["cycle"])):
            raise ValueError(f"short-window rows misaligned with {split} features")
        parts.append(S.drop(columns=["engine_id", "cycle"]).add_suffix(f"_w{w}"))
    return pd.concat(parts, axis=1)


def make_agent(tcfg, out_dir, fd):
    params = inspect.signature(TriageAgent.__init__).parameters
    return TriageAgent(tcfg, out_dir, **({"subset": fd} if "subset" in params else {}))


def run(fd, cfg):
    t, ev = cfg["triage"], cfg["eval"]
    P = load_prepared(cfg, fd)
    cap, F, Ft = P["cap"], P["Ftr"], P["Fte"]
    B = joblib.load(art(cfg, "models", fd, f"rul_bundle_v2{TAG}.joblib"))     # trusted: produced by 03b
    zp = art(cfg, "triage_inputs", f"{fd}.npz")
    if not zp.exists():
        raise FileNotFoundError(f"{zp.name} missing; run 09a_anomaly_inputs.py first (separate process by design)")
    Z = np.load(zp)
    score, pers = Z["score"], Z["pers"].astype(bool)
    cut = {"medium": float(Z["cut_medium"]), "high": float(Z["cut_high"]), "critical": float(Z["cut_critical"])}
    W = int(Z["window"])
    Xdf = design(P, "te", B["windows"])
    if list(Xdf.columns) != B["cols"]:
        raise ValueError(f"{fd}: feature order differs from bundle")
    X = Xdf.to_numpy(float)
    eng, cyc = Ft.engine_id.to_numpy(), Ft.cycle.to_numpy()
    if len(score) != len(eng):
        raise ValueError(f"{fd}: anomaly inputs have {len(score)} rows, test features {len(eng)}")
    rul_true, rul_cap = Ft["rul"].to_numpy(float), Ft["rul_cap"].to_numpy(float)
    model = B["models"]["xgb"]
    base = T.predict(model, X)

    # conservative offset and residual spread from OUT-OF-FOLD training predictions only
    oof = np.load(art(cfg, "models", fd, f"rul_boost_preds{TAG}.npz"))["oof_xgb"]
    y = F["rul_cap"].to_numpy(float)
    ok = (F.cycle.to_numpy() >= ev["min_cut_len"]) & np.isfinite(oof)
    bins = [30, 60]
    bi, err = np.digitize(oof[ok], bins), (oof - y)[ok]
    q = cfg["rul"]["cons_q"]
    deltas = np.array([max(0.0, float(np.quantile(err[bi == b], q))) if (bi == b).any() else 0.0 for b in range(3)])
    sigmas = np.array([float(err[bi == b].std()) if (bi == b).sum() > 1 else 0.0 for b in range(3)])
    cons = np.clip(base - deltas[np.digitize(base, bins)], 0, None)
    sig = sigmas[np.digitize(base, bins)]          # rul_std field = out-of-fold residual std in the prediction bin, not an ensemble spread

    tcfg = {"triage": {**t, "medium_hi": cut["medium"], "high_hi": cut["high"], "critical_hi": cut["critical"]}, "fleet": fd}
    out_dir = art(cfg, "work_orders_v2", fd, "x").parent
    agent = make_agent(tcfg, out_dir, fd)
    sev = [agent.classify(float(c), float(s), bool(p), bool(h))[0] for c, s, p, h in zip(cons, score, pers, cyc >= W)]
    si = np.array([LVL[s] for s in sev])

    # cycle-by-cycle replay: the agent is called only when the severity level rises for an engine; the agent deduplicates again itself
    booster, seen, new_orders = model.get_booster(), {}, []
    for i in range(len(eng)):
        lv, e = int(si[i]), int(eng[i])
        if lv < 1 or lv <= seen.get(e, 0):
            continue
        seen[e] = lv
        contrib = booster.predict(xgb.DMatrix(X[i:i + 1]), pred_contribs=True)[0][:-1]
        top = pd.Series(np.abs(contrib), index=B["cols"]).groupby(lambda c: c.split("__")[0]).sum() \
            .sort_values(ascending=False).index[:3].tolist()
        wo = agent.work_order(e, int(cyc[i]), float(base[i]), float(sig[i]), float(cons[i]), float(score[i]),
                              bool(pers[i]), top, B["version"])
        if wo is not None:
            new_orders.append(wo)
    rec = art(cfg, "work_orders_v2", f"{fd}_orders.json")
    if new_orders or not rec.exists():
        rec.write_text(json.dumps(new_orders, indent=2, default=str))
    else:
        print(f"{fd}: agent issued no new orders (already issued by an earlier run); keeping the previous record", flush=True)
    orders = json.loads(rec.read_text())

    last = last_rows(eng)
    truth = np.select([rul_true <= t["critical_rul"], rul_true <= t["high_rul"], rul_true <= t["medium_rul"]], [3, 2, 1], 0)
    sl = si[last]
    m = sl >= 0
    conf = np.zeros((4, 4), int)
    np.add.at(conf, (truth[last][m], sl[m]), 1)
    le30, gt60 = rul_true[last] <= 30, rul_true[last] > 60
    valid = cyc >= W
    healthy, near = (rul_cap >= cap) & valid, (rul_true <= 30) & valid
    d = pd.DataFrame({"e": eng, "r": rul_true, "s": si})
    first = d[d.s >= 2].groupby("e")["r"].first()
    mn = d.groupby("e")["r"].min()
    elig = mn[mn <= 30].index
    yc = np.minimum(rul_true[last], cap)

    def block(p):
        s = summarize(yc, p[last])
        s["late_rate"] = float((p[last] > yc).mean())
        return s

    pd.DataFrame({"engine_id": eng, "cycle": cyc, "rul_true": rul_true, "rul_base": base, "rul_cons": cons,
                  "anomaly_score": score, "persistent": pers, "severity": sev}
                 ).to_csv(art(cfg, "predictions", f"triage_v2_{fd}.csv"), index=False)
    res = {"n_engines": int(len(last)), "cons_deltas": deltas.tolist(), "rul_std_by_bin": sigmas.tolist(),
           "anomaly_cut": cut, "persist_k": int(Z["persist_k"]), "confusion": conf.tolist(),
           "n_review_last_cycle": int((sl < 0).sum()),
           "last_cycle": {"n_le30": int(le30.sum()), "high_plus_of_le30": int((sl[le30] >= 2).sum()),
                          "low_of_le30": int((sl[le30] == 0).sum()),
                          "n_gt60": int(gt60.sum()), "high_plus_of_gt60": int((sl[gt60] >= 2).sum())},
           "rows": {"n_healthy": int(healthy.sum()), "false_high_plus_rate": float((si[healthy] >= 2).mean()),
                    "n_near": int(near.sum()), "high_plus_rate_near_failure": float((si[near] >= 2).mean())},
           "lead": [float(v) for v in first.to_numpy()], "n_eligible": int(len(elig)),
           "n_eligible_never_high_plus": int((~elig.isin(first.index)).sum()),
           "n_first_alert_gt60": int((first > 60).sum()),
           "wo_total": len(orders), "wo_by_sev": {str(k): v for k, v in Counter(w.get("severity") for w in orders).items()},
           "rul_base": block(base), "rul_cons": block(cons)}
    lc, rw = res["last_cycle"], res["rows"]
    print(f"{fd}: engines={res['n_engines']} review@last={res['n_review_last_cycle']} | last-cycle RUL<=30: HIGH+ {lc['high_plus_of_le30']}/{lc['n_le30']}, "
          f"LOW {lc['low_of_le30']} | RUL>60 flagged HIGH+ {lc['high_plus_of_gt60']}/{lc['n_gt60']} | row false HIGH+ (healthy) "
          f"{rw['false_high_plus_rate']:.4f} | work orders {res['wo_total']} {res['wo_by_sev']}", flush=True)
    print(f"      RUL last cycle base: rmse {res['rul_base']['rmse']:.2f} nasa {res['rul_base']['nasa']:.0f} late {res['rul_base']['late_rate']:.2f} "
          f"| conservative: rmse {res['rul_cons']['rmse']:.2f} nasa {res['rul_cons']['nasa']:.0f} late {res['rul_cons']['late_rate']:.2f} "
          f"| deltas {np.round(deltas, 1).tolist()}", flush=True)
    return res


def main():
    cfg = load_config()
    out = {fd: run(fd, cfg) for fd in cfg["data"]["subsets"]}
    art(cfg, "triage_v2_summary.json").write_text(json.dumps(out, indent=2))
    print("saved triage_v2_summary.json, predictions/triage_v2_FD00X.csv, work_orders_v2/FD00X_orders.json")


if __name__ == "__main__":
    main()
