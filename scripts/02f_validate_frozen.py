import argparse, json, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from fleetpm.config import load_config, art
from fleetpm.pipeline import load_prepared, feat_cols
from fleetpm.models.anomaly import AEDetector, persistent

ap = argparse.ArgumentParser()
ap.add_argument("--fd", nargs="*"); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--split", default="0.5,0.2,0.3"); ap.add_argument("--boot", type=int, default=500)
args = ap.parse_args()
cfg = load_config(); a = cfg["anomaly"]; W = cfg["features"]["window"]; h0, h1 = a["healthy_cycle_range"]
qf = a["fpr_target"]; pr = a["positive_rul"]; sp = [float(x) for x in args.split.split(",")]
rng = np.random.default_rng(0)
TH = dict(fpr=0.02, det=0.90, lead=15, fpr_std=0.01)          # provisional targets
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)


def ci_ratio(en, num, den, B):
    """Engine-bootstrap CI of sum(num)/sum(den); resamples whole engines."""
    if B <= 0: return None
    _, inv = np.unique(en, return_inverse=True)
    n = np.bincount(inv, weights=num.astype(float)); d = np.bincount(inv, weights=den.astype(float))
    r = []
    for _ in range(B):
        i = rng.integers(0, len(n), len(n)); ds = d[i].sum()
        r.append(n[i].sum() / ds if ds > 0 else np.nan)
    return np.round(np.nanpercentile(r, [2.5, 97.5]), 4).tolist()


def evaluate(s, th, K, D, B):
    et, neg, pos, cap = D["et"], D["neg"], D["pos"], D["cap"]
    fl = (s > th) & D["valid"]; pk = persistent(fl, et, K)
    fa = pd.DataFrame({"e": et, "r": D["rt"], "p": pk}).query("p").groupby("e")["r"].first()
    first = fa.reindex(D["ue"]); el = D["el"]
    det = el & first.notna() & (first < cap)
    lead = first[det]
    cz = pd.Series(neg).groupby(et).sum().reindex(D["ue"]) >= K
    fa_eng = pd.Series(pk & neg).groupby(et).any().reindex(D["ue"])
    fp_e = pd.Series(fl & neg).groupby(et).sum().sort_values(ascending=False)
    rate_e = pd.Series(fl[neg]).groupby(et[neg]).mean()
    v = det[el].to_numpy().astype(float)
    dci = None if B <= 0 else np.round(np.percentile([rng.choice(v, len(v)).mean() for _ in range(B)], [2.5, 97.5]), 3).tolist()
    return {"fpr_row": float(fl[neg].mean()), "tpr_row": float(fl[pos].mean()),
            "fpr_persist": float(pk[neg].mean()), "tpr_persist": float(pk[pos].mean()),
            "eng_eligible": int(el.sum()), "eng_detected": int(det.sum()), "det_rate": float(det.mean() if False else det.sum() / max(el.sum(), 1)),
            "eng_missed": int((el & first.isna()).sum()), "eng_early_only": int((el & (first >= cap)).sum()),
            "lead_median": float(lead.median()) if len(lead) else np.nan,
            "lead_p10": float(lead.quantile(.1)) if len(lead) else np.nan,
            "eng_false_alarm_rate": float(fa_eng[cz].mean()) if cz.any() else np.nan,
            "unassessable": int((D["last_cycle"] < W + K - 1).sum()),
            "top3_fp_share": float(fp_e.head(3).sum() / max(fp_e.sum(), 1)),
            "engines_fp_gt10pct": int((rate_e > .10).sum()),
            "ci_fpr_row": ci_ratio(et, fl & neg, neg, B), "ci_fpr_persist": ci_ratio(et, pk & neg, neg, B),
            "ci_tpr_persist": ci_ratio(et, pk & pos, pos, B), "ci_det_rate": dci}


def regime_table(s, th, D):
    reg = D["reg"]
    if len(np.unique(reg)) < 2: return None
    fl = (s > th) & D["valid"]; rows = []
    for r in np.unique(reg):
        m = reg == r
        rows.append({"regime": int(r), "neg_rows": int((m & D["neg"]).sum()), "pos_rows": int((m & D["pos"]).sum()),
                     "fpr_row": round(float(fl[m & D["neg"]].mean()), 4) if (m & D["neg"]).any() else np.nan,
                     "tpr_row": round(float(fl[m & D["pos"]].mean()), 4) if (m & D["pos"]).any() else np.nan})
    return pd.DataFrame(rows)


out_dir = art(cfg, "validation", "x").parent
for fd in args.fd or cfg["data"]["subsets"]:
    P = load_prepared(cfg, fd); cap = P["cap"]; F, T = P["Ftr"], P["Fte"]; cols = feat_cols(F)
    X = F[cols].to_numpy(np.float32); eng = F.engine_id.to_numpy(); cyc = F.cycle.to_numpy()
    Xt = T[cols].to_numpy(np.float32); et = T.engine_id.to_numpy(); ct = T.cycle.to_numpy(); rt = T["rul"].to_numpy()
    valid = ct >= W; neg = (T["rul_cap"].to_numpy() >= cap) & valid; pos = (rt <= pr) & valid; sel = neg | pos
    ue = np.unique(et)
    D = dict(et=et, rt=rt, valid=valid, neg=neg, pos=pos, cap=cap, ue=ue, reg=P["Zte"]["regime_id"].to_numpy(),
             el=pd.Series(pos).groupby(et).any().reindex(ue), last_cycle=pd.Series(ct).groupby(et).max().reindex(ue).to_numpy())
    mid = valid & ~neg & ~pos
    print(f"\n================ {fd} ================")
    print(f"Unassessed middle band ({pr}<RUL<{cap}): {int(mid.sum())} rows in {len(np.unique(et[mid]))} engines "
          f"(not scored); negatives {int(neg.sum())} rows, positives {int(pos.sum())} rows; "
          f"eligible engines {int(D['el'].sum())}/{len(ue)}")
    recs = []
    hr = (cyc >= W) & (F["rul_cap"].to_numpy() >= cap); hw = (cyc >= h0) & (cyc <= h1)
    for j in range(args.seeds):
        seed = cfg["seed"] + j; B = args.boot if j == 0 else 0
        ids = np.unique(eng); np.random.default_rng(seed).shuffle(ids); n = len(ids)
        i1 = int(sp[0] * n); i2 = i1 + int(sp[1] * n)
        G = {"tr": ids[:i1], "es": ids[i1:i2], "cal": ids[i2:]}
        Htr = hw & np.isin(eng, G["tr"]); Hes = hw & np.isin(eng, G["es"])
        det = AEDetector(cfg, seed).fit(X[Htr], X[Hes])
        ifl = IsolationForest(n_estimators=200, random_state=seed).fit(np.c_[det.latent(X[Htr]), det.errors(X[Htr])])

        def score(A):
            e = det.errors(A); return {"HI": e, "IFl": -ifl.decision_function(np.c_[det.latent(A), e])}
        St = score(Xt)
        refs = {"CAL_only": hr & np.isin(eng, G["cal"]), "pooled": hr & np.isin(eng, np.r_[G["es"], G["cal"]])}
        for mode, rows in refs.items():
            Sr = score(X[rows])
            for dn in ("HI", "IFl"):
                th = float(np.quantile(Sr[dn], 1 - qf)); sat = bool((Sr[dn] >= th).mean() > 2 * qf)
                K = next((k for k in range(1, 9) if persistent(Sr[dn] > th, eng[rows], k).mean() <= qf / 2), 8)
                r = evaluate(St[dn], th, K, D, B)
                r.update(seed=seed, mode=mode, det=dn, k=K, thr=th, saturated=sat, n_thr_eng=len(np.unique(eng[rows])),
                         cal_fpr_row=float((Sr[dn] > th).mean()), auroc=float(roc_auc_score(pos[sel], St[dn][sel])))
                recs.append(r)
                if j == 0 and mode == "CAL_only":
                    rt_ = regime_table(St[dn], th, D)
                    if rt_ is not None: print(f"\n[{dn}] per-regime (seed {seed}, CAL_only threshold):\n{rt_.to_string(index=False)}")
                    if dn == "IFl":
                        g = []
                        for q in (0.98, 0.99, 0.995):
                            t2 = float(np.quantile(Sr[dn], q))
                            for k in (3, 4, 5):
                                e2 = evaluate(St[dn], t2, k, D, 0)
                                g.append({"q": q, "k": k, "fpr_persist": round(e2["fpr_persist"], 4), "tpr_persist": round(e2["tpr_persist"], 3),
                                          "det_rate": round(e2["det_rate"], 3), "lead_median": e2["lead_median"]})
                        print(f"\n[IFl] sensitivity (threshold quantile x k), seed {seed}:\n{pd.DataFrame(g).to_string(index=False)}")
    R = pd.DataFrame(recs); R.to_csv(out_dir / f"{fd}_validation_runs.csv", index=False)
    keys = ["cal_fpr_row", "fpr_row", "fpr_persist", "tpr_persist", "det_rate", "lead_median", "eng_false_alarm_rate",
            "unassessable", "top3_fp_share", "auroc", "k", "n_thr_eng"]
    agg = R.groupby(["mode", "det"])[keys].agg(["mean", "std"]).round(4)
    print(f"\n--- {fd}: mean/std over {args.seeds} engine-level splits (thresholds and k frozen per split from calibration engines) ---")
    print(agg.to_string())
    first = R[R.seed == R.seed.min()]
    print("\nSeed-0 engine-bootstrap 95% CIs (resampling whole engines):")
    print(first[["mode", "det", "ci_fpr_row", "ci_fpr_persist", "ci_tpr_persist", "ci_det_rate"]].to_string(index=False))
    print("\nSeed-0 engine outcomes:")
    print(first[["mode", "det", "eng_eligible", "eng_detected", "eng_missed", "eng_early_only", "engines_fp_gt10pct"]].to_string(index=False))
    print("\nProvisional pass check (CAL_only, mean over splits):")
    for dn in ("HI", "IFl"):
        s = R[(R["mode"] == "CAL_only") & (R.det == dn)]
        chk = {"persist FPR<=%.2f" % TH["fpr"]: s.fpr_persist.mean() <= TH["fpr"],
               "eligible det>=%.2f" % TH["det"]: s.det_rate.mean() >= TH["det"],
               "no saturation": not s.saturated.any(),
               "lead median>=%d" % TH["lead"]: s.lead_median.mean() >= TH["lead"],
               "FPR std<=%.2f" % TH["fpr_std"]: (s.fpr_persist.std() <= TH["fpr_std"]) if len(s) > 1 else None}
        print(f" {dn}: " + " | ".join(f"{k}: {'PASS' if v else ('n/a' if v is None else 'FAIL')}" for k, v in chk.items()))