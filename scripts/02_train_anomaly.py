import _bootstrap
import json, joblib, numpy as np, pandas as pd, mlflow
import re
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from fleetpm.config import load_config, art
from fleetpm.pipeline import load_prepared, feat_cols
from fleetpm.models.anomaly import AEDetector, persistent
from fleetpm.mlops import tracking as mt

cfg = load_config(); a = cfg["anomaly"]; seed = cfg["seed"]; mt.setup(cfg)
lo, hi = a["healthy_cycle_range"]
for fd in cfg["data"]["subsets"]:
    P = load_prepared(cfg, fd); F = P["Ftr"]; cols = feat_cols(F)
    X = F[cols].to_numpy(np.float32); eng = F.engine_id.to_numpy(); cyc = F.cycle.to_numpy()
    ids = np.unique(eng); np.random.default_rng(seed).shuffle(ids)
    n = len(ids); i1 = int(a["split"]["train"] * n); i2 = i1 + int(a["split"]["es"] * n)
    g = {"tr": ids[:i1], "es": ids[i1:i2], "cal": ids[i2:]}
    h = (cyc >= lo) & (cyc <= hi)
    M = {k: h & np.isin(eng, v) for k, v in g.items()}
    W = cfg["features"]["window"]
    healthy = (cyc >= W) & (F["rul_cap"].to_numpy() >= P["cap"])
    thr_rows = healthy & np.isin(eng, np.r_[g["es"], g["cal"]])   # pooled ES + CAL engines
    if not thr_rows.any():
        raise ValueError(f"{fd}: no healthy threshold rows")
    det = AEDetector(cfg, seed).fit(X[M["tr"]], X[M["es"]]).calibrate(X[thr_rows])
    T = P["Fte"]; Xt = T[cols].to_numpy(np.float32); et = T.engine_id.to_numpy()
    rt = T["rul"].to_numpy(); HI = det.index(Xt)
    ct = T["cycle"].to_numpy(); valid = ct >= cfg["features"]["window"]   # exclude feature warm-up rows
    neg = (T["rul_cap"].to_numpy() >= P["cap"]) & valid
    pos = (rt <= a["positive_rul"]) & valid; sel = neg | pos
    yb = pos[sel]
    auc = lambda s: float(roc_auc_score(yb, s[sel])) if yb.min() != yb.max() else float("nan")
    err_tr = det.errors(X[M["tr"]])
    if_e = IsolationForest(n_estimators=200, random_state=seed).fit(err_tr[:, None])
    if_l = IsolationForest(n_estimators=200, random_state=seed).fit(
        np.c_[det.latent(X[M["tr"]]), err_tr])
    s_e = -if_e.decision_function(det.errors(Xt)[:, None])
    s_l = -if_l.decision_function(np.c_[det.latent(Xt), det.errors(Xt)])
    pers = persistent((HI > 1.0) & valid, et, a["persist_k"])
    det_rul = pd.DataFrame({"e": et, "rul": rt, "p": pers}).query("p").groupby("e")["rul"].first()
    res = {"fd": fd, "tau": det.tau, "n_healthy_train_rows": int(M["tr"].sum()),
           "auroc_HI": auc(HI), "auroc_IF_error_1d": auc(s_e), "auroc_IF_latent_error": auc(s_l),
           "fpr_hi_gt1_neg": float((HI[neg] > 1).mean()) if neg.any() else None,
           "tpr_hi_gt1_pos": float((HI[pos] > 1).mean()) if pos.any() else None,
           "fpr_hi_gt1_neg_incl_warmup": float((HI[neg | ((T['rul_cap'].to_numpy() >= P['cap']) & ~valid)] > 1).mean()),
           "engines_detected": int(len(det_rul)), "engines_total": int(len(np.unique(et))),
           "median_true_rul_at_first_detection": float(det_rul.median()) if len(det_rul) else None,
           "detections_rul_ge_cap": int((det_rul >= P["cap"]).sum())}
    qf = a["fpr_target"]; K = a["persist_k"]; cap = P["cap"]; pr = a["positive_rul"]; W = cfg["features"]["window"]
    Xc = X[thr_rows]; ec = det.errors(Xc); eng_cal = eng[thr_rows]
    S = {"HI": HI, "IFe": s_e, "IFl": s_l}
    Sc = {"HI": det.index(Xc), "IFe": -if_e.decision_function(ec[:, None]),
          "IFl": -if_l.decision_function(np.c_[det.latent(Xc), ec])}
    neg_all = T["rul_cap"].to_numpy() >= cap                      # includes warm-up rows
    el = pd.Series(pos).groupby(et).any()                         # eligible: has a post-warm-up row with RUL<=pr
    tab = pd.DataFrame({"last_cycle": pd.Series(ct).groupby(et).max(),
                        "last_rul": pd.Series(rt).groupby(et).last(), "eligible": el})
    tab["assessable"] = tab.last_cycle >= W + K - 1               # can a persistent alert exist at all?
    res["eng_eligible"] = float(el.sum()); res["eng_unassessable"] = float((~tab.assessable).sum())
    es_rows = healthy & np.isin(eng, g["es"])
    Xe = X[es_rows]; ee = det.errors(Xe)
    Se = {"HI": det.index(Xe), "IFe": -if_e.decision_function(ee[:, None]),
          "IFl": -if_l.decision_function(np.c_[det.latent(Xe), ee])}
    res["n_cal_engines"] = float(len(g["cal"])); res["n_es_engines"] = float(len(g["es"]))
    res["n_thr_engines"] = float(len(np.unique(eng[thr_rows])))
    cap_ok = pd.Series(neg).groupby(et).sum() >= K       # engines with enough post-warm-up cap-zone rows
    dump = {}
    for k in S:
        s, sc = S[k], Sc[k]
        th = float(np.quantile(sc, 1 - qf))                       # own threshold from own calibration scores
        raw = s > th; f = raw & valid; pk = persistent(f, et, K)
        res[f"{k}_es_fpr_row"] = float((Se[k] > th).mean())            # second 20-engine sample, same threshold
        res[f"{k}_op_valid"] = float(not ((sc >= th).mean() > 2 * qf))  # 0 = threshold sits on a score plateau
        fa_eng = pd.Series(pk & neg).groupby(et).any()
        res[f"{k}_eng_fa_rate"] = float(fa_eng[cap_ok].mean())         # share of cap-zone engines with a persistent alert there
        fc = sc > th; pkc = persistent(fc, eng_cal, K)            # calibration rows are contiguous per engine
        fa = pd.DataFrame({"e": et, "rul": rt, "p": pk}).query("p").groupby("e")["rul"].first()
        tab[f"{k}_first_alert_rul"] = fa; fa = tab[f"{k}_first_alert_rul"]
        # mutually exclusive, by first persistent alert: none | early (RUL>=cap) | warn (pr<RUL<cap) | in_pos (RUL<=pr)
        tab[f"{k}_cat"] = np.select([fa.isna(), fa >= cap, fa > pr], ["none", "early_ge_cap", "warn_pr_cap"], "in_pos")
        for flag, nm in ((True, "E"), (False, "I")):
            for c in ("none", "early_ge_cap", "warn_pr_cap", "in_pos"):
                res[f"{k}_{nm}_{c}"] = float(((tab.eligible == flag) & (tab[f"{k}_cat"] == c)).sum())
        lead = fa[tab.eligible & (fa < cap)]
        res.update({f"{k}_thr": th,
                    f"{k}_cal_fpr_row": float(fc.mean()), f"{k}_cal_fpr_persist": float(pkc.mean()),
                    f"{k}_cal_frac_ge_thr": float((sc >= th).mean()),
                    f"{k}_saturated": float((sc >= th).mean() > 2 * qf),
                    f"{k}_fpr_row": float(f[neg].mean()), f"{k}_tpr_row": float(f[pos].mean()),
                    f"{k}_fpr_persist": float(pk[neg].mean()), f"{k}_tpr_persist": float(pk[pos].mean()),
                    f"{k}_fpr_row_incl_warmup": float(raw[neg_all].mean()),
                    f"{k}_fpr_row_warmup_only": float(raw[neg_all & ~valid].mean()),
                    f"{k}_lead_median": float(lead.median()) if len(lead) else float("nan"),
                    f"{k}_lead_p10": float(lead.quantile(0.1)) if len(lead) else float("nan")})
        dump.update({f"S_{k}": s, f"C_{k}": sc, f"th_{k}": th})
    np.savez_compressed(art(cfg, "models", fd, "anomaly_scores.npz"), et=et, rt=rt, ct=ct, valid=valid,
                        neg_all=neg_all, pos=pos, eng_cal=eng_cal, cyc_cal=cyc[thr_rows], **dump)
    tab.to_csv(art(cfg, "models", fd, "anomaly_engine_table.csv"))
        # ---- deployable IFl bundle: threshold, cutoffs, persist_k from ES+CAL engines only ----
    pool = np.isin(eng, np.r_[g["es"], g["cal"]]) & (cyc >= W)
    th_l = float(dump["th_IFl"]); Xp = X[pool]
    s_pool = -if_l.decision_function(np.c_[det.latent(Xp), det.errors(Xp)])
    r_pool = F["rul"].to_numpy()[pool]
    hi_c = max(float(np.median(s_pool[r_pool <= 60])), th_l)
    cr_c = max(float(np.median(s_pool[r_pool <= 30])), hi_c)
    healthy_p = F["rul_cap"].to_numpy()[pool] >= P["cap"]
    fpk = {kk: float(persistent(s_pool > th_l, eng[pool], kk)[healthy_p].mean()) for kk in range(1, 9)}
    k_sel = K
    res.update({"IFl_cut_medium": th_l, "IFl_cut_high": hi_c, "IFl_cut_critical": cr_c,
                "IFl_k_sel": float(k_sel), **{f"IFl_pool_persist_fpr_k{kk}": v for kk, v in fpk.items()}})
    joblib.dump({"det": det, "if_l": if_l, "thr": th_l, "persist_k": k_sel, "window": W,
                 "cut": {"medium": th_l, "high": hi_c, "critical": cr_c},
                 "version": f"{fd}-IFl-pooled"},
                art(cfg, "models", fd, "anomaly_ifl.joblib"))
    print(json.dumps(res, indent=1))
    joblib.dump(det, art(cfg, "models", fd, "anomaly.joblib"))
    art(cfg, "models", fd, "anomaly_metrics.json").write_text(json.dumps(res, indent=1))
    with mlflow.start_run(run_name=f"{fd}_anomaly_ae"):
        mlflow.log_params({"fd": fd, **{k: v for k, v in a.items() if not isinstance(v, (dict, list))}})
        mlflow.log_metrics({re.sub(r"[^A-Za-z0-9_.\-/ :]", "_", k): v for k, v in res.items()
                            if isinstance(v, float) and v == v})