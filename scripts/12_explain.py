import _bootstrap
import argparse
import copy
import time
import numpy as np
import xgboost as xgb
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _ext_common as C
from fleetpm.config import load_config, art
from fleetpm.pipeline import load_prepared, feat_cols
from fleetpm.features.rolling import build_features
from fleetpm.models import rul_tabular as T
from fleetpm.models.metrics import cut_eval

ap = argparse.ArgumentParser()
ap.add_argument("--fd", nargs="*")
ap.add_argument("--draws", type=int, default=3)
ap.add_argument("--ks", default="1,2,3,5")
ap.add_argument("--trees", type=int, default=300, help="XGB trees for faithfulness retraining (lr rescaled so trees*lr is constant)")
ap.add_argument("--skip-faith", action="store_true")
ap.add_argument("--check", action="store_true", help="fit fold 1 at deployed settings (500 trees), compare with saved OOF RMSE, exit")
ap.add_argument("--skip-cluster", action="store_true")
ap.add_argument("--boot", type=int, default=100)
args = ap.parse_args()
cfg = load_config()
seed, ev = cfg["seed"], cfg["eval"]
KS = [int(x) for x in args.ks.split(",")]
NAMES = ["T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB",
         "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"]
SMAP = {f"sensor_{i}": n for i, n in enumerate(NAMES, 1)}  # ordering matches Table 2 of the Damage Propagation paper


def design(P):
    F = P["Ftr"]
    cols = feat_cols(F)
    S = build_features(P["Ztr"], P["pre"].sensors, 10, ["mean", "slope"])
    if not ((S.engine_id.to_numpy() == F.engine_id.to_numpy()).all() and (S.cycle.to_numpy() == F.cycle.to_numpy()).all()):
        raise ValueError("Ztr and Ftr rows not aligned")
    S = S.drop(columns=["engine_id", "cycle"])
    S.columns = [c.replace("__mean", "__w10mean").replace("__slope", "__w10slope") for c in S.columns]
    X = np.hstack([F[cols].to_numpy(float), S.to_numpy(float)])
    return X, cols + list(S.columns), F.rul_cap.to_numpy(float), F.engine_id.to_numpy()


def tci(d):
    d = np.asarray(d, float)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    return float(m), float(m - h), float(m + h)


def faith(fd):
    P = load_prepared(cfg, fd)
    X, names, y, eng = design(P)
    so = np.array([n.split("__")[0] for n in names])
    sensors = sorted(set(so))
    r = copy.deepcopy(cfg["rul"])
    o = cfg["rul"]["xgb"]
    r["xgb"]["n_estimators"] = args.trees
    r["xgb"]["learning_rate"] = o["learning_rate"] * o["n_estimators"] / args.trees
    if args.check:
        r["xgb"]["n_estimators"], r["xgb"]["learning_rate"] = o["n_estimators"], o["learning_rate"]
    folds = list(GroupKFold(n_splits=ev["cv_folds"]).split(X, y, eng))
    print(f"[{fd}] {len(sensors)} sensors, {X.shape[1]} features, trees {args.trees}, lr {r['xgb']['learning_rate']:.3f}", flush=True)
    rng = np.random.default_rng(seed)

    def fit_eval(tr, va, keep):
        m = T.fit(T.make("xgb", r, seed), "xgb", X[tr][:, keep], y[tr], T.weights(y[tr], r))
        p = T.predict(m, X[va][:, keep])
        return m, cut_eval(y[va], p, eng[va], ev["min_cut_len"], ev["n_cut_reps"], seed)["rmse"]

    R = {k: dict(rem_top=[], rem_rand=[], keep_top=[], keep_rand=[]) for k in KS}
    full, imp_all, ref_rm = [], [], []
    zref = np.clip(np.load(art(cfg, "models", fd, "rul_boost_preds_gru.npz"))["oof_xgb"], 0, None)
    for f, (tr, va) in enumerate(folds, 1):
        t0 = time.time()
        allk = np.ones(X.shape[1], bool)
        m0, r0 = fit_eval(tr, va, allk)
        full.append(r0)
        ref_rm.append(cut_eval(y[va], zref[va], eng[va], ev["min_cut_len"], ev["n_cut_reps"], seed)["rmse"])
        if args.check:
            print(f"[{fd}] check fold 1: refit RMSE {r0:.4f} vs saved deployed OOF RMSE {ref_rm[-1]:.4f} (trees {r['xgb']['n_estimators']})", flush=True)
            raise SystemExit(0)
        sub = rng.choice(tr, min(4000, len(tr)), replace=False)
        con = m0.get_booster().predict(xgb.DMatrix(X[sub]), pred_contribs=True)[:, :-1]
        imp = np.array([np.abs(con[:, so == s]).sum(1).mean() for s in sensors])
        imp_all.append(imp)
        rank = [sensors[i] for i in np.argsort(-imp)]
        for k in KS:
            top = set(rank[:k])
            R[k]["rem_top"].append(fit_eval(tr, va, ~np.isin(so, list(top)))[1])
            R[k]["keep_top"].append(fit_eval(tr, va, np.isin(so, list(top)))[1])
            rr, kr = [], []
            for _ in range(args.draws):
                rs = list(rng.choice(sensors, k, replace=False))
                rr.append(fit_eval(tr, va, ~np.isin(so, rs))[1])
                kr.append(fit_eval(tr, va, np.isin(so, rs))[1])
            R[k]["rem_rand"].append(float(np.mean(rr)))
            R[k]["keep_rand"].append(float(np.mean(kr)))
        print(f"[{fd}] fold {f}/{len(folds)} full RMSE {r0:.2f} top3 {rank[:3]} deployed-OOF ref {ref_rm[-1]:.2f} ({time.time() - t0:.0f}s)", flush=True)
    out = dict(full_rmse_folds=full, full_rmse_mean=float(np.mean(full)), mean_abs_shap_by_sensor=dict(zip(sensors, np.mean(imp_all, 0))), k={})
    print(f" full-model grouped-CV RMSE {np.mean(full):.2f}")
    print(" k | remove-top minus remove-random RMSE (faithful if >0) | keep-top minus keep-random (faithful if <0)   [mean, 95% t-CI over folds]")
    for k in KS:
        dr = tci(np.array(R[k]["rem_top"]) - np.array(R[k]["rem_rand"]))
        dk = tci(np.array(R[k]["keep_top"]) - np.array(R[k]["keep_rand"]))
        out["k"][str(k)] = dict(raw=R[k], remove_diff=dr, keep_diff=dk)
        print(f" {k} | {dr[0]:+.3f} [{dr[1]:+.3f}, {dr[2]:+.3f}] | {dk[0]:+.3f} [{dk[1]:+.3f}, {dk[2]:+.3f}]")
    return out


def cluster(fd):
    P = load_prepared(cfg, fd)
    Z, sens = P["Ztr"], P["pre"].sensors
    rows = []
    for _, g in Z.groupby("engine_id", sort=True):
        a = g[sens].to_numpy(float)[-50:]
        x = np.arange(len(a)) - (len(a) - 1) / 2
        rows.append((x @ (a - a.mean(0))) / (x @ x))
    S = np.array(rows)
    Xs = StandardScaler().fit_transform(S)
    rng = np.random.default_rng(seed)
    out = dict(n_engines=len(Xs), silhouette={})
    labs = {}
    for k in (2, 3, 4):
        km = KMeans(k, n_init=10, random_state=seed).fit(Xs)
        out["silhouette"][str(k)] = float(silhouette_score(Xs, km.labels_))
        labs[k] = km
    ref = labs[2].labels_
    ari = []
    for _ in range(args.boot):
        i = rng.integers(0, len(Xs), len(Xs))
        ari.append(adjusted_rand_score(ref, KMeans(2, n_init=5, random_state=seed).fit(Xs[i]).predict(Xs)))
    null = []
    cov = np.cov(Xs, rowvar=False)
    for _ in range(30):
        Xn = rng.multivariate_normal(np.zeros(Xs.shape[1]), cov, size=len(Xs))
        null.append(silhouette_score(Xn, KMeans(2, n_init=5, random_state=seed).fit(Xn).labels_))
    d = labs[2].cluster_centers_[0] - labs[2].cluster_centers_[1]
    top = np.argsort(-np.abs(d))[:5]
    life = Z.groupby("engine_id", sort=True).cycle.max().to_numpy(float)
    out.update(life_spearman_k2=float(stats.spearmanr(ref, life)[0]), ari_k2_mean=float(np.mean(ari)), ari_k2_ci=C.pct(ari), cluster_sizes_k2=np.bincount(ref).tolist(),
               null_sil_k2_mean=float(np.mean(null)), null_sil_k2_p95=float(np.percentile(null, 95)),
               top_discriminating=[dict(sensor=sens[i], symbol=SMAP.get(sens[i]), centroid_diff=float(d[i])) for i in top])
    print(f" n={len(Xs)} silhouette {out['silhouette']} ARI(k=2) {out['ari_k2_mean']:.3f} {np.round(out['ari_k2_ci'], 3).tolist()} "
          f"sizes {out['cluster_sizes_k2']} | Gaussian-null silhouette mean {out['null_sil_k2_mean']:.3f} p95 {out['null_sil_k2_p95']:.3f} "
          f"spearman(cluster,life) {out['life_spearman_k2']:+.2f}")
    print("  top discriminating sensors (HYPOTHESIS only):", [(t["sensor"], t["symbol"]) for t in out["top_discriminating"]])
    return out, PCA(2, random_state=seed).fit_transform(Xs), ref


fds = args.fd or cfg["data"]["subsets"]
path = art(cfg, "ext_explain.json")
scat = {}
if not args.skip_cluster:
    print("NOTE: readme.txt lists two fault modes for FD003/FD004 (HPC + Fan); the paper text (Sec. V.C) states only HPC degradation was included in the challenge data. Unresolved here.")
for fd in fds:
    rec = {}
    if not args.skip_faith:
        print(f"\n===== {fd}: faithfulness =====", flush=True)
        rec["faithfulness"] = faith(fd)
    if not args.skip_cluster:
        print(f"\n===== {fd}: fault-mode clustering (unsupervised, no labels) =====", flush=True)
        rec["clustering"], xy, lab = cluster(fd)
        scat[fd] = (xy, lab)
    C.save_merge(path, fd, rec)
    lines = []
    if "faithfulness" in rec:
        lines.append("faithfulness remove-diff by k: " + str({k: round(v["remove_diff"][0], 3) for k, v in rec["faithfulness"]["k"].items()}))
    if "clustering" in rec:
        c = rec["clustering"]
        lines.append(f"cluster k=2 silhouette {c['silhouette']['2']:.3f}, ARI {c['ari_k2_mean']:.3f}, null p95 {c['null_sil_k2_p95']:.3f}; mapping to fan/HPC is HYPOTHESIS")
    C.log(f"Task3 explain {fd}", lines)
if scat:
    fig, axs = plt.subplots(1, len(scat), figsize=(4 * len(scat), 3.6), squeeze=False)
    for ax, (fd, (xy, lab)) in zip(axs[0], scat.items()):
        ax.scatter(xy[:, 0], xy[:, 1], c=lab, s=10, cmap="coolwarm")
        ax.set_title(f"{fd} degradation signatures (PCA, k=2 labels)", fontsize=8)
    fig.tight_layout()
    fig.savefig(art(cfg, "figures", "ext_clusters.png"), dpi=150)
print("\nSaved artifacts/ext_explain.json")
