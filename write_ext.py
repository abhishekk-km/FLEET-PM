import py_compile, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TS = time.strftime("%Y%m%d_%H%M%S")
FILES = {}

FILES["scripts/_ext_common.py"] = r'''import _bootstrap
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
    d[key] = obj
    path.write_text(json.dumps(d, indent=1, default=_jd))


def log(title, lines):
    p = ROOT / "docs" / "SYSTEM_LOG.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with p.open("a") as f:
        f.write(f"\n## {title} ({stamp})\nAuto-appended from script output. Audit pending until reviewed.\n")
        for ln in lines:
            f.write(f"- {ln}\n")
'''

FILES["scripts/10_conformal.py"] = r'''import _bootstrap
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _ext_common as C
from fleetpm.config import load_config, art
from fleetpm.models.metrics import summarize, cut_rows

ap = argparse.ArgumentParser()
ap.add_argument("--fd", nargs="*")
ap.add_argument("--nested", type=int, default=100)
ap.add_argument("--boot", type=int, default=1000)
args = ap.parse_args()
cfg = load_config()
seed = cfg["seed"]
ev = cfg["eval"]
cq = cfg["rul"].get("cons_q", 0.8)
ALPHAS = (0.1, 0.2)
VARS = ("all_rows", "one_cut_per_engine")
CAL_REPS, EV_REPS = 20, 10
OLD_SPREAD = {"FD002": 0.44, "FD003": 0.58, "FD004": 0.50}  # from handoff, old pipeline, NOT recomputed here


def cal_data(D, ids, variant, rng):
    idx = np.where(np.isin(D["eng"], ids))[0]
    if variant == "all_rows":
        idx = idx[D["cyc"][idx] >= C.MIN_CYCLE]
    else:
        sub = D["eng"][idx]
        idx = np.concatenate([idx[cut_rows(sub, C.MIN_CYCLE, rng)] for _ in range(CAL_REPS)])
    return D["oof"][idx], D["oof"][idx] - D["y"][idx]


def cover(pred, y, qs):
    c = C.apply_q(pred, qs) <= y
    band = np.digitize(pred, C.BANDS)
    return float(c.mean()), [float(c[band == b].mean()) if (band == b).any() else np.nan for b in range(3)]


def eval_cuts(D, ids, qs, rng):
    idx = np.where(np.isin(D["eng"], ids))[0]
    sub = D["eng"][idx]
    r = np.concatenate([idx[cut_rows(sub, C.MIN_CYCLE, rng)] for _ in range(EV_REPS)])
    return cover(D["oof"][r], D["y"][r], qs)


def eval_rows(D, ids, qs):
    idx = np.where(np.isin(D["eng"], ids) & (D["cyc"] >= C.MIN_CYCLE))[0]
    return cover(D["oof"][idx], D["y"][idx], qs)


def nested(D, R, rng):
    ue = np.unique(D["eng"])
    acc = {(v, a): dict(cm=[], bb=[], rm=[], dev=[]) for v in VARS for a in ALPHAS}
    for _ in range(R):
        p = rng.permutation(ue)
        h = len(p) // 2
        cal, held = p[:h], p[h:]
        for v in VARS:
            pr, s = cal_data(D, cal, v, rng)
            for a in ALPHAS:
                qs = C.band_q(pr, s, C.conf_qfun(a))
                cm, bb = eval_cuts(D, held, qs, rng)
                rm, _ = eval_rows(D, held, qs)
                A = acc[(v, a)]
                A["cm"].append(cm); A["bb"].append(bb); A["rm"].append(rm)
                A["dev"].append(np.nanmean(np.abs(np.array(bb) - (1 - a))))
    out = {}
    for (v, a), A in acc.items():
        out[f"{v}|{a}"] = dict(cut_cov_mean=float(np.mean(A["cm"])), cut_cov_sd=float(np.std(A["cm"])),
                               cut_band_cov=np.nanmean(np.array(A["bb"]), 0).tolist(),
                               allrows_cov_mean=float(np.mean(A["rm"])),
                               marg_dev=float(abs(np.mean(A["cm"]) - (1 - a))),
                               band_dev=float(np.mean(A["dev"])))
    score = {v: float(np.mean([out[f"{v}|{a}"]["marg_dev"] + out[f"{v}|{a}"]["band_dev"] for a in ALPHAS])) for v in VARS}
    return out, score


def boot_cov(c, band, idx):
    cb = c[idx].astype(float)
    out = {"marg": C.pct(cb.mean(1))}
    for b in range(3):
        mb = band[idx] == b
        cnt = mb.sum(1)
        v = np.where(cnt > 0, (cb * mb).sum(1) / np.maximum(cnt, 1), np.nan)
        out[f"b{b}"] = C.pct(v)
    return out


def test_eval(D, methods, B, rng):
    n = len(D["true_cap"])
    idx = rng.integers(0, n, (B, n))
    pred = D["pred_t"]
    band = np.digitize(pred, C.BANDS)
    res = {}
    for name, L in methods.items():
        c = L <= D["true_cap"]
        slack = pred - L
        sc, sr = summarize(D["true_cap"], L), summarize(D["true_raw"], L)
        res[name] = dict(n=n, cov_cap=float(c.mean()), cov_raw=float((L <= D["true_raw"]).mean()),
                         band_n=[int((band == b).sum()) for b in range(3)],
                         band_cov=[float(c[band == b].mean()) if (band == b).any() else None for b in range(3)],
                         ci=boot_cov(c, band, idx), slack_mean=float(slack.mean()),
                         slack_ci=C.pct(slack[idx].mean(1)), late_rate=float((L > D["true_cap"]).mean()),
                         rmse_cap=sc["rmse"], nasa_cap=sc["nasa"], rmse_raw=sr["rmse"], nasa_raw=sr["nasa"])
    return res


allres = {}
for fd in args.fd or cfg["data"]["subsets"]:
    rng = np.random.default_rng(seed)
    D = C.load_oof(cfg, fd)
    print(f"\n===== {fd}: engines train {len(np.unique(D['eng']))}, test {len(D['true_cap'])}, cap {D['cap']} =====", flush=True)
    nres, score = nested(D, args.nested, rng)
    for k, v in nres.items():
        print(f" nested {k:<24} cut_cov {v['cut_cov_mean']:.3f}+-{v['cut_cov_sd']:.3f} allrows {v['allrows_cov_mean']:.3f} "
              f"band_cov {np.round(v['cut_band_cov'], 3).tolist()} band_dev {v['band_dev']:.3f}")
    best = min(score, key=score.get)
    print(f" variant scores (lower better): {score} -> frozen variant: {best}", flush=True)
    pr, s = cal_data(D, np.unique(D["eng"]), best, rng)
    qs = {a: C.band_q(pr, s, C.conf_qfun(a)) for a in ALPHAS}
    m19 = D["cyc"] >= ev["min_cut_len"]
    q_off = C.band_q(D["oof"][m19], (D["oof"] - D["y"])[m19], C.quant_qfun(cq))
    p = D["pred_t"]
    methods = {"raw": p, f"offset_q{cq}": C.apply_q(p, q_off),
               "conformal_a0.2": C.apply_q(p, qs[0.2]), "conformal_a0.1": C.apply_q(p, qs[0.1])}
    tres = test_eval(D, methods, args.boot, rng)
    print(f" q bands (<30,30-60,>=60): offset {np.round(q_off, 2).tolist()} conf0.2 {np.round(qs[0.2], 2).tolist()} conf0.1 {np.round(qs[0.1], 2).tolist()}")
    print(" TEST (official, evaluated once; capped labels unless stated)")
    for name, r in tres.items():
        bc = [None if x is None else round(x, 3) for x in r["band_cov"]]
        print(f"  {name:<16} cov {r['cov_cap']:.3f} CI{np.round(r['ci']['marg'], 3).tolist()} cov_raw {r['cov_raw']:.3f} "
              f"band_cov {bc} n {r['band_n']} slack {r['slack_mean']:.2f} CI{np.round(r['slack_ci'], 2).tolist()} "
              f"rmse {r['rmse_cap']:.2f} nasa {r['nasa_cap']:.0f} | raw rmse {r['rmse_raw']:.2f} nasa {r['nasa_raw']:.0f}")
    print(f"  old bootstrap-ensemble 95% coverage (handoff, not recomputed, different quantity): {OLD_SPREAD.get(fd)}")
    np.savez_compressed(art(cfg, f"ext_conformal_bounds_{fd}.npz"), pred=p, L10=methods["conformal_a0.1"],
                        L20=methods["conformal_a0.2"], Loff=methods[f"offset_q{cq}"], true_cap=D["true_cap"], true_raw=D["true_raw"])
    rec = dict(nested=nres, nested_score=score, variant=best, q_conf={str(a): v for a, v in qs.items()}, q_offset=q_off,
               test=tres, old_ensemble_cov95=OLD_SPREAD.get(fd), nested_repeats=args.nested, boot=args.boot)
    C.save_merge(art(cfg, "ext_conformal.json"), fd, rec)
    allres[fd] = (tres, list(methods))
    C.log(f"Task1 conformal {fd}", [f"variant {best}; test cov a0.2 {tres['conformal_a0.2']['cov_cap']:.3f}, a0.1 {tres['conformal_a0.1']['cov_cap']:.3f}, "
          f"offset {tres[f'offset_q{cq}']['cov_cap']:.3f}, raw {tres['raw']['cov_cap']:.3f}; empirical coverage only, no guarantee claimed"])

fig, axs = plt.subplots(1, len(allres), figsize=(4 * len(allres), 3.6), squeeze=False)
for ax, (fd, (tres, names)) in zip(axs[0], allres.items()):
    cov = [tres[m]["cov_cap"] for m in names]
    err = [[tres[m]["cov_cap"] - tres[m]["ci"]["marg"][0] for m in names], [tres[m]["ci"]["marg"][1] - tres[m]["cov_cap"] for m in names]]
    ax.bar(range(len(names)), cov, yerr=np.clip(err, 0, None), capsize=3, color="#4472c4")
    ax.axhline(0.8, ls="--", c="gray"); ax.axhline(0.9, ls=":", c="gray")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_title(f"{fd} test coverage (capped), 95% engine CI", fontsize=8)
fig.tight_layout()
fig.savefig(art(cfg, "figures", "ext_conformal.png"), dpi=150)
print("\nSaved artifacts/ext_conformal.json, figures/ext_conformal.png, ext_conformal_bounds_FD*.npz")
'''

FILES["scripts/11_cost_policy.py"] = r'''import _bootstrap
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _ext_common as C
from fleetpm.config import load_config, art

ap = argparse.ArgumentParser()
ap.add_argument("--fd", nargs="*")
ap.add_argument("--boot", type=int, default=300)
args = ap.parse_args()
cfg = load_config()
seed = cfg["seed"]
cq = cfg["rul"].get("cons_q", 0.8)
R_SEL = np.array([1, 3, 10, 30, 100, 300, 1000], float)
R_FINE = np.logspace(0, 3, 200)
KF = 10
THR = np.arange(0, 126, 1.0)
TRI = ("rul_raw", "rul_offset", "rul_conformal10")
ASSUME = "Replay assumptions: immediate action, simulated, failure at final training cycle, no anomaly signal (OOF anomaly scores unavailable)"


def crossfit(D, rng):
    eng, cyc, oof, y = D["eng"], D["cyc"], D["oof"], D["y"]
    ue = np.unique(eng)
    fold_e = rng.permutation(len(ue)) % KF
    rowfold = fold_e[np.searchsorted(ue, eng)]
    s = oof - y
    elig = cyc >= C.MIN_CYCLE
    Lo, Lc = np.zeros_like(oof), np.zeros_like(oof)
    for k in range(KF):
        cal, te = elig & (rowfold != k), rowfold == k
        qo = C.band_q(oof[cal], s[cal], C.quant_qfun(cq))
        qc = C.band_q(oof[cal], s[cal], C.conf_qfun(0.1))
        Lo[te] = C.apply_q(oof[te], qo)
        Lc[te] = C.apply_q(oof[te], qc)
    return {"rul_raw": oof, "rul_offset": Lo, "rul_conformal10": Lc}


def policy_arrays(D, V):
    eng, cyc = D["eng"], D["cyc"]
    ids, start, cnt = np.unique(eng, return_index=True, return_counts=True)
    if not ((cyc[start] == 1).all() and (cyc[start + cnt - 1] == cnt).all()):
        raise ValueError("training trajectories are not contiguous from cycle 1")
    n = len(ids)
    NS = np.arange(30, int(cnt.max()) + 1, 5)
    fam = {"run_to_failure": (np.array([0.0]), np.ones((1, n)), np.zeros((1, n)))}
    fixF, fixW = np.ones((len(NS), n)), np.zeros((len(NS), n))
    Fa = {k: np.ones((len(THR), n)) for k in V}
    Wa = {k: np.zeros((len(THR), n)) for k in V}
    for j in range(n):
        a, Tl = start[j], cnt[j]
        f = NS < Tl
        fixF[:, j] = ~f
        fixW[:, j] = np.where(f, Tl - NS, 0)
        for k, v in V.items():
            seg = v[a + 29:a + Tl - 1]
            if len(seg) == 0:
                continue
            m = np.minimum.accumulate(seg)
            pos = np.searchsorted(-m, -THR, side="left")
            fired = pos < len(m)
            Fa[k][:, j] = ~fired
            Wa[k][:, j] = np.where(fired, Tl - 30 - pos, 0)
    fam["fixed_interval"] = (NS.astype(float), fixF, fixW)
    for k in V:
        fam[k] = (THR, Fa[k], Wa[k])
    return fam, n


def best_curve(Fa, Wa, r, idx=None):
    if idx is not None:
        Fa, Wa = Fa[:, idx], Wa[:, idx]
    cost = r[:, None] * Fa.mean(1)[None, :] + Wa.mean(1)[None, :]
    return cost.min(1), cost.argmin(1)


def cv_curve(Fa, Wa, r, folds):
    tot = np.zeros(len(r))
    for k in np.unique(folds):
        te = folds == k
        cost = r[:, None] * Fa[:, ~te].mean(1)[None, :] + Wa[:, ~te].mean(1)[None, :]
        p = cost.argmin(1)
        tot += r * Fa[p][:, te].sum(1) + Wa[p][:, te].sum(1)
    return tot / Fa.shape[1]


def first_below(a, b, r):
    m = a < b
    return float(r[np.argmax(m)]) if m.any() else None


def breakevens(fam, r, idx=None):
    cur = {k: best_curve(v[1], v[2], r, idx)[0] for k, v in fam.items()}
    out = {}
    for k in TRI:
        out[k] = dict(vs_rtf=first_below(cur[k], cur["run_to_failure"], r), vs_fixed=first_below(cur[k], cur["fixed_interval"], r))
    return out


allres = {}
for fd in args.fd or cfg["data"]["subsets"]:
    rng = np.random.default_rng(seed)
    D = C.load_oof(cfg, fd)
    V = crossfit(D, rng)
    fam, n = policy_arrays(D, V)
    print(f"\n===== {fd}: {n} training engines replayed =====", flush=True)
    folds5 = rng.permutation(n) % 5
    rec = dict(n_engines=n, assumptions=ASSUME, curves={}, cv_curves={}, at_r={})
    for k, (par, Fa, Wa) in fam.items():
        c, _ = best_curve(Fa, Wa, R_FINE)
        rec["curves"][k] = c
        rec["cv_curves"][k] = cv_curve(Fa, Wa, R_FINE, folds5)
        c2, pi = best_curve(Fa, Wa, R_SEL)
        rec["at_r"][k] = [dict(r=float(R_SEL[i]), param=float(par[pi[i]]), cost=float(c2[i]),
                               failures_avoided=float(1 - Fa[pi[i]].mean()), mean_wasted_all=float(Wa[pi[i]].mean()),
                               mean_wasted_when_maintained=float(Wa[pi[i]][Fa[pi[i]] == 0].mean()) if (Fa[pi[i]] == 0).any() else None)
                          for i in range(len(R_SEL))]
    be = breakevens(fam, R_FINE)
    bs = {k: dict(vs_rtf=[], vs_fixed=[]) for k in TRI}
    for _ in range(args.boot):
        b = breakevens(fam, R_FINE, rng.integers(0, n, n))
        for k in TRI:
            for w in ("vs_rtf", "vs_fixed"):
                bs[k][w].append(b[k][w])
    rec["break_even"] = {}
    for k in TRI:
        rec["break_even"][k] = {}
        for w in ("vs_rtf", "vs_fixed"):
            v = [x for x in bs[k][w] if x is not None]
            rec["break_even"][k][w] = dict(point=be[k][w], ci=C.pct(v) if v else None, boot_never=int(len(bs[k][w]) - len(v)))
    cvbe = {k: dict(vs_rtf=first_below(rec["cv_curves"][k], rec["cv_curves"]["run_to_failure"], R_FINE),
                    vs_fixed=first_below(rec["cv_curves"][k], rec["cv_curves"]["fixed_interval"], R_FINE)) for k in TRI}
    rec["break_even_cv"] = cvbe
    print(" best policy per r (in-sample param; cost in C_waste units per engine):")
    for i, r in enumerate(R_SEL):
        row = {k: round(rec["at_r"][k][i]["cost"], 1) for k in fam}
        best = min(row, key=row.get)
        print(f"  r={r:<6.0f} {row} -> {best}")
    for k in TRI:
        a10 = [x for x in rec["at_r"][k] if x["r"] == 100][0]
        print(f"  {k}: r=100 param {a10['param']:.0f} failures_avoided {a10['failures_avoided']:.3f} mean_wasted {a10['mean_wasted_all']:.1f} | "
              f"break-even vs RTF {rec['break_even'][k]['vs_rtf']} vs fixed {rec['break_even'][k]['vs_fixed']} (cv: {cvbe[k]})")
    C.save_merge(art(cfg, "ext_cost_policy.json"), fd, rec)
    allres[fd] = rec
    C.log(f"Task2 cost policy {fd}", [f"break-even r vs run-to-failure: " + ", ".join(f"{k} {rec['break_even'][k]['vs_rtf']['point']}" for k in TRI),
                                      "replay on training trajectories with cross-fitted OOF bounds; simulated; no invented costs"])

fig, axs = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
for ax, (fd, rec) in zip(axs.ravel(), allres.items()):
    for k, c in rec["curves"].items():
        ax.loglog(R_FINE, c, label=k)
    ax.loglog(R_FINE, rec["cv_curves"]["rul_conformal10"], "--", c="k", lw=1, label="conformal10 (CV param)")
    ax.set_title(fd, fontsize=9); ax.set_xlabel("r = C_fail / C_waste"); ax.set_ylabel("mean cost per engine (C_waste units)")
    ax.legend(fontsize=6)
fig.suptitle(ASSUME, fontsize=8)
fig.tight_layout()
fig.savefig(art(cfg, "figures", "ext_cost_policy.png"), dpi=150)
print("\nSaved artifacts/ext_cost_policy.json, figures/ext_cost_policy.png")
'''

FILES["scripts/12_explain.py"] = r'''import _bootstrap
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
    folds = list(GroupKFold(n_splits=ev["cv_folds"]).split(X, y, eng))
    print(f"[{fd}] {len(sensors)} sensors, {X.shape[1]} features, trees {args.trees}, lr {r['xgb']['learning_rate']:.3f}", flush=True)
    rng = np.random.default_rng(seed)

    def fit_eval(tr, va, keep):
        m = T.fit(T.make("xgb", r, seed), "xgb", X[tr][:, keep], y[tr], T.weights(y[tr], r))
        p = T.predict(m, X[va][:, keep])
        return m, cut_eval(y[va], p, eng[va], ev["min_cut_len"], ev["n_cut_reps"], seed)["rmse"]

    R = {k: dict(rem_top=[], rem_rand=[], keep_top=[], keep_rand=[]) for k in KS}
    full, imp_all = [], []
    for f, (tr, va) in enumerate(folds, 1):
        t0 = time.time()
        allk = np.ones(X.shape[1], bool)
        m0, r0 = fit_eval(tr, va, allk)
        full.append(r0)
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
        print(f"[{fd}] fold {f}/{len(folds)} full RMSE {r0:.2f} top3 {rank[:3]} ({time.time() - t0:.0f}s)", flush=True)
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
    for _ in range(30):
        Xn = np.column_stack([rng.permutation(Xs[:, j]) for j in range(Xs.shape[1])])
        null.append(silhouette_score(Xn, KMeans(2, n_init=5, random_state=seed).fit(Xn).labels_))
    d = labs[2].cluster_centers_[0] - labs[2].cluster_centers_[1]
    top = np.argsort(-np.abs(d))[:5]
    out.update(ari_k2_mean=float(np.mean(ari)), ari_k2_ci=C.pct(ari), cluster_sizes_k2=np.bincount(ref).tolist(),
               null_sil_k2_mean=float(np.mean(null)), null_sil_k2_p95=float(np.percentile(null, 95)),
               top_discriminating=[dict(sensor=sens[i], symbol=SMAP.get(sens[i]), centroid_diff=float(d[i])) for i in top])
    print(f" n={len(Xs)} silhouette {out['silhouette']} ARI(k=2) {out['ari_k2_mean']:.3f} {np.round(out['ari_k2_ci'], 3).tolist()} "
          f"sizes {out['cluster_sizes_k2']} | column-shuffle null silhouette mean {out['null_sil_k2_mean']:.3f} p95 {out['null_sil_k2_p95']:.3f}")
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
'''

FILES["dashboard/app.py"] = r'''import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fleetpm.config import load_config  # noqa: E402
from fleetpm.pipeline import load_prepared  # noqa: E402
from fleetpm.models.metrics import summarize, last_rows  # noqa: E402

st.set_page_config(page_title="FleetPM monitor", layout="wide")
cfg = load_config()
ART = ROOT / cfg["paths"]["artifacts"]
PRED = "rul_boost_preds_gru.npz"
st.title("FleetPM: predictive maintenance monitor")
st.caption("Benchmark replay of NASA C-MAPSS. All work orders and actions are simulated. Not a production system.")


@st.cache_resource
def prepared(fd):
    return load_prepared(cfg, fd)


@st.cache_data
def read_json(p, mtime):
    return json.loads(Path(p).read_text())


def ts(p):
    return pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")


have = [fd for fd in cfg["data"]["subsets"] if (ART / "models" / fd / PRED).exists() and (ART / "features" / f"{fd}.joblib").exists()]
st.sidebar.write("Subsets with predictions and features:", ", ".join(have) if have else "none")
if not have:
    st.error("No prediction artifacts found. Run scripts/01_build_features.py and scripts/03b_rul_boost.py first.")
    st.stop()
fd = st.sidebar.selectbox("Subset", have)
t1, t2, t3, t4, t5 = st.tabs(["Fleet RUL", "Anomaly", "Work orders", "Extensions", "Artifacts"])

with t1:
    P = prepared(fd)
    T = P["Fte"]
    z = np.load(ART / "models" / fd / PRED)
    L = last_rows(T.engine_id.to_numpy())
    pred = np.clip(np.asarray(z["test_xgb"], float)[L], 0, None)
    raw, cap = np.asarray(P["rul"], float), float(P["cap"])
    if len(pred) != len(raw):
        st.error("Prediction/label length mismatch")
    else:
        c1, c2, c3, c4 = st.columns(4)
        sc, sr = summarize(np.minimum(raw, cap), pred), summarize(raw, pred)
        c1.metric("Engines (test, last cycle)", len(pred))
        c2.metric(f"RMSE vs capped labels (cap {cap:.0f})", f"{sc['rmse']:.2f}")
        c3.metric("RMSE vs raw labels", f"{sr['rmse']:.2f}")
        c4.metric("NASA score (capped)", f"{sc['nasa']:.0f}")
        tr = cfg["triage"]
        bucket = pd.cut(pred, [-1, tr["critical_rul"], tr["high_rul"], tr["medium_rul"], 1e9], labels=["CRITICAL", "HIGH", "MEDIUM", "LOW"])
        st.write("Severity counts, RUL-only provisional buckets on the base prediction (not the deployed triage, no anomaly or persistence):")
        st.bar_chart(pd.Series(bucket).value_counts().reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).fillna(0))
        st.write("Predicted RUL distribution")
        st.bar_chart(pd.Series(np.histogram(pred, bins=range(0, 135, 10))[0], index=[f"{b}-{b + 9}" for b in range(0, 130, 10)]))
        df = pd.DataFrame({"engine_id": np.arange(1, len(pred) + 1), "pred_rul": pred.round(1), "true_rul_raw": raw, "error": (pred - raw).round(1)})
        st.write("Largest overestimates against raw labels (error > 0 is late, the dangerous direction):")
        st.dataframe(df.sort_values("error", ascending=False).head(15), hide_index=True)

with t2:
    p = ART / "models" / fd / "anomaly_metrics.json"
    if not p.exists():
        st.info("anomaly_metrics.json not found for this subset.")
    else:
        m = read_json(str(p), p.stat().st_mtime)
        keys = [k for k in m if k.startswith("IFl_") and not k.startswith("IFl_pool")] + ["auroc_IF_latent_error", "engines_total", "eng_eligible", "eng_unassessable"]
        st.dataframe(pd.DataFrame({"metric": keys, "value": [m.get(k) for k in keys]}), hide_index=True)
        st.caption("Anomaly rates are measured against a benchmark-derived definition of healthy vs RUL<=30 rows; the benchmark has no failure-onset labels.")

with t3:
    files = sorted((ART / "work_orders").rglob("*.json")) if (ART / "work_orders").exists() else []
    rows = []
    for f in files:
        try:
            rows.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            st.warning(f"Unreadable work order file: {f.name}")
    if not rows:
        st.info("No simulated work orders on disk. Run the triage script to generate them.")
    else:
        w = pd.DataFrame(rows)
        show = [c for c in ["work_order_id", "subset", "engine_id", "severity", "rule_fired", "rul_conservative", "health_index", "model_version", "created_at"] if c in w.columns]
        st.write(f"{len(w)} simulated work orders on disk (all subsets).")
        if "severity" in w.columns:
            st.bar_chart(w["severity"].value_counts())
        st.dataframe(w[show].tail(50), hide_index=True)

with t4:
    for fname, label in (("ext_conformal.json", "Conformal lower bounds"), ("ext_cost_policy.json", "Cost-aware policy replay"), ("ext_explain.json", "Explanation faithfulness and clustering")):
        p = ART / fname
        st.subheader(label)
        if not p.exists():
            st.info(f"{fname} not generated yet.")
            continue
        d = read_json(str(p), p.stat().st_mtime)
        if fd not in d:
            st.info(f"No {fd} entry in {fname}.")
            continue
        r = d[fd]
        if fname == "ext_conformal.json":
            st.write(f"Calibration variant frozen from nested training-only split: {r['variant']}. Empirical coverage, no guarantee.")
            st.dataframe(pd.DataFrame({k: dict(coverage=v["cov_cap"], slack=v["slack_mean"], rmse=v["rmse_cap"], nasa=v["nasa_cap"], late_rate=v["late_rate"]) for k, v in r["test"].items()}).T)
        elif fname == "ext_cost_policy.json":
            st.write(r["assumptions"])
            st.dataframe(pd.DataFrame({k: {"break-even vs run-to-failure": v["vs_rtf"]["point"], "vs best fixed interval": v["vs_fixed"]["point"]} for k, v in r["break_even"].items()}).T)
        else:
            if "faithfulness" in r:
                st.dataframe(pd.DataFrame({k: dict(remove_diff=v["remove_diff"][0], keep_diff=v["keep_diff"][0]) for k, v in r["faithfulness"]["k"].items()}).T)
            if "clustering" in r:
                c = r["clustering"]
                st.write(f"k=2 silhouette {c['silhouette']['2']:.3f}, bootstrap ARI {c['ari_k2_mean']:.3f}, shuffle-null p95 {c['null_sil_k2_p95']:.3f}. Fault-mode mapping is a hypothesis.")

with t5:
    rows = []
    for p in [ART / "features" / f"{fd}.joblib", ART / "models" / fd / PRED, ART / "models" / fd / "rul_bundle_v2_gru.joblib", ART / "models" / fd / "anomaly_ifl.joblib", ART / "mlflow.db"]:
        rows.append(dict(file=str(p.relative_to(ROOT)), exists=p.exists(), modified=ts(p) if p.exists() else None))
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    fe, pr = ART / "features" / f"{fd}.joblib", ART / "models" / fd / PRED
    if fe.exists() and pr.exists() and fe.stat().st_mtime > pr.stat().st_mtime:
        st.warning("Features are newer than the saved predictions: predictions may be stale.")
    st.caption("Retraining history is not shown: the retrain demo is FD001 with synthetic drift and writes only to MLflow.")
'''

for rel, text in FILES.items():
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.read_text() != text:
        shutil.copy2(p, p.with_name(p.name + f".bak_{TS}"))
        print("backed up", rel)
    p.write_text(text)
    py_compile.compile(str(p), doraise=True)
    print("wrote + compiled", rel)