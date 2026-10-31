import _bootstrap
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
    rec["fixed_thr"] = {}
    print(" fixed thresholds chosen in advance (no sweep): failures avoided / mean wasted life (all engines) / cost at r=10, r=100")
    for k in TRI:
        _, Fa_k, Wa_k = fam[k]
        rec["fixed_thr"][k] = {}
        for t in (15, 30, 60):
            i = int(t)
            fa, wa = float(1 - Fa_k[i].mean()), float(Wa_k[i].mean())
            cst = {str(int(r)): float(r * Fa_k[i].mean() + wa) for r in R_SEL}
            rec["fixed_thr"][k][str(t)] = dict(failures_avoided=fa, mean_wasted=wa, cost=cst)
            print(f"  {k:<16} thr {t:<3} avoided {fa:.3f} wasted {wa:5.1f} cost r=10 {cst['10']:.1f} r=100 {cst['100']:.1f}")
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
