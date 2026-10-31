import _bootstrap
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
                         cov_lt_cap=float(c[D["true_raw"] < D["cap"]].mean()), n_lt_cap=int((D["true_raw"] < D["cap"]).sum()),
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
        print(f"  {name:<16} cov {r['cov_cap']:.3f} CI{np.round(r['ci']['marg'], 3).tolist()} cov_raw {r['cov_raw']:.3f} cov_lt_cap {r['cov_lt_cap']:.3f} (n {r['n_lt_cap']}) "
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
