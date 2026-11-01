import py_compile, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "scripts" / "13_deck_v2.py"
DST = ROOT / "scripts" / "14_deck_v3.py"
if not SRC.exists():
    raise SystemExit("scripts/13_deck_v2.py not found")
MARK = "# ------------------------------------------------------------------ slide 1\n"
txt = SRC.read_text()
if txt.count(MARK) != 1:
    raise SystemExit("slide-1 marker not found exactly once in 13_deck_v2.py")
PRE = txt[:txt.index(MARK)]

BODY = r'''
from pptx.enum.dml import MSO_LINE
plt.rcParams.update({"font.size": 6})
TITLE = "SENTINEL: Fleet Health and RUL Intelligence"
FEAT = "2E75B6"
SUBF = dict(zip(FDS, PAL))
BORD = "D6DEE9"


def finish(fig, name, rect_=(0, 0, 1, 1)):
    fig.tight_layout(pad=0.35, rect=rect_)
    p = art(cfg, "figures", f"deck3_{name}.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return str(p)


def mini(name, w, h, draw, title=None, nr=1, nc=1):
    fig, axs = plt.subplots(nr, nc, figsize=(w, h), squeeze=False)
    draw(axs)
    for ax in axs.ravel():
        if not ax.get_visible():
            continue
        clean(ax)
        ax.tick_params(labelsize=5.5, length=2, pad=1.5)
        ax.xaxis.label.set_size(5.8)
        ax.yaxis.label.set_size(5.8)
        ax.title.set_size(6.3)
        hd, lb = ax.get_legend_handles_labels()
        if lb:
            ax.legend(fontsize=5, frameon=False, handlelength=1.2, borderpad=0.2, labelspacing=0.25)
    if title:
        fig.suptitle(title, fontsize=6.6, fontweight="bold", color="#0B2A4A")
    return finish(fig, name, (0, 0, 1, 0.93) if title else (0, 0, 1, 1))


def cimg(s, name, x, y, w, h, draw, title=None, nr=1, nc=1):
    pic(s, mini(name, w, h, draw, title, nr, nc), x, y, w, h)
    rect(s, x, y, w, h, None, BORD)


def pimg(s, path, x, y, w, h):
    if path:
        pic(s, path, x, y, w, h)
    rect(s, x, y, w, h, None, BORD)


def rng(v, f=".1f"):
    return f"{min(v):{f}} to {max(v):{f}}"


def lifes(f):
    return np.unique(D[f]["eng"], return_counts=True)[1]


cap_ = int(D["FD001"]["cap"])
an_, tg_, dcfg = cfg["anomaly"], cfg["triage"], cfg["data"]
nfeat = [len(D[f]["P"]["pre"].sensors) * (len(cfg["features"]["stats"]) + 2) for f in FDS]
FA = [f for f in FDS if anom[f]]


def band_rmse(f, lo_, hi_):
    d = D[f]
    e, tr = d["pred_t"] - d["true_cap"], d["true_raw"]
    m = (tr > lo_) & (tr <= hi_)
    return float(np.sqrt((e[m] ** 2).mean())) if m.any() else float("nan")


# ------------------------------------------------------------------ chart drawers
def d_scatter(axs):
    ax = axs[0, 0]
    for f in FDS:
        ax.scatter(D[f]["true_cap"], D[f]["pred_t"], s=3, color=H(SUBF[f]), label=f, linewidths=0)
    ax.plot([0, 125], [0, 125], c=H(GREY), ls="--", lw=0.8)
    ax.set_xlabel("True RUL (capped)")
    ax.set_ylabel("Predicted RUL")


def d_scatter4(axs):
    for ax, f in zip(axs.ravel(), FDS):
        t, p_ = D[f]["true_cap"], D[f]["pred_t"]
        late = p_ > t
        ax.scatter(t[~late], p_[~late], s=5, c=H(BLUE), linewidths=0, label="early")
        ax.scatter(t[late], p_[late], s=5, c=H(RED), linewidths=0, label="late")
        ax.plot([0, 125], [0, 125], c=H(GREY), ls="--", lw=0.8)
        ax.set_title(f"{f}  RMSE {ST[f]['rmse']:.1f}")


def d_hist(axs):
    ax = axs[0, 0]
    err = np.concatenate([D[f]["pred_t"] - D[f]["true_cap"] for f in FDS])
    b = np.arange(-60, 65, 5)
    ax.hist(np.clip(err[err <= 0], -60, 60), bins=b, color=H(BLUE), label="early")
    ax.hist(np.clip(err[err > 0], -60, 60), bins=b, color=H(RED), label="late")
    ax.set_xlabel("Predicted minus true RUL (cycles)")
    ax.set_ylabel("Engines")


def d_band(axs):
    ax = axs[0, 0]
    wd = 0.2
    for i, f in enumerate(FDS):
        e, tr, cp = D[f]["pred_t"] - D[f]["true_cap"], D[f]["true_raw"], D[f]["cap"]
        ms = [tr <= 30, (tr > 30) & (tr <= 60), (tr > 60) & (tr <= cp), tr > cp]
        ax.bar(np.arange(4) + (i - 1.5) * wd, [float(np.sqrt((e[m] ** 2).mean())) if m.any() else 0 for m in ms], wd, color=H(SUBF[f]), label=f)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["<=30", "31-60", "61-cap", ">cap"])
    ax.set_xlabel("True RUL band (cycles)")
    ax.set_ylabel("RMSE (cycles)")


def d_box(axs):
    ax = axs[0, 0]
    bp = ax.boxplot([D[f]["pred_t"] - D[f]["true_cap"] for f in FDS], widths=0.5, patch_artist=True, medianprops={"color": "black"}, flierprops={"markersize": 2})
    for pt, f in zip(bp["boxes"], FDS):
        pt.set_facecolor(H(SUBF[f]))
        pt.set_alpha(0.85)
    ax.axhline(0, c=H(GREY), lw=0.8)
    ax.set_xticklabels(FDS)
    ax.set_ylabel("Predicted minus true RUL")


def d_cdf(axs):
    ax = axs[0, 0]
    for f in FDS:
        v = np.sort(np.abs(D[f]["pred_t"] - D[f]["true_cap"]))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=H(SUBF[f]), label=f, lw=1.3)
    ax.set_xlabel("Absolute error (cycles)")
    ax.set_ylabel("Share of engines")


def d_nasa(axs):
    ax = axs[0, 0]
    ax.bar(range(4), [ST[f]["nasa"] / ST[f]["n"] for f in FDS], 0.55, color=[H(SUBF[f]) for f in FDS])
    ax.set_xticks(range(4))
    ax.set_xticklabels(FDS)
    ax.set_ylabel("NASA score per engine")


def d_auroc(axs):
    ax = axs[0, 0]
    for j, (k, lab, col) in enumerate([("auroc_HI", "Health index", GREY), ("auroc_IF_error_1d", "IF error", TEAL), ("auroc_IF_latent_error", "IF latent+error", NAVY)]):
        ax.scatter([anom[f][k] for f in FA], np.arange(len(FA)) + (j - 1) * 0.18, s=22, c=H(col), label=lab)
    ax.set_yticks(range(len(FA)))
    ax.set_yticklabels(FA)
    ax.set_xlim(0.9, 1.003)
    ax.set_xlabel("AUROC")


def d_tprfpr(axs):
    ax = axs[0, 0]
    x = np.arange(len(FA))
    ax.bar(x - 0.2, [anom[f]["IFl_tpr_persist"] for f in FA], 0.4, color=H(TEAL), label="persistent TPR")
    ax.bar(x + 0.2, [anom[f]["IFl_fpr_persist"] for f in FA], 0.4, color=H(RED), label="persistent FPR")
    ax.set_xticks(x)
    ax.set_xticklabels(FA)
    ax.set_ylim(0, 1.2)


def d_lead(axs):
    ax = axs[0, 0]
    ax.bar(range(len(FA)), [anom[f]["IFl_lead_median"] for f in FA], 0.5, color=H(NAVY), label="median lead")
    ax.scatter(range(len(FA)), [anom[f]["IFl_lead_p10"] for f in FA], c=H(ORANGE), zorder=3, s=18, label="10th percentile")
    ax.set_xticks(range(len(FA)))
    ax.set_xticklabels(FA)
    ax.set_ylabel("Cycles of warning")


def d_outcome(axs):
    ax = axs[0, 0]
    bot = np.zeros(len(FA))
    for k, lab, col in [("IFl_E_in_pos", "alert at RUL<=30", GREEN), ("IFl_E_warn_pr_cap", "warning 30 to cap", TEAL), ("IFl_E_early_ge_cap", "early (>= cap)", AMBER), ("IFl_E_none", "missed", RED)]:
        v = np.array([anom[f].get(k, 0) for f in FA], float)
        ax.bar(range(len(FA)), v, 0.6, bottom=bot, color=H(col), label=lab)
        bot += v
    ax.set_xticks(range(len(FA)))
    ax.set_xticklabels(FA)
    ax.set_ylabel("Eligible engines")


def d_fprk(axs):
    ax = axs[0, 0]
    ks = list(range(1, 9))
    for f in FA:
        ax.semilogy(ks, [max(anom[f].get(f"IFl_pool_persist_fpr_k{k}", np.nan), 1e-5) for k in ks], marker="o", ms=2, color=H(SUBF[f]), label=f)
    ax.set_xlabel("Persistence k")
    ax.set_ylabel("Persistent FPR")


def d_cost(axs):
    ax = axs[0, 0]
    for k in PLAB:
        ax.loglog(RGRID, np.mean([cost[f]["curves"][k] for f in FDS], 0), color=H(PCOL[k]), lw=1.3, label=PLAB[k])
    ax.set_xlabel("r = C_fail / C_waste")
    ax.set_ylabel("Cost per engine")


def d_slack(axs):
    ax = axs[0, 0]
    mk = ["o", "s", "^", "D"]
    for i, f in enumerate(FDS):
        for m in M:
            r = conf[f]["test"][m]
            ax.scatter(r["slack_mean"], r["cov_lt_cap"], s=16, marker=mk[i], c=H(MCOL[m]), label=MLAB[m] if i == 0 else None)
    for y_ in (0.8, 0.9):
        ax.axhline(y_, ls=":", c=H(GREY), lw=0.8)
    ax.set_xlabel("Mean slack (cycles)")
    ax.set_ylabel("Coverage below cap")


def d_cov(axs):
    ax = axs[0, 0]
    v = np.array([[conf[f]["test"][m]["cov_lt_cap"] for f in FDS] for m in M])
    mu = v.mean(1)
    ax.bar(range(4), mu, color=[H(MCOL[m]) for m in M], yerr=[mu - v.min(1), v.max(1) - mu], capsize=2, ecolor="#334155")
    for y_ in (0.8, 0.9):
        ax.axhline(y_, ls=":", c=H(GREY), lw=0.9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Raw", "Offset", "Conf 80%", "Conf 90%"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage")


def d_life(axs):
    ax = axs[0, 0]
    bp = ax.boxplot([lifes(f) for f in FDS], widths=0.5, patch_artist=True, medianprops={"color": "black"}, flierprops={"markersize": 2})
    for pt, f in zip(bp["boxes"], FDS):
        pt.set_facecolor(H(SUBF[f]))
        pt.set_alpha(0.85)
    ax.set_xticklabels(FDS)
    ax.set_ylabel("Cycles to failure")


def d_cut(axs):
    ax = axs[0, 0]
    for f in FDS:
        n_ = D[f]["P"]["Fte"].groupby("engine_id").size().to_numpy()
        ax.scatter(n_, D[f]["true_raw"], s=4, color=H(SUBF[f]), linewidths=0, label=f)
    ax.set_xlabel("Observed test cycles")
    ax.set_ylabel("True RUL (raw)")


def d_sens(axs):
    ax = axs[0, 0]
    kept = np.array([len(D[f]["P"]["pre"].sensors) for f in FDS])
    const = np.array([len(D[f]["P"]["pre"].const) for f in FDS])
    wk = 21 - kept - const
    ax.bar(range(4), kept, 0.6, color=H(TEAL), label="kept")
    ax.bar(range(4), wk, 0.6, bottom=kept, color=H(AMBER), label="dropped: weak trend")
    ax.bar(range(4), const, 0.6, bottom=kept + wk, color=H(GREY), label="dropped: constant")
    ax.set_xticks(range(4))
    ax.set_xticklabels(FDS)
    ax.set_ylabel("Sensors (of 21)")


def d_reg(axs):
    ax = axs[0, 0]
    for f in FDS:
        v = D[f]["P"]["Ztr"]["regime_id"].value_counts(normalize=True).sort_values(ascending=False).to_numpy()
        ax.plot(range(1, len(v) + 1), v, marker="o", ms=3, color=H(SUBF[f]), label=f)
    ax.set_xlabel("Regime rank")
    ax.set_ylabel("Share of train rows")


def d_nom(axs):
    for ax, al in zip(axs[0], ("0.1", "0.2")):
        for j, (var, col, mk) in enumerate([("all_rows", GREY, "s"), ("one_cut_per_engine", NAVY, "o")]):
            ax.errorbar(np.arange(4) + (j - 0.5) * 0.2, [conf[f]["nested"][f"{var}|{al}"]["cut_cov_mean"] for f in FDS],
                        yerr=[conf[f]["nested"][f"{var}|{al}"]["cut_cov_sd"] for f in FDS], fmt=mk, ms=3, color=H(col), capsize=2, lw=0.8, label=var)
        ax.axhline(1 - float(al), ls=":", c=H(RED), lw=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(FDS)
        ax.set_title(f"nominal {1 - float(al):.0%}")
    axs[0, 0].set_ylabel("Held-out coverage")


def d_bandcov(axs):
    ax = axs[0, 0]
    for f in FDS:
        for al, ls_ in (("0.1", "-"), ("0.2", "--")):
            v = conf[f]["nested"][f"{conf[f]['variant']}|{al}"]["cut_band_cov"]
            ax.plot([0, 1, 2], v, ls=ls_, marker="o", ms=2.5, color=H(SUBF[f]), label=f"{f} a={al}" if al == "0.1" else None, lw=1)
    for y_ in (0.8, 0.9):
        ax.axhline(y_, ls=":", c=H(GREY), lw=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["<30", "30-60", ">=60"])
    ax.set_xlabel("Predicted-RUL band (solid: 90%, dashed: 80%)")
    ax.set_ylabel("Held-out coverage")


def d_qband(axs):
    for ax, f in zip(axs[0], FDS):
        x = np.arange(3)
        ax.bar(x - 0.27, conf[f]["q_offset"], 0.27, color=H(TEAL), label="offset 0.8 q")
        ax.bar(x, conf[f]["q_conf"]["0.2"], 0.27, color=H(BLUE), label="conformal 80%")
        ax.bar(x + 0.27, conf[f]["q_conf"]["0.1"], 0.27, color=H(NAVY), label="conformal 90%")
        ax.set_xticks(x)
        ax.set_xticklabels(["<30", "30-60", ">=60"])
        ax.set_title(f)
    axs[0, 0].set_ylabel("Bound offset q (cycles)")


def d_covci(axs):
    ax = axs[0, 0]
    for j, m in enumerate(M):
        r_ = [conf[f]["test"][m] for f in FDS]
        y = [q["cov_cap"] for q in r_]
        lo = [q["cov_cap"] - q["ci"]["marg"][0] for q in r_]
        hi = [q["ci"]["marg"][1] - q["cov_cap"] for q in r_]
        ax.errorbar([i * 5 + j for i in range(4)], y, yerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)], fmt="o", ms=3, color=H(MCOL[m]), capsize=2, lw=0.9, label=MLAB[m])
    for y_ in (0.8, 0.9):
        ax.axhline(y_, ls=":", c=H(GREY), lw=0.8)
    ax.set_xticks([i * 5 + 1.5 for i in range(4)])
    ax.set_xticklabels(FDS)
    ax.set_ylim(0.2, 1.02)
    ax.set_ylabel("Coverage with 95% engine CI")


def d_nasarat(axs):
    ax = axs[0, 0]
    for j, m in enumerate(M[1:]):
        ax.bar(np.arange(4) + (j - 1) * 0.27, [conf[f]["test"][m]["nasa_cap"] / max(conf[f]["test"]["raw"]["nasa_cap"], 1e-9) for f in FDS], 0.27, color=H(MCOL[m]), label=MLAB[m])
    ax.axhline(1, c=H(GREY), lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(FDS)
    ax.set_ylabel("NASA score relative to raw")


def d_atr(axs):
    for ax, key, lab in zip(axs[0], ("failures_avoided", "mean_wasted_all"), ("Failures avoided at r=100", "Mean wasted life at r=100 (cycles)")):
        for j, k in enumerate(("rul_raw", "rul_offset", "rul_conformal10", "fixed_interval")):
            vals = [[y[key] for y in cost[f]["at_r"][k] if y["r"] == 100.0][0] for f in FDS]
            ax.bar(np.arange(4) + (j - 1.5) * 0.2, vals, 0.2, color=H(PCOL[k]), label=PLAB[k] if key == "failures_avoided" else None)
        ax.set_xticks(range(4))
        ax.set_xticklabels(FDS)
        ax.set_title(lab)


def d_be(axs):
    ax = axs[0, 0]
    for j, k in enumerate(("rul_raw", "rul_offset", "rul_conformal10")):
        xs, ys, lo, hi = [], [], [], []
        for i, f in enumerate(FDS):
            b = cost[f]["break_even"][k]["vs_rtf"]
            if b["point"] is None:
                continue
            xs.append(i * 4 + j)
            ys.append(b["point"])
            ci = b.get("ci") or [b["point"], b["point"]]
            lo.append(max(0, b["point"] - ci[0]))
            hi.append(max(0, ci[1] - b["point"]))
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=3.5, capsize=2, color=H(PCOL[k]), label=PLAB[k])
    ax.set_xticks([i * 4 + 1 for i in range(4)])
    ax.set_xticklabels(FDS)
    ax.set_ylabel("Break-even r vs run-to-failure (95% CI)")


def d_fixed(metric):
    def dr(axs):
        for ax, f in zip(axs[0], FDS):
            for k in ("rul_raw", "rul_offset", "rul_conformal10"):
                v = [(cost[f]["fixed_thr"][k][t]["cost"]["100"] if metric == "cost100" else cost[f]["fixed_thr"][k][t][metric]) for t in ("15", "30", "60")]
                ax.plot([15, 30, 60], v, marker="o", ms=3, color=H(PCOL[k]), label=PLAB[k].replace("RUL threshold, ", ""))
            ax.set_xticks([15, 30, 60])
            ax.set_title(f)
            ax.set_xlabel("threshold (cycles)")
    return dr


# ================================================================== SLIDE 1
def ring_img(name, fr, col, txt, fr2=None, col2=None):
    fig, ax = plt.subplots(figsize=(0.9, 0.9))
    ax.pie([fr, 1 - fr], startangle=90, counterclock=False, colors=[H(col), "#E6EBF2"], radius=1.0, wedgeprops=dict(width=0.26, edgecolor="white"))
    if fr2 is not None:
        ax.pie([fr2, 1 - fr2], startangle=90, counterclock=False, colors=[H(col2), "#E6EBF2"], radius=0.7, wedgeprops=dict(width=0.26, edgecolor="white"))
    ax.text(0, 0, txt, ha="center", va="center", fontsize=6.5, fontweight="bold", color="#0B2A4A")
    ax.set_aspect("equal")
    fig.subplots_adjust(0, 0, 1, 1)
    p = art(cfg, "figures", f"deck3_{name}.png")
    fig.savefig(p, dpi=250, transparent=True)
    plt.close(fig)
    return str(p)


rm = [ST[f]["rmse"] for f in FDS]
cv = [conf[f]["test"]["conformal_a0.1"]["cov_lt_cap"] for f in FDS]
lr = float(np.mean([conf[f]["test"]["raw"]["late_rate"] for f in FDS]))
lc = float(np.mean([conf[f]["test"]["conformal_a0.1"]["late_rate"] for f in FDS]))
be = [b for b in (cost[f]["break_even"]["rul_conformal10"]["vs_rtf"]["point"] for f in FDS) if b is not None]
af = [anom[f]["auroc_IF_latent_error"] for f in FA]
pf = [anom[f]["IFl_fpr_persist"] for f in FA]
tp = [anom[f]["IFl_tpr_persist"] for f in FA]
sl = [conf[f]["test"]["conformal_a0.1"]["slack_mean"] for f in FDS]
b30 = [band_rmse(f, -1, 30) for f in FDS]
bmid = [band_rmse(f, 60, D[f]["cap"]) for f in FDS]
nr_ = float(np.mean(rm)) / cap_

s = prs.slides.add_slide(BL)
rect(s, 0, 0, 13.333, 0.46, NAVY)
rect(s, 0, 0.46, 13.333, 0.04, TEAL)
text(s, 0.12, 0.03, 8.4, 0.4, [(TITLE, 16, True, WHITE)], anchor=MSO_ANCHOR.MIDDLE)
text(s, 8.3, 0.03, 4.9, 0.4, [("NASA C-MAPSS benchmark replay | simulated actions | technical review", 8, False, "C9D6E6")], align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)

gauges = [("Engines with work order", f"{N_ORD} / {NTOT}", f"{len(FDS)} subsets, official test", N_ORD / NTOT, BLUE, f"{N_ORD / NTOT:.0%}", None, None),
          ("RMSE vs capped RUL", rng(rm), "cycles; ring = share of cap", nr_, TEAL, f"{nr_:.0%}", None, None)]
if af:
    gauges.append(("Anomaly AUROC", rng(af, ".3f"), "Isolation Forest, latent + error", min(af), PURPLE, f"{min(af):.2f}", None, None))
gauges.append(("90% bound coverage", rng(cv, ".2f"), "engines below cap; measured", min(cv), NAVY, f"{min(cv):.2f}", None, None))
gauges.append(("Late predictions", f"{lr:.0%} to {lc:.0%}", "outer raw, inner 90% bound", lr, RED, "", lc, GREEN))
if tp:
    gauges.append(("Persistent TPR", rng(tp, ".2f"), "anomaly, RUL <= 30 proxy", min(tp), AMBER, f"{min(tp):.0%}", None, None))
for i, (lab, val, sub, fr, col, ctr, fr2, col2) in enumerate(gauges):
    x = 0.1 + i * 2.205
    rect(s, x, 0.55, 2.15, 0.76, LIGHT)
    pic(s, ring_img(f"k{i}", fr, col, ctr, fr2, col2), x + 0.05, 0.59, 0.68, 0.68)
    text(s, x + 0.76, 0.56, 1.38, 0.18, [(lab.upper(), 5.6, True, "51627A")])
    text(s, x + 0.76, 0.72, 1.38, 0.3, [(val, 11, True, NAVY)])
    text(s, x + 0.76, 1.02, 1.38, 0.28, [(sub, 5.6, False, "64748B")])

RY, RH = 1.38, 2.72
rect(s, 0.1, RY, 2.75, RH, None, BORD)
sec(s, 0.14, RY + 0.01, 2.65, "Pipeline ring")
stg = [BLUE, BLUE, FEAT, TEAL, TEAL, TEAL, ORANGE, ORANGE]
cx, cy, R = 1.475, RY + 1.52, 0.85
o = rect(s, cx - 1.12, cy - 1.12, 2.24, 2.24, None, GREY, MSO_SHAPE.OVAL)
o.line.dash_style = MSO_LINE.DASH
rect(s, cx - R, cy - R, 2 * R, 2 * R, None, BORD, MSO_SHAPE.OVAL)
pts = [(cx + R * np.cos(np.radians(-90 + 45 * i)), cy + R * np.sin(np.radians(-90 + 45 * i))) for i in range(8)]
for (px, py) in pts:
    v = np.array([px - cx, py - cy])
    u = v / np.linalg.norm(v)
    a_, b_ = np.array([cx, cy]) + u * 0.42, np.array([px, py]) - u * 0.24
    line(s, a_[0], a_[1], b_[0], b_[1], BORD, False, 0.75)
for i in range(8):
    a_, b_ = np.array(pts[i]), np.array(pts[(i + 1) % 8])
    u = (b_ - a_) / np.linalg.norm(b_ - a_)
    p0, p1 = a_ + u * 0.25, b_ - u * 0.25
    line(s, p0[0], p0[1], p1[0], p1[1], "334155", True, 1.2)
for i, (px, py) in enumerate(pts):
    put(rect(s, px - 0.22, py - 0.22, 0.44, 0.44, stg[i], WHITE, MSO_SHAPE.OVAL), [(str(i + 1), 10, True, WHITE)], align=PP_ALIGN.CENTER, m=0.0)
put(rect(s, cx - 0.4, cy - 0.4, 0.8, 0.8, NAVY, None, MSO_SHAPE.OVAL),
    [("CORE", 7, True, WHITE), ("MLflow", 5, False, "C9D6E6"), ("drift gate", 5, False, "C9D6E6"), ("dashboard", 5, False, "C9D6E6")], align=PP_ALIGN.CENTER, m=0.0)

leg = [("Ingest and validate", "26 columns; NaN, duplicate, cycle-gap and RUL alignment checks"),
       ("Regime normalisation", f"KMeans k={dcfg['n_regimes']}; per-regime z-score; median w={dcfg['smooth_window']}"),
       ("Causal features", f"windows 30, 10; {min(nfeat)} to {max(nfeat)} columns; |Spearman| >= {dcfg['min_abs_trend']}"),
       ("Anomaly detector", f"autoencoder (cycles {an_['healthy_cycle_range'][0]}-{an_['healthy_cycle_range'][1]}) + Isolation Forest; k={an_['persist_k']}"),
       ("RUL regressor", f"XGBoost {cfg['rul']['xgb']['n_estimators']} trees, depth {cfg['rul']['xgb']['max_depth']}; cap {cap_}; grouped CV"),
       ("Risk bound", "0.8-quantile offset deployed; conformal 90% offline"),
       ("Deterministic triage", f"5 severities; cutoffs {tg_['critical_rul']}/{tg_['high_rul']}/{tg_['medium_rul']} (provisional)"),
       ("Simulated work order", "deterministic ID; TreeSHAP sensors; restart-safe")]
rect(s, 2.9, RY, 3.0, RH, None, BORD)
sec(s, 2.94, RY + 0.01, 2.9, "Stage specification")
for i, (t1, t2) in enumerate(leg):
    y = RY + 0.34 + i * 0.295
    put(rect(s, 2.97, y + 0.03, 0.22, 0.22, stg[i], None, MSO_SHAPE.OVAL), [(str(i + 1), 7, True, WHITE)], align=PP_ALIGN.CENTER, m=0.0)
    text(s, 3.22, y, 2.66, 0.29, [(t1, 6.5, True, NAVY), (t2, 5.2, False, "334155")], anchor=MSO_ANCHOR.MIDDLE)

FX, FW = 5.95, 1.95
rect(s, FX, RY, FW, RH, None, BORD)
sec(s, FX + 0.04, RY + 0.01, FW - 0.08, "Triage funnel (engines)")
fun = [("Test engines replayed", NTOT, BLUE), ("Work order issued", N_ORD, TEAL), ("HIGH or CRITICAL", N_HIGH, ORANGE), ("CRITICAL", N_CRIT, RED)]
tops, bots = [1.7, 1.42, 1.14, 0.86], [1.42, 1.14, 0.86, 0.58]
fcx = FX + FW / 2
for i, (lab, n, col) in enumerate(fun):
    y = RY + 0.34 + i * 0.585
    fb = s.shapes.build_freeform(E(fcx - tops[i] / 2), E(y), scale=1.0)
    fb.add_line_segments([(E(fcx + tops[i] / 2), E(y)), (E(fcx + bots[i] / 2), E(y + 0.56)), (E(fcx - bots[i] / 2), E(y + 0.56))], close=True)
    sh = fb.convert_to_shape()
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(col)
    sh.line.fill.background()
    sh.shadow.inherit = False
    put(sh, [(f"{n} ({100 * n / NTOT:.0f}%)", 8.5, True, WHITE), (lab, 5.8, False, "E3ECF7")], align=PP_ALIGN.CENTER, m=0.0)

cimg(s, "s1_sc", 7.95, RY, 2.6, RH, d_scatter, "Predicted vs true RUL (test, capped)")
cimg(s, "s1_cov", 10.6, RY, 2.63, RH, d_cov, "Lower-bound coverage, engines below cap")

CY, CH, CWD = 4.17, 1.5, 2.15
row3 = [("s1_band", d_band, "RMSE by true-RUL band"), ("s1_hist", d_hist, "Test error distribution"),
        ("s1_auc", d_auroc if FA else d_hist, "Anomaly AUROC by scorer"), ("s1_tf", d_tprfpr if FA else d_hist, "Persistent TPR and FPR"),
        ("s1_cost", d_cost, "Cost replay, mean of subsets"), ("s1_slack", d_slack, "Coverage vs bound slack")]
for i, (nm, fn, ttl) in enumerate(row3):
    cimg(s, nm, 0.1 + i * 2.205, CY, CWD, CH, fn, ttl)

BY, BH = 5.73, 1.4
th = [("Target and features", BLUE, f"RUL capped at {cap_} cycles: early-life degradation is not observable. Causal trailing windows (30 and 10 cycles) over regime-normalised, median-smoothed sensors: mean, std, min, max, slope, kurtosis. Weight 3 below RUL 30."),
      ("Anomaly detection", TEAL, f"An autoencoder fitted on healthy cycles {an_['healthy_cycle_range'][0]}-{an_['healthy_cycle_range'][1]} sets a normal envelope; an Isolation Forest scores latent state and error. An alert needs {an_['persist_k']} consecutive flags. Output is unusual behaviour, not a confirmed fault."),
      ("Conformal lower bound", NAVY, "Out-of-fold over-prediction residuals, calibrated per predicted-RUL band; bound = prediction minus the finite-sample (1 - alpha) quantile. Coverage is measured on held-out engines, not guaranteed. Offline; deployed triage uses the 0.8-quantile offset."),
      ("Cost-aware maintenance", ORANGE, "r = C_fail / C_waste compares failure with premature maintenance. Policies fire causally from cycle 30. Break-even r shows where predictive triage beats run-to-failure and fixed interval. Training-trajectory replay; no monetary values.")]
for i, (t_, col, b_) in enumerate(th):
    tile(s, 0.1 + i * 2.6, BY, 2.55, BH, t_, b_, col, fs=6.6)
fnd = [f"RMSE {rng(b30)} cycles at true RUL <= 30 versus {rng(bmid)} at RUL 60 to 125.",
       f"90% bound: late rate {lr:.0%} to {lc:.0%}, slack {rng(sl, '.0f')} cycles, coverage {rng(cv, '.2f')}."]
if af and pf:
    fnd.append(f"Anomaly AUROC {rng(af, '.3f')}; persistent FPR {min(pf):.2%} to {max(pf):.2%}.")
if be:
    fnd.append(f"Break-even r {rng(be)} vs run-to-failure (in-sample, simulated).")
tile(s, 10.55, BY, 2.68, BH, "Key findings", fnd, NAVY, fs=6.8)
text(s, 0.1, 7.18, 13.1, 0.25, [("Benchmark replay of simulated NASA C-MAPSS data; all actions are simulated; thresholds are provisional. Numbers, intervals and assumptions are in the appendix.", 6.5, False, GREY)])

C4 = [0.15 + i * 3.25 for i in range(4)]
W4 = 3.2

# ================================================================== A1 data
s = frame("Data, label construction and evaluation protocol", BLUE, "data/CMAPSSData readme.txt, loaded files, configs/config.yaml")
rows = [["Subset", "Op. conditions", "Fault modes (readme)", "Train engines", "Test engines", "Train rows", "Test rows", "Sensors kept", "Features", "Mean train life", "RUL cap"]]
for f in FDS:
    P_ = D[f]["P"]
    ns = len(P_["pre"].sensors)
    rows.append([f, META[f][0], META[f][1], len(lifes(f)), ST[f]["n"], len(D[f]["eng"]), len(P_["Fte"]), ns, ns * (len(cfg["features"]["stats"]) + 2), f"{lifes(f).mean():.0f}", int(D[f]["cap"])])
table(s, 0.15, 0.75, 13.0, 1.45, rows, None, 9)
for i, (nm, fn, ttl) in enumerate([("a1_life", d_life, "Training engine life"), ("a1_cut", d_cut, "Test cut: observed cycles vs true RUL"),
                                   ("a1_sens", d_sens, "Sensor selection"), ("a1_reg", d_reg, "Operating-regime share")]):
    cimg(s, nm, C4[i], 2.3, W4, 2.35, fn, ttl)
tl = [("Labels and target", BLUE, [f"Train RUL = maximum cycle minus cycle; test RUL = official RUL vector plus remaining cycles to the last observed cycle.", f"Target capped at {cap_} (assumption); samples with RUL below 30 carry weight 3.", "Raw labels are kept for evaluation."]),
      ("Evaluation scope", TEAL, ["Official test: prediction at the last observed cycle per engine, aligned with the official RUL vector (engine order validated).", "Capped labels are primary; raw-label RMSE is inflated where true RUL exceeds the cap."]),
      ("Validation protocol", NAVY, [f"{cfg['eval']['cv_folds']}-fold GroupKFold by engine for every selection; seed 42.", "Out-of-fold predictions feed offsets, conformal bounds and cost replay.", "Bootstrap intervals resample whole engines. Test labels are never used for fitting or selection."]),
      ("Leakage status", ORANGE, ["Regime clusters, normalisation statistics and sensor selection are fitted on all training engines before cross-validation.", "Strict fold-wise refitting has not been run; a mild optimistic CV bias is possible and unquantified."])]
for i, (t_, col, b_) in enumerate(tl):
    tile(s, C4[i], 4.75, W4, 2.4, t_, b_, col, 9)

# ================================================================== A2 accuracy
s = frame("RUL accuracy on the official test set", TEAL, "recomputed from rul_boost_preds_gru.npz and official RUL files")
rows = [["Subset", "Engines", "RMSE (capped)", "MAE (capped)", "NASA (capped)", "NASA per engine", "RMSE (raw)", "Late rate", "Bias (cycles)"]]
for f in FDS:
    v = ST[f]
    rows.append([f, v["n"], f"{v['rmse']:.2f}", f"{v['mae']:.2f}", f"{v['nasa']:.0f}", f"{v['nasa'] / v['n']:.2f}", f"{v['rmse_raw']:.2f}", f"{v['late']:.2f}", f"{v['bias']:+.2f}"])
table(s, 0.15, 0.75, 13.0, 1.4, rows, None, 9.5)
cimg(s, "a2_sc", 0.15, 2.25, 4.7, 2.65, d_scatter4, None, 2, 2)
cimg(s, "a2_band", 4.9, 2.25, 4.0, 2.65, d_band, "RMSE by true-RUL band")
cimg(s, "a2_box", 8.95, 2.25, 4.2, 2.65, d_box, "Signed error by subset")
cimg(s, "a2_cdf", 0.15, 5.0, 4.3, 2.15, d_cdf, "Absolute-error CDF")
cimg(s, "a2_nasa", 4.5, 5.0, 3.4, 2.15, d_nasa, "NASA score per engine")
tile(s, 7.95, 5.0, 5.2, 2.15, "Reading",
     [f"RMSE {rng(rm)} cycles on capped labels; raw-label RMSE {rng([ST[f]['rmse_raw'] for f in FDS])} is inflated by engines with true RUL above the cap.",
      f"Accuracy is best near end of life ({rng(b30)} cycles at RUL <= 30) and degrades mid-life ({rng(bmid)} at RUL 60 to 125).",
      f"Late predictions (the unsafe side) are {min(ST[f]['late'] for f in FDS):.0%} to {max(ST[f]['late'] for f in FDS):.0%} of engines before any conservative adjustment."], TEAL, 8.5)

# ================================================================== A3 model selection
ab, md = opt(ch_abl, 4.3, 3.0), opt(ch_models, 4.4, 3.0)
vl = opt(ch_valid, 4.2, 3.0) if S else None
s = frame("Model selection evidence", TEAL, "rul_study*.json, rul_boost*.json, final_summary.json; table values copied from docs/SYSTEM_LOG.md section 6.3")
pimg(s, ab, 0.15, 0.75, 4.3, 3.0)
pimg(s, md, 4.55, 0.75, 4.4, 3.0)
pimg(s, vl, 9.05, 0.75, 4.1, 3.0)
rows = [["Experiment", "FD001", "FD002", "FD003", "FD004", "Decision"],
        ["Tuned XGBoost, CV RMSE default to tuned", "13.85 to 13.61", "17.06 to 16.84", "13.61 to 13.12", "15.81 to 15.69", "selection-biased; not adopted"],
        ["Window 10 features, CV RMSE without to with", "14.04 to 13.85", "17.18 to 17.06", "13.72 to 13.61", "15.98 to 15.81", "kept"],
        ["Window 10 features, test RMSE without to with", "12.89 to 13.18", "14.62 to 14.27", "13.62 to 13.96", "14.20 to 14.24", "test difference is noise"],
        ["ExtraTrees blend, test RMSE", "12.98", "14.17", "13.98", "14.51", "CV chose XGBoost alone"],
        ["GRU blend minus XGBoost, paired 95% CI", "[-1.42, 0.63]", "[-0.50, 0.53]", "[-1.31, -0.01]", "[-0.75, 0.38]", "not adopted"],
        ["Final XGBoost test RMSE (recomputed)"] + [f"{ST[f]['rmse']:.2f}" for f in FDS] + ["deployed"]]
table(s, 0.15, 3.9, 8.9, 3.2, rows, [3.0, 1.1, 1.1, 1.1, 1.1, 1.8], 8)
tile(s, 9.15, 3.9, 4.0, 3.2, "Reading",
     ["Slope features carry most of the signal; window-10 features give a small consistent CV gain and a test-neutral result.",
      "Tuning, an ExtraTrees blend, causal smoothing and a GRU blend did not give a reliable gain under engine-bootstrap intervals.",
      "Baseline models come from a single earlier run; selection used engine-grouped CV, never the test score."], TEAL, 8.5)

# ================================================================== A4 conformal calibration
s = frame("Conformal bound: calibration on training engines", NAVY, "artifacts/ext_conformal.json (nested engine split, training out-of-fold predictions only)")
rows = [["Subset", "Variant", "Nominal", "Held-out coverage (SD)", "Band coverage <30 / 30-60 / >=60", "Selected"]]
for f in FDS:
    for k, v in conf[f]["nested"].items():
        var, al = k.split("|")
        rows.append([f, var, f"{1 - float(al):.2f}", f"{v['cut_cov_mean']:.3f} ({v['cut_cov_sd']:.3f})", " / ".join(f"{x:.2f}" for x in v["cut_band_cov"]), "yes" if var == conf[f]["variant"] else ""])
table(s, 0.15, 0.75, 7.4, 3.9, rows, [0.7, 1.6, 0.7, 1.4, 2.0, 0.7], 7.5)
cimg(s, "a4_nom", 7.7, 0.75, 5.45, 1.9, d_nom, None, 1, 2)
cimg(s, "a4_bc", 7.7, 2.7, 5.45, 1.95, d_bandcov, "Held-out coverage by predicted-RUL band")
cimg(s, "a4_q", 0.15, 4.75, 8.4, 2.35, d_qband, "Bound offset q by predicted-RUL band", 1, 4)
tile(s, 8.65, 4.75, 4.5, 2.35, "Selection rule",
     ["Variants compared on a 50/50 nested engine split of training out-of-fold predictions only.",
      "Rule: lowest sum of marginal and per-band coverage deviation under cut-based evaluation.",
      "The official test set is read once, after the variant is frozen."], NAVY, 8.5)

# ================================================================== A5 conformal test
s = frame("Conformal bound: official test coverage", NAVY, "artifacts/ext_conformal.json; 95% engine-bootstrap intervals; capped labels")
rows = [["Subset", "Method", "Coverage", "95% CI", "Below cap", "Slack", "Late", "RMSE", "NASA"]]
for f in FDS:
    for m in M:
        r = conf[f]["test"][m]
        lo, hi = r["ci"]["marg"]
        rows.append([f, MLAB[m], f"{r['cov_cap']:.3f}", f"{lo:.2f} to {hi:.2f}", f"{r['cov_lt_cap']:.3f}", f"{r['slack_mean']:.1f}", f"{r['late_rate']:.2f}", f"{r['rmse_cap']:.1f}", f"{r['nasa_cap']:.0f}"])
table(s, 0.15, 0.75, 7.1, 6.35, rows, [0.7, 1.4, 0.8, 1.1, 0.8, 0.6, 0.6, 0.6, 0.7], 7.5)
cimg(s, "a5_ci", 7.35, 0.75, 5.8, 2.15, d_covci, "Coverage with 95% engine bootstrap CI")
cimg(s, "a5_sl", 7.35, 2.95, 5.8, 2.1, d_slack, "Coverage below cap vs mean slack")
cimg(s, "a5_na", 7.35, 5.1, 5.8, 2.0, d_nasarat, "NASA score relative to the raw prediction")

# ================================================================== A6 cost replay
s = frame("Cost-aware policy replay", ORANGE, "artifacts/ext_cost_policy.json; training trajectories; cross-fitted out-of-fold signals")
pimg(s, opt(ch_cost4, 6.4, 3.3), 0.15, 0.75, 6.4, 3.3)
cimg(s, "a6_atr", 0.15, 4.1, 6.4, 2.05, d_atr, None, 1, 2)
tile(s, 0.15, 6.2, 6.4, 0.95, "Replay assumptions", [cost[FDS[0]]["assumptions"] + ". Cost in C_waste units per engine; r = C_fail / C_waste."], ORANGE, 7.8)
cimg(s, "a6_be", 6.7, 0.75, 6.45, 2.5, d_be, "Break-even r versus run-to-failure")
rows = [["Subset", "Signal", "Break-even r vs run-to-failure", "vs best fixed interval"]]
for f in FDS:
    for k in ("rul_raw", "rul_offset", "rul_conformal10"):
        b = cost[f]["break_even"][k]
        rows.append([f, PLAB[k].replace("RUL threshold, ", ""), fmt_be(b["vs_rtf"]), fmt_be(b["vs_fixed"])])
table(s, 6.7, 3.35, 6.45, 3.75, rows, [0.7, 1.0, 2.0, 2.0], 8)

# ================================================================== A7 fixed thresholds
if all(cost[f].get("fixed_thr") for f in FDS):
    s = frame("Cost replay at thresholds fixed in advance", ORANGE, "artifacts/ext_cost_policy.json (thresholds 15, 30, 60 cycles; no sweep)")
    cimg(s, "a7_fa", 0.15, 0.75, 13.0, 2.0, d_fixed("failures_avoided"), "Share of engines maintained before failure", 1, 4)
    cimg(s, "a7_mw", 0.15, 2.85, 13.0, 2.0, d_fixed("mean_wasted"), "Mean wasted life at maintenance (cycles, all engines)", 1, 4)
    cimg(s, "a7_c1", 0.15, 4.95, 13.0, 2.0, d_fixed("cost100"), "Cost at r = 100 (C_waste units per engine)", 1, 4)
    text(s, 0.15, 6.97, 13.0, 0.22, [("Thresholds are not tuned on data. Lines: raw prediction, 0.8-quantile offset, conformal 90% bound.", 7.5, False, "334155")])
else:
    print("fixed-threshold slide skipped: rerun scripts/11_cost_policy.py")

# ================================================================== A8 anomaly
if FA:
    s = frame("Anomaly detection", TEAL, "artifacts/models/FD00X/anomaly_metrics.json (deployed Isolation Forest on latent and error)")
    for i, (nm, fn, ttl) in enumerate([("a8_auc", d_auroc, "AUROC by scorer"), ("a8_tf", d_tprfpr, "Persistent TPR and FPR"), ("a8_ld", d_lead, "Alert lead time"),
                                       ("a8_out", d_outcome, "Eligible-engine outcome"), ("a8_k", d_fprk, "Persistent FPR vs k")]):
        cimg(s, nm, 0.15 + i * 2.62, 0.75, 2.55, 2.55, fn, ttl)
    rows = [["Subset", "AUROC HI", "AUROC IFe", "AUROC IFl", "Persist TPR", "Persist FPR", "Lead median", "Lead p10", "Engine false alarm", "Unassessable"]]
    for f in FA:
        m = anom[f]
        rows.append([f, f"{m['auroc_HI']:.4f}", f"{m['auroc_IF_error_1d']:.4f}", f"{m['auroc_IF_latent_error']:.4f}", f"{m['IFl_tpr_persist']:.3f}", f"{m['IFl_fpr_persist']:.4f}", f"{m['IFl_lead_median']:.1f}", f"{m['IFl_lead_p10']:.1f}", f"{m['IFl_eng_fa_rate']:.3f}", int(m["eng_unassessable"])])
    table(s, 0.15, 3.4, 13.0, 1.6, rows, None, 9)
    tile(s, 0.15, 5.1, 6.4, 2.05, "Definitions",
         ["Positives: rows with RUL <= 30 after cycle 30. Negatives: rows at the RUL cap. Both are benchmark proxies; no failure-onset labels exist.",
          "Thresholds and k are frozen from calibration engines; persistence requires consecutive flagged cycles."], TEAL, 8.5)
    tile(s, 6.75, 5.1, 6.4, 2.05, "Reading",
         [f"Persistent true-positive rate {rng(tp, '.2f')}; lowest on {FA[int(np.argmin(tp))]}.",
          "Rows before cycle 30 are false-alarm heavy and are routed to REVIEW.",
          "Anomaly output indicates unusual behaviour, not a confirmed fault."], ORANGE, 8.5)

# ================================================================== A9 explainability
fs_ = [f for f in FDS if ex.get(f, {}).get("faithfulness")]
if fs_:
    def d_faith(axs):
        for ax, key, ttl in zip(axs[0], ("remove_diff", "keep_diff"), ("Remove top-k minus random-k (faithful if above 0)", "Keep top-k minus random-k (faithful if below 0)")):
            for i, f in enumerate(fs_):
                kd = ex[f]["faithfulness"]["k"]
                ks = list(kd)
                m_ = np.array([kd[k][key][0] for k in ks])
                lo = np.array([kd[k][key][1] for k in ks])
                hi = np.array([kd[k][key][2] for k in ks])
                ax.errorbar(np.arange(len(ks)) + (i - (len(fs_) - 1) / 2) * 0.15, m_, yerr=[m_ - lo, hi - m_], fmt="o", ms=3, capsize=2, lw=0.8, color=H(SUBF[f]), label=f)
            ax.axhline(0, c=H(GREY), lw=0.8)
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels([f"k={k}" for k in ks])
            ax.set_title(ttl)
        axs[0, 0].set_ylabel("RMSE difference (cycles)")
    s = frame("Explainability: attribution and retraining faithfulness", TEAL, "artifacts/ext_explain.json; XGBoost exact TreeSHAP; retraining with 300 trees, engine-grouped 5-fold")
    pimg(s, opt(ch_shap, 6.4, 3.3, fs_), 0.15, 0.75, 6.4, 3.3)
    cimg(s, "a9_f", 6.7, 0.75, 6.45, 3.3, d_faith)
    vt = []
    for f in fs_:
        kd = ex[f]["faithfulness"]["k"]
        zero = all(v["remove_diff"][1] <= 0 <= v["remove_diff"][2] for v in kd.values())
        kp_ = [k for k, v in kd.items() if v["keep_diff"][2] < 0]
        vt.append(f"{f}: remove-top " + ("intervals all contain zero" if zero else "is worse than random for some k") + ("; keep-only-top better than random for k = " + ", ".join(kp_) if kp_ else "; keep-top not distinguishable from random") + ".")
    miss = [f for f in FDS if f not in fs_]
    if miss:
        vt.append("Not yet computed: " + ", ".join(miss) + ".")
    vt.append("Redundant sensors are a plausible explanation (HYPOTHESIS). SHAP ranks describe the fitted model, not physical cause.")
    tile(s, 0.15, 4.15, 4.3, 2.95, "Verdict (computed from stored intervals)", vt, TEAL, 8.5)
    rows = [["Subset", "k", "Remove-top minus random (95% CI)", "Keep-top minus random (95% CI)"]]
    for f in fs_:
        for k, v in ex[f]["faithfulness"]["k"].items():
            a_, b_ = v["remove_diff"], v["keep_diff"]
            rows.append([f, k, f"{a_[0]:+.2f} [{a_[1]:+.2f}, {a_[2]:+.2f}]", f"{b_[0]:+.2f} [{b_[1]:+.2f}, {b_[2]:+.2f}]"])
    table(s, 4.55, 4.15, 8.6, 2.95, rows, [0.8, 0.4, 2.6, 2.6], 7)

# ================================================================== A10 clustering
cl = {f: ex[f]["clustering"] for f in FDS if ex.get(f, {}).get("clustering")}
if cl:
    cf_ = list(cl)

    def d_sil(axs):
        ax = axs[0, 0]
        ax.scatter(range(len(cf_)), [cl[f]["silhouette"]["2"] for f in cf_], c=H(NAVY), s=28, label="silhouette k=2")
        ax.scatter(range(len(cf_)), [cl[f]["null_sil_k2_p95"] for f in cf_], c=H(RED), marker="x", s=28, label="Gaussian null p95")
        ax.set_xticks(range(len(cf_)))
        ax.set_xticklabels(cf_)

    def d_ari(axs):
        ax = axs[0, 0]
        m_ = np.array([cl[f]["ari_k2_mean"] for f in cf_])
        lo = np.array([cl[f]["ari_k2_ci"][0] for f in cf_])
        hi = np.array([cl[f]["ari_k2_ci"][1] for f in cf_])
        ax.errorbar(range(len(cf_)), m_, yerr=[m_ - lo, hi - m_], fmt="o", ms=4, capsize=3, color=H(TEAL), label="ARI (95% interval)")
        ax.set_xticks(range(len(cf_)))
        ax.set_xticklabels(cf_)
        ax.set_ylim(0, 1.05)

    def d_spc(axs):
        ax = axs[0, 0]
        ax.bar(range(len(cf_)), [cl[f].get("life_spearman_k2", 0) for f in cf_], 0.55, color=H(AMBER), label="Spearman(cluster, life)")
        ax.axhline(0, c=H(GREY), lw=0.8)
        ax.set_xticks(range(len(cf_)))
        ax.set_xticklabels(cf_)
    s = frame("Unsupervised degradation-signature clustering", TEAL, "artifacts/ext_explain.json; normalised sensor slopes over the last 50 cycles before failure")
    rows = [["Subset", "Engines", "Silhouette (k=2)", "Gaussian-null p95", "ARI (k=2)", "ARI 95% interval", "Spearman(cluster, life)", "Cluster sizes"]]
    for f, c in cl.items():
        rows.append([f, c["n_engines"], f"{c['silhouette']['2']:.3f}", f"{c['null_sil_k2_p95']:.3f}", f"{c['ari_k2_mean']:.3f}", f"{c['ari_k2_ci'][0]:.2f} to {c['ari_k2_ci'][1]:.2f}", f"{c.get('life_spearman_k2', float('nan')):+.2f}", str(c["cluster_sizes_k2"])])
    table(s, 0.15, 0.75, 13.0, 1.5, rows, None, 9)
    im = art(cfg, "figures", "ext_clusters.png")
    if im.exists():
        pic(s, str(im), 0.15, 2.3, 12.7, 2.86)
    cimg(s, "a10_s", 0.15, 5.25, 3.2, 1.9, d_sil)
    cimg(s, "a10_a", 3.4, 5.25, 3.2, 1.9, d_ari)
    cimg(s, "a10_p", 6.65, 5.25, 3.2, 1.9, d_spc)
    tile(s, 9.9, 5.25, 3.25, 1.9, "Reading",
         ["FD001 and FD002 (one fault mode) are negative controls; FD003 and FD004 exceed the Gaussian-covariance null most clearly.",
          "Clusters track engine life, so degradation speed is a confound. Fan versus compressor mapping is a HYPOTHESIS."], TEAL, 7.2)

# ================================================================== A11 triage and work orders
t = cfg["triage"]
SEVL = ["REVIEW", "MEDIUM", "HIGH", "CRITICAL"]
SEVCOL = {"REVIEW": GREY, "MEDIUM": AMBER, "HIGH": ORANGE, "CRITICAL": RED}


def d_wo(axs):
    ax = axs[0, 0]
    bot = np.zeros(4)
    for sv in SEVL:
        v = np.array([CNT[f][sv] for f in FDS], float)
        ax.bar(range(4), v, 0.6, bottom=bot, color=H(SEVCOL[sv]), label=sv)
        bot += v
    ax.set_xticks(range(4))
    ax.set_xticklabels(FDS)
    ax.set_ylabel("Simulated work orders")


def d_maxsev(axs):
    ax = axs[0, 0]
    cnt = np.zeros((5, 4))
    for (f, e_), lv in mx.items():
        cnt[lv + 1, FDS.index(f)] += 1
    cnt[0] = [ST[f]["n"] - cnt[1:, i].sum() for i, f in enumerate(FDS)]
    bot = np.zeros(4)
    for j, (lab, col) in enumerate([("no order", GREEN), ("REVIEW", GREY), ("MEDIUM", AMBER), ("HIGH", ORANGE), ("CRITICAL", RED)]):
        ax.bar(range(4), cnt[j], 0.6, bottom=bot, color=H(col), label=lab)
        bot += cnt[j]
    ax.set_xticks(range(4))
    ax.set_xticklabels(FDS)
    ax.set_ylabel("Engines by highest severity")


s = frame("Triage rules and simulated work orders", ORANGE, f"artifacts/{WOB.name}/FD00X/WO-*.json; configs/config.yaml (thresholds provisional)")
rows = [["Subset", "REVIEW", "MEDIUM", "HIGH", "CRITICAL", "Orders", "Engines with order", "Engines tested"]]
for f in FDS:
    c = CNT[f]
    rows.append([f, c["REVIEW"], c["MEDIUM"], c["HIGH"], c["CRITICAL"], sum(c.values()), sum(1 for k in mx if k[0] == f), ST[f]["n"]])
rows.append(["Total"] + [sum(CNT[f][sv] for f in FDS) for sv in SEVL] + [sum(sum(CNT[f].values()) for f in FDS), N_ORD, NTOT])
table(s, 0.15, 0.75, 7.6, 2.1, rows, None, 9)
cimg(s, "a11_wo", 7.9, 0.75, 5.25, 2.1, d_wo, "Simulated work orders by severity")
rows = [["Severity", "Rule (first match)"],
        ["CRITICAL", f"Conservative RUL <= {t['critical_rul']}, or persistent anomaly at or above the critical score cut"],
        ["HIGH", f"Conservative RUL <= {t['high_rul']}, or persistent anomaly at or above the high score cut"],
        ["MEDIUM", f"Conservative RUL <= {t['medium_rul']}, or persistent anomaly at or above the medium score cut"],
        ["LOW", "None of the above"],
        ["REVIEW", "Invalid input (NaN, infinite, negative) or fewer than 30 cycles of history; never LOW"]]
table(s, 0.15, 3.0, 13.0, 1.9, rows, [1.0, 9.0], 9.5)
cimg(s, "a11_mx", 0.15, 5.0, 4.6, 2.15, d_maxsev)
tile(s, 4.85, 5.0, 8.3, 2.15, "Work-order lifecycle",
     [f"Persistence k = {cfg['anomaly']['persist_k']} consecutive cycles. Status OPEN_SIMULATED, deterministic identifiers, one order per engine per severity level, issued only on escalation, restart-safe.",
      "Conservative RUL is the base prediction minus the 0.8 quantile of out-of-fold over-prediction per predicted-RUL band. The conformal bound is not yet wired into the deployed triage."], ORANGE, 9)

# ================================================================== A12 triage evaluation
cf = opt(ch_conf4, 6.4, 2.5) if TR else None
ld = opt(ch_lead2, 6.4, 2.5) if TR else None
if TR and cf and ld:
    def d_trl(axs):
        ax = axs[0, 0]
        ax.bar(np.arange(4) - 0.2, [TR[f]["rul_base"]["late_rate"] for f in FDS], 0.4, color=H(GREY), label="base")
        ax.bar(np.arange(4) + 0.2, [TR[f]["rul_cons"]["late_rate"] for f in FDS], 0.4, color=H(TEAL), label="conservative")
        ax.set_xticks(range(4))
        ax.set_xticklabels(FDS)
        ax.set_ylabel("Late-prediction rate")

    def d_trn(axs):
        ax = axs[0, 0]
        ax.bar(np.arange(4) - 0.2, [TR[f]["rul_base"]["nasa"] for f in FDS], 0.4, color=H(GREY), label="base")
        ax.bar(np.arange(4) + 0.2, [TR[f]["rul_cons"]["nasa"] for f in FDS], 0.4, color=H(TEAL), label="conservative")
        ax.set_xticks(range(4))
        ax.set_xticklabels(FDS)
        ax.set_ylabel("Last-cycle NASA score")
    s = frame("Triage evaluation at the last observed cycle", ORANGE, "artifacts/triage_v2_summary.json (true buckets reuse the triage cutoffs: assumption)")
    rows = [["Subset", "Engines", "True RUL<=30", "Triaged HIGH+", "Late base", "Late conservative", "NASA base", "NASA conservative", "REVIEW"]]
    for f in FDS:
        r = TR[f]
        rows.append([f, r["n_engines"], r["last_cycle"]["n_le30"], r["last_cycle"]["high_plus_of_le30"], f"{r['rul_base']['late_rate']:.2f}", f"{r['rul_cons']['late_rate']:.2f}", f"{r['rul_base']['nasa']:.0f}", f"{r['rul_cons']['nasa']:.0f}", r["n_review_last_cycle"]])
    table(s, 0.15, 0.75, 13.0, 1.6, rows, None, 9)
    pimg(s, cf, 0.15, 2.45, 6.4, 2.5)
    pimg(s, ld, 6.75, 2.45, 6.4, 2.5)
    cimg(s, "a12_l", 0.15, 5.05, 4.2, 2.1, d_trl)
    cimg(s, "a12_n", 4.45, 5.05, 4.2, 2.1, d_trn)
    worse = [f for f in FDS if TR[f]["rul_cons"]["nasa"] > TR[f]["rul_base"]["nasa"]]
    tile(s, 8.75, 5.05, 4.4, 2.1, "Reading",
         ["Recall near failure is driven mainly by the RUL model, which is most accurate there.",
          "The conservative offset lowers late predictions" + (f" but raises last-cycle NASA on {', '.join(worse)}: a safety trade-off, not an accuracy gain." if worse else " and did not raise NASA on any subset."),
          "Confusion cells: L, M, H, C; rows true bucket, columns triage severity."], ORANGE, 8)

# ================================================================== A13 retraining and work order
s = frame("Retraining demonstration and a sample work order", GREEN, "artifacts/retrain_demo/*.json (FD001, SYNTHETIC drift); persisted work-order JSON")
rd = sorted(art(cfg, "retrain_demo", "x").parent.glob("*.json"))
if rd:
    R_ = sorted((json.loads(p_.read_text()) for p_ in rd), key=lambda r: (r["synthetic_drift_std"], r["days_since"]))
    labs, rows = [], [["Scenario", "Max PSI", "v1 val RMSE", "v1 arrivals RMSE", "Trigger", "v2 val RMSE", "v2 promoted"]]
    for r in R_:
        lab = f"Drift {r['synthetic_drift_std']:g} sd" if r["synthetic_drift_std"] else (f"Scheduled {r['days_since']} d" if r["days_since"] else "No drift")
        labs.append(lab)
        trig = "+".join(k.upper() for k, v in r["trigger"].items() if v) or "none"
        rows.append([lab, f"{r['max_psi']:.2f}", f"{r['v1_val']['rmse']:.1f}", f"{r['v1_arrivals']['rmse']:.1f}", trig, f"{r['v2_val']['rmse']:.1f}" if "v2_val" in r else "n/a", ("yes" if r["v2_promoted"] else "no") if "v2_val" in r else "n/a"])
    table(s, 0.15, 0.75, 6.6, 1.7, rows, [1.5, 0.6, 0.8, 0.9, 0.8, 0.8, 0.7], 8)

    def d_rt(axs):
        ax = axs[0, 0]
        x = np.arange(len(R_))
        ax.bar(x - 0.27, [r["v1_val"]["rmse"] for r in R_], 0.27, color=H(GREY), label="v1 validation")
        ax.bar(x, [r["v1_arrivals"]["rmse"] for r in R_], 0.27, color=H(AMBER), label="v1 arrivals")
        ax.bar(x + 0.27, [r["v2_val"]["rmse"] if "v2_val" in r else 0 for r in R_], 0.27, color=H(GREEN), label="v2 validation")
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.set_ylabel("RMSE (cycles)")
    cimg(s, "a13_rt", 0.15, 2.55, 6.6, 2.3, d_rt, "Retraining scenarios (log scale)")
tile(s, 0.15, 4.95, 6.6, 2.2, "Reading",
     ["Triggers: RMSE above 1.25 times baseline, 7 days elapsed, or PSI above 0.25 on any feature. Promotion only if the candidate beats the champion on engine-held-out validation RMSE; the previous champion stays under the registry alias.",
      "Drift is injected, not observed. The validation set is shifted too, so only the PSI trigger fires. v2 also trains on 60% more labelled engines. Shown on FD001 only."], GREEN, 8.3)
wo = WOF.get("FD001")
if wo:
    keys = ["work_order_id", "incident_key", "status", "simulated", "engine_id", "replay_cycle", "current_cycle", "rul_mean", "rul_conservative", "anomaly_score", "health_index", "persistent_anomaly", "severity", "rule_fired", "recommendation", "requires_human_review", "model_version", "top_degradation_sensors"]
    ks = [k for k in keys if k in wo] or list(wo)
    sec(s, 6.9, 0.72, 6.25, "Simulated work order (FD001 replay, persisted JSON)", PURPLE)
    bx = rect(s, 6.9, 1.1, 6.25, 6.05, LIGHT)
    put(bx, [(f"{k}: {', '.join(map(str, wo[k])) if isinstance(wo[k], list) else wo[k]}", 8.5, False, INK) for k in ks], anchor=MSO_ANCHOR.TOP, m=0.12, font="Courier New")

# ================================================================== A14 methodology and sources
rl = cfg["rul"]
s = frame("Methodology and sources", NAVY, "configs/config.yaml; scripts 02, 03b, 09, 10, 11; references on this slide")
tile(s, 0.15, 0.75, 6.4, 3.1, "Evaluation protocol",
     [f"RUL label: max_cycle minus cycle, capped at {cap_}; test labels from the official RUL files, scored at the last observed cycle.",
      f"Features: trailing window {cfg['features']['window']} ({', '.join(cfg['features']['stats'])}) plus window 10 (mean, slope); causal, per engine.",
      f"Validation: {cfg['eval']['cv_folds']}-fold GroupKFold by engine; random truncation at cycle >= {cfg['eval']['min_cut_len']}, {cfg['eval']['n_cut_reps']} repeats; test never used for selection.",
      "NASA score: sum of exp(-d/13) - 1 for d < 0 and exp(d/10) - 1 for d >= 0, d = predicted minus true RUL; lower is better."], BLUE, 9.5)
tile(s, 6.7, 0.75, 6.45, 3.1, "Anomaly, conservative RUL and conformal bound",
     [f"Autoencoder {an_['hidden'][0]}-{an_['hidden'][1]}-{an_['latent']} trained on healthy cycles {an_['healthy_cycle_range'][0]} to {an_['healthy_cycle_range'][1]}; Isolation Forest on latent state plus error; persistence k = {an_['persist_k']}.",
      f"Conservative RUL: base prediction minus the {int(rl['cons_q'] * 100)}th percentile of out-of-fold over-prediction per predicted-RUL band (<30, 30-60, >=60).",
      "Conformal bound: finite-sample (1 - alpha) quantile of the same residuals per band, calibrated on one random cut per engine."], TEAL, 9.5)
refs = ["A. Saxena, K. Goebel, D. Simon, N. Eklund, Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation, PHM08, Denver, 2008.",
        "A. Saxena et al., Metrics for Evaluating Performance of Prognostics Techniques, PHM08, Denver, 2008.",
        "D. Frederick, J. DeCastro, J. Litt, User's Guide for the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS), NASA/ARL TM-2007-215026, 2007.",
        "T. Chen, C. Guestrin, XGBoost: A Scalable Tree Boosting System, KDD, 2016.",
        "F. T. Liu, K. M. Ting, Z.-H. Zhou, Isolation Forest, IEEE ICDM, 2008.",
        "S. M. Lundberg et al., From local explanations to global understanding with explainable AI for trees, Nature Machine Intelligence 2, 2020.",
        "V. Vovk, A. Gammerman, G. Shafer, Algorithmic Learning in a Random World, Springer, 2005 (conformal prediction).",
        "F. Pedregosa et al., Scikit-learn, JMLR 12, 2011.   A. Paszke et al., PyTorch, NeurIPS, 2019.   MLflow, mlflow.org.",
        "Reference details other than the C-MAPSS and PHM08 papers are written from memory and must be verified before submission."]
tile(s, 0.15, 4.0, 13.0, 3.15, "Sources", refs, NAVY, 9.2)

# ================================================================== A15 limitations register
pl = ""
for pp in (art(cfg, "pytest_out.txt"), ROOT / "pytest_out.txt"):
    if pp.exists():
        ls_ = [l_.strip() for l_ in pp.read_text().splitlines() if l_.strip()]
        pl = ls_[-1] if ls_ else ""
        if pl:
            break
s = frame("Assumptions, limitations and open items", RED, "docs/SYSTEM_LOG.md; scripts 10, 11 and 12 audits")
rows = [["Item", "Evidence and impact", "Status", "Next step"],
        ["Scope", "Historical replay of simulated data; no real telemetry, field validation or maintenance-system integration" + (f"; unit tests: {pl}" if pl else ""), "Scope", "Pilot on real fleet data"],
        ["Triage thresholds", "Cutoffs 15, 30, 60 are provisional; true buckets reuse the same cutoffs", "Assumption", "Calibrate with maintenance cost data"],
        ["Anomaly labels", "Positives defined as RUL <= 30; no failure-onset labels exist", "Proxy", "Obtain fault-onset annotations"],
        ["Preprocessing leakage", "Regimes, normalisation and sensor selection fitted on all training engines before CV", "Unverified", "Run fold-wise refit check"],
        ["Conformal coverage", "Empirical; calibration cuts uniform in life, test cuts about 60% into life; cap-saturated engines inflate marginal coverage", "Measured", "Report below-cap coverage"],
        ["Bound in deployment", "Deployed triage uses the 0.8-quantile offset; the conformal bound is evaluated offline", "Verified", "Decide after cost review"],
        ["Cost replay", "Immediate action, failure at final cycle, no anomaly signal, no monetary cost, in-sample parameters", "Verified run", "Use real costs; CV curve given"],
        ["Explainability", "SHAP describes the model; faithfulness retraining uses 300 trees; sensors may be redundant", "Hypothesis", "Physically grouped ablation"],
        ["Fault-mode clusters", "readme lists two modes for FD003/FD004, paper text describes HPC only; clusters track engine life", "Hypothesis", "Obtain labelled fault modes"],
        ["Retraining", "FD001 only with synthetic drift; v2 trained on 60% more engines", "Verified demo", "Real drift with delayed labels"],
        ["Raw-label RMSE", "Inflated by engines with true RUL above the cap", "Verified", "Capped labels are primary"],
        ["Persistent TPR", f"Varies by subset ({rng(tp, '.2f') if tp else 'n/a'}); lowest on {FA[int(np.argmin(tp))] if tp else 'n/a'}", "Verified", "Investigate weakest subset"],
        ["Dashboard", "Read-only monitor, not validated for operational use", "Unvalidated", "Review with operators"]]
table(s, 0.15, 0.75, 13.0, 6.35, rows, [1.7, 6.2, 1.1, 2.6], 8.5)

out = art(cfg, "figures", "fleetpm_deck_v3.pptx")
prs.save(out)
print("saved", out, "| slides:", len(prs.slides))
'''

if DST.exists():
    shutil.copy2(DST, DST.with_name(DST.name + f".bak_{time.strftime('%Y%m%d_%H%M%S')}"))
DST.write_text(PRE + BODY)
py_compile.compile(str(DST), doraise=True)
print("wrote + compiled scripts/14_deck_v3.py")