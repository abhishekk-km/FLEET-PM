import _bootstrap

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from fleetpm.config import ROOT, art, load_config
from fleetpm.pipeline import load_prepared

FDS = ["FD001", "FD002", "FD003", "FD004"]
NAVY, TEAL, GOLD, GOLD2, RUST, SLATE, LIGHT = "1F3A5F", "1B998B", "E09F3E", "D08C2B", "C8553D", "5C6B7A", "EEF2F6"
PURPLE, GREEN, DARK = "7B4B94", "2E7D5B", "1B1B1B"
FDC = {"FD001": "#1F3A5F", "FD002": "#1B998B", "FD003": "#E09F3E", "FD004": "#C8553D"}
SEVC = {"LOW": "#2E7D5B", "MEDIUM": "#E09F3E", "HIGH": "#C8553D", "CRITICAL": "#7A1F3D"}
FD_DESC = {"FD001": "1c/1f", "FD002": "6c/1f", "FD003": "1c/2f", "FD004": "6c/2f"}   # operating conditions / fault modes (readme.txt)
plt.rcParams.update({"font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
                     "legend.fontsize": 5.5, "axes.linewidth": .5, "xtick.major.width": .5, "ytick.major.width": .5})


def jload(p):
    if not p.exists():
        raise FileNotFoundError(f"{p.relative_to(ROOT)} missing; run the earlier pipeline steps first")
    return json.loads(p.read_text())


def despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=1.5, pad=1)


def save(fig, path, rect=(0, 0, 1, 1)):
    fig.tight_layout(pad=.3, rect=rect)
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------- figures
def fig_pred(cfg, S, path, size):
    fig, axs = plt.subplots(2, 2, figsize=size)
    for ax, fd in zip(axs.ravel(), FDS):
        d = pd.read_csv(art(cfg, "predictions", f"rul_final_{fd}.csv"))
        ax.scatter(d.rul_true_cap, d.rul_pred, s=3, alpha=.7, color=FDC[fd], linewidths=0)
        ax.plot([0, 125], [0, 125], color="#" + SLATE, lw=.6, ls="--")
        ax.set_title(f"{fd}  RMSE {S[fd]['cap']['rmse']:.1f}  n={len(d)}", pad=1.5)
        ax.set_xlim(0, 130); ax.set_ylim(0, 130)
        despine(ax)
    fig.supxlabel("True RUL (cap 125)", fontsize=6); fig.supylabel("Predicted RUL", fontsize=6)
    save(fig, path, (.02, .03, 1, 1))


def fig_band(S, path, size):
    names = [("le30", "RUL<=30"), ("30_60", "30<RUL<=60"), ("60_cap", "60<RUL<=125"), ("gt_cap", "RUL>125 (vs uncapped)")]
    x, w = np.arange(len(names)), .2
    fig, ax = plt.subplots(figsize=size)
    for i, fd in enumerate(FDS):
        ax.bar(x + (i - 1.5) * w, [S[fd]["bands"][k]["rmse"] for k, _ in names], w, color=FDC[fd], label=fd)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in names]); ax.set_ylabel("RMSE (cycles)")
    ax.legend(ncol=4, frameon=False, loc="upper left")
    ax.set_ylim(0, max(S[f]["bands"]["gt_cap"]["rmse"] for f in FDS) * 1.2)
    despine(ax)
    save(fig, path)


def fig_valid(S, path, size):
    x, w = np.arange(len(FDS)), .36
    cv = np.array([S[f]["cv_rmse_k1"] for f in FDS]); te = np.array([S[f]["cap"]["rmse"] for f in FDS])
    ci = np.array([S[f]["cap"]["ci"] for f in FDS])
    lo, hi = np.maximum(0, te - ci[:, 0]), np.maximum(0, ci[:, 1] - te)
    fig, ax = plt.subplots(figsize=size)
    ax.bar(x - w / 2, cv, w, color="#" + NAVY, label="Engine-grouped 5-fold CV")
    ax.bar(x + w / 2, te, w, color="#" + TEAL, yerr=[lo, hi], capsize=1.5, error_kw={"lw": .6}, label="Official test, 95% bootstrap CI")
    for xi, a, b, h in zip(x, cv, te, hi):
        ax.text(xi - w / 2, a + .3, f"{a:.1f}", ha="center", fontsize=5.5)
        ax.text(xi + w / 2, b + h + .3, f"{b:.1f}", ha="center", fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(FDS); ax.set_ylabel("RMSE (cycles)")
    ax.set_ylim(0, (ci[:, 1].max()) * 1.5)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    despine(ax)
    save(fig, path)


def fig_anom(cfg, path, size):
    am = {fd: jload(art(cfg, "models", fd, "anomaly_metrics.json")) for fd in FDS}
    va = {}
    for fd in FDS:
        R = pd.read_csv(art(cfg, "validation", f"{fd}_validation_runs.csv"))
        s = R[(R["mode"] == "pooled") & (R["det"] == "IFl")]
        va[fd] = (s.tpr_persist.mean(), s.det_rate.mean(), s.fpr_persist.mean())
    x = np.arange(4)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=size, gridspec_kw={"width_ratios": [1, 1.3]})
    a1.bar(x - .18, [am[f]["auroc_HI"] for f in FDS], .36, color="#" + NAVY, label="AE health index")
    a1.bar(x + .18, [am[f]["auroc_IF_latent_error"] for f in FDS], .36, color="#" + GOLD, label="IF on latent+error")
    a1.set_ylim(.9, 1.0); a1.set_xticks(x); a1.set_xticklabels([f[2:] for f in FDS]); a1.set_title("AUROC (RUL<=30 vs healthy)", pad=1.5)
    a1.legend(frameon=False, loc="lower left"); despine(a1)
    for j, (nm, col) in enumerate((("Persistent TPR", TEAL), ("Eligible-engine detection", NAVY), ("Persistent FPR", RUST))):
        a2.bar(x + (j - 1) * .26, [va[f][j] for f in FDS], .26, color="#" + col, label=nm)
    a2.set_ylim(0, 1.25); a2.set_xticks(x); a2.set_xticklabels([f[2:] for f in FDS]); a2.set_title("Frozen-threshold validation (IF, mean of splits)", pad=1.5)
    a2.legend(frameon=False, loc="upper left", ncol=1); despine(a2)
    save(fig, path)
    return va


def fig_conf(TR, path, size):
    fig, axs = plt.subplots(1, 4, figsize=size)
    for k, (ax, fd) in enumerate(zip(axs, FDS)):
        c = np.array(TR[fd]["confusion"]); rs = np.maximum(c.sum(1, keepdims=True), 1)
        ax.imshow(c / rs, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, int(c[i, j]), ha="center", va="center", fontsize=5.5, color="white" if c[i, j] / rs[i, 0] > .5 else "black")
        ax.set_xticks(range(4)); ax.set_xticklabels(list("LMHC")); ax.set_yticks(range(4))
        ax.set_yticklabels(list("LMHC") if k == 0 else [])
        ax.set_title(fd, pad=1.5); ax.set_xlabel("triage severity", labelpad=1)
        if k == 0:
            ax.set_ylabel("true bucket", labelpad=1)
        ax.tick_params(length=1, pad=1)
    save(fig, path)


def fig_lead(TR, path, size):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=size, gridspec_kw={"width_ratios": [1.2, 1]})
    data = [TR[f]["lead"] if TR[f]["lead"] else [np.nan] for f in FDS]
    bp = a1.boxplot(data, widths=.55, patch_artist=True, flierprops={"markersize": 1.5}, medianprops={"color": "black", "lw": .8},
                    whiskerprops={"lw": .5}, capprops={"lw": .5}, boxprops={"lw": .5})
    for patch, fd in zip(bp["boxes"], FDS):
        patch.set_facecolor(FDC[fd]); patch.set_alpha(.85)
    a1.set_xticks(range(1, 5)); a1.set_xticklabels(FDS)
    a1.axhline(30, color="#" + RUST, lw=.6, ls="--")
    a1.set_title("True RUL at first HIGH or CRITICAL", pad=1.5); a1.set_ylabel("cycles before failure"); despine(a1)
    bottom = np.zeros(4)
    for s in ("MEDIUM", "HIGH", "CRITICAL"):
        v = np.array([TR[f]["wo_by_sev"].get(s, 0) for f in FDS], float)
        a2.bar(range(4), v, .6, bottom=bottom, color=SEVC[s], label=s)
        bottom += v
    a2.set_xticks(range(4)); a2.set_xticklabels([f[2:] for f in FDS]); a2.set_title("Simulated work orders (replay)", pad=1.5)
    a2.legend(frameon=False, loc="upper left"); despine(a2)
    save(fig, path)


def fig_abl(cfg, path, size):
    ns = len(load_prepared(cfg, "FD001")["pre"].sensors)
    m1, m2 = jload(art(cfg, "models", "FD001", "rul_study_abl_mean.json")), jload(art(cfg, "models", "FD001", "rul_study_abl_mean_slope.json"))
    m3, m4 = jload(art(cfg, "models", "FD001", "rul_boost_nowin.json"))["candidates"], jload(art(cfg, "models", "FD001", "rul_boost_gru.json"))["candidates"]
    xg = [m1["xgb"]["cv_rmse"], m2["xgb"]["cv_rmse"], m3["xgb"]["cv_rmse_by_k"]["1"], m4["xgb"]["cv_rmse_by_k"]["1"]]
    et = [m1["extratrees"]["cv_rmse"], m2["extratrees"]["cv_rmse"], m3["extratrees"]["cv_rmse_by_k"]["1"], m4["extratrees"]["cv_rmse_by_k"]["1"]]
    labs = [f"mean\n({ns})", f"+slope\n({2 * ns})", f"+std,min,\nmax,kurt\n({6 * ns})", f"+window 10\n({8 * ns})"]
    x, w = np.arange(4), .36
    fig, ax = plt.subplots(figsize=size)
    ax.bar(x - w / 2, xg, w, color="#" + NAVY, label="XGBoost"); ax.bar(x + w / 2, et, w, color="#" + GOLD, label="ExtraTrees")
    for xi, a, b in zip(x, xg, et):
        ax.text(xi - w / 2, a + .3, f"{a:.1f}", ha="center", fontsize=5.5); ax.text(xi + w / 2, b + .3, f"{b:.1f}", ha="center", fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel("CV RMSE (cycles)"); ax.set_ylim(0, max(xg + et) * 1.2)
    ax.set_title("FD001 feature ablation, CV RMSE (bars 1-2 fold-averaged, 3-4 pooled out-of-fold)", pad=2); ax.legend(frameon=False, loc="upper right")
    despine(ax)
    save(fig, path)


def fig_models(cfg, S, path, size):
    names = [("ridge", "Ridge"), ("extratrees", "ExtraTrees"), ("lstm", "LSTM"), ("gru", "GRU"), ("cnn1d", "CNN1D"), ("cnnlstm", "CNN-LSTM")]
    st = {fd: jload(art(cfg, "models", fd, "rul_study.json")) for fd in FDS}
    cols = ["#9AA5B1", "#E09F3E", "#7B4B94", "#2E7D5B", "#D08C2B", "#C8553D"]
    x, w = np.arange(4), .115
    fig, ax = plt.subplots(figsize=size)
    mx = 0
    for i, (k, n) in enumerate(names):
        v = [st[fd][k]["test_cap_rmse"] for fd in FDS]; mx = max(mx, max(v))
        ax.bar(x + (i - 3) * w, v, w, color=cols[i], label=n)
    ax.bar(x + 3 * w, [S[fd]["cap"]["rmse"] for fd in FDS], w, color="#" + NAVY, label="XGBoost (final)")
    ax.set_xticks(x); ax.set_xticklabels(FDS); ax.set_ylabel("Test RMSE, capped (cycles)"); ax.set_ylim(0, mx * 1.35)
    ax.set_title("Model comparison (baselines: single run from earlier study; selection by CV)", pad=2)
    ax.legend(frameon=False, ncol=4, loc="upper left"); despine(ax)
    save(fig, path)


# ----------------------------------------------------------------------------- pptx helpers
def rgb(h):
    return RGBColor.from_string(h)


def write(shape, items, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margin=.04, font="Calibri"):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(.02)
    for i, (t, size, bold, color) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = t
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, font
        r.font.color.rgb = rgb(color)


def box(slide, kind, x, y, w, h, fill):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def label(slide, x, y, w, txt, col=NAVY):
    box(slide, MSO_SHAPE.RECTANGLE, x, y + .02, .05, .16, col)
    tb = slide.shapes.add_textbox(Inches(x + .09), Inches(y), Inches(w - .09), Inches(.2))
    write(tb, [(txt, 8, True, NAVY)], align=PP_ALIGN.LEFT, margin=0)


def pic(slide, path, x, y, w):
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))


def table(slide, x, y, rows, colw, size=7, hdr_h=.34, row_h=.24, first_left=True):
    gs = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(sum(colw)), Inches(hdr_h + row_h * (len(rows) - 1)))
    t = gs.table
    for j, cw in enumerate(colw):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(hdr_h if i == 0 else row_h)
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(NAVY if i == 0 else (LIGHT if i % 2 else "FFFFFF"))
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(.04)
            c.margin_top = c.margin_bottom = Inches(.01)
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if (j == 0 and first_left) else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(v)
            r.font.size, r.font.name = Pt(size), "Calibri"
            r.font.bold = i == 0 or j == 0
            r.font.color.rgb = rgb("FFFFFF" if i == 0 else DARK)


def bullets(slide, x, y, w, h, items, size=6.5, color=DARK, fill=None):
    if fill:
        box(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill)
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    write(tb, [("\u2022 " + t, size, False, color) for t in items], align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, margin=.06)


def title_bar(slide, title, sub):
    bar = box(slide, MSO_SHAPE.RECTANGLE, 0, 0, 13.333, .72, NAVY)
    write(bar, [(title, 18, True, "FFFFFF"), (sub, 9, False, "D6E2F0")], align=PP_ALIGN.LEFT, margin=.3)


def kpi(slide, i, val, lab, icon, col):
    x = .3 + i * 2.13
    box(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, .8, 2.05, .72, LIGHT)
    box(slide, icon, x + .1, 1.03, .24, .24, col)
    tb = slide.shapes.add_textbox(Inches(x + .4), Inches(.82), Inches(1.62), Inches(.68))
    write(tb, [(val, 13, True, NAVY), (lab, 6.3, False, SLATE)], align=PP_ALIGN.LEFT, margin=.02)


# ----------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    S = jload(art(cfg, "final_summary.json"))
    TR = jload(art(cfg, "triage_v2_summary.json"))
    if list(S) != FDS or list(TR) != FDS:
        raise ValueError("final_summary.json or triage_v2_summary.json does not contain exactly FD001-FD004")
    fd_ = art(cfg, "figures", "x").parent
    f = {k: fd_ / f"fig_{k}.png" for k in ("pred", "band", "valid", "anom", "conf", "lead", "abl", "models")}
    fig_pred(cfg, S, f["pred"], (3.9, 2.0)); fig_band(S, f["band"], (3.9, 1.35)); fig_valid(S, f["valid"], (3.9, 1.35))
    va = fig_anom(cfg, f["anom"], (3.75, 1.5)); fig_conf(TR, f["conf"], (3.75, 1.45)); fig_lead(TR, f["lead"], (3.75, 1.75))
    fig_abl(cfg, f["abl"], (3.9, 2.3)); fig_models(cfg, S, f["models"], (4.35, 2.3))

    ft, an, ev, tg, rl = cfg["features"], cfg["anomaly"], cfg["eval"], cfg["triage"], cfg["rul"]
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # ================= slide 1
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title_bar(s, "Self-Healing Machines: Predictive Maintenance for a Connected Fleet",
              "NASA C-MAPSS benchmark replay: anomaly detection, RUL prediction, deterministic triage, simulated work orders")
    rm = [v["cap"]["rmse"] for v in S.values()]
    le = [v["bands"]["le30"]["rmse"] for v in S.values()]
    tp, fp = [va[k][0] for k in FDS], [va[k][2] for k in FDS]
    n_le = sum(TR[k]["last_cycle"]["n_le30"] for k in FDS); hp = sum(TR[k]["last_cycle"]["high_plus_of_le30"] for k in FDS)
    ne = sum(TR[k]["n_engines"] for k in FDS)
    lb = sum(TR[k]["rul_base"]["late_rate"] * TR[k]["n_engines"] for k in FDS) / ne
    lc = sum(TR[k]["rul_cons"]["late_rate"] * TR[k]["n_engines"] for k in FDS) / ne
    cards = [(f"{len(FDS)} of 4", "C-MAPSS subsets trained and evaluated independently", MSO_SHAPE.HEXAGON, NAVY),
             (f"{min(rm):.1f} to {max(rm):.1f}", "Test RMSE, cycles (RUL cap 125), XGBoost", MSO_SHAPE.DONUT, TEAL),
             (f"{min(le):.1f} to {max(le):.1f}", "RMSE, cycles, when true RUL is 30 or less", MSO_SHAPE.DIAMOND, GOLD),
             (f"{min(tp):.0%} to {max(tp):.0%}", f"Anomaly persistent TPR (RUL<=30); persistent FPR {min(fp):.1%} to {max(fp):.1%}", MSO_SHAPE.CHEVRON, RUST),
             (f"{hp} of {n_le}", "Engines with true RUL<=30 triaged HIGH or CRITICAL at last cycle", MSO_SHAPE.OVAL, PURPLE),
             (f"{lb:.0%} to {lc:.0%}", "Late-prediction rate, raw vs conservative RUL (pooled, last cycle)", MSO_SHAPE.ISOSCELES_TRIANGLE, GREEN)]
    for i, c in enumerate(cards):
        kpi(s, i, *c)

    label(s, .3, 1.62, 4.9, "System architecture (benchmark replay of historical run-to-failure data)", NAVY)
    cx = [.3 + i * 1.005 for i in range(5)]
    specs = [(0, 2.2, .95, NAVY, "C-MAPSS data", "4 subsets; schema, integrity and RUL label checks"),
             (1, 2.2, .95, TEAL, "Features", "Regime-normalised, causal rolling stats, windows 30 and 10"),
             (2, 1.85, .82, GOLD, "Anomaly", "Autoencoder + Isolation Forest, persistence k=4"),
             (2, 2.77, .82, GOLD2, "RUL", "XGBoost, cap 125, conservative offset"),
             (3, 2.2, .95, PURPLE, "Triage", "Deterministic rules, 4 severities"),
             (4, 2.2, .95, GREEN, "Work orders", "Simulated, deduplicated, persisted")]
    for c, y, h, col, t1, t2 in specs:
        b = box(s, MSO_SHAPE.ROUNDED_RECTANGLE, cx[c], y, .88, h, col)
        write(b, [(t1, 8, True, "FFFFFF"), (t2, 6.3, False, "FFFFFF")])
    for c, ys in ((0, [2.675]), (1, [2.26, 3.18]), (2, [2.26, 3.18]), (3, [2.675])):
        for y in ys:
            box(s, MSO_SHAPE.RIGHT_ARROW, cx[c] + .895, y - .09, .1, .18, SLATE)
    band = box(s, MSO_SHAPE.ROUNDED_RECTANGLE, .3, 3.66, 4.9, .28, "DDE3EA")
    write(band, [("MLflow tracking and registry  |  PSI drift and gated retraining (demonstration)  |  Dashboard (in development)", 6.3, True, NAVY)])

    label(s, .3, 3.98, 4.9, "Held-out test results: official test set, last observed cycle per engine", TEAL)
    rows = [["Subset", "Test RMSE (95% CI)", "NASA", "RMSE RUL<=30", "RMSE 30-60", "RMSE 60-125"]]
    for fd in FDS:
        c, b = S[fd]["cap"], S[fd]["bands"]
        rows.append([f"{fd} ({FD_DESC[fd]})", f"{c['rmse']:.2f} ({c['ci'][0]:.1f}-{c['ci'][1]:.1f})", f"{c['nasa']:.0f}",
                     f"{b['le30']['rmse']:.1f} ({b['le30']['n']})", f"{b['30_60']['rmse']:.1f} ({b['30_60']['n']})",
                     f"{b['60_cap']['rmse']:.1f} ({b['60_cap']['n']})"])
    table(s, .3, 4.2, rows, [1.1, 1.2, .5, .72, .68, .70], size=6.8, hdr_h=.34, row_h=.25)

    label(s, .3, 5.6, 4.9, "Triage policy: provisional thresholds, simulated actions", PURPLE)
    rules = [["Severity", "Trigger (conservative RUL, anomaly)", "Simulated action"],
             ["CRITICAL", f"RUL<={tg['critical_rul']}, or persistent anomaly above critical cut", "Remove from service, full inspection, human review"],
             ["HIGH", f"RUL<={tg['high_rul']}, or persistent anomaly above high cut", "Maintenance within 2 cycles"],
             ["MEDIUM", f"RUL<={tg['medium_rul']}, or persistent anomaly above medium cut", "Inspection within 10 cycles"],
             ["LOW", "None of the above", "No work order"]]
    table(s, .3, 5.82, rules, [.62, 2.45, 1.83], size=6.2, hdr_h=.22, row_h=.22)

    cB, cC = 5.3, 9.3
    label(s, cB, 1.62, 3.9, "Predicted vs true RUL (test engines)", TEAL); pic(s, f["pred"], cB, 1.82, 3.9)
    label(s, cB, 3.84, 3.9, "RMSE by true-RUL band", GOLD); pic(s, f["band"], cB, 4.04, 3.9)
    label(s, cB, 5.41, 3.9, "Validation: CV vs held-out test", RUST); pic(s, f["valid"], cB, 5.61, 3.9)
    label(s, cC, 1.62, 3.75, "Anomaly detection: ranking and frozen-threshold validation", GOLD); pic(s, f["anom"], cC, 1.82, 3.75)
    label(s, cC, 3.34, 3.75, "Triage at last cycle: true bucket vs severity (L, M, H, C)", PURPLE); pic(s, f["conf"], cC, 3.54, 3.75)
    label(s, cC, 5.01, 3.75, "Replay: alert timing and work orders", GREEN); pic(s, f["lead"], cC, 5.21, 3.75)

    raw = ", ".join(f"{fd} {v['raw']['rmse']:.1f}" for fd, v in S.items())
    n_eng = ", ".join(str(v["n_test_engines"]) for v in S.values())
    t = s.shapes.add_textbox(Inches(.3), Inches(7.03), Inches(12.73), Inches(.42))
    write(t, [(f"Scope: test engines {n_eng}; c = operating conditions, f = fault modes. RMSE and NASA use RUL labels capped at 125; RMSE against uncapped labels is {raw}, "
               "inflated by engines with true RUL above 125 that are under-predicted by construction. Severity buckets use the triage cutoffs as ground truth (assumption). "
               "Benchmark replay only: no real telemetry, field validation or maintenance-system integration; all work orders are simulated.", 6.3, False, SLATE)],
          align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, margin=0)

    # ================= slide 2 (appendix)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    title_bar(s, "Appendix: data, methodology, ablations, operations, sources and limitations",
              "Every figure and table is generated from the persisted pipeline artifacts of this repository")
    label(s, .3, .85, 4.3, "Datasets and final model inputs", NAVY)
    rows = [["Subset", "Train / test engines", "Cond. / faults", "Sensors", "Train / test rows", "Features"]]
    for fd in FDS:
        P = load_prepared(cfg, fd)
        ns = len(P["pre"].sensors)
        rows.append([fd, f"{P['report']['train_engines']} / {P['report']['test_engines']}", FD_DESC[fd], str(ns),
                     f"{len(P['Ftr'])} / {len(P['Fte'])}", str(ns * (len(ft["stats"]) + 2 * len(S[fd]["windows"])))])
    table(s, .3, 1.07, rows, [.55, .85, .75, .55, 1.1, .5], size=6.8, hdr_h=.34, row_h=.25)
    bullets(s, .3, 2.5, 4.3, .95, [
        f"Constant sensors are dropped; sensors with mean |Spearman(sensor, cycle)| below {cfg['data']['min_abs_trend']} per engine are dropped (see Sensors).",
        f"FD002 and FD004: operating settings clustered into {cfg['data']['n_regimes']} regimes (KMeans); each sensor z-scored per regime using training statistics.",
        f"Trailing median smoothing (window {cfg['data']['smooth_window']}), causal; rolling features never cross engine boundaries.",
        "c = operating conditions, f = fault modes (readme.txt: FD001 1c/1f, FD002 6c/1f, FD003 1c/2f, FD004 6c/2f)."], size=6.2, fill=LIGHT)
    label(s, 4.7, .85, 3.9, "Feature-family ablation", TEAL); pic(s, f["abl"], 4.7, 1.07, 3.9)
    label(s, 8.7, .85, 4.35, "Model comparison", GOLD); pic(s, f["models"], 8.7, 1.07, 4.35)

    label(s, .3, 3.55, 4.3, "Gated retraining demonstration (FD001, XGBoost, SYNTHETIC drift)", GREEN)
    fs = sorted(art(cfg, "retrain_demo", "x").parent.glob("*.json"))
    if not fs:
        raise FileNotFoundError("artifacts/retrain_demo/*.json missing; run 05_retrain_demo.py scenarios first")
    R = sorted((json.loads(p.read_text()) for p in fs), key=lambda r: (r["synthetic_drift_std"], r["days_since"]))
    rows = [["Scenario", "Max PSI", "v1 val RMSE", "v1 arrivals RMSE", "Trigger", "v2 val RMSE", "v2 promoted"]]
    for r in R:
        lab = f"Synthetic drift {r['synthetic_drift_std']:g} sd" if r["synthetic_drift_std"] else (f"Scheduled, {r['days_since']} d" if r["days_since"] else "No drift, 0 d")
        trig = "+".join(k.upper() for k, v in r["trigger"].items() if v) or "none"
        rows.append([lab, f"{r['max_psi']:.2f}", f"{r['v1_val']['rmse']:.1f}", f"{r['v1_arrivals']['rmse']:.1f}", trig,
                     f"{r['v2_val']['rmse']:.1f}" if "v2_val" in r else "n/a", ("yes" if r["v2_promoted"] else "no") if "v2_val" in r else "n/a"])
    table(s, .3, 3.77, rows, [1.15, .5, .55, .65, .55, .5, .4], size=6.2, hdr_h=.34, row_h=.24)
    bullets(s, .3, 4.95, 4.3, .5, ["Retrain triggers: RMSE factor, days since last training, PSI alarm. Promotion only if the candidate beats the champion on engine-held-out "
                                   "validation RMSE; the previous champion is retained under the registry alias. Drift is injected, not observed; v2 also trains on 60% more labelled engines, so its gain combines adaptation and extra data. In the drift scenario the validation set is shifted too (a stress test), so only the PSI trigger fires."], size=5.5, fill=LIGHT)

    label(s, 4.7, 3.55, 3.9, "Simulated work order (FD001 replay, persisted JSON)", PURPLE)
    op = art(cfg, "work_orders_v2", "FD001_orders.json")
    if not op.exists():
        raise FileNotFoundError("artifacts/work_orders_v2/FD001_orders.json missing; run 09_triage_v2.py first")
    wos = json.loads(op.read_text())
    if not wos:
        raise ValueError("FD001_orders.json holds no work orders")
    wo = ([w for w in wos if w.get("severity") == "CRITICAL"] or wos)[0]
    keys = ["work_order_id", "incident_key", "status", "simulated", "engine_id", "replay_cycle", "rul_mean", "rul_conservative", "anomaly_score",
            "persistent_anomaly", "severity", "rule_fired", "recommendation", "requires_human_review", "model_version", "top_degradation_sensors"]
    ks = [k for k in keys if k in wo] or list(wo)
    txt = [(f"{k}: {', '.join(map(str, wo[k])) if isinstance(wo[k], list) else wo[k]}", 5.6, False, DARK) for k in ks]
    mono = box(s, MSO_SHAPE.ROUNDED_RECTANGLE, 4.7, 3.77, 3.9, 1.68, LIGHT)
    write(mono, txt, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, margin=.08, font="Courier New")

    label(s, 8.7, 3.55, 4.35, "Methodology and evaluation protocol", RUST)
    cap = S["FD001"]["cap"]
    bullets(s, 8.7, 3.77, 4.35, 1.68, [
        "RUL label: max_cycle - cycle on training trajectories, capped at 125; test labels from the official RUL files, scored at the last observed cycle.",
        f"Features: trailing window {ft['window']} ({', '.join(ft['stats'])}) plus window 10 (mean, slope); causal, per engine.",
        f"Validation: {ev['cv_folds']}-fold GroupKFold by engine; random truncation at cycle >= {ev['min_cut_len']}, {ev['n_cut_reps']} repeats; test never used for selection.",
        "NASA score: sum of exp(-d/13)-1 for d<0 and exp(d/10)-1 for d>=0, d = predicted minus true RUL; lower is better.",
        f"Anomaly: autoencoder {an['hidden'][0]}-{an['hidden'][1]}-{an['latent']} on healthy cycles {an['healthy_cycle_range'][0]} to {an['healthy_cycle_range'][1]}; Isolation Forest on latent + error; k={an['persist_k']}.",
        f"Conservative RUL: base minus the {int(rl['cons_q'] * 100)}th percentile of out-of-fold over-prediction per bin (<30, 30-60, >=60), training engines only."],
        size=5.9, fill=LIGHT)

    label(s, .3, 5.55, 6.4, "Sources", NAVY)
    refs = ["A. Saxena, K. Goebel, D. Simon, N. Eklund, Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation, PHM08, Denver, 2008.",
            "NASA, C-MAPSS Jet Engine Simulated Data, data.nasa.gov/dataset/cmapss-jet-engine-simulated-data (official source of all four subsets).",
            "A. Saxena et al., Metrics for Evaluating Performance of Prognostic Techniques, PHM08, Denver, 2008 (asymmetric scoring function).",
            "T. Chen, C. Guestrin, XGBoost: A Scalable Tree Boosting System, KDD, 2016.",
            "F. T. Liu, K. M. Ting, Z.-H. Zhou, Isolation Forest, IEEE ICDM, 2008.",
            "G. E. Hinton, R. R. Salakhutdinov, Reducing the Dimensionality of Data with Neural Networks, Science 313, 2006 (autoencoder).",
            "S. M. Lundberg, S.-I. Lee, A Unified Approach to Interpreting Model Predictions, NeurIPS, 2017 (TreeSHAP attribution).",
            "F. Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 12, 2011.  |  A. Paszke et al., PyTorch, NeurIPS, 2019.",
            "MLflow, mlflow.org (experiment tracking and model registry).  |  N. Siddiqi, Credit Risk Scorecards, Wiley, 2006 (population stability index)."]
    bullets(s, .3, 5.77, 6.4, 1.62, refs, size=5.6, fill=LIGHT)

    label(s, 6.8, 5.55, 6.25, "Limitations, assumptions and verification status", RUST)
    pl = ""
    pp = ROOT / "pytest_out.txt"
    if pp.exists():
        ls_ = [l.strip() for l in pp.read_text().splitlines() if l.strip()]
        pl = ls_[-1] if ls_ else ""
    if not pl:
        raise FileNotFoundError("pytest_out.txt missing or empty; run: python -m pytest -q 2>&1 | tee pytest_out.txt")
    lim = [f"Unit tests (pytest): {pl}. Passing tests do not imply full coverage.",
           "Regime clustering and sensor selection were fitted on all training engines before CV; a strict fold-wise refit check has not been run.",
           "Raw-label RMSE is dominated by engines with true RUL above 125 (cap artifact); the capped yardstick is reported as primary.",
           "A GRU blend was evaluated; paired test CIs did not show a reliable gain, so the final RUL model is XGBoost alone, no smoothing.",
           "Anomaly positives are defined as RUL<=30 (no failure-onset labels exist); thresholds and triage cutoffs are provisional, not industrial limits.",
           "Retraining was demonstrated with synthetic drift; real drift adaptation requires newly labelled run-to-failure data.",
           "Benchmark replay only: no real telemetry ingestion, field validation or maintenance-system integration; all actions are simulated.",
           "Monitoring dashboard is in development and is not part of the verified results."]
    worse = [fd for fd in FDS if TR[fd]["rul_cons"]["nasa"] > TR[fd]["rul_base"]["nasa"]]   # cons_cost
    fewer_late = all(TR[fd]["rul_cons"]["late_rate"] < TR[fd]["rul_base"]["late_rate"] for fd in FDS)
    lim.append(f"Conservative RUL offset {'lowers late predictions on all subsets' if fewer_late else 'does not lower late predictions on every subset'}; "
               + (f"it raises last-cycle NASA score on {', '.join(worse)}, so it is a safety trade-off, not an accuracy gain." if worse else "it did not raise the NASA score on any subset."))
    nrev = sum(TR[fd]["n_review_last_cycle"] for fd in FDS)
    if nrev:
        lim.append(f"{nrev} test engines have fewer cycles than the feature window and are routed to REVIEW instead of a risk level.")
    bullets(s, 6.8, 5.77, 6.25, 1.62, lim, size=5.3, fill=LIGHT)

    out = fd_ / "fleetpm_deck.pptx"
    prs.save(str(out))
    print("saved", out.name, "| figures in artifacts/figures/")


if __name__ == "__main__":
    main()