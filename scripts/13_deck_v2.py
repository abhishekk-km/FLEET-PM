import _bootstrap
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
import _ext_common as C
from fleetpm.config import load_config, art, ROOT
from fleetpm.pipeline import load_prepared
from fleetpm.models.metrics import summarize

cfg = load_config()
FDS = cfg["data"]["subsets"]
NAVY, BLUE, TEAL, AMBER, ORANGE, RED, GREEN, GREY, LIGHT, INK, WHITE, PURPLE = (
    "0B2A4A", "0B5CAD", "14808C", "D99A00", "E36414", "B3202A", "2E8B57", "8A97A8", "F3F6FA", "111827", "FFFFFF", "6B4C9A")
H = lambda c: "#" + c
PAL = [BLUE, TEAL, AMBER, NAVY]
NAMES = ["T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB",
         "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"]
M = ["raw", "offset_q0.8", "conformal_a0.2", "conformal_a0.1"]
MLAB = {"raw": "Raw", "offset_q0.8": "Offset (0.8 q)", "conformal_a0.2": "Conformal 80%", "conformal_a0.1": "Conformal 90%"}
MCOL = {"raw": GREY, "offset_q0.8": TEAL, "conformal_a0.2": BLUE, "conformal_a0.1": NAVY}
PLAB = {"run_to_failure": "Run to failure", "fixed_interval": "Fixed interval", "rul_raw": "RUL threshold, raw",
        "rul_offset": "RUL threshold, offset", "rul_conformal10": "RUL threshold, 90% bound"}
PCOL = {"run_to_failure": GREY, "fixed_interval": AMBER, "rul_raw": BLUE, "rul_offset": TEAL, "rul_conformal10": NAVY}
SEVC = {"MEDIUM": AMBER, "HIGH": ORANGE, "CRITICAL": RED}
RGRID = np.logspace(0, 3, 200)
RANK = {"REVIEW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
META = {"FD001": ("1", "1 (HPC)"), "FD002": ("6", "1 (HPC)"), "FD003": ("1", "2 (HPC, fan)"), "FD004": ("6", "2 (HPC, fan)")}


def jl(*parts, required=True):
    p = art(cfg, *parts)
    if not p.exists():
        if required:
            raise SystemExit(f"missing {p}: run the producing script first")
        print("optional input missing:", p.relative_to(ROOT))
        return None
    return json.loads(p.read_text())


def opt(fn, *a):
    try:
        return fn(*a)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
        print(f"skipped {fn.__name__}: {type(e).__name__} {e}")
        return None


conf, cost = jl("ext_conformal.json"), jl("ext_cost_policy.json")
ex = jl("ext_explain.json", required=False) or {}
anom = {f: jl("models", f, "anomaly_metrics.json", required=False) for f in FDS}
S = jl("final_summary.json", required=False)
TR = jl("triage_v2_summary.json", required=False)
for f in FDS:
    if f not in conf or f not in cost:
        raise SystemExit(f"{f} missing from ext_conformal.json or ext_cost_policy.json")
    for m in M:
        if "cov_lt_cap" not in conf[f]["test"][m]:
            raise SystemExit("rerun scripts/10_conformal.py (cov_lt_cap missing)")
D = {f: C.load_oof(cfg, f) for f in FDS}
ST = {}
for f, d in D.items():
    p, tc, tr = d["pred_t"], d["true_cap"], d["true_raw"]
    s_, r_ = summarize(tc, p), summarize(tr, p)
    ST[f] = dict(n=len(p), rmse=s_["rmse"], mae=s_["mae"], nasa=s_["nasa"], rmse_raw=r_["rmse"],
                 late=float((p > tc).mean()), bias=float((p - tc).mean()))
NTOT = sum(v["n"] for v in ST.values())

WOB = art(cfg, "work_orders_v2", "x").parent
mx, CNT, WOF = {}, {f: {k: 0 for k in RANK} for f in FDS}, {}
for p in sorted(WOB.rglob("WO-*.json")):
    d = json.loads(p.read_text())
    fd = p.name.split("-")[1]
    sev = d.get("severity")
    if fd not in FDS or sev not in RANK:
        continue
    CNT[fd][sev] += 1
    k = (fd, d.get("engine_id"))
    mx[k] = max(mx.get(k, 0), RANK[sev])
    if fd == "FD001" and (sev == "CRITICAL" or "FD001" not in WOF):
        WOF["FD001"] = d
if not mx:
    raise SystemExit(f"no work orders under {WOB}")
N_ORD, N_HIGH, N_CRIT = len(mx), sum(v >= 2 for v in mx.values()), sum(v == 3 for v in mx.values())

# ------------------------------------------------------------------ pptx helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BL = prs.slide_layouts[6]
rgb = lambda h: RGBColor.from_string(h)
E = lambda v: int(Inches(v))
APX = [0]


def rect(s, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.shadow.inherit = False
    if fill:
        sh.fill.solid()
        sh.fill.fore_color.rgb = rgb(fill)
    else:
        sh.fill.background()
    if line:
        sh.line.color.rgb = rgb(line)
        sh.line.width = Pt(0.75)
    else:
        sh.line.fill.background()
    return sh


def put(sh, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE, m=0.06, font="Arial"):
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(m)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, (t, sz, b, col) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = t
        r.font.size, r.font.bold, r.font.name = Pt(round(sz * 0.94, 1)), b, font
        r.font.color.rgb = rgb(col)


def text(s, x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    put(tb, paras, align, anchor)
    return tb


def line(s, x1, y1, x2, y2, color=GREY, arrow=False, wd=1.25):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(wd)
    if arrow:
        ln = c.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))


def pic(s, path, x, y, w, h):
    s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))


def tile(s, x, y, w, h, title, body, color, fs=9):
    hd = rect(s, x, y, w, 0.27, color)
    put(hd, [(title, 10, True, WHITE)])
    b = rect(s, x, y + 0.27, w, h - 0.27, LIGHT)
    put(b, [(t, fs, False, INK) for t in (body if isinstance(body, list) else [body])], anchor=MSO_ANCHOR.TOP, m=0.08)


def table(s, x, y, w, h, rows, widths=None, fs=8.5):
    nr, nc = len(rows), len(rows[0])
    t = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(h)).table
    if widths:
        for j, wj in enumerate(widths):
            t.columns[j].width = Inches(wj * w / sum(widths))
    for i in range(nr):
        t.rows[i].height = Inches(h / nr)
        for j, v in enumerate(rows[i]):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(NAVY if i == 0 else (WHITE if i % 2 else LIGHT))
            c.margin_left = c.margin_right = Inches(0.05)
            c.margin_top = c.margin_bottom = Inches(0.015)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(v)
            r.font.size, r.font.bold, r.font.name = Pt(round(fs * 0.94, 1)), i == 0, "Arial"
            r.font.color.rgb = rgb(WHITE if i == 0 else INK)


def frame(title, color, src):
    APX[0] += 1
    s = prs.slides.add_slide(BL)
    rect(s, 0, 0, 13.333, 0.6, NAVY)
    rect(s, 0, 0.6, 13.333, 0.05, color)
    text(s, 0.25, 0.08, 10.5, 0.45, [(title, 18, True, WHITE)], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 11.0, 0.08, 2.1, 0.45, [(f"Appendix {APX[0]}", 11, False, "C9D6E6")], align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.25, 7.2, 12.8, 0.26, [("Source: " + src + " | Benchmark replay, simulated actions only.", 8, False, GREY)])
    return s


def sec(s, x, y, w, label, color=NAVY):
    text(s, x, y, w, 0.28, [(label, 10.5, True, color)])
    line(s, x + 0.05, y + 0.29, x + w, y + 0.29, "D6DEE9", wd=1)


# ------------------------------------------------------------------ charts
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#9AA7B8", "axes.labelcolor": "#334155",
                     "xtick.color": "#334155", "ytick.color": "#334155"})


def fig_(w, h, **kw):
    return plt.subplots(figsize=(w, h), **kw)


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#E6EBF2", lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    p = art(cfg, "figures", f"deck2_{name}.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return str(p)


def ch_cov(w, h):
    v = np.array([[conf[f]["test"][m]["cov_lt_cap"] for f in FDS] for m in M])
    mu = v.mean(1)
    fig, ax = fig_(w, h)
    ax.bar(range(4), mu, color=[H(MCOL[m]) for m in M], yerr=[mu - v.min(1), v.max(1) - mu], capsize=3, ecolor="#334155")
    for y in (0.8, 0.9):
        ax.axhline(y, ls=":", c=H(GREY), lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Raw", "Offset\n0.8 q", "Conf.\n80%", "Conf.\n90%"], fontsize=6.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Lower-bound coverage, engines below cap\nbar: mean of subsets, whisker: range; dotted: 80%, 90%", fontsize=7)
    clean(ax)
    return save(fig, "cov")


def ch_cost(w, h):
    fig, ax = fig_(w, h)
    for k in PLAB:
        ax.loglog(RGRID, np.mean([cost[f]["curves"][k] for f in FDS], 0), color=H(PCOL[k]), lw=1.8, label=PLAB[k])
    ax.set_xlabel("r = C_fail / C_waste")
    ax.set_ylabel("Mean cost per engine (C_waste units)")
    ax.set_title("Cost-aware policy replay, training trajectories, mean of 4 subsets (simulated)", fontsize=7.5)
    ax.legend(fontsize=6.5, frameon=False)
    clean(ax)
    return save(fig, "cost")


def ch_cost4(w, h):
    fig, axs = fig_(w, h, nrows=2, ncols=2, sharex=True)
    for ax, f in zip(axs.ravel(), FDS):
        for k, c in cost[f]["curves"].items():
            ax.loglog(RGRID, c, color=H(PCOL[k]), lw=1.5, label=PLAB[k])
        ax.set_title(f, fontsize=8)
        clean(ax)
    axs[0, 0].legend(fontsize=5.5, frameon=False)
    fig.supxlabel("r = C_fail / C_waste", fontsize=8)
    fig.supylabel("Mean cost per engine", fontsize=8)
    return save(fig, "cost4")


def ch_scatter(w, h):
    fig, axs = fig_(w, h, nrows=2, ncols=2)
    for ax, f in zip(axs.ravel(), FDS):
        t, p, cap = D[f]["true_cap"], D[f]["pred_t"], D[f]["cap"]
        late = p > t
        ax.scatter(t[~late], p[~late], s=7, c=H(BLUE), label="early")
        ax.scatter(t[late], p[late], s=7, c=H(RED), label="late")
        ax.plot([0, cap], [0, cap], c=H(GREY), ls="--", lw=1)
        ax.set_title(f, fontsize=8)
        clean(ax)
    axs[0, 0].legend(fontsize=6, frameon=False)
    fig.supxlabel("True RUL (capped)", fontsize=8)
    fig.supylabel("Predicted RUL", fontsize=8)
    return save(fig, "scatter")


def ch_band(w, h):
    fig, ax = fig_(w, h)
    wd = 0.2
    for i, f in enumerate(FDS):
        d = D[f]
        e, tr, cap = d["pred_t"] - d["true_cap"], d["true_raw"], d["cap"]
        ms = [tr <= 30, (tr > 30) & (tr <= 60), (tr > 60) & (tr <= cap), tr > cap]
        ax.bar(np.arange(4) + (i - 1.5) * wd, [float(np.sqrt((e[m] ** 2).mean())) if m.any() else 0.0 for m in ms], wd, color=H(PAL[i]), label=f)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["<=30", "31-60", "61-cap", ">cap (capped)"])
    ax.set_xlabel("True RUL band (cycles)")
    ax.set_ylabel("RMSE vs capped label")
    ax.legend(fontsize=6.5, frameon=False)
    clean(ax)
    return save(fig, "band")


def ch_slack(w, h):
    fig, ax = fig_(w, h)
    mk = ["o", "s", "^", "D"]
    for i, f in enumerate(FDS):
        for m in M:
            r = conf[f]["test"][m]
            ax.scatter(r["slack_mean"], r["cov_lt_cap"], s=45, marker=mk[i], c=H(MCOL[m]))
    for y in (0.8, 0.9):
        ax.axhline(y, ls=":", c=H(GREY), lw=1)
    hs = [Line2D([], [], marker="o", ls="", c=H(MCOL[m]), label=MLAB[m]) for m in M]
    hs += [Line2D([], [], marker=mk[i], ls="", c=H(GREY), label=f) for i, f in enumerate(FDS)]
    ax.legend(handles=hs, fontsize=6, frameon=False)
    ax.set_xlabel("Mean slack, prediction minus bound (cycles)")
    ax.set_ylabel("Coverage, engines below cap")
    clean(ax)
    return save(fig, "slack")


def ch_anom(w, h):
    fs = [f for f in FDS if anom[f]]
    fig, ax = fig_(w, h)
    for j, (k, lab, col) in enumerate([("auroc_HI", "Health index", GREY), ("auroc_IF_error_1d", "IF on error", TEAL),
                                       ("auroc_IF_latent_error", "IF on latent and error (deployed)", NAVY)]):
        ax.scatter([anom[f][k] for f in fs], np.arange(len(fs)) + (j - 1) * 0.15, s=60, c=H(col), label=lab)
    ax.set_yticks(range(len(fs)))
    ax.set_yticklabels(fs)
    ax.set_xlim(0.9, 1.0)
    ax.set_xlabel("AUROC, held-out test engines")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    clean(ax)
    return save(fig, "anom")


def ch_lead(w, h):
    fs = [f for f in FDS if anom[f]]
    fig, ax = fig_(w, h)
    ax.bar(range(len(fs)), [anom[f]["IFl_lead_median"] for f in fs], color=H(NAVY), width=0.5, label="median lead")
    ax.scatter(range(len(fs)), [anom[f]["IFl_lead_p10"] for f in fs], c=H(ORANGE), zorder=3, s=40, label="10th percentile")
    ax.set_xticks(range(len(fs)))
    ax.set_xticklabels(fs)
    ax.set_ylabel("Cycles of warning before failure")
    ax.legend(fontsize=6.5, frameon=False)
    clean(ax)
    return save(fig, "lead")


def ch_shap(w, h, fs):
    nr = math.ceil(len(fs) / 2)
    fig, axs = fig_(w, h, nrows=nr, ncols=2, squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, f in zip(axs.ravel(), fs):
        top = sorted(ex[f]["faithfulness"]["mean_abs_shap_by_sensor"].items(), key=lambda kv: -kv[1])[:8][::-1]
        ax.axis("on")
        ax.barh([NAMES[int(k.split("_")[1]) - 1] for k, _ in top], [v for _, v in top], color=H(BLUE))
        ax.set_title(f, fontsize=8)
        clean(ax)
    fig.supxlabel("Mean |SHAP| (cycles), top 8 sensors", fontsize=8)
    return save(fig, "shap")


def ch_conf4(w, h):
    fig, axs = fig_(w, h, nrows=1, ncols=4)
    for k, (ax, f) in enumerate(zip(axs, FDS)):
        c = np.array(TR[f]["confusion"])
        rs = np.maximum(c.sum(1, keepdims=True), 1)
        ax.imshow(c / rs, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, int(c[i, j]), ha="center", va="center", fontsize=6, color="white" if c[i, j] / rs[i, 0] > .5 else "black")
        ax.set_xticks(range(4))
        ax.set_xticklabels(list("LMHC"))
        ax.set_yticks(range(4))
        ax.set_yticklabels(list("LMHC") if k == 0 else [])
        ax.set_title(f, fontsize=8)
        ax.set_xlabel("triage severity")
        if k == 0:
            ax.set_ylabel("true bucket")
    return save(fig, "conf4")


def ch_lead2(w, h):
    fig, (a1, a2) = fig_(w, h, nrows=1, ncols=2)
    bp = a1.boxplot([TR[f]["lead"] if TR[f]["lead"] else [np.nan] for f in FDS], widths=.55, patch_artist=True,
                    medianprops={"color": "black"}, flierprops={"markersize": 2})
    for patch, c in zip(bp["boxes"], PAL):
        patch.set_facecolor(H(c))
        patch.set_alpha(.85)
    a1.set_xticklabels(FDS)
    a1.axhline(30, color=H(RED), lw=.8, ls="--")
    a1.set_title("True RUL at first HIGH or CRITICAL", fontsize=8)
    a1.set_ylabel("cycles before failure")
    clean(a1)
    bot = np.zeros(4)
    for sv in ("MEDIUM", "HIGH", "CRITICAL"):
        v = np.array([TR[f]["wo_by_sev"].get(sv, 0) for f in FDS], float)
        a2.bar(range(4), v, .6, bottom=bot, color=H(SEVC[sv]), label=sv)
        bot += v
    a2.set_xticks(range(4))
    a2.set_xticklabels([f[2:] for f in FDS])
    a2.set_title("Simulated work orders (replay)", fontsize=8)
    a2.legend(fontsize=6.5, frameon=False)
    clean(a2)
    return save(fig, "lead2")


def ch_valid(w, h):
    cv = np.array([S[f]["cv_rmse_k1"] for f in FDS])
    te = np.array([ST[f]["rmse"] for f in FDS])
    ci = np.array([S[f]["cap"]["ci"] for f in FDS])
    fig, ax = fig_(w, h)
    x = np.arange(4)
    ax.bar(x - .18, cv, .36, color=H(NAVY), label="Engine-grouped 5-fold CV")
    ax.bar(x + .18, te, .36, color=H(TEAL), yerr=[np.maximum(0, te - ci[:, 0]), np.maximum(0, ci[:, 1] - te)], capsize=2, label="Official test, 95% CI")
    ax.set_xticks(x)
    ax.set_xticklabels(FDS)
    ax.set_ylabel("RMSE (cycles)")
    ax.set_ylim(0, ci[:, 1].max() * 1.4)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    clean(ax)
    return save(fig, "valid")


def ch_models(w, h):
    names = [("ridge", "Ridge"), ("extratrees", "ExtraTrees"), ("lstm", "LSTM"), ("gru", "GRU"), ("cnn1d", "CNN1D"), ("cnnlstm", "CNN-LSTM")]
    st = {f: json.loads(art(cfg, "models", f, "rul_study.json").read_text()) for f in FDS}
    cols = [GREY, AMBER, PURPLE, GREEN, ORANGE, RED]
    fig, ax = fig_(w, h)
    x, wd, top = np.arange(4), .115, 0
    for i, (k, n) in enumerate(names):
        v = [st[f][k]["test_cap_rmse"] for f in FDS]
        top = max(top, max(v))
        ax.bar(x + (i - 3) * wd, v, wd, color=H(cols[i]), label=n)
    ax.bar(x + 3 * wd, [ST[f]["rmse"] for f in FDS], wd, color=H(NAVY), label="XGBoost (final)")
    ax.set_xticks(x)
    ax.set_xticklabels(FDS)
    ax.set_ylabel("Test RMSE, capped")
    ax.set_ylim(0, top * 1.4)
    ax.legend(fontsize=5.5, frameon=False, ncol=4, loc="upper left")
    clean(ax)
    return save(fig, "models")


def ch_abl(w, h):
    ns = len(D["FD001"]["P"]["pre"].sensors)
    rd = lambda n: json.loads(art(cfg, "models", "FD001", n).read_text())
    m1, m2 = rd("rul_study_abl_mean.json"), rd("rul_study_abl_mean_slope.json")
    m3, m4 = rd("rul_boost_nowin.json")["candidates"], rd("rul_boost_gru.json")["candidates"]
    xg = [m1["xgb"]["cv_rmse"], m2["xgb"]["cv_rmse"], m3["xgb"]["cv_rmse_by_k"]["1"], m4["xgb"]["cv_rmse_by_k"]["1"]]
    et = [m1["extratrees"]["cv_rmse"], m2["extratrees"]["cv_rmse"], m3["extratrees"]["cv_rmse_by_k"]["1"], m4["extratrees"]["cv_rmse_by_k"]["1"]]
    fig, ax = fig_(w, h)
    x = np.arange(4)
    ax.bar(x - .18, xg, .36, color=H(NAVY), label="XGBoost")
    ax.bar(x + .18, et, .36, color=H(AMBER), label="ExtraTrees")
    for xi, a, b in zip(x, xg, et):
        ax.text(xi - .18, a + .3, f"{a:.1f}", ha="center", fontsize=6)
        ax.text(xi + .18, b + .3, f"{b:.1f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"mean\n({ns})", f"+slope\n({2 * ns})", f"+std,min,\nmax,kurt\n({6 * ns})", f"+window 10\n({8 * ns})"], fontsize=6.5)
    ax.set_ylabel("CV RMSE (cycles)")
    ax.set_ylim(0, max(xg + et) * 1.2)
    ax.set_title("FD001 feature ablation (bars 1-2 fold-averaged, 3-4 pooled OOF)", fontsize=7)
    ax.legend(fontsize=6.5, frameon=False)
    clean(ax)
    return save(fig, "abl")


def fmt_be(x):
    if x["point"] is None:
        return "never"
    return f"{x['point']:.1f}" + (f" ({x['ci'][0]:.1f} to {x['ci'][1]:.1f})" if x.get("ci") else "")


# ------------------------------------------------------------------ slide 1
from pptx.enum.dml import MSO_LINE
TITLE = "SENTINEL: Fleet Health and RUL Intelligence"
SUBT = "NASA C-MAPSS benchmark replay | simulated actions | technical review"
FEAT = "2E75B6"
s = prs.slides.add_slide(BL)
rect(s, 0, 0, 13.333, 0.5, NAVY)
rect(s, 0, 0.5, 13.333, 0.04, TEAL)
text(s, 0.15, 0.04, 8.4, 0.42, [(TITLE, 17, True, WHITE)], anchor=MSO_ANCHOR.MIDDLE)
text(s, 8.2, 0.04, 5.0, 0.42, [(SUBT, 8.5, False, "C9D6E6")], align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)


def rng(v, f=".1f"):
    return f"{min(v):{f}} to {max(v):{f}}"


def band_rmse(f, lo_, hi_):
    d = D[f]
    e, tr = d["pred_t"] - d["true_cap"], d["true_raw"]
    m = (tr > lo_) & (tr <= hi_)
    return float(np.sqrt((e[m] ** 2).mean())) if m.any() else float("nan")


def ring_img(name, fr, col, txt, fr2=None, col2=None):
    fig, ax = plt.subplots(figsize=(0.9, 0.9))
    ax.pie([fr, 1 - fr], startangle=90, counterclock=False, colors=[H(col), "#E6EBF2"], radius=1.0,
           wedgeprops=dict(width=0.26, edgecolor="white"))
    if fr2 is not None:
        ax.pie([fr2, 1 - fr2], startangle=90, counterclock=False, colors=[H(col2), "#E6EBF2"], radius=0.7,
               wedgeprops=dict(width=0.26, edgecolor="white"))
    ax.text(0, 0, txt, ha="center", va="center", fontsize=6.5, fontweight="bold", color="#0B2A4A")
    ax.set_aspect("equal")
    fig.subplots_adjust(0, 0, 1, 1)
    p = art(cfg, "figures", f"deck3_{name}.png")
    fig.savefig(p, dpi=250, transparent=True)
    plt.close(fig)
    return str(p)


cap_ = int(D["FD001"]["cap"])
rm = [ST[f]["rmse"] for f in FDS]
cv = [conf[f]["test"]["conformal_a0.1"]["cov_lt_cap"] for f in FDS]
lr = float(np.mean([conf[f]["test"]["raw"]["late_rate"] for f in FDS]))
lc = float(np.mean([conf[f]["test"]["conformal_a0.1"]["late_rate"] for f in FDS]))
be = [b for b in (cost[f]["break_even"]["rul_conformal10"]["vs_rtf"]["point"] for f in FDS) if b is not None]
af = [anom[f]["auroc_IF_latent_error"] for f in FDS if anom[f]]
pf = [anom[f]["IFl_fpr_persist"] for f in FDS if anom[f]]
tp = [anom[f]["IFl_tpr_persist"] for f in FDS if anom[f]]
sl = [conf[f]["test"]["conformal_a0.1"]["slack_mean"] for f in FDS]
b30 = [band_rmse(f, -1, 30) for f in FDS]
bmid = [band_rmse(f, 60, D[f]["cap"]) for f in FDS]
nr = float(np.mean(rm)) / cap_

gauges = [("Engines with work order", f"{N_ORD} / {NTOT}", f"{NTOT} test engines, 4 subsets", N_ORD / NTOT, BLUE, f"{N_ORD / NTOT:.0%}", None, None),
          ("RMSE vs capped RUL", rng(rm), "cycles; mean = share of cap", nr, TEAL, f"{nr:.0%}", None, None)]
if af:
    gauges.append(("Anomaly AUROC (min)", rng(af, ".3f"), "Isolation Forest, latent + error", min(af), PURPLE, f"{min(af):.2f}", None, None))
gauges.append(("90% bound coverage", rng(cv, ".2f"), "engines below cap; measured", min(cv), NAVY, f"{min(cv):.2f}", None, None))
gauges.append(("Late predictions", f"{lr:.0%} to {lc:.0%}", "outer: raw, inner: 90% bound", lr, RED, "", lc, GREEN))
if tp:
    gauges.append(("Persistent TPR (min)", rng(tp, ".2f"), "anomaly, RUL <= 30 proxy", min(tp), AMBER, f"{min(tp):.0%}", None, None))
for i, (lab, val, sub, fr, col, ctr, fr2, col2) in enumerate(gauges):
    x = 0.15 + i * 2.19
    rect(s, x, 0.6, 2.1, 0.84, LIGHT)
    pic(s, ring_img(f"k{i}", fr, col, ctr, fr2, col2), x + 0.06, 0.65, 0.74, 0.74)
    text(s, x + 0.84, 0.62, 1.25, 0.2, [(lab.upper(), 5.8, True, "51627A")])
    text(s, x + 0.84, 0.8, 1.25, 0.3, [(val, 11.5, True, NAVY)])
    text(s, x + 0.84, 1.08, 1.25, 0.34, [(sub, 5.8, False, "64748B")])

# ---------------- radial architecture
AY, AH = 1.52, 3.43
rect(s, 0.15, AY, 3.35, AH, None, "D6DEE9")
sec(s, 0.2, AY + 0.01, 3.25, "Pipeline ring: eight stages")
cx, cy, R = 1.825, AY + 1.92, 1.1
o = rect(s, cx - 1.46, cy - 1.46, 2.92, 2.92, None, GREY, MSO_SHAPE.OVAL)
o.line.dash_style = MSO_LINE.DASH
rect(s, cx - R, cy - R, 2 * R, 2 * R, None, "D6DEE9", MSO_SHAPE.OVAL)
stg = [BLUE, BLUE, FEAT, TEAL, TEAL, TEAL, ORANGE, ORANGE]
pts = [(cx + R * np.cos(np.radians(-90 + 45 * i)), cy + R * np.sin(np.radians(-90 + 45 * i))) for i in range(8)]
for (px, py) in pts:
    v = np.array([px - cx, py - cy])
    u = v / np.linalg.norm(v)
    a, b = np.array([cx, cy]) + u * 0.56, np.array([px, py]) - u * 0.31
    line(s, a[0], a[1], b[0], b[1], "D6DEE9", False, 0.75)
for i in range(8):
    a, b = np.array(pts[i]), np.array(pts[(i + 1) % 8])
    u = (b - a) / np.linalg.norm(b - a)
    p0, p1 = a + u * 0.33, b - u * 0.33
    line(s, p0[0], p0[1], p1[0], p1[1], "334155", True, 1.25)
for i, (px, py) in enumerate(pts):
    put(rect(s, px - 0.3, py - 0.3, 0.6, 0.6, stg[i], WHITE, MSO_SHAPE.OVAL), [(str(i + 1), 12, True, WHITE)], align=PP_ALIGN.CENTER, m=0.0)
put(rect(s, cx - 0.52, cy - 0.52, 1.04, 1.04, NAVY, None, MSO_SHAPE.OVAL),
    [("FLEET HEALTH", 6.5, True, WHITE), ("CORE", 6.5, True, WHITE), ("MLflow | drift gate | dashboard", 4.8, False, "C9D6E6")],
    align=PP_ALIGN.CENTER, m=0.03)

# ---------------- numbered technical legend
dcfg, an_, tg_ = cfg["data"], cfg["anomaly"], cfg["triage"]
nfeat = [len(D[f]["P"]["pre"].sensors) * (len(cfg["features"]["stats"]) + 2) for f in FDS]
leg = [("Ingest and validate", "26 columns, NaN, duplicates, cycle gaps, official RUL alignment"),
       ("Regime normalisation", f"KMeans k={dcfg['n_regimes']} on settings; per-regime z-score; median w={dcfg['smooth_window']}"),
       ("Causal features", f"windows 30 and 10; {min(nfeat)} to {max(nfeat)} columns; sensors by |Spearman| >= {dcfg['min_abs_trend']}"),
       ("Anomaly detector", f"autoencoder (healthy {an_['healthy_cycle_range'][0]}-{an_['healthy_cycle_range'][1]}) + Isolation Forest; k={an_['persist_k']}"),
       ("RUL regressor", f"XGBoost {cfg['rul']['xgb']['n_estimators']} trees, depth {cfg['rul']['xgb']['max_depth']}; cap {cap_}; {cfg['eval']['cv_folds']}-fold grouped CV"),
       ("Risk bound", "0.8-quantile offset deployed; conformal 90% lower bound offline"),
       ("Deterministic triage", f"CRITICAL, HIGH, MEDIUM, LOW, REVIEW; cutoffs {tg_['critical_rul']}, {tg_['high_rul']}, {tg_['medium_rul']} (provisional)"),
       ("Simulated work order", "deterministic ID, TreeSHAP sensors, on escalation, restart-safe")]
rect(s, 3.57, AY, 3.7, AH, None, "D6DEE9")
sec(s, 3.62, AY + 0.01, 3.6, "Stage specification")
for i, (t1, t2) in enumerate(leg):
    y = AY + 0.38 + i * 0.380
    put(rect(s, 3.65, y + 0.05, 0.27, 0.27, stg[i], None, MSO_SHAPE.OVAL), [(str(i + 1), 8, True, WHITE)], align=PP_ALIGN.CENTER, m=0.0)
    tb = text(s, 3.96, y, 3.28, 0.37, [(t1, 7.5, True, NAVY), (t2, 6, False, "334155")], anchor=MSO_ANCHOR.MIDDLE)

# ---------------- funnel
FX, FW = 7.35, 2.2
rect(s, FX, AY, FW, AH, None, "D6DEE9")
sec(s, FX + 0.05, AY + 0.01, FW - 0.1, "Triage funnel (engines)")
fun = [("Test engines replayed", NTOT, BLUE), ("Work order issued", N_ORD, TEAL), ("HIGH or CRITICAL", N_HIGH, ORANGE), ("CRITICAL", N_CRIT, RED)]
tops, bots = [2.0, 1.66, 1.32, 0.98], [1.66, 1.32, 0.98, 0.64]
fcx = FX + FW / 2
for i, (lab, n, col) in enumerate(fun):
    y = AY + 0.4 + i * 0.75
    fb = s.shapes.build_freeform(E(fcx - tops[i] / 2), E(y), scale=1.0)
    fb.add_line_segments([(E(fcx + tops[i] / 2), E(y)), (E(fcx + bots[i] / 2), E(y + 0.71)), (E(fcx - bots[i] / 2), E(y + 0.71))], close=True)
    sh = fb.convert_to_shape()
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(col)
    sh.line.fill.background()
    sh.shadow.inherit = False
    put(sh, [(f"{n} ({100 * n / NTOT:.0f}%)", 9.5, True, WHITE), (lab, 6.5, False, "E3ECF7")], align=PP_ALIGN.CENTER, m=0.02)

# ---------------- coverage chart
CX2, CW2 = 9.62, 3.56
rect(s, CX2, AY, CW2, AH, None, "D6DEE9")


def ch_cov3(w, h):
    v = np.array([[conf[f]["test"][m]["cov_lt_cap"] for f in FDS] for m in M])
    mu = v.mean(1)
    fig, ax = fig_(w, h)
    ax.bar(range(4), mu, color=[H(MCOL[m]) for m in M], yerr=[mu - v.min(1), v.max(1) - mu], capsize=3, ecolor="#334155")
    for y in (0.8, 0.9):
        ax.axhline(y, ls=":", c=H(GREY), lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Raw", "Offset 0.8 q", "Conformal 80%", "Conformal 90%"], fontsize=6)
    ax.tick_params(labelsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage", fontsize=6.5)
    ax.set_title("Lower-bound coverage, engines below cap", fontsize=7.5)
    clean(ax)
    return save(fig, "cov3")


pic(s, ch_cov3(3.48, 2.55), CX2 + 0.04, AY + 0.04, 3.48, 2.55)
text(s, CX2 + 0.04, AY + 2.64, CW2 - 0.08, 0.78,
     [("Bars: mean over four subsets; whiskers: minimum to maximum; dotted lines: nominal 80% and 90%. Measured on official test engines with true RUL below the cap. Empirical coverage, not a guarantee.", 6.8, False, "334155")])

# ---------------- bottom row
BY = 5.03
tl = [("Target and features", BLUE,
       f"RUL is capped at {cap_} cycles because sensors carry no observable degradation signal early in life. Features are strictly causal: trailing windows of 30 and 10 cycles over regime-normalised, median-smoothed sensors (mean, standard deviation, minimum, maximum, slope, kurtosis). Sample weight 3 below RUL 30."),
      ("Anomaly detection", TEAL,
       f"An autoencoder trained on healthy cycles {an_['healthy_cycle_range'][0]} to {an_['healthy_cycle_range'][1]} defines a normal operating envelope; an Isolation Forest scores latent state and reconstruction error. An alert needs {an_['persist_k']} consecutive flagged cycles, which suppresses isolated noise. The output is unusual behaviour, not a confirmed fault."),
      ("Conformal lower bound", NAVY,
       "Out-of-fold over-prediction residuals are calibrated per predicted-RUL band; the bound is the prediction minus the finite-sample (1 - alpha) quantile. Coverage is measured on held-out engines, not guaranteed. Evaluated offline; the deployed triage still uses the 0.8-quantile offset."),
      ("Cost-aware maintenance", ORANGE,
       "Failure and premature maintenance are compared through r = C_fail / C_waste. Policies fire causally from cycle 30. The break-even r states when predictive triage beats run-to-failure and fixed-interval maintenance. Replay on training trajectories; no monetary values are assumed.")]
for i, (t_, col, b_) in enumerate(tl):
    tile(s, 0.15 + (i % 2) * 3.4, BY + (i // 2) * 1.06, 3.34, 1.02, t_, b_, col, fs=8.3)


def ch_cost3(w, h):
    fig, ax = fig_(w, h)
    for k in PLAB:
        ax.loglog(RGRID, np.mean([cost[f]["curves"][k] for f in FDS], 0), color=H(PCOL[k]), lw=1.6, label=PLAB[k])
    ax.set_xlabel("r = C_fail / C_waste", fontsize=6.5)
    ax.set_ylabel("Mean cost per engine", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.set_title("Cost-aware replay, mean of 4 subsets (simulated)", fontsize=7.5)
    ax.legend(fontsize=5.8, frameon=False)
    clean(ax)
    return save(fig, "cost3")


pic(s, ch_cost3(3.1, 2.1), 6.98, BY, 3.1, 2.1)
fnd = [f"Accuracy peaks near end of life: RMSE {rng(b30)} cycles when true RUL is 30 or less, against {rng(bmid)} cycles for RUL 60 to 125.",
       f"The 90% lower bound cuts the late-prediction rate from {lr:.0%} to {lc:.0%} at a mean slack of {rng(sl, '.0f')} cycles; coverage {rng(cv, '.2f')}, measured."]
if af and pf:
    fnd.append(f"Anomaly AUROC {rng(af, '.3f')}; persistent false-positive rate {min(pf):.2%} to {max(pf):.2%} on healthy rows.")
if be:
    fnd.append(f"Predictive thresholds beat run-to-failure above r = {rng(be)} (90% bound policy); in-sample, simulated.")
tile(s, 10.14, BY, 3.04, 2.1, "Key findings", fnd, NAVY, fs=7.6)
text(s, 0.15, 7.17, 13.0, 0.25, [("Benchmark replay of simulated NASA C-MAPSS data; all actions are simulated; thresholds are provisional. Detailed numbers, intervals and assumptions are in the appendix.", 6.5, False, GREY)])

# ------------------------------------------------------------------ appendix
s = frame("Data and evaluation protocol", BLUE, "data/CMAPSSData readme.txt, loaded files, configs/config.yaml")
rows = [["Subset", "Operating conditions", "Fault modes (readme)", "Train engines", "Test engines", "Train rows", "Sensors kept", "RUL cap"]]
for f in FDS:
    d = D[f]
    rows.append([f, META[f][0], META[f][1], len(np.unique(d["eng"])), ST[f]["n"], len(d["eng"]), len(d["P"]["pre"].sensors), int(d["cap"])])
table(s, 0.3, 0.85, 12.7, 1.9, rows, fs=10)
tile(s, 0.3, 3.0, 6.25, 1.9, "Labels and target",
     ["Train RUL = maximum cycle minus cycle. Test RUL = official RUL vector plus remaining cycles to the last observed cycle.",
      "Target capped at 125 (assumption). Low-RUL samples (RUL below 30) carry weight 3."], BLUE, 10)
tile(s, 6.75, 3.0, 6.25, 1.9, "Evaluation scope",
     ["Official test: prediction at the last observed cycle per engine, aligned with the official RUL vector.",
      "Capped labels are primary. Raw-label RMSE is inflated where true RUL exceeds the cap."], TEAL, 10)
tile(s, 0.3, 5.1, 6.25, 1.95, "Validation protocol",
     [f"{cfg['eval']['cv_folds']}-fold GroupKFold by engine for every selection. Out-of-fold predictions feed offsets, conformal bounds and cost replay.",
      "Seed 42. Bootstrap intervals resample whole engines. Official test labels never used for fitting or selection."], NAVY, 10)
tile(s, 6.75, 5.1, 6.25, 1.95, "Leakage status",
     ["Regime clusters, normalisation statistics and sensor selection are fitted on all training engines before cross-validation.",
      "Strict fold-wise refitting has not been run; a mild optimistic bias in CV is possible and unquantified."], ORANGE, 10)

s = frame("RUL accuracy on the official test set", TEAL, "recomputed from rul_boost_preds_gru.npz and official RUL files")
rows = [["Subset", "Engines", "RMSE (capped)", "MAE (capped)", "NASA (capped)", "RMSE (raw)", "Late rate", "Bias (cycles)"]]
for f in FDS:
    v = ST[f]
    rows.append([f, v["n"], f"{v['rmse']:.2f}", f"{v['mae']:.2f}", f"{v['nasa']:.0f}", f"{v['rmse_raw']:.2f}", f"{v['late']:.2f}", f"{v['bias']:+.2f}"])
table(s, 0.3, 0.85, 12.7, 1.6, rows, fs=10)
pic(s, ch_scatter(6.3, 4.3), 0.3, 2.65, 6.3, 4.3)
pic(s, ch_band(6.3, 4.3), 6.8, 2.65, 6.3, 4.3)

ab, md, vl = opt(ch_abl, 4.1, 3.0), opt(ch_models, 4.3, 3.0), (opt(ch_valid, 4.0, 3.0) if S else None)
if ab or md or vl:
    s = frame("Model selection evidence", TEAL, "rul_study*.json, rul_boost*.json, final_summary.json")
    for im, x, w in ((ab, 0.2, 4.1), (md, 4.4, 4.3), (vl, 8.8, 4.0)):
        if im:
            pic(s, im, x, 0.9, w, 3.0)
    tile(s, 0.3, 4.2, 12.7, 2.8, "Reading",
         ["Trend (slope) features carry most of the signal; adding window-10 features gives a consistent small CV gain on all subsets and a test-neutral result.",
          "Tuned hyperparameters, an ExtraTrees blend, causal prediction smoothing and a GRU blend gave no reliable gain under engine-bootstrap intervals, so the final model is XGBoost with default parameters.",
          "Baseline models come from a single run of an earlier study; selection was by engine-grouped CV, not by test score."], TEAL, 11)

s = frame("Conformal bound: calibration on training engines", NAVY, "artifacts/ext_conformal.json (nested engine split, training out-of-fold predictions only)")
rows = [["Subset", "Variant", "Nominal", "Held-out coverage", "SD", "All-rows coverage", "Band coverage (<30 / 30-60 / >=60)", "Selected"]]
for f in FDS:
    for k, v in conf[f]["nested"].items():
        var, al = k.split("|")
        rows.append([f, var, f"{1 - float(al):.2f}", f"{v['cut_cov_mean']:.3f}", f"{v['cut_cov_sd']:.3f}", f"{v['allrows_cov_mean']:.3f}",
                     " / ".join(f"{x:.2f}" for x in v["cut_band_cov"]), "yes" if var == conf[f]["variant"] else ""])
table(s, 0.3, 0.85, 12.7, 5.5, rows, [0.8, 1.6, 0.8, 1.2, 0.7, 1.2, 2.4, 0.8], fs=8.5)
text(s, 0.3, 6.45, 12.7, 0.7, [("Selection rule: lowest sum of marginal and per-band coverage deviation under cut-based evaluation, using training engines only. "
                                "The official test set is evaluated once, after the variant is frozen.", 9.5, False, INK)])

s = frame("Conformal bound: official test coverage", NAVY, "artifacts/ext_conformal.json; 95% engine-bootstrap intervals; capped labels")
rows = [["Subset", "Method", "Coverage", "95% CI", "Cov. below cap", "Mean slack", "Late rate", "RMSE", "NASA"]]
for f in FDS:
    for m in M:
        r = conf[f]["test"][m]
        lo, hi = r["ci"]["marg"]
        rows.append([f, MLAB[m], f"{r['cov_cap']:.3f}", f"{lo:.2f} to {hi:.2f}", f"{r['cov_lt_cap']:.3f}", f"{r['slack_mean']:.1f}",
                     f"{r['late_rate']:.2f}", f"{r['rmse_cap']:.2f}", f"{r['nasa_cap']:.0f}"])
table(s, 0.3, 0.85, 8.5, 6.2, rows, [0.7, 1.4, 0.8, 1.1, 1.0, 0.8, 0.7, 0.7, 0.7], fs=8)
pic(s, ch_slack(4.1, 3.4), 9.0, 0.9, 4.1, 3.4)
sl = [conf[f]["test"]["conformal_a0.1"]["slack_mean"] for f in FDS]
tile(s, 9.0, 4.5, 4.1, 2.5, "Reading",
     [f"The 90% bound costs {min(sl):.0f} to {max(sl):.0f} cycles of mean slack and raises RMSE and NASA score relative to the raw prediction.",
      "Coverage below the cap is the stricter measure; engines at the cap are covered almost trivially."], NAVY, 9.5)

s = frame("Cost-aware policy replay", ORANGE, "artifacts/ext_cost_policy.json; training trajectories; cross-fitted out-of-fold signals")
pic(s, ch_cost4(6.3, 4.6), 0.3, 0.85, 6.3, 4.6)
rows = [["Subset", "Signal", "Break-even r vs run-to-failure", "vs best fixed interval"]]
for f in FDS:
    for k in ("rul_raw", "rul_offset", "rul_conformal10"):
        b = cost[f]["break_even"][k]
        rows.append([f, PLAB[k].replace("RUL threshold, ", ""), fmt_be(b["vs_rtf"]), fmt_be(b["vs_fixed"])])
table(s, 6.8, 0.85, 6.3, 4.6, rows, [0.7, 1.1, 1.9, 1.9], fs=8.5)
tile(s, 0.3, 5.65, 12.8, 1.4, "Replay assumptions",
     [cost[FDS[0]]["assumptions"] + ".",
      "r = C_fail / C_waste; costs are in units of C_waste per engine. Swept thresholds make the raw, offset and bound policies near-reparametrisations, so compare them at fixed thresholds (next slide)."],
     ORANGE, 9.5)

if all(cost[f].get("fixed_thr") for f in FDS):
    s = frame("Cost replay at thresholds fixed in advance", ORANGE, "artifacts/ext_cost_policy.json (thresholds 15, 30, 60 cycles; no sweep)")
    for j, k in enumerate(("rul_raw", "rul_offset", "rul_conformal10")):
        x = 0.3 + j * 4.3
        sec(s, x, 0.8, 4.1, PLAB[k], PCOL[k])
        rows = [["Subset", "Thr", "Avoided", "Wasted", "Cost r=10", "Cost r=100"]]
        for f in FDS:
            for t in ("15", "30", "60"):
                v = cost[f]["fixed_thr"][k][t]
                rows.append([f, t, f"{v['failures_avoided']:.3f}", f"{v['mean_wasted']:.1f}", f"{v['cost']['10']:.1f}", f"{v['cost']['100']:.1f}"])
        table(s, x, 1.2, 4.1, 5.0, rows, [0.8, 0.5, 0.8, 0.8, 0.9, 0.9], fs=8)
    text(s, 0.3, 6.35, 12.7, 0.7, [("Avoided: share of engines maintained before failure. Wasted: mean remaining life at maintenance, over all engines. "
                                    "Cost in C_waste units per engine. Thresholds are not tuned on data.", 9.5, False, INK)])
else:
    print("fixed-threshold slide skipped: rerun scripts/11_cost_policy.py")

s = frame("Anomaly detection", TEAL, "artifacts/models/FD00X/anomaly_metrics.json (deployed Isolation Forest on latent and error)")
pic(s, ch_anom(6.3, 3.0), 0.3, 0.85, 6.3, 3.0)
pic(s, ch_lead(6.3, 3.0), 6.8, 0.85, 6.3, 3.0)
rows = [["Subset", "AUROC HI", "AUROC IFe", "AUROC IFl", "Persist TPR", "Persist FPR", "Lead median", "Lead p10", "Engine false alarm", "Unassessable"]]
for f in FDS:
    m = anom[f]
    if m:
        rows.append([f, f"{m['auroc_HI']:.4f}", f"{m['auroc_IF_error_1d']:.4f}", f"{m['auroc_IF_latent_error']:.4f}", f"{m['IFl_tpr_persist']:.3f}",
                     f"{m['IFl_fpr_persist']:.4f}", f"{m['IFl_lead_median']:.1f}", f"{m['IFl_lead_p10']:.1f}", f"{m['IFl_eng_fa_rate']:.3f}", int(m["eng_unassessable"])])
table(s, 0.3, 4.05, 12.7, 1.9, rows, fs=9)
text(s, 0.3, 6.05, 12.7, 1.1, [("Positives are rows with RUL <= 30 after cycle 30; negatives are rows at the RUL cap. Both are benchmark proxies because no failure-onset labels exist, "
                                "so precision and recall of fault onset cannot be established. Rows before cycle 30 are false-alarm heavy and are routed to REVIEW. "
                                "Persistent TPR is lowest on FD003.", 9.5, False, INK)])

fs_ = [f for f in FDS if ex.get(f, {}).get("faithfulness")]
if fs_:
    s = frame("Explainability: attribution and retraining faithfulness", TEAL, "artifacts/ext_explain.json; XGBoost exact TreeSHAP; retraining with 300 trees, engine-grouped 5-fold")
    pic(s, ch_shap(6.3, 4.3, fs_), 0.3, 0.85, 6.3, 4.3)
    rows = [["Subset", "k", "Remove-top minus random (95% CI)", "Keep-top minus random (95% CI)"]]
    for f in fs_:
        for k, v in ex[f]["faithfulness"]["k"].items():
            a, b = v["remove_diff"], v["keep_diff"]
            rows.append([f, k, f"{a[0]:+.2f} [{a[1]:+.2f}, {a[2]:+.2f}]", f"{b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}]"])
    table(s, 6.8, 0.85, 6.3, 4.3, rows, [0.8, 0.4, 2.4, 2.4], fs=8)
    vt = []
    for f in fs_:
        rem = [v["remove_diff"] for v in ex[f]["faithfulness"]["k"].values()]
        pos = [k for k, v in ex[f]["faithfulness"]["k"].items() if v["remove_diff"][1] > 0]
        vt.append(f"{f}: " + ("all remove-top intervals contain zero; no evidence that top-attributed sensors are uniquely necessary."
                              if all(a[1] <= 0 <= a[2] for a in rem) else "remove-top is worse than remove-random for k = " + ", ".join(pos) + "."))
    miss = [f for f in FDS if f not in fs_]
    if miss:
        vt.append("Faithfulness not yet computed for: " + ", ".join(miss) + ".")
    vt.append("Redundant sensors are a plausible explanation (HYPOTHESIS). SHAP ranks describe the fitted model, not physical cause.")
    tile(s, 0.3, 5.3, 12.8, 1.8, "Verdict (computed from the stored intervals)", vt, TEAL, 9.5)

cl = {f: ex[f]["clustering"] for f in FDS if ex.get(f, {}).get("clustering")}
if cl:
    s = frame("Unsupervised degradation-signature clustering", TEAL, "artifacts/ext_explain.json; normalised sensor slopes over the last 50 cycles before failure")
    rows = [["Subset", "Engines", "Silhouette (k=2)", "Gaussian-null p95", "ARI (k=2)", "ARI 95% interval", "Spearman(cluster, life)", "Cluster sizes"]]
    for f, c in cl.items():
        rows.append([f, c["n_engines"], f"{c['silhouette']['2']:.3f}", f"{c['null_sil_k2_p95']:.3f}", f"{c['ari_k2_mean']:.3f}",
                     f"{c['ari_k2_ci'][0]:.2f} to {c['ari_k2_ci'][1]:.2f}", f"{c.get('life_spearman_k2', float('nan')):+.2f}", str(c["cluster_sizes_k2"])])
    table(s, 0.3, 0.85, 12.7, 1.7, rows, fs=9.5)
    im = art(cfg, "figures", "ext_clusters.png")
    if im.exists():
        pic(s, im, 0.3, 2.7, 12.7, 2.86)
    text(s, 0.3, 5.7, 12.7, 1.4, [("FD001 and FD002 have one fault mode and act as negative controls; FD003 and FD004 exceed the Gaussian-covariance null most clearly. "
                                   "Clusters correlate with engine life, so degradation speed is a confound. Any mapping to fan or compressor degradation is a HYPOTHESIS: "
                                   "readme.txt lists two fault modes for FD003 and FD004 while the paper text describes HPC degradation only.", 9.5, False, INK)])

t = cfg["triage"]
s = frame("Triage rules and simulated work orders", ORANGE, f"artifacts/{WOB.name}/FD00X/WO-*.json; configs/config.yaml (thresholds provisional)")
rows = [["Subset", "REVIEW", "MEDIUM", "HIGH", "CRITICAL", "Orders", "Engines with order", "Engines tested"]]
for f in FDS:
    c = CNT[f]
    rows.append([f, c["REVIEW"], c["MEDIUM"], c["HIGH"], c["CRITICAL"], sum(c.values()), sum(1 for k in mx if k[0] == f), ST[f]["n"]])
rows.append(["Total", sum(CNT[f]["REVIEW"] for f in FDS), sum(CNT[f]["MEDIUM"] for f in FDS), sum(CNT[f]["HIGH"] for f in FDS),
             sum(CNT[f]["CRITICAL"] for f in FDS), sum(sum(CNT[f].values()) for f in FDS), N_ORD, NTOT])
table(s, 0.3, 0.85, 12.7, 2.1, rows, fs=10)
rows = [["Severity", "Rule (first match)"],
        ["CRITICAL", f"Conservative RUL <= {t['critical_rul']}, or persistent anomaly at or above the critical score cut"],
        ["HIGH", f"Conservative RUL <= {t['high_rul']}, or persistent anomaly at or above the high score cut"],
        ["MEDIUM", f"Conservative RUL <= {t['medium_rul']}, or persistent anomaly at or above the medium score cut"],
        ["LOW", "None of the above"],
        ["REVIEW", "Invalid input (NaN, infinite, negative) or fewer than 30 cycles of history; never LOW"]]
table(s, 0.3, 3.2, 12.7, 2.3, rows, [1.0, 8.0], fs=10)
tile(s, 0.3, 5.7, 12.7, 1.4, "Work-order lifecycle",
     [f"Persistence k = {cfg['anomaly']['persist_k']} consecutive cycles. Status OPEN_SIMULATED, deterministic identifiers, one order per engine per severity level, issued only on escalation, restart-safe.",
      "Conservative RUL is the base prediction minus the 0.8 quantile of out-of-fold over-prediction per predicted-RUL band. The conformal bound is not yet wired into the deployed triage."],
     ORANGE, 9.5)

cf, ld = (opt(ch_conf4, 6.3, 2.6), opt(ch_lead2, 6.3, 2.6)) if TR else (None, None)
if TR and cf and ld:
    s = frame("Triage evaluation at the last observed cycle", ORANGE, "artifacts/triage_v2_summary.json (true buckets reuse the triage cutoffs: assumption)")
    rows = [["Subset", "Engines", "True RUL<=30", "Triaged HIGH+", "Late rate base", "Late rate conservative", "NASA base", "NASA conservative", "REVIEW"]]
    for f in FDS:
        r = TR[f]
        rows.append([f, r["n_engines"], r["last_cycle"]["n_le30"], r["last_cycle"]["high_plus_of_le30"], f"{r['rul_base']['late_rate']:.2f}",
                     f"{r['rul_cons']['late_rate']:.2f}", f"{r['rul_base']['nasa']:.0f}", f"{r['rul_cons']['nasa']:.0f}", r["n_review_last_cycle"]])
    table(s, 0.3, 0.85, 12.7, 1.8, rows, fs=9.5)
    pic(s, cf, 0.3, 2.85, 6.3, 2.6)
    pic(s, ld, 6.8, 2.85, 6.3, 2.6)
    worse = [f for f in FDS if TR[f]["rul_cons"]["nasa"] > TR[f]["rul_base"]["nasa"]]
    tile(s, 0.3, 5.6, 12.7, 1.5, "Reading",
         ["Recall of engines near failure is driven mainly by the RUL model, which is most accurate there.",
          "The conservative offset lowers late predictions" + (f" but raises the last-cycle NASA score on {', '.join(worse)}: a safety trade-off, not an accuracy gain." if worse else " and did not raise NASA on any subset."),
          "Confusion cells: L, M, H, C = LOW, MEDIUM, HIGH, CRITICAL; rows are the true bucket, columns the triage severity."], ORANGE, 9.5)

s = frame("Retraining demonstration and a sample work order", GREEN, "artifacts/retrain_demo/*.json (FD001, SYNTHETIC drift); persisted work-order JSON")
rd = sorted(art(cfg, "retrain_demo", "x").parent.glob("*.json"))
if rd:
    R = sorted((json.loads(p.read_text()) for p in rd), key=lambda r: (r["synthetic_drift_std"], r["days_since"]))
    rows = [["Scenario", "Max PSI", "v1 val RMSE", "v1 arrivals RMSE", "Trigger", "v2 val RMSE", "v2 promoted"]]
    for r in R:
        lab = f"Synthetic drift {r['synthetic_drift_std']:g} sd" if r["synthetic_drift_std"] else (f"Scheduled, {r['days_since']} d" if r["days_since"] else "No drift, 0 d")
        trig = "+".join(k.upper() for k, v in r["trigger"].items() if v) or "none"
        rows.append([lab, f"{r['max_psi']:.2f}", f"{r['v1_val']['rmse']:.1f}", f"{r['v1_arrivals']['rmse']:.1f}", trig,
                     f"{r['v2_val']['rmse']:.1f}" if "v2_val" in r else "n/a", ("yes" if r["v2_promoted"] else "no") if "v2_val" in r else "n/a"])
    table(s, 0.3, 0.85, 6.3, 2.0, rows, [1.6, 0.6, 0.8, 0.9, 0.8, 0.8, 0.7], fs=8.5)
else:
    print("retrain_demo JSON missing; table omitted")
tile(s, 0.3, 3.1, 6.3, 3.9, "Reading",
     ["Triggers: RMSE above 1.25 times baseline, 7 days elapsed, or population stability index above 0.25 on any feature. Promotion only if the candidate beats the champion on engine-held-out validation RMSE; the previous champion stays under the registry alias.",
      "Drift is injected, not observed. In the drift scenario the validation set is shifted too, so only the PSI trigger fires. v2 also trains on 60% more labelled engines, so its gain combines adaptation and extra data.",
      "New labelled data arrives only after failures in a real deployment. Shown on FD001 only."], GREEN, 10)
wo = WOF.get("FD001")
if wo:
    keys = ["work_order_id", "incident_key", "status", "simulated", "engine_id", "replay_cycle", "current_cycle", "rul_mean", "rul_conservative", "anomaly_score",
            "health_index", "persistent_anomaly", "severity", "rule_fired", "recommendation", "requires_human_review", "model_version", "top_degradation_sensors"]
    ks = [k for k in keys if k in wo] or list(wo)
    sec(s, 6.8, 0.8, 6.3, "Simulated work order (FD001 replay, persisted JSON)", PURPLE)
    box_ = rect(s, 6.8, 1.2, 6.3, 5.8, LIGHT)
    put(box_, [(f"{k}: {', '.join(map(str, wo[k])) if isinstance(wo[k], list) else wo[k]}", 8.5, False, INK) for k in ks], anchor=MSO_ANCHOR.TOP, m=0.12, font="Courier New")

s = frame("Methodology and sources", NAVY, "configs/config.yaml; scripts 02, 03b, 09; references listed on this slide")
cap = int(D["FD001"]["cap"])
an, ft, ev, rl = cfg["anomaly"], cfg["features"], cfg["eval"], cfg["rul"]
tile(s, 0.3, 0.85, 6.3, 3.4, "Evaluation protocol",
     [f"RUL label: max_cycle minus cycle on training trajectories, capped at {cap}; test labels from the official RUL files, scored at the last observed cycle.",
      f"Features: trailing window {ft['window']} ({', '.join(ft['stats'])}) plus window 10 (mean, slope); causal, per engine.",
      f"Validation: {ev['cv_folds']}-fold GroupKFold by engine; random truncation at cycle >= {ev['min_cut_len']}, {ev['n_cut_reps']} repeats; test never used for selection.",
      "NASA score: sum of exp(-d/13) - 1 for d < 0 and exp(d/10) - 1 for d >= 0, with d = predicted minus true RUL; lower is better."], BLUE, 9.5)
tile(s, 6.8, 0.85, 6.3, 3.4, "Anomaly and conservative RUL",
     [f"Autoencoder {an['hidden'][0]}-{an['hidden'][1]}-{an['latent']} trained on healthy cycles {an['healthy_cycle_range'][0]} to {an['healthy_cycle_range'][1]}; Isolation Forest on latent state plus error; persistence k = {an['persist_k']}.",
      f"Conservative RUL: base prediction minus the {int(rl['cons_q'] * 100)}th percentile of out-of-fold over-prediction per predicted-RUL band (<30, 30-60, >=60), training engines only.",
      "Conformal bound: finite-sample (1 - alpha) quantile of the same residuals, per band, calibrated on one random cut per engine."], TEAL, 9.5)
refs = ["A. Saxena, K. Goebel, D. Simon, N. Eklund, Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation, PHM08, Denver, 2008.",
        "A. Saxena et al., Metrics for Evaluating Performance of Prognostics Techniques, PHM08, Denver, 2008.",
        "D. Frederick, J. DeCastro, J. Litt, User's Guide for C-MAPSS, NASA/ARL TM-2007-215026, 2007.",
        "T. Chen, C. Guestrin, XGBoost: A Scalable Tree Boosting System, KDD, 2016.",
        "F. T. Liu, K. M. Ting, Z.-H. Zhou, Isolation Forest, IEEE ICDM, 2008.",
        "G. E. Hinton, R. R. Salakhutdinov, Reducing the Dimensionality of Data with Neural Networks, Science 313, 2006.",
        "S. M. Lundberg et al., From local explanations to global understanding with explainable AI for trees, Nature Machine Intelligence 2, 2020.",
        "V. Vovk, A. Gammerman, G. Shafer, Algorithmic Learning in a Random World, Springer, 2005 (conformal and Mondrian conformal prediction).",
        "A. N. Angelopoulos, S. Bates, A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification, arXiv:2107.07511, 2021.",
        "F. Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 12, 2011.   |   A. Paszke et al., PyTorch, NeurIPS, 2019.   |   MLflow, mlflow.org."]
tile(s, 0.3, 4.4, 12.8, 2.7, "Sources", refs, NAVY, 8.5)

pl = ""
for pp in (art(cfg, "pytest_out.txt"), ROOT / "pytest_out.txt"):
    if pp.exists():
        ls_ = [l.strip() for l in pp.read_text().splitlines() if l.strip()]
        pl = ls_[-1] if ls_ else ""
        if pl:
            break
s = frame("Assumptions, limitations and open items", RED, "docs/SYSTEM_LOG.md; scripts 10, 11 and 12 audits")
tl2 = [("Replay scope", BLUE, "Historical replay of simulated data. No real telemetry, verified fault-onset labels, maintenance-system integration or field validation. All work orders and actions are simulated."
        + (f" Unit tests: {pl}. Passing tests do not imply full coverage." if pl else "")),
       ("Thresholds and proxies", TEAL, "Triage cutoffs of 15, 30 and 60 cycles are provisional, not industrial limits. True buckets in the triage evaluation reuse the same cutoffs. Anomaly positives are defined as RUL <= 30, a proxy."),
       ("Conformal bound", NAVY, "Coverage is empirical. Calibration cuts are uniform in life while test cuts sit about 60 percent into life. Cap-saturated engines inflate marginal coverage, so coverage below the cap is reported separately."),
       ("Cost replay", ORANGE, "Training trajectories with cross-fitted out-of-fold signals. Immediate action, failure at the final training cycle, no anomaly signal, no monetary costs. Parameters selected in sample; a cross-validated curve is provided."),
       ("Explainability", TEAL, "SHAP ranks describe the model, not physical cause. Faithfulness retraining uses 300 trees. Fan versus compressor mapping of clusters is a hypothesis."),
       ("Open items", RED, "Strict fold-wise preprocessing leakage check not run. Retraining shown on FD001 with synthetic drift only. Conformal bound not wired into deployed triage. Persistent true-positive rate varies across subsets. Dashboard not part of the verified results.")]
for i, (t_, col, b) in enumerate(tl2):
    tile(s, 0.3 + (i % 3) * 4.3, 0.85 + (i // 3) * 3.1, 4.1, 2.95, t_, b, col, 10.5)

out = art(cfg, "figures", "fleetpm_deck_v2.pptx")
prs.save(out)
print("saved", out, "| slides:", len(prs.slides))
