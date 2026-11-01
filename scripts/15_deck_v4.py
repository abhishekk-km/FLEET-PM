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



import io
import sys
from pathlib import Path
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION

# ------------------------------------------------------------------ template
def find_template():
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.exists():
            raise SystemExit(f"template not found: {p}")
        return p
    c = [p for p in ROOT.glob("**/*.pptx") if ("teliport" in p.name.lower() or "template" in p.name.lower())
         and "artifacts" not in p.parts and "backup" not in p.name.lower()]
    if not c:
        raise SystemExit("Teliport template not found. Save it as data/teliport_template.pptx or pass its path as an argument.")
    return sorted(c)[0]


TPL = find_template()
prs = Presentation(str(TPL))
if len(prs.slides) < 2:
    raise SystemExit("template must contain at least 2 slides (cover and content)")
lst = prs.slides._sldIdLst
for sid in list(lst)[2:]:
    prs.part.drop_rel(sid.rId)
    lst.remove(sid)
S2 = prs.slides[1]
SC = prs.slide_width / Inches(13.333)
print("template:", TPL.name, "| scale", round(SC, 3), "| slides kept 2")
for ph in list(S2.placeholders):
    if not ph.has_text_frame or not ph.text_frame.text.strip():
        ph._element.getparent().remove(ph._element)
LAYOUT2 = S2.slide_layout
dec_layout = any(sh.shape_type == 13 for sh in list(LAYOUT2.shapes) + list(LAYOUT2.slide_master.shapes))
S2_DECOR = []
for sh in S2.shapes:
    if sh.shape_type == 13:
        S2_DECOR.append(("pic", sh.image.blob, sh.left, sh.top, sh.width, sh.height, (sh.crop_left, sh.crop_right, sh.crop_top, sh.crop_bottom)))
    elif sh.shape_type == 1 and not (sh.has_text_frame and sh.text_frame.text.strip()):
        try:
            lc, lw = sh.line.color.rgb, sh.line.width
        except Exception:
            lc, lw = None, None
        S2_DECOR.append(("rect", sh.auto_shape_type, sh.left, sh.top, sh.width, sh.height, (lc, lw)))
NEED_COPY = bool(S2_DECOR) and not dec_layout
print("decor on slide 2:", len(S2_DECOR), "| layout provides logos:", dec_layout, "| copying to appendix:", NEED_COPY)


def new_slide():
    s_ = prs.slides.add_slide(LAYOUT2)
    for ph in list(s_.placeholders):
        ph._element.getparent().remove(ph._element)
    if NEED_COPY:
        for kind, a, l, t, w, h, ex_ in S2_DECOR:
            if kind == "pic":
                s_pc = s_.shapes.add_picture(io.BytesIO(a), l, t, w, h)
                if ex_:
                    s_pc.crop_left, s_pc.crop_right, s_pc.crop_top, s_pc.crop_bottom = ex_
            else:
                r_ = s_.shapes.add_shape(a, l, t, w, h)
                r_.fill.background()
                r_.shadow.inherit = False
                if ex_ and ex_[0] is not None:
                    r_.line.color.rgb = ex_[0]
                    if ex_[1]:
                        r_.line.width = ex_[1]
                else:
                    r_.line.fill.background()
    return s_


# ------------------------------------------------------------------ coordinate transform (template inches)
ID = dict(ax=SC, bx=0.0, ay=SC, by=0.0, f=SC)
APP = dict(ax=0.988 * SC, bx=0.102 * SC, ay=0.96875 * SC, by=(0.9 - 0.75 * 0.96875) * SC, f=0.97 * SC)
TR = dict(ID)
_rect, _text, _line, _pic, _table, _put = rect, text, line, pic, table, put


def rect(s, x, y, w, h, *a, **k):
    return _rect(s, TR["ax"] * x + TR["bx"], TR["ay"] * y + TR["by"], TR["ax"] * w, TR["ay"] * h, *a, **k)


def text(s, x, y, w, h, *a, **k):
    return _text(s, TR["ax"] * x + TR["bx"], TR["ay"] * y + TR["by"], TR["ax"] * w, TR["ay"] * h, *a, **k)


def line(s, x1, y1, x2, y2, *a, **k):
    return _line(s, TR["ax"] * x1 + TR["bx"], TR["ay"] * y1 + TR["by"], TR["ax"] * x2 + TR["bx"], TR["ay"] * y2 + TR["by"], *a, **k)


def pic(s, path, x, y, w, h):
    return _pic(s, path, TR["ax"] * x + TR["bx"], TR["ay"] * y + TR["by"], TR["ax"] * w, TR["ay"] * h)


def table(s, x, y, w, h, rows, widths=None, fs=8.5):
    return _table(s, TR["ax"] * x + TR["bx"], TR["ay"] * y + TR["by"], TR["ax"] * w, TR["ay"] * h, rows, widths, fs * TR["f"])


def put(sh, paras, *a, **k):
    return _put(sh, [(t, sz * TR["f"], b, c) for (t, sz, b, c) in paras], *a, **k)


def Xe(v): return int(Inches(TR["ax"] * v + TR["bx"]))
def Ye(v): return int(Inches(TR["ay"] * v + TR["by"]))
def We(v): return int(Inches(TR["ax"] * v))
def He(v): return int(Inches(TR["ay"] * v))


# ------------------------------------------------------------------ native charts
def style_chart(ch, title):
    ch.font.size = Pt(7 * TR["f"])
    ch.font.name = "Arial"
    ch.has_title = True
    tf = ch.chart_title.text_frame
    tf.text = title
    r_ = tf.paragraphs[0].runs[0]
    r_.font.size, r_.font.bold, r_.font.color.rgb = Pt(7.5 * TR["f"]), True, rgb(NAVY)
    ch.chart_title.include_in_layout = False


def native_cov(s, x, y, w, h):
    cd = CategoryChartData()
    cd.categories = ["Raw", "Offset 0.8 q", "Conformal 80%", "Conformal 90%"]
    cd.add_series("Mean coverage", [float(np.mean([conf[f]["test"][m]["cov_lt_cap"] for f in FDS])) for m in M])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Xe(x), Ye(y), We(w), He(h), cd).chart
    style_chart(ch, "Lower-Bound Coverage, Engines Below Cap (Mean of 4 Subsets)")
    ch.has_legend = False
    pl = ch.plots[0]
    pl.gap_width = 45
    pl.has_data_labels = True
    dl = pl.data_labels
    dl.number_format, dl.number_format_is_linked = "0.00", False
    dl.font.size = Pt(7 * TR["f"])
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for i, m in enumerate(M):
        pt = pl.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(MCOL[m])
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = 0, 1.0, 0.2
    va.major_gridlines.format.line.color.rgb = rgb("E6EBF2")
    va.tick_labels.font.size = Pt(6.5 * TR["f"])
    ch.category_axis.tick_labels.font.size = Pt(6.5 * TR["f"])


def set_log(ax):
    sc = ax._element.find(qn("c:scaling"))
    sc.insert(0, sc.makeelement(qn("c:logBase"), {"val": "10"}))


def native_cost(s, x, y, w, h):
    cd = XyChartData()
    idx = np.unique(np.linspace(0, len(RGRID) - 1, 40).astype(int))
    for k in PLAB:
        se = cd.add_series(PLAB[k])
        c = np.mean([cost[f]["curves"][k] for f in FDS], 0)
        for i in idx:
            se.add_data_point(float(RGRID[i]), float(max(c[i], 0.1)))
    ch = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Xe(x), Ye(y), We(w), He(h), cd).chart
    style_chart(ch, "Cost-aware replay, mean of 4 subsets (simulated)")
    for ser, k in zip(ch.plots[0].series, PLAB):
        ser.format.line.color.rgb = rgb(PCOL[k])
        ser.format.line.width = Pt(1.75)
        ser.smooth = False
    for ax, ttl, lo_, hi_ in ((ch.category_axis, "r = C_fail / C_waste", 1, 1000), (ch.value_axis, "Mean cost per engine", 0.1, 1000)):
        try:
            set_log(ax)
        except Exception as e_:
            print("log axis not applied:", e_)
        ax.minimum_scale, ax.maximum_scale = lo_, hi_
        ax.has_title = True
        ax.axis_title.text_frame.text = ttl
        ax.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(6.5 * TR["f"])
        ax.tick_labels.font.size = Pt(6.5 * TR["f"])
        ax.major_gridlines.format.line.color.rgb = rgb("E6EBF2")
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(6 * TR["f"])


def ring_img(name, fr, col, txt, fr2=None, col2=None):
    fig, ax = plt.subplots(figsize=(0.9, 0.9))
    ax.pie([fr, 1 - fr], startangle=90, counterclock=False, colors=[H(col), "#E6EBF2"], radius=1.0, wedgeprops=dict(width=0.26, edgecolor="white"))
    if fr2 is not None:
        ax.pie([fr2, 1 - fr2], startangle=90, counterclock=False, colors=[H(col2), "#E6EBF2"], radius=0.7, wedgeprops=dict(width=0.26, edgecolor="white"))
    ax.text(0, 0, txt, ha="center", va="center", fontsize=6.5, fontweight="bold", color="#0B2A4A")
    ax.set_aspect("equal")
    fig.subplots_adjust(0, 0, 1, 1)
    p = art(cfg, "figures", f"deck4_{name}.png")
    fig.savefig(p, dpi=250, transparent=True)
    plt.close(fig)
    return str(p)


# ------------------------------------------------------------------ shared numbers
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

# ================================================================== SLIDE 2 (the one-pager, on the template's content page)
TB_NAVY, TB_BLUE, TB_SKY, TB_CYAN, TB_GOLD, TB_BG, TB_LINE, TB_AMB = "0B2F6B", "1566C0", "2F8FD0", "0E9AA7", "F2A900", "EEF5FC", "BCD2EA", "C98A00"
TITLE = "SENTINEL: Closed-Loop Predictive Maintenance for a Connected Fleet"


def tint(h, a=0.86):
    c = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "".join(f"{int(v + (255 - v) * a):02X}" for v in c)


def ring_icon(name, fr, col, kind, fr2=None, col2=None):
    fig, ax = plt.subplots(figsize=(0.9, 0.9))
    ax.pie([fr, 1 - fr], startangle=90, counterclock=False, colors=[H(col), "#E6EBF2"], radius=1.0, wedgeprops=dict(width=0.26, edgecolor="white"))
    if fr2 is not None:
        ax.pie([fr2, 1 - fr2], startangle=90, counterclock=False, colors=[H(col2), "#E6EBF2"], radius=0.7, wedgeprops=dict(width=0.26, edgecolor="white"))
    k = 0.62 if fr2 is not None else 1.0
    ic, t = "#0B2F6B", np.linspace(0, 2 * np.pi, 100)
    pl = lambda x, y, lw=1.3: ax.plot(np.asarray(x) * k, np.asarray(y) * k, c=ic, lw=lw, solid_capstyle="round")
    if kind == "gear":
        pl(.2 * np.cos(t), .2 * np.sin(t))
        for a in np.arange(0, 2 * np.pi, np.pi / 4):
            pl([.28 * np.cos(a), .42 * np.cos(a)], [.28 * np.sin(a), .42 * np.sin(a)], 2)
    elif kind == "target":
        for r_ in (.42, .26, .1):
            pl(r_ * np.cos(t), r_ * np.sin(t))
    elif kind == "pulse":
        pl([-.45, -.22, -.12, 0, .12, .24, .45], [0, 0, .32, -.32, .18, 0, 0], 1.6)
    elif kind == "shield":
        pl([-.3, .3, .3, 0, -.3, -.3], [.38, .38, 0, -.42, 0, .38])
        pl([-.12, -.02, .14], [0, -.1, .12], 1.6)
    elif kind == "clock":
        pl(.4 * np.cos(t), .4 * np.sin(t))
        pl([0, 0], [0, .26], 1.5)
        pl([0, .18], [0, 0], 1.5)
    elif kind == "bell":
        a = np.linspace(0, np.pi, 40)
        pl(.3 * np.cos(a), .05 + .3 * np.sin(a))
        pl([-.3, -.38, .38, .3], [.05, -.22, -.22, .05])
        pl(.05 * np.cos(t), -.32 + .05 * np.sin(t))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    p = art(cfg, "figures", f"deck5_{name}.png")
    fig.savefig(p, dpi=250, transparent=True)
    plt.close(fig)
    return str(p)


def native_cost2(s_, x, y, w, h):
    cd = XyChartData()
    idx = np.unique(np.linspace(0, len(RGRID) - 1, 40).astype(int))
    for k in PLAB:
        se = cd.add_series(PLAB[k])
        c = np.mean([cost[f]["curves"][k] for f in FDS], 0)
        for i in idx:
            se.add_data_point(float(np.log10(RGRID[i])), float(np.log10(max(c[i], 0.5))))
    ch = s_.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Xe(x), Ye(y), We(w), He(h), cd).chart
    style_chart(ch, "Cost-aware replay, mean of 4 subsets (simulated, log10 axes)")
    for ser, k in zip(ch.plots[0].series, PLAB):
        ser.format.line.color.rgb = rgb(PCOL[k])
        ser.format.line.width = Pt(1.75)
        ser.smooth = False
    for ax, ttl in ((ch.category_axis, "log10 r, r = C_fail / C_waste"), (ch.value_axis, "log10 cost per engine")):
        ax.minimum_scale, ax.maximum_scale, ax.major_unit = 0, 3, 1
        ax.has_title = True
        ax.axis_title.text_frame.text = ttl
        ax.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(6 * TR["f"])
        ax.tick_labels.font.size = Pt(6.5 * TR["f"])
        ax.major_gridlines.format.line.color.rgb = rgb("E6EBF2")
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(5.8 * TR["f"])


from matplotlib.patches import Circle, Polygon


def ring_icon(name, fr, col, kind, fr2=None, col2=None):
    fig, ax = plt.subplots(figsize=(0.9, 0.9))
    ax.pie([fr, 1 - fr], startangle=90, counterclock=False, colors=[H(col), "#E6EBF2"], radius=1.0, wedgeprops=dict(width=0.26, edgecolor="white"))
    if fr2 is not None:
        ax.pie([fr2, 1 - fr2], startangle=90, counterclock=False, colors=[H(col2), "#E6EBF2"], radius=0.7, wedgeprops=dict(width=0.26, edgecolor="white"))
    k = 0.6 if fr2 is not None else 1.0
    ic = "#0B2F6B"
    SC_ = lambda v: np.asarray(v, float) * k
    poly = lambda xs, ys, fc=ic: ax.add_patch(Polygon(np.c_[SC_(xs), SC_(ys)], closed=True, fc=fc, ec="none", zorder=5))
    circ = lambda r, fc=ic, x=0.0, y=0.0: ax.add_patch(Circle((x * k, y * k), r * k, fc=fc, ec="none", zorder=5))
    ln = lambda xs, ys, lw=2.4, c=ic: ax.plot(SC_(xs), SC_(ys), c=c, lw=lw, solid_capstyle="round", zorder=6)
    if kind == "gear":
        px, py = [], []
        for i in range(8):
            for da, r_ in ((-10, .38), (-7, .5), (7, .5), (10, .38)):
                a = np.radians(i * 45 + da)
                px.append(r_ * np.cos(a))
                py.append(r_ * np.sin(a))
        poly(px, py)
        circ(.15, "white")
    elif kind == "target":
        circ(.46)
        circ(.33, "white")
        circ(.21)
        circ(.09, "white")
    elif kind == "pulse":
        ln([-.5, -.25, -.13, 0, .13, .26, .5], [0, 0, .36, -.36, .2, 0, 0], 2.8)
    elif kind == "shield":
        poly([-.34, .34, .34, .26, 0, -.26, -.34], [.4, .4, .05, -.22, -.46, -.22, .05])
        ln([-.14, -.03, .17], [0, -.12, .14], 2.4, "white")
    elif kind == "clock":
        circ(.45)
        ln([0, 0], [0, .28], 2.4, "white")
        ln([0, .2], [0, 0], 2.4, "white")
    elif kind == "bell":
        a = np.linspace(0, np.pi, 30)
        poly(list(.3 * np.cos(a)) + [.42, -.42], list(.08 + .32 * np.sin(a)) + [-.2, -.2])
        circ(.07, ic, 0, -.33)
        circ(.05, ic, 0, .44)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    p = art(cfg, "figures", f"deck6_{name}.png")
    fig.savefig(p, dpi=250, transparent=True)
    plt.close(fig)
    return str(p)


# MINDMAP SLIDE (final structured slide)
from pptx.enum.chart import XL_MARKER_STYLE
s = S2
TR.update(ID)
MIDB, DCY, ORG = "3B7DD8", "0B7F8A", "E36414"
BASE = "https://abhishekk-km.github.io/FLEET-PM/interactive/"
REPO = "https://github.com/abhishekk-km/FLEET-PM"
NAV_ON = False
TITLE = "SENTINEL: Closed-Loop Predictive Maintenance for a Connected Fleet"
an_, tg_, dcfg = cfg["anomaly"], cfg["triage"], cfg["data"]
nfeat = [len(D[f]["P"]["pre"].sensors) * (len(cfg["features"]["stats"]) + 2) for f in FDS]
nsen = [len(D[f]["P"]["pre"].sensors) for f in FDS]
rc_ = cfg["mlops"]["retrain"]
NA = lambda v, fm=".2f": rng(v, fm) if v else "n/a"
lead = [anom[f]["IFl_lead_median"] for f in FA]
sl = [conf[f]["test"]["conformal_a0.1"]["slack_mean"] for f in FDS]
CEN, LEFT, MID_ = PP_ALIGN.CENTER, PP_ALIGN.LEFT, MSO_ANCHOR.MIDDLE


def lnk(sh, url):
    sh.click_action.hyperlink.address = url


def lbtn(x, y, w, h, label, url, fill=TB_NAVY, col="FFFFFF", fs=6.4):
    b = rect(s, x, y, w, h, fill, None, MSO_SHAPE.ROUNDED_RECTANGLE)
    put(b, [(label, fs, True, col)], align=CEN, m=0.03)
    lnk(b, url)


def badge(x, y, n):
    put(rect(s, x, y, 0.22, 0.22, TB_GOLD, "FFFFFF", MSO_SHAPE.OVAL), [(str(n), 7.5, True, TB_NAVY)], align=CEN, m=0.0)


def itag(x, y, w, h, url):
    lbtn(x + w - 0.62, y + h - 0.15, 0.58, 0.13, "Interactive", url, TB_GOLD, TB_NAVY, 4.6)


# ---------------- title
text(s, 2.7, 0.08, 9.4, 0.4, [(TITLE, 16, True, TB_NAVY)], align=CEN, anchor=MID_)
text(s, 2.7, 0.46, 9.4, 0.3, [("From sensor readings to a maintenance work order: sense, prepare, detect, predict, decide, act, learn | NASA turbofan engine data, four subsets | Benchmark replay with simulated actions", 6.6, False, TB_BLUE)], align=CEN, anchor=MID_)
rect(s, 2.7, 0.78, 9.4, 0.03, TB_GOLD)

# ---------------- KPI row
kp = [("Engines Escalated", f"{N_ORD} / {NTOT}", "engines given a simulated work order", N_ORD / NTOT, TB_BLUE, "gear", None, None),
      ("Life-Prediction Error", rng(rm), "cycles, root mean squared; ring = share of the 125-cycle cap", nr_, TB_CYAN, "target", None, None)]
if af:
    kp.append(("Anomaly Detection Score", NA(af, ".3f"), "area under the curve; 1.0 is perfect", min(af), TB_SKY, "pulse", None, None))
kp.append(("Safe-Bound Reliability", rng(cv, ".2f"), "engines whose true life stayed above the 90% bound", min(cv), TB_NAVY, "shield", None, None))
kp.append(("Late (Unsafe) Predictions", f"{lr:.0%} to {lc:.0%}", "overestimates: raw (outer) vs with bound (inner)", lr, RED, "clock", lc, GREEN))
if tp:
    kp.append(("Early-Warning Recall", NA(tp), "near-failure readings flagged in time", min(tp), TB_GOLD, "bell", None, None))
KW = (12.83 - 0.07 * (len(kp) - 1)) / len(kp)
for i, (lab, val, sub, fr, col, kind, fr2, col2) in enumerate(kp):
    x = 0.25 + i * (KW + 0.07)
    rect(s, x, 0.86, KW, 0.69, TB_BG, TB_LINE)
    rect(s, x, 0.86, 0.05, 0.69, col)
    pic(s, ring_icon(f"v14k{i}", fr, col, kind, fr2, col2), x + 0.08, 0.9, 0.6, 0.6)
    text(s, x + 0.72, 0.87, KW - 0.74, 0.17, [(lab.upper(), 5.6, True, TB_BLUE)])
    text(s, x + 0.72, 1.02, KW - 0.74, 0.28, [(val, 11, True, TB_NAVY)])
    text(s, x + 0.72, 1.29, KW - 0.74, 0.26, [(sub, 5.4, False, "5B6B80")])

# ---------------- 1 story band
badge(0.13, 1.6, 1)
story = [("The Problem", TB_NAVY, "Machines such as engines are repaired after they break or on a fixed calendar. Breakdowns cause unplanned downtime; fixed servicing replaces parts that are still healthy. Yet sensor readings change gradually long before failure, which makes earlier action possible."),
         ("The Objective", TB_BLUE, "Build a closed loop that reads sensor data, spots abnormal behaviour, estimates how many operating cycles remain before failure, decides urgency with fixed rules and raises a maintenance work order automatically. People review only the exceptions."),
         ("Constraints and Coverage", TB_CYAN, f"No large language model; open-source tools only; failures are rare, so models learn from healthy behaviour. All four NASA turbofan subsets are processed ({NTOT} test engines; 1 or 6 operating conditions; 1 or 2 fault modes).")]
SW = (12.83 - 0.12 * 2) / 3
for i, (lab, col, body) in enumerate(story):
    x = 0.25 + i * (SW + 0.12)
    rect(s, x, 1.61, 0.05, 0.52, col)
    put(rect(s, x + 0.05, 1.61, SW - 0.05, 0.52, tint(col, 0.9), TB_LINE), [(lab.upper(), 6, True, col), (body, 5.9, False, INK)], align=LEFT, m=0.08 if i else 0.2)

# ---------------- 2 architecture
rect(s, 0.25, 2.2, 8.8, 2.82, "F7FAFE", TB_LINE)
hdr = rect(s, 0.25, 2.2, 8.8, 0.26, TB_NAVY)
put(rect(s, 0.3, 2.22, 0.22, 0.22, "FFFFFF", None, MSO_SHAPE.OVAL), [("2", 7.5, True, TB_NAVY)], align=CEN, m=0.0)
text(s, 0.58, 2.2, 6.2, 0.26, [("ARCHITECTURE: DATA \u2192 FEATURES \u2192 MODELS \u2192 DECISION AGENT \u2192 WORK ORDER", 7.4, True, "FFFFFF")], anchor=MID_)
text(s, 6.6, 2.2, 2.4, 0.26, [("closed loop, no manual steps", 6, False, "C9D6E6")], align=PP_ALIGN.RIGHT, anchor=MID_)
chev = [("1 Sense", "telemetry in", TB_NAVY, "FFFFFF"), ("2 Prepare", "validate, normalise", TB_BLUE, "FFFFFF"), ("3 Engineer", "causal features", TB_SKY, "FFFFFF"),
        ("4 Detect", "anomaly scoring", TB_CYAN, "FFFFFF"), ("5 Predict", "life and safe bound", DCY, "FFFFFF"), ("6 Decide", "rule triage", TB_GOLD, TB_NAVY),
        ("7 Act", "work orders", TB_AMB, "FFFFFF"), ("8 Learn", "gated retraining", MIDB, "FFFFFF")]
for i, (a_, b_, col, tcol) in enumerate(chev):
    put(rect(s, 0.3 + i * 1.085, 2.5, 1.12, 0.3, col, None, MSO_SHAPE.PENTAGON), [(a_, 7, True, tcol), (b_, 5.2, False, tcol)], align=CEN, m=0.02)
SX = lambda i: 0.3 + i * 1.76
for lab, c0, c1, col in [("DATA ENGINEERING: PREPARE THE DATA", 0, 1, TB_NAVY), ("MACHINE LEARNING: DETECT, PREDICT", 2, 2, TB_CYAN), ("DECISION AGENT WITHOUT A LARGE LANGUAGE MODEL: DECIDE AND ACT", 3, 4, TB_AMB)]:
    put(rect(s, SX(c0), 2.84, SX(c1) + 1.64 - SX(c0), 0.15, tint(col, 0.8), col), [(lab, 5, True, col)], align=CEN, m=0.01)
stg = [
    ("Ingest and Validate", TB_NAVY, TB_NAVY, [
        "Reads 26 values per operating cycle (3 operating settings, 21 sensors) from the NASA files; a live data feed can replace the files.",
        "Rejects missing values, duplicate cycles and gaps; aligns the official remaining-life labels.",
        f"Rescales sensors within each operating condition; keeps the {min(nsen)} to {max(nsen)} sensors that trend with wear."]),
    ("Causal Features", TB_BLUE, TB_BLUE, [
        f"Smooths each sensor with a median over the last {dcfg['smooth_window']} cycles, using past data only (no peeking at the future).",
        "Summarises the last 30 and 10 cycles: mean, spread, minimum, maximum, slope and distribution shape (kurtosis).",
        f"Target: remaining useful life, capped at {cap_} cycles; {min(nfeat)} to {max(nfeat)} features per row."]),
    ("Models", TB_CYAN, TB_CYAN, [
        f"Anomaly: an autoencoder learns healthy behaviour; an Isolation Forest scores deviations; {an_['persist_k']} flagged cycles in a row raise an alert.",
        f"Life: gradient-boosted trees (XGBoost, {cfg['rul']['xgb']['n_estimators']} trees) estimate remaining cycles; SHAP names the responsible sensors.",
        "Safe bound: the prediction minus a calibrated margin, so overestimates become rare."]),
    ("Triage Rules", TB_GOLD, TB_AMB, [
        "Fixed, auditable rules; the first match wins and is recorded.",
        f"Critical at {tg_['critical_rul']} cycles or fewer, high at {tg_['high_rul']}, medium at {tg_['medium_rul']} (provisional limits), or on a persistent anomaly.",
        "Invalid input, or under 30 cycles of history, goes to a person for REVIEW; it is never marked healthy."]),
    ("Work Orders", TB_SKY, TB_SKY, [
        "Writes a simulated maintenance order with the rule that fired, the estimated life and the top 3 responsible sensors.",
        "The same input always gives the same order identifier; one order per severity level, issued only on escalation.",
        "Critical orders are flagged for human review. No physical action is ever triggered."])]
for i, (nm, col, dc, bl) in enumerate(stg):
    x = SX(i)
    put(rect(s, x, 3.03, 0.4, 0.4, col, None, MSO_SHAPE.OVAL), [(str(i + 1), 11, True, TB_NAVY if col == TB_GOLD else "FFFFFF")], align=CEN, m=0.0)
    text(s, x + 0.44, 3.03, 1.2, 0.4, [(nm, 7.2, True, dc)], anchor=MID_)
    put(rect(s, x, 3.46, 1.64, 1.14, tint(col, 0.9), col), [("\u2022 " + b, 5.2, False, INK) for b in bl], align=LEFT, anchor=MSO_ANCHOR.TOP, m=0.05)
    if i < 4:
        rect(s, x + 1.65, 3.15, 0.11, 0.16, "334155", None, MSO_SHAPE.RIGHT_ARROW)
put(rect(s, 0.3, 4.64, 0.9, 0.36, TB_NAVY, None, MSO_SHAPE.ROUNDED_RECTANGLE), [("MODEL OPERATIONS LOOP", 5.4, True, "FFFFFF")], align=CEN, m=0.02)
ops = ["MLflow: records the settings, scores and model files of every run",
       "Registry: a new model replaces the current one only if validation error improves",
       f"Drift gate: retrain when the population stability index exceeds {rc_['psi_alarm']}, error exceeds {rc_['rmse_factor']} times baseline, or {rc_['days']} days pass",
       "Retraining: validated on held-out engines; the previous model is kept for rollback",
       "Monitor: read-only Streamlit dashboard of predictions, bounds and work orders"]
CWp = (8.75 - 0.95 - 0.05 * 4) / 5
for j, t_ in enumerate(ops):
    put(rect(s, 1.25 + j * (CWp + 0.05), 4.64, CWp, 0.36, tint(TB_NAVY, 0.84), TB_NAVY, MSO_SHAPE.ROUNDED_RECTANGLE), [(t_, 4.8, True, TB_NAVY)], align=CEN, m=0.02)

# ---------------- 3 evidence: native charts on the right
badge(9.05, 2.2, 3)


def n_cov(x, y, w, h):
    cd = CategoryChartData()
    cd.categories = ["Raw prediction", "Offset bound", "Conformal 80%", "Conformal 90%"]
    cd.add_series("Share of engines", [float(np.mean([conf[f]["test"][m]["cov_lt_cap"] for f in FDS])) for m in M])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Xe(x), Ye(y), We(w), He(h), cd).chart
    style_chart(ch, "Does the Lower Bound Stay Below the True Life? (Share of Engines; Targets 0.80 and 0.90)")
    ch.has_legend = False
    pl = ch.plots[0]
    pl.gap_width = 45
    pl.has_data_labels = True
    pl.data_labels.number_format, pl.data_labels.number_format_is_linked = "0.00", False
    pl.data_labels.font.size = Pt(7 * TR["f"])
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    for i, m in enumerate(M):
        pt = pl.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(MCOL[m])
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = 0, 1.0, 0.2
    va.major_gridlines.format.line.color.rgb = rgb("E6EBF2")
    va.tick_labels.font.size = Pt(6.5 * TR["f"])
    ch.category_axis.tick_labels.font.size = Pt(6.5 * TR["f"])


def n_cost(x, y, w, h):
    rs = [v["r"] for v in cost[FDS[0]]["at_r"]["rul_raw"]]
    cd = CategoryChartData()
    cd.categories = [f"{r:g}" for r in rs]
    order = [("run_to_failure", "Run to failure (baseline)"), ("fixed_interval", "Fixed service interval"), ("rul_raw", "Predictive, raw prediction"),
             ("rul_offset", "Predictive, offset bound"), ("rul_conformal10", "Predictive, 90% bound")]
    for k, lab in order:
        cd.add_series(lab, [float(np.mean([cost[f]["at_r"][k][j]["cost"] / rs[j] for f in FDS])) for j in range(len(rs))])
    ch = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Xe(x), Ye(y), We(w), He(h), cd).chart
    style_chart(ch, "Maintenance Cost Relative to Running to Failure (Lower Is Better; Simulated)")
    for ser, (k, _) in zip(ch.plots[0].series, order):
        ser.format.line.color.rgb = rgb(PCOL[k])
        ser.format.line.width = Pt(1.75)
        ser.smooth = False
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 4
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(PCOL[k])
    va, ca = ch.value_axis, ch.category_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = 0, 1.1, 0.2
    va.major_gridlines.format.line.color.rgb = rgb("E6EBF2")
    va.tick_labels.font.size = Pt(6.5 * TR["f"])
    ca.tick_labels.font.size = Pt(6.5 * TR["f"])
    ca.has_title = True
    ca.axis_title.text_frame.text = "r = cost of one failure \u00f7 cost of one wasted operating cycle"
    ca.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(6 * TR["f"])
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(5.5 * TR["f"])


n_cov(9.12, 2.2, 3.96, 1.38)
itag(9.12, 2.2, 3.96, 1.38, BASE + "coverage.html")
n_cost(9.12, 3.62, 3.96, 1.4)
itag(9.12, 3.62, 3.96, 1.4, BASE + "cost_replay.html")

# ---------------- bottom row: design reasoning table
RY, RH = 5.08, 2.0
rows = [["Design Decision", "Reason", "Evidence"],
        ["Cap Remaining Life at 125 Cycles", "Early in life the sensors show no wear, so the model is not asked to guess beyond 125 cycles.", f"Error {rng(rm)} cycles"],
        ["Gradient-Boosted Trees", "Strong on tabular window features and explainable sensor by sensor; seven model families were compared and none was reliably better.", "Seven families compared"],
        ["Learn from Healthy Behaviour", "Failures are rare, so the anomaly model trains on healthy cycles only and needs no failure labels.", f"Detection score {NA(af, '.3f')}"],
        [f"Require {an_['persist_k']} Flags in a Row", "A single noisy reading should not trigger an alert; flags must persist.", f"False alarms {min(pf):.2%} to {max(pf):.2%}" if pf else "n/a"],
        ["Shift Predictions Earlier", "Overestimating remaining life is the dangerous error, so the safe bound deliberately errs early.", f"Late {lr:.0%} to {lc:.0%}"],
        ["Validate by Whole Engine", "Rows from one engine resemble each other; splitting by engine prevents leakage into validation.", f"{cfg['eval']['cv_folds']} folds, grouped"]]
table(s, 0.25, RY, 4.1, RH, rows, [1.05, 2.15, 0.9], 5.1)


# ---------------- bottom row: native charts
def n_models(x, y, w, h):
    names = [("ridge", "Ridge regression"), ("extratrees", "Extra trees"), ("lstm", "Long short-term memory"), ("gru", "Gated recurrent unit"),
             ("cnn1d", "1-D convolutional network"), ("cnnlstm", "Convolution plus memory")]
    st = {f: json.loads(art(cfg, "models", f, "rul_study.json").read_text()) for f in FDS}
    vals = [(lab, float(np.mean([st[f][k]["test_cap_rmse"] for f in FDS]))) for k, lab in names]
    vals.append(("Gradient-boosted trees (selected)", float(np.mean(rm))))
    vals.sort(key=lambda t: -t[1])
    cd = CategoryChartData()
    cd.categories = [a for a, _ in vals]
    cd.add_series("Test error, cycles", [b for _, b in vals])
    ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Xe(x), Ye(y), We(w), He(h), cd).chart
    style_chart(ch, "Test Error by Model Family (Mean of Four Subsets, Cycles; Lower Is Better)")
    ch.has_legend = False
    pl = ch.plots[0]
    pl.gap_width = 40
    pl.has_data_labels = True
    pl.data_labels.number_format, pl.data_labels.number_format_is_linked = "0.0", False
    pl.data_labels.font.size = Pt(6 * TR["f"])
    pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    for i, (a, _) in enumerate(vals):
        pt = pl.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(TB_NAVY if "selected" in a else "8FB3DE")
    ch.value_axis.minimum_scale = 0
    ch.value_axis.major_gridlines.format.line.color.rgb = rgb("E6EBF2")
    ch.value_axis.tick_labels.font.size = Pt(6 * TR["f"])
    ch.category_axis.tick_labels.font.size = Pt(5.4 * TR["f"])


def n_scatter(x, y, w, h):
    cd = XyChartData()
    for f in FDS:
        se = cd.add_series(f)
        for a, b in zip(D[f]["true_cap"], D[f]["pred_t"]):
            se.add_data_point(float(a), float(b))
    se = cd.add_series("Perfect prediction")
    se.add_data_point(0, 0)
    se.add_data_point(125, 125)
    ch = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Xe(x), Ye(y), We(w), He(h), cd).chart
    style_chart(ch, "Predicted vs True Remaining Life, All Test Engines")
    sers = list(ch.plots[0].series)
    for ser, f in zip(sers[:4], FDS):
        ser.marker.style = XL_MARKER_STYLE.CIRCLE
        ser.marker.size = 3
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = rgb(SUBF[f])
        ser.marker.format.line.fill.background()
        ser.format.line.fill.background()
    sers[4].marker.style = XL_MARKER_STYLE.NONE
    sers[4].format.line.color.rgb = rgb(GREY)
    sers[4].format.line.width = Pt(1)
    for ax, ttl in ((ch.category_axis, "True remaining life (cycles, capped at 125)"), (ch.value_axis, "Predicted (cycles)")):
        ax.minimum_scale, ax.maximum_scale, ax.major_unit = 0, 130, 25
        ax.has_title = True
        ax.axis_title.text_frame.text = ttl
        ax.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(5.6 * TR["f"])
        ax.tick_labels.font.size = Pt(5.6 * TR["f"])
        ax.major_gridlines.format.line.color.rgb = rgb("E6EBF2")
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(5.2 * TR["f"])


try:
    n_models(4.4, RY, 2.3, RH)
except (OSError, KeyError, ValueError, TypeError) as e_:
    print("model comparison chart skipped:", type(e_).__name__, e_)
    put(rect(s, 4.4, RY, 2.3, RH, TB_BG, TB_LINE), [("Model comparison unavailable: rul_study.json missing", 6, False, INK)], align=CEN)
n_scatter(6.75, RY, 1.8, RH)
itag(6.75, RY, 1.8, RH, BASE + "pred_vs_true.html")

# ---------------- 4 triage decision logic (replaces the funnel)
FXp, FWp = 8.6, 2.5
rect(s, FXp, RY, FWp, RH, "F7FAFE", TB_LINE)
badge(FXp + 0.04, RY + 0.02, 4)
text(s, FXp + 0.28, RY + 0.01, FWp - 0.3, 0.24, [("Triage Decision Logic: How Urgency Is Decided", 6.4, True, TB_NAVY)], anchor=MID_)
tiers = [("All readings", "Input: every reading of every engine", TB_NAVY, "FFFFFF"),
         ("Valid, 30+ cycles?", "REVIEW: a person checks; never marked healthy", "8A97A8", "FFFFFF"),
         ("Life \u2264 60 cycles?", f"MEDIUM: inspect within 10 cycles", TB_GOLD, TB_NAVY),
         ("Life \u2264 30 cycles?", f"HIGH: maintain within 2 cycles", ORG, "FFFFFF"),
         ("Life \u2264 15 cycles?", "CRITICAL: remove from service; human review", RED, "FFFFFF")]
tops_, bots_ = [1.45, 1.27, 1.09, 0.91, 0.73], [1.27, 1.09, 0.91, 0.73, 0.55]
fcx_ = FXp + 0.08 + 0.725
for i, (inner, outer, col, tcol) in enumerate(tiers):
    y = RY + 0.27 + i * 0.3
    fb = s.shapes.build_freeform(Xe(fcx_ - tops_[i] / 2), Ye(y), scale=1.0)
    fb.add_line_segments([(Xe(fcx_ + tops_[i] / 2), Ye(y)), (Xe(fcx_ + bots_[i] / 2), Ye(y + 0.28)), (Xe(fcx_ - bots_[i] / 2), Ye(y + 0.28))], close=True)
    sh = fb.convert_to_shape()
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(col)
    sh.line.fill.background()
    sh.shadow.inherit = False
    put(sh, [(inner, 5.4, True, tcol)], align=CEN, m=0.0)
    line(s, fcx_ + (tops_[i] + bots_[i]) / 4, y + 0.14, FXp + 1.6, y + 0.14, col, False, 1.0)
    put(rect(s, FXp + 1.6, y + 0.01, 0.88, 0.27, tint(col, 0.86), col), [(outer, 4.8, True, INK)], align=LEFT, m=0.03)
text(s, FXp + 0.04, RY + 1.78, FWp - 0.08, 0.22, [("Life = estimated remaining life after the safety margin. A persistent anomaly also raises severity. Limits are provisional.", 4.6, False, "334155")], align=CEN, anchor=MID_)

# ---------------- 5 key findings
KX, KW2 = 11.15, 1.93
hd = rect(s, KX, RY, KW2, 0.27, TB_NAVY)
badge(KX + 0.04, RY + 0.025, 5)
text(s, KX + 0.3, RY, KW2 - 0.3, 0.27, [("Key Findings", 9, True, "FFFFFF")], anchor=MID_)
fnd = [f"Accuracy: predictions are most accurate near failure, with an error of {rng(b30)} cycles when 30 or fewer remain, versus {rng(bmid)} cycles when 60 to 125 remain.",
       f"Safety: the 90% lower bound reduces dangerous overestimates of remaining life from {lr:.0%} to {lc:.0%} of engines, at a cost of {rng(sl, '.0f')} cycles of earlier warning.",
       f"Detection: abnormal behaviour is found with an area under the curve of {NA(af, '.3f')}; alerts arrive {rng(lead, '.0f')} cycles before failure." if lead else "Detection: see the anomaly results in the appendix.",
       f"Action: {N_ORD} of {NTOT} test engines triggered simulated work orders; {N_CRIT} reached the critical level for human review.",
       (f"Economics: in simulation, predictive maintenance is cheaper than running to failure once a failure costs about {rng(be)} times a wasted cycle." if be else "Economics: see the cost chart.")]
put(rect(s, KX, RY + 0.27, KW2, RH - 0.27, TB_BG, TB_LINE), [("\u2022 " + t_, 5.6, False, INK) for t_ in fnd], align=LEFT, anchor=MSO_ANCHOR.TOP, m=0.07)

# ---------------- links footer
text(s, 0.25, 7.1, 3.6, 0.3, [("Charts tagged Interactive open live, hoverable versions:", 6, True, TB_NAVY)], anchor=MID_)
lbtn(3.9, 7.12, 3.0, 0.26, "Live Dashboard (Streamlit, localhost:8501)", "http://localhost:8501")
lbtn(7.0, 7.12, 2.8, 0.26, "All Interactive Charts (Index)", BASE + "index.html", TB_BLUE)
lbtn(9.9, 7.12, 3.18, 0.26, "Source Code: github.com/abhishekk-km/FLEET-PM", REPO, TB_CYAN)

# ------------------------------------------------------------------ appendix frame with navigation
TR.update(APP)
APPS, NAVB = [], []
SHORT = {"Data, label construction and evaluation protocol": "Data", "RUL accuracy on the official test set": "Accuracy", "Model selection evidence": "Models",
         "Conformal bound: calibration on training engines": "Conf. calib.", "Conformal bound: official test coverage": "Conf. test", "Cost-aware policy replay": "Cost replay",
         "Cost replay at thresholds fixed in advance": "Fixed thr.", "Anomaly detection": "Anomaly", "Explainability: attribution and retraining faithfulness": "Explain",
         "Unsupervised degradation-signature clustering": "Clusters", "Triage rules and simulated work orders": "Triage", "Triage evaluation at the last observed cycle": "Triage eval",
         "Retraining demonstration and a sample work order": "Retrain", "Methodology and sources": "Method", "Assumptions, limitations and open items": "Limits"}


def btn(s_, x, y, w, h, label, fs=6.5, fill=LIGHT, col=NAVY):
    b = rect(s_, x, y, w, h, fill, BORD, MSO_SHAPE.ROUNDED_RECTANGLE)
    put(b, [(label, fs, True, col)], align=PP_ALIGN.CENTER, m=0.02)
    return b


def frame(title, color, src):
    APX[0] += 1
    s_ = new_slide()
    keep = dict(TR)
    TR.update(ID)
    text(s_, 2.7, 0.1, 9.4, 0.42, [(title, 16, True, NAVY)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s_, 2.7, 0.5, 9.4, 0.3, [(f"Appendix {APX[0]}  |  Source: {src}", 6.5, False, "64748B")], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    rect(s_, 2.7, 0.82, 9.4, 0.025, color)
    home = btn(s_, 0.25, 7.15, 0.75, 0.25, "Home")
    prev = btn(s_, 1.06, 7.15, 0.75, 0.25, "< Prev")
    nxt = btn(s_, 1.87, 7.15, 0.75, 0.25, "Next >")
    home.click_action.target_slide = S2
    live = btn(s_, 10.9, 7.15, 2.18, 0.25, "Live dashboard (localhost:8501)", 6, NAVY, WHITE)
    live.click_action.hyperlink.address = "http://localhost:8501"
    text(s_, 2.7, 7.15, 8.1, 0.25, [("Benchmark replay, simulated actions only.", 6.5, False, GREY)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    APPS.append((SHORT.get(title, title[:12]), s_))
    NAVB.append((prev, nxt))
    TR.update(keep)
    return s_


def finalize_nav():
    n = len(APPS)
    for i, (pv, nx) in enumerate(NAVB):
        pv.click_action.target_slide = APPS[i - 1][1] if i > 0 else S2
        nx.click_action.target_slide = APPS[i + 1][1] if i < n - 1 else S2
    keep = dict(TR)
    TR.update(ID)
    text(S2, 0.25, 7.14, 0.75, 0.26, [("Appendix:" if NAV_ON else "", 6.5, True, NAVY)], anchor=MSO_ANCHOR.MIDDLE)
    bw = (12.0 - 0.05 * (n - 1)) / n
    for i, (lab, sl_) in enumerate(APPS if NAV_ON else []):
        b = btn(S2, 1.0 + i * (bw + 0.05), 7.14, bw, 0.26, f"A{i + 1} {lab}", 5.8)
        b.click_action.target_slide = sl_
    TR.update(keep)
    print("navigation built:", n, "appendix slides")


# TITLE-CASE WRAPPERS
_SMALL = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "vs", "with"}


def _cap(p, small):
    k = next((i for i, c in enumerate(p) if c.isalpha()), None)
    if k is None or any(c.isupper() for c in p) or any(c.isdigit() for c in p) or small:
        return p
    return p[:k] + p[k].upper() + p[k + 1:]


def tc(t):
    out = []
    for i, w in enumerate(str(t).split(" ")):
        parts = w.split("-")
        out.append("-".join(_cap(p, (i > 0 or j > 0) and p.lower() in _SMALL) for j, p in enumerate(parts)))
    return " ".join(out)


_sec0, _tile0, _mini0, _table0, _frame0 = sec, tile, mini, table, frame


def sec(s, x, y, w, label, *a, **k):
    return _sec0(s, x, y, w, tc(label), *a, **k)


def tile(s, x, y, w, h, title, *a, **k):
    return _tile0(s, x, y, w, h, tc(title), *a, **k)


def mini(name, w, h, draw, title=None, *a, **k):
    return _mini0(name, w, h, draw, tc(title) if title else None, *a, **k)


def table(s, x, y, w, h, rows, *a, **k):
    return _table0(s, x, y, w, h, [[tc(c) for c in rows[0]]] + [list(r) for r in rows[1:]], *a, **k)


def frame(title, color, src):
    return _frame0(tc(title), color, src)


SHORT = {tc(k_): v_ for k_, v_ in SHORT.items()}
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

finalize_nav()
out = art(cfg, "figures", "fleetpm_deck_v4.pptx")
prs.save(out)
_q = Presentation(str(out))
_l = _q.slides._sldIdLst
for _sid in list(_l)[2:]:
    _q.part.drop_rel(_sid.rId)
    _l.remove(_sid)
_q.save(str(art(cfg, "figures", "fleetpm_submit_2slides.pptx")))
print("saved fleetpm_submit_2slides.pptx (cover + slide 2)")
print("saved", out, "| slides:", len(prs.slides))
