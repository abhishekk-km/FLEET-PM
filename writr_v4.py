import py_compile, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TS = time.strftime("%Y%m%d_%H%M%S")


def backup(p):
    if p.exists():
        shutil.copy2(p, p.with_name(p.name + f".bak_{TS}"))


def patch(rel, pairs):
    p = ROOT / rel
    s = p.read_text()
    for old, new in pairs:
        if new in s:
            continue
        if s.count(old) != 1:
            raise SystemExit(f"{rel}: expected exactly 1 match, found {s.count(old)} for: {old[:70]!r}")
        s = s.replace(old, new)
    backup(p)
    p.write_text(s)
    py_compile.compile(str(p), doraise=True)
    print("patched + compiled", rel)


patch("dashboard/app.py", [
    (".block-container{padding-top:1rem;max-width:1450px}", ".block-container{padding-top:4.5rem;max-width:1450px}"),
    ('''    show(alt.Chart(m).mark_bar().encode(
        x=alt.X("subset:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="label:N",
        y=alt.Y("RMSE:Q", title="RMSE (cycles)"),
''', '''    show(alt.Chart(m).mark_point(filled=True, size=160).encode(
        y=alt.Y("subset:N", title=None),
        x=alt.X("RMSE:Q", title="RMSE (cycles)", scale=alt.Scale(zero=False)),
'''),
    ('''        show(alt.Chart(m).mark_bar().encode(x=alt.X("subset:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="scorer:N",
                                            y=alt.Y("AUROC:Q", scale=alt.Scale(domain=[0.8, 1.0])),
''', '''        show(alt.Chart(m).mark_point(filled=True, size=170).encode(y=alt.Y("subset:N", title=None),
                                            x=alt.X("AUROC:Q", scale=alt.Scale(domain=[0.9, 1.0])),
'''),
])

DECK = r'''import _bootstrap
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
from fleetpm.config import load_config, art
from fleetpm.models.metrics import summarize

cfg = load_config()
FDS = cfg["data"]["subsets"]
NAVY, BLUE, TEAL, AMBER, ORANGE, RED, GREEN, GREY, LIGHT, INK, WHITE = (
    "0B2A4A", "0B5CAD", "14808C", "D99A00", "E36414", "B3202A", "2E8B57", "8A97A8", "F3F6FA", "111827", "FFFFFF")
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
RGRID = np.logspace(0, 3, 200)
RANK = {"REVIEW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def jl(*parts, required=True):
    p = art(cfg, *parts)
    if not p.exists():
        if required:
            raise SystemExit(f"missing {p}: run the producing script first")
        return None
    return json.loads(p.read_text())


conf, cost = jl("ext_conformal.json"), jl("ext_cost_policy.json")
ex = jl("ext_explain.json", required=False) or {}
anom = {f: jl("models", f, "anomaly_metrics.json", required=False) for f in FDS}
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
    s, r = summarize(tc, p), summarize(tr, p)
    ST[f] = dict(n=len(p), rmse=s["rmse"], mae=s["mae"], nasa=s["nasa"], rmse_raw=r["rmse"],
                 late=float((p > tc).mean()), bias=float((p - tc).mean()))
NTOT = sum(v["n"] for v in ST.values())

WOB = art(cfg, "work_orders_v2", "x").parent
mx, CNT = {}, {f: {k: 0 for k in RANK} for f in FDS}
for p in WOB.rglob("WO-*.json"):
    d = json.loads(p.read_text())
    fd = p.name.split("-")[1]
    sev = d.get("severity")
    if fd not in FDS or sev not in RANK:
        continue
    CNT[fd][sev] += 1
    k = (fd, d.get("engine_id"))
    mx[k] = max(mx.get(k, 0), RANK[sev])
if not mx:
    raise SystemExit(f"no work orders under {WOB}")
N_ORD, N_HIGH, N_CRIT = len(mx), sum(v >= 2 for v in mx.values()), sum(v == 3 for v in mx.values())

# ------------------------------------------------------------------ pptx helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BL = prs.slide_layouts[6]
rgb = lambda h: RGBColor.from_string(h)
E = lambda v: int(Inches(v))


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


def put(sh, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE, m=0.06):
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
        r.font.size, r.font.bold, r.font.name = Pt(sz), b, "Calibri"
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
    s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))


def tile(s, x, y, w, h, title, body, color, fs=9):
    rect(s, x, y, w, 0.27, color)
    put(s.shapes[-1], [(title, 10, True, WHITE)])
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
            r.font.size, r.font.bold, r.font.name = Pt(fs), i == 0, "Calibri"
            r.font.color.rgb = rgb(WHITE if i == 0 else INK)


def frame(title, tag, color, src):
    s = prs.slides.add_slide(BL)
    rect(s, 0, 0, 13.333, 0.6, NAVY)
    rect(s, 0, 0.6, 13.333, 0.05, color)
    text(s, 0.25, 0.08, 10.5, 0.45, [(title, 18, True, WHITE)], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 11.0, 0.08, 2.1, 0.45, [(tag, 11, False, "C9D6E6")], align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.25, 7.2, 12.8, 0.26, [("Source: " + src + " | Benchmark replay, simulated actions only.", 8, False, GREY)])
    return s


def sec(s, x, y, w, label, color=NAVY):
    text(s, x, y, w, 0.28, [(label, 10.5, True, color)])
    line(s, x + 0.05, y + 0.29, x + w, y + 0.29, "D6DEE9", wd=1)


# ------------------------------------------------------------------ chart helpers
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


def mean_curves(keys):
    return {k: np.mean([cost[f]["curves"][k] for f in FDS], 0) for k in keys}


def ch_cost(w, h):
    cu = mean_curves(list(PLAB))
    fig, ax = fig_(w, h)
    for k, c in cu.items():
        ax.loglog(RGRID, c, color=H(PCOL[k]), lw=1.8, label=PLAB[k])
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
        r = [float(np.sqrt((e[m] ** 2).mean())) if m.any() else 0.0 for m in ms]
        ax.bar(np.arange(4) + (i - 1.5) * wd, r, wd, color=H(PAL[i]), label=f)
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
    n = len(fs)
    nr = math.ceil(n / 2)
    fig, axs = fig_(w, h, nrows=nr, ncols=2, squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, f in zip(axs.ravel(), fs):
        sh = ex[f]["faithfulness"]["mean_abs_shap_by_sensor"]
        top = sorted(sh.items(), key=lambda kv: -kv[1])[:8][::-1]
        ax.axis("on")
        ax.barh([NAMES[int(k.split("_")[1]) - 1] for k, _ in top], [v for _, v in top], color=H(BLUE))
        ax.set_title(f, fontsize=8)
        clean(ax)
    fig.supxlabel("Mean |SHAP| (cycles), top 8 sensors", fontsize=8)
    return save(fig, "shap")


def fmt_be(x):
    if x["point"] is None:
        return "never"
    return f"{x['point']:.1f}" + (f" ({x['ci'][0]:.1f} to {x['ci'][1]:.1f})" if x.get("ci") else "")


# ------------------------------------------------------------------ slide 1
s = prs.slides.add_slide(BL)
rect(s, 0, 0, 13.333, 0.62, NAVY)
text(s, 0.25, 0.06, 8.6, 0.5, [("FleetPM: Predictive Maintenance for a Connected Fleet", 22, True, WHITE)], anchor=MSO_ANCHOR.MIDDLE)
text(s, 8.7, 0.06, 4.5, 0.5, [("NASA C-MAPSS benchmark replay | simulated actions | technical review", 10, False, "C9D6E6")],
     align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
rm = [ST[f]["rmse"] for f in FDS]
cv = [conf[f]["test"]["conformal_a0.1"]["cov_lt_cap"] for f in FDS]
lr = np.mean([conf[f]["test"]["raw"]["late_rate"] for f in FDS])
lc = np.mean([conf[f]["test"]["conformal_a0.1"]["late_rate"] for f in FDS])
be = [cost[f]["break_even"]["rul_conformal10"]["vs_rtf"]["point"] for f in FDS]
be = [b for b in be if b is not None]
kp = [("Engines replayed", f"{NTOT}", f"{len(FDS)} subsets, official test, last cycle", BLUE, MSO_SHAPE.GEAR_6),
      ("RMSE vs capped RUL", f"{min(rm):.1f} to {max(rm):.1f}", "cycles; confidence intervals in appendix", TEAL, MSO_SHAPE.DONUT),
      ("90% bound coverage", f"{min(cv):.2f} to {max(cv):.2f}", "engines below cap; measured, not guaranteed", NAVY, MSO_SHAPE.DIAMOND),
      ("Late predictions", f"{lr:.0%} to {lc:.0%}", "raw to 90% bound, mean of subsets", RED, MSO_SHAPE.LIGHTNING_BOLT),
      ("Break-even r vs run-to-failure", f"{min(be):.1f} to {max(be):.1f}" if be else "n/a",
       "90% bound policy; in-sample, simulated", AMBER, MSO_SHAPE.OVAL)]
for i, (lab, val, sub, col, ic) in enumerate(kp):
    x = 0.18 + i * 2.62
    rect(s, x, 0.72, 2.5, 0.8, LIGHT)
    rect(s, x, 0.72, 0.06, 0.8, col)
    text(s, x + 0.12, 0.74, 2.0, 0.22, [(lab.upper(), 7.5, True, "51627A")])
    text(s, x + 0.12, 0.93, 2.2, 0.34, [(val, 18, True, NAVY)])
    text(s, x + 0.12, 1.27, 2.35, 0.22, [(sub, 7.5, False, "64748B")])
    rect(s, x + 2.1, 0.78, 0.28, 0.28, col, shape=ic)

rect(s, 0.18, 1.6, 7.45, 2.97, None, "D6DEE9")
sec(s, 0.23, 1.62, 7.3, "System architecture")


def node(x, y, w, h, t, sub, col):
    sh = rect(s, x, y, w, h, col, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    put(sh, [(t, 9.5, True, WHITE), (sub, 7.5, False, "E3ECF7")], align=PP_ALIGN.CENTER)


node(0.23, 1.97, 2.1, 0.6, "Raw C-MAPSS", "FD001 to FD004, 26 columns", BLUE)
node(2.83, 1.97, 2.1, 0.6, "Validate and normalise", "schema, gaps, per-regime z-score", BLUE)
node(5.43, 1.97, 2.1, 0.6, "Causal features", "windows 30 and 10, six statistics", BLUE)
line(s, 2.33, 2.27, 2.83, 2.27, GREY, True)
line(s, 4.93, 2.27, 5.43, 2.27, GREY, True)
line(s, 6.48, 2.57, 6.48, 2.71)
line(s, 1.98, 2.71, 6.48, 2.71)
line(s, 1.98, 2.71, 1.98, 2.87, GREY, True)
line(s, 5.7, 2.71, 5.7, 2.87, GREY, True)
node(0.23, 2.87, 3.5, 0.6, "Anomaly detection", "autoencoder on healthy cycles + Isolation Forest, persistence k=4", TEAL)
node(3.9, 3.87 - 1.0, 3.6, 0.6, "RUL prediction and risk bound", "XGBoost, capped target; offset deployed, conformal bound evaluated", TEAL)
line(s, 1.98, 3.47, 1.98, 3.61)
line(s, 5.7, 3.47, 5.7, 3.61)
line(s, 1.03, 3.61, 5.7, 3.61)
line(s, 1.03, 3.61, 1.03, 3.77, GREY, True)
cx = [0.23, 2.13, 4.03, 5.93]
node(cx[0], 3.77, 1.6, 0.6, "Deterministic triage", "RUL bound + anomaly rules", ORANGE)
node(cx[1], 3.77, 1.6, 0.6, "Simulated work orders", "idempotent JSON", ORANGE)
node(cx[2], 3.77, 1.6, 0.6, "Monitoring dashboard", "read-only", NAVY)
node(cx[3], 3.77, 1.6, 0.6, "MLflow, retraining gate", "champion / challenger", NAVY)
for a in cx[:3]:
    line(s, a + 1.6, 4.07, a + 1.9, 4.07, GREY, True)
for i, (lab, col) in enumerate([("Data", BLUE), ("Models", TEAL), ("Decision", ORANGE), ("Operations", NAVY)]):
    rect(s, 0.27 + i * 1.2, 4.43, 0.12, 0.12, col)
    text(s, 0.42 + i * 1.2, 4.37, 1.0, 0.22, [(lab, 7.5, False, "334155")])

rect(s, 7.72, 1.6, 2.55, 2.97, None, "D6DEE9")
sec(s, 7.75, 1.62, 2.5, "Triage funnel: highest severity per engine")
fun = [("Test engines replayed", NTOT, BLUE), ("With simulated work order", N_ORD, TEAL),
       ("HIGH or CRITICAL", N_HIGH, ORANGE), ("CRITICAL", N_CRIT, RED)]
tops, bots = [2.4, 1.95, 1.5, 1.05], [1.95, 1.5, 1.05, 0.7]
for i, (lab, n, col) in enumerate(fun):
    y = 1.98 + i * 0.64
    fb = s.shapes.build_freeform(E(8.995 - tops[i] / 2), E(y), scale=1.0)
    fb.add_line_segments([(E(8.995 + tops[i] / 2), E(y)), (E(8.995 + bots[i] / 2), E(y + 0.6)),
                          (E(8.995 - bots[i] / 2), E(y + 0.6))], close=True)
    sh = fb.convert_to_shape()
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(col)
    sh.line.fill.background()
    sh.shadow.inherit = False
    put(sh, [(f"{n} ({100 * n / NTOT:.0f}%)", 10.5, True, WHITE), (lab, 7.5 if i < 3 else 7, False, "E3ECF7")], align=PP_ALIGN.CENTER)

rect(s, 10.35, 1.6, 2.8, 2.97, None, "D6DEE9")
pic(s, ch_cov(2.7, 2.9), 10.4, 1.63, 2.7, 2.9)

tl = [("Target and features", BLUE,
       "RUL labels are capped at 125 cycles because early-life degradation is not observable from sensors. Features are strictly causal: trailing windows of 30 and 10 cycles over regime-normalised, median-smoothed sensors (mean, std, min, max, slope, kurtosis)."),
      ("Anomaly detection", TEAL,
       "An autoencoder trained on healthy cycles 30 to 70 defines a normal envelope. An Isolation Forest scores latent state and reconstruction error. An alert requires 4 consecutive flagged cycles, which suppresses isolated noise."),
      ("Conformal lower bound", NAVY,
       "Out-of-fold over-prediction residuals are calibrated per predicted-RUL band. The bound is the prediction minus the finite-sample (1 - alpha) quantile. Coverage is measured on held-out engines, not guaranteed. Evaluated offline; deployed triage uses the 0.8-quantile offset."),
      ("Cost-aware maintenance", ORANGE,
       "Failure and premature maintenance are compared through r = C_fail / C_waste. Policies fire causally from cycle 30. Break-even r states when predictive triage beats run-to-failure and fixed-interval maintenance. No monetary values are assumed.")]
for i, (t, col, b) in enumerate(tl):
    tile(s, 0.18 + (i % 2) * 3.96, 4.65 + (i // 2) * 1.27, 3.86, 1.2, t, b, col, fs=8.5)
pic(s, ch_cost(5.05, 2.47), 8.1, 4.65, 5.05, 2.47)
text(s, 0.18, 7.2, 13.0, 0.26, [("Benchmark replay of simulated NASA C-MAPSS data. Work orders are simulated. Triage thresholds are provisional. "
                                 "Full numbers, confidence intervals and assumptions are in the appendix.", 8, False, GREY)])

# ------------------------------------------------------------------ appendix
meta = {"FD001": ("1", "1 (HPC)"), "FD002": ("6", "1 (HPC)"), "FD003": ("1", "2 (HPC, fan)"), "FD004": ("6", "2 (HPC, fan)")}
s = frame("Data and evaluation protocol", "Appendix 1", BLUE, "data/CMAPSSData readme.txt, loaded files, configs/config.yaml")
rows = [["Subset", "Operating conditions", "Fault modes (readme)", "Train engines", "Test engines", "Train rows", "Sensors kept", "RUL cap"]]
for f in FDS:
    d = D[f]
    rows.append([f, meta[f][0], meta[f][1], len(np.unique(d["eng"])), ST[f]["n"], len(d["eng"]), len(d["P"]["pre"].sensors), int(d["cap"])])
table(s, 0.3, 0.85, 12.7, 1.9, rows, fs=10)
tile(s, 0.3, 3.0, 6.25, 1.9, "Labels and target",
     ["Train RUL = maximum cycle minus cycle. Test RUL = official RUL vector plus remaining cycles to the last observed cycle.",
      "Target capped at 125 (assumption). Low-RUL samples (RUL below 30) carry weight 3."], BLUE, 10)
tile(s, 6.75, 3.0, 6.25, 1.9, "Evaluation scope",
     ["Official test: prediction at the last observed cycle per engine, aligned with the official RUL vector.",
      "Capped labels are primary. Raw-label RMSE is inflated where true RUL exceeds the cap."], TEAL, 10)
tile(s, 0.3, 5.1, 6.25, 1.95, "Validation protocol",
     ["5-fold GroupKFold by engine for every selection. Out-of-fold predictions feed offsets, conformal bounds and cost replay.",
      "Seed 42. Bootstrap intervals resample whole engines. Official test labels never used for fitting or selection."], NAVY, 10)
tile(s, 6.75, 5.1, 6.25, 1.95, "Leakage status",
     ["Regime clusters, normalisation statistics and sensor selection are fitted on all training engines before cross-validation.",
      "Strict fold-wise refitting has not been run; a mild optimistic bias in CV is possible and unquantified."], ORANGE, 10)

s = frame("RUL accuracy on the official test set", "Appendix 2", TEAL, "scripts/06_final_eval.py outputs, recomputed from rul_boost_preds_gru.npz")
rows = [["Subset", "Engines", "RMSE (capped)", "MAE (capped)", "NASA (capped)", "RMSE (raw)", "Late rate", "Bias (cycles)"]]
for f in FDS:
    v = ST[f]
    rows.append([f, v["n"], f"{v['rmse']:.2f}", f"{v['mae']:.2f}", f"{v['nasa']:.0f}", f"{v['rmse_raw']:.2f}", f"{v['late']:.2f}", f"{v['bias']:+.2f}"])
table(s, 0.3, 0.85, 12.7, 1.6, rows, fs=10)
pic(s, ch_scatter(6.3, 4.3), 0.3, 2.65, 6.3, 4.3)
pic(s, ch_band(6.3, 4.3), 6.8, 2.65, 6.3, 4.3)

s = frame("Conformal bound: calibration on training engines", "Appendix 3", NAVY, "artifacts/ext_conformal.json (nested engine split, training out-of-fold predictions only)")
rows = [["Subset", "Variant", "Nominal", "Held-out coverage", "SD", "All-rows coverage", "Band coverage (<30 / 30-60 / >=60)", "Selected"]]
for f in FDS:
    for k, v in conf[f]["nested"].items():
        var, al = k.split("|")
        rows.append([f, var, f"{1 - float(al):.2f}", f"{v['cut_cov_mean']:.3f}", f"{v['cut_cov_sd']:.3f}", f"{v['allrows_cov_mean']:.3f}",
                     " / ".join(f"{x:.2f}" for x in v["cut_band_cov"]), "yes" if var == conf[f]["variant"] else ""])
table(s, 0.3, 0.85, 12.7, 5.5, rows, [0.8, 1.6, 0.8, 1.2, 0.7, 1.2, 2.4, 0.8], fs=8.5)
text(s, 0.3, 6.45, 12.7, 0.7, [("Selection rule: lowest sum of marginal and per-band coverage deviation under cut-based evaluation, using training engines only. "
                                "The official test set is evaluated once, after the variant is frozen.", 9.5, False, INK)])

s = frame("Conformal bound: official test coverage", "Appendix 4", NAVY, "artifacts/ext_conformal.json; 95% engine-bootstrap intervals; capped labels")
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

s = frame("Cost-aware policy replay", "Appendix 5", ORANGE, "artifacts/ext_cost_policy.json; training trajectories; cross-fitted out-of-fold signals")
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
    s = frame("Cost replay at thresholds fixed in advance", "Appendix 6", ORANGE, "artifacts/ext_cost_policy.json (thresholds 15, 30, 60 cycles; no sweep)")
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

s = frame("Anomaly detection", "Appendix 7", TEAL, "artifacts/models/FD00X/anomaly_metrics.json (deployed Isolation Forest on latent and error)")
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
    s = frame("Explainability: attribution and retraining faithfulness", "Appendix 8", TEAL, "artifacts/ext_explain.json; XGBoost exact TreeSHAP; retraining with 300 trees, engine-grouped 5-fold")
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
    s = frame("Unsupervised degradation-signature clustering", "Appendix 9", TEAL, "artifacts/ext_explain.json; per-engine normalised sensor slopes over the last 50 cycles before failure")
    rows = [["Subset", "Engines", "Silhouette (k=2)", "Gaussian-null p95", "ARI (k=2)", "ARI 95% interval", "Spearman(cluster, life)", "Cluster sizes"]]
    for f, c in cl.items():
        rows.append([f, c["n_engines"], f"{c['silhouette']['2']:.3f}", f"{c['null_sil_k2_p95']:.3f}", f"{c['ari_k2_mean']:.3f}",
                     f"{c['ari_k2_ci'][0]:.2f} to {c['ari_k2_ci'][1]:.2f}", f"{c.get('life_spearman_k2', float('nan')):+.2f}", str(c["cluster_sizes_k2"])])
    table(s, 0.3, 0.85, 12.7, 1.7, rows, fs=9.5)
    im = art(cfg, "figures", "ext_clusters.png")
    if im.exists():
        pic(s, str(im), 0.3, 2.7, 12.7, 2.86)
    text(s, 0.3, 5.7, 12.7, 1.4, [("FD001 and FD002 have one fault mode and act as negative controls; FD003 and FD004 exceed the Gaussian-covariance null most clearly. "
                                   "Clusters correlate with engine life, so degradation speed is a confound. Any mapping to fan or compressor degradation is a HYPOTHESIS: "
                                   "readme.txt lists two fault modes for FD003 and FD004 while the paper text describes HPC degradation only.", 9.5, False, INK)])

t = cfg["triage"]
s = frame("Triage rules and simulated work orders", "Appendix 10", ORANGE, f"artifacts/{WOB.name}/FD00X/WO-*.json; configs/config.yaml (thresholds provisional)")
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

s = frame("MLOps and gated retraining", "Appendix 11", NAVY, "docs/SYSTEM_LOG.md section 8 (FD001, XGBoost, 50 / 30 / 20 percent engine split); artifacts/mlflow.db")
rows = [["Scenario", "v1 validation RMSE", "Trigger", "v2 validation RMSE", "Promoted"],
        ["Control, no drift", "17.009", "none (max PSI 0.138, alarm 0.25)", "not trained", "no"],
        ["Synthetic drift, +1.5 SD on all features", "61.42", "PSI above alarm on 84 of 84 features", "17.95", "yes"],
        ["Scheduled, 7 days", "17.01", "days since training", "14.27 (confounded: 60% more labelled engines)", "yes"]]
table(s, 0.3, 0.85, 12.7, 1.9, rows, [2.6, 1.3, 2.8, 2.8, 0.9], fs=9.5)
tile(s, 0.3, 3.0, 6.25, 2.0, "Tracking and registry",
     ["MLflow local SQLite store with file artifacts. Parameters, metrics and model artifacts logged per run.",
      "Champion and challenger aliases: a candidate is promoted only if its validation RMSE beats the champion."], NAVY, 10)
tile(s, 6.75, 3.0, 6.25, 2.0, "Retraining gate",
     ["Triggers: RMSE above 1.25 times baseline, 7 days elapsed, or population stability index above 0.25 on any feature.",
      "New labelled data arrives only after failures in a real deployment; the demonstration states this."], TEAL, 10)
tile(s, 0.3, 5.2, 12.7, 1.85, "Limits of the demonstration",
     ["Shown on FD001 only, with synthetic drift. A stress test, not realistic drift. The scheduled scenario is confounded by training-set size.",
      "Values in the table are copied from docs/SYSTEM_LOG.md, not recomputed here."], ORANGE, 10)

s = frame("Assumptions, limitations and open items", "Appendix 12", RED, "docs/SYSTEM_LOG.md; scripts/10, 11 and 12 audits")
tl = [("Replay scope", BLUE, "Historical replay of simulated data. No real telemetry, verified fault-onset labels, maintenance-system integration or field validation. All work orders and actions are simulated."),
      ("Thresholds and proxies", TEAL, "Triage cutoffs of 15, 30 and 60 cycles are provisional, not industrial limits. True buckets in the triage evaluation reuse the same cutoffs. Anomaly positives are defined as RUL <= 30, a proxy."),
      ("Conformal bound", NAVY, "Coverage is empirical. Calibration cuts are uniform in life while test cuts sit about 60 percent into life. Cap-saturated engines inflate marginal coverage, so coverage below the cap is reported separately."),
      ("Cost replay", ORANGE, "Training trajectories with cross-fitted out-of-fold signals. Immediate action, failure at the final training cycle, no anomaly signal, no monetary costs. Parameters selected in sample; a cross-validated curve is provided."),
      ("Explainability", TEAL, "SHAP ranks describe the model, not physical cause. Faithfulness retraining uses 300 trees. Fan versus compressor mapping of clusters is a hypothesis."),
      ("Open items", RED, "Strict fold-wise preprocessing leakage check not run. Retraining shown on FD001 with synthetic drift only. Conformal bound not wired into deployed triage. Persistent true-positive rate varies across subsets.")]
for i, (t_, col, b) in enumerate(tl):
    tile(s, 0.3 + (i % 3) * 4.3, 0.85 + (i // 3) * 2.75, 4.1, 2.6, t_, b, col, 10)
text(s, 0.3, 6.4, 12.7, 0.7, [("Reference: Saxena A., Goebel K., Simon D., Eklund N. Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. "
                               "Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, 2008.", 9, False, "334155")])

out = art(cfg, "figures", "fleetpm_deck_v2.pptx")
prs.save(out)
print("saved", out, "| slides:", len(prs.slides))
'''

p = ROOT / "scripts" / "13_deck_v2.py"
backup(p)
p.write_text(DECK)
py_compile.compile(str(p), doraise=True)
print("wrote + compiled scripts/13_deck_v2.py")