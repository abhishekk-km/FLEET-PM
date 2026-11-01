import py_compile, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "scripts" / "14_deck_v3.py"
DST = ROOT / "scripts" / "15_deck_v4.py"
if not SRC.exists():
    raise SystemExit("scripts/14_deck_v3.py not found; run write_v8.py first")
txt = SRC.read_text()
M1 = "# ================================================================== SLIDE 1\n"
M2 = "C4 = [0.15 + i * 3.25 for i in range(4)]\n"
OUTL = 'out = art(cfg, "figures", "fleetpm_deck_v3.pptx")'
for m in (M1, M2, OUTL):
    if txt.count(m) != 1:
        raise SystemExit(f"marker not found exactly once: {m[:60]!r}")
HEAD = txt[:txt.index(M1)]
TAIL = txt[txt.index(M2):].replace(OUTL, 'finalize_nav()\nout = art(cfg, "figures", "fleetpm_deck_v4.pptx")')

MID = r'''
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
        S2_DECOR.append(("pic", sh.image.blob, sh.left, sh.top, sh.width, sh.height, None))
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
                s_.shapes.add_picture(io.BytesIO(a), l, t, w, h)
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
    style_chart(ch, "Lower-bound coverage, engines below cap (mean of 4 subsets)")
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
s = S2
TR.update(ID)
text(s, 2.7, 0.1, 9.4, 0.42, [(TITLE, 18, True, NAVY)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 2.7, 0.5, 9.4, 0.28, [("NASA C-MAPSS benchmark replay | simulated actions | technical review", 8, False, "51627A")], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
rect(s, 2.7, 0.82, 9.4, 0.025, TEAL)

gauges = [("Engines with work order", f"{N_ORD} / {NTOT}", f"{len(FDS)} subsets, official test", N_ORD / NTOT, BLUE, f"{N_ORD / NTOT:.0%}", None, None),
          ("RMSE vs capped RUL", rng(rm), "cycles; ring = share of cap", nr_, TEAL, f"{nr_:.0%}", None, None)]
if af:
    gauges.append(("Anomaly AUROC", rng(af, ".3f"), "Isolation Forest, latent + error", min(af), PURPLE, f"{min(af):.2f}", None, None))
gauges.append(("90% bound coverage", rng(cv, ".2f"), "engines below cap; measured", min(cv), NAVY, f"{min(cv):.2f}", None, None))
gauges.append(("Late predictions", f"{lr:.0%} to {lc:.0%}", "outer raw, inner 90% bound", lr, RED, "", lc, GREEN))
if tp:
    gauges.append(("Persistent TPR", rng(tp, ".2f"), "anomaly, RUL <= 30 proxy", min(tp), AMBER, f"{min(tp):.0%}", None, None))
KW = (12.83 - 0.07 * (len(gauges) - 1)) / len(gauges)
for i, (lab, val, sub, fr, col, ctr, fr2, col2) in enumerate(gauges):
    x = 0.25 + i * (KW + 0.07)
    rect(s, x, 0.9, KW, 0.72, LIGHT)
    pic(s, ring_img(f"k{i}", fr, col, ctr, fr2, col2), x + 0.04, 0.95, 0.62, 0.62)
    text(s, x + 0.68, 0.92, KW - 0.7, 0.17, [(lab.upper(), 5.6, True, "51627A")])
    text(s, x + 0.68, 1.07, KW - 0.7, 0.28, [(val, 11, True, NAVY)])
    text(s, x + 0.68, 1.36, KW - 0.7, 0.25, [(sub, 5.6, False, "64748B")])

# ---------------- architecture (primary element)
PX, PY, PW, PH = 0.25, 1.7, 8.6, 3.38
rect(s, PX, PY, PW, PH, None, BORD)
sec(s, PX + 0.05, PY + 0.01, PW - 0.1, "System architecture: end-to-end data flow (benchmark replay)")
an_, tg_, dcfg = cfg["anomaly"], cfg["triage"], cfg["data"]
nfeat = [len(D[f]["P"]["pre"].sensors) * (len(cfg["features"]["stats"]) + 2) for f in FDS]
nsen = [len(D[f]["P"]["pre"].sensors) for f in FDS]
CWc, GAP, X0, CY0, CH0 = 1.5, 0.2, PX + 0.15, PY + 0.4, 2.2
stages = [
    ("Ingest and validate", BLUE, [f"C-MAPSS FD001 to FD004, 26 columns", "schema, NaN, duplicate, cycle-gap checks", "official RUL vector alignment",
                                   f"KMeans k={dcfg['n_regimes']} regime z-score (FD002, FD004)", f"causal median smoothing w={dcfg['smooth_window']}",
                                   f"sensors kept {min(nsen)} to {max(nsen)} (|Spearman| >= {dcfg['min_abs_trend']})"]),
    ("Causal features", FEAT, ["trailing windows 30 and 10 cycles", "mean, std, min, max, slope, kurtosis", f"{min(nfeat)} to {max(nfeat)} feature columns",
                               "never crosses engine boundaries", f"RUL label capped at {cap_} cycles", "sample weight 3 below RUL 30"]),
    ("Models", TEAL, None),
    ("Deterministic triage", ORANGE, ["CRITICAL, HIGH, MEDIUM, LOW, REVIEW", f"RUL cutoffs {tg_['critical_rul']} / {tg_['high_rul']} / {tg_['medium_rul']} (provisional)",
                                      f"anomaly needs k={an_['persist_k']} persistent flags", "invalid input or short history: REVIEW, never LOW",
                                      "rules fire in fixed order, first match", "no LLM, fully auditable"]),
    ("Work orders and monitor", GREEN, ["simulated orders only (OPEN_SIMULATED)", "deterministic ID, restart-safe", "issued once per severity, on escalation",
                                         "TreeSHAP top-3 sensors attached", "read-only Streamlit monitor", "no physical action is triggered"]),
]
for i, (nm, col, bl) in enumerate(stages):
    x = X0 + i * (CWc + GAP)
    rect(s, x, CY0, CWc, 0.36, col)
    put(rect(s, x + 0.05, CY0 + 0.05, 0.26, 0.26, WHITE, None, MSO_SHAPE.OVAL), [(str(i + 1), 8, True, col)], align=PP_ALIGN.CENTER, m=0.0)
    text(s, x + 0.34, CY0, CWc - 0.36, 0.36, [(nm, 7.2, True, WHITE)], anchor=MSO_ANCHOR.MIDDLE)
    body = rect(s, x, CY0 + 0.36, CWc, CH0 - 0.36, WHITE, col)
    if bl:
        put(body, [("\u2022 " + t, 6.2, False, INK) for t in bl], align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, m=0.06)
    else:
        subs = [("Anomaly detector", f"autoencoder (healthy cycles {an_['healthy_cycle_range'][0]}-{an_['healthy_cycle_range'][1]}) + Isolation Forest, k={an_['persist_k']}", TEAL),
                ("RUL regressor", f"XGBoost {cfg['rul']['xgb']['n_estimators']} trees, depth {cfg['rul']['xgb']['max_depth']}, engine-grouped {cfg['eval']['cv_folds']}-fold CV", "0B5CAD"),
                ("Risk bound", "0.8-quantile offset (deployed); conformal 90% lower bound (offline)", NAVY)]
        for j, (t1, t2, c_) in enumerate(subs):
            put(rect(s, x + 0.06, CY0 + 0.42 + j * 0.58, CWc - 0.12, 0.54, c_, None, MSO_SHAPE.ROUNDED_RECTANGLE),
                [(t1, 6.6, True, WHITE), (t2, 5.6, False, "E3ECF7")], align=PP_ALIGN.CENTER, m=0.03)
    if i < 4:
        rect(s, x + CWc + 0.02, CY0 + 1.1, GAP - 0.04, 0.22, "334155", None, MSO_SHAPE.RIGHT_ARROW)
rc_ = cfg["mlops"]["retrain"]
BY0 = CY0 + CH0 + 0.12
put(rect(s, X0, BY0, 0.95, 0.5, NAVY, None, MSO_SHAPE.ROUNDED_RECTANGLE), [("CONTROL PLANE", 6.4, True, WHITE)], align=PP_ALIGN.CENTER, m=0.02)
chips = [("MLflow tracking", "local SQLite store: parameters, metrics, model artifacts"), ("Registry gate", "champion and challenger aliases; promote only if validation RMSE improves"),
         ("Drift gate", f"PSI > {rc_['psi_alarm']}, RMSE > {rc_['rmse_factor']}x baseline, or {rc_['days']} days"), ("Gated retraining", "engine-held-out validation; previous champion kept for rollback")]
CWh = (PW - 0.3 - 1.0 - 0.06 * 4) / 4
for j, (t1, t2) in enumerate(chips):
    put(rect(s, X0 + 1.0 + j * (CWh + 0.06), BY0, CWh, 0.5, "DDE3EA", None, MSO_SHAPE.ROUNDED_RECTANGLE), [(t1, 6.4, True, NAVY), (t2, 5.4, False, "334155")], align=PP_ALIGN.CENTER, m=0.03)
for i in (2, 4):
    xc = X0 + i * (CWc + GAP) + CWc / 2
    line(s, xc, BY0, xc, CY0 + CH0, "334155", True, 1.0)

# ---------------- native charts (right column)
native_cov(s, 8.95, 1.7, 4.13, 1.66)
native_cost(s, 8.95, 3.4, 4.13, 1.68)

# ---------------- bottom row
BY = 5.16
th = [("Target and features", BLUE, f"RUL is capped at {cap_} cycles because early-life degradation is not observable. Features are causal trailing windows (30 and 10 cycles) over regime-normalised, smoothed sensors: mean, std, min, max, slope, kurtosis."),
      ("Anomaly detection", TEAL, f"An autoencoder fitted on healthy cycles {an_['healthy_cycle_range'][0]} to {an_['healthy_cycle_range'][1]} defines a normal envelope; an Isolation Forest scores latent state and error. An alert needs {an_['persist_k']} consecutive flags. It signals unusual behaviour, not a confirmed fault."),
      ("Conformal lower bound", NAVY, "Out-of-fold over-prediction residuals are calibrated per predicted-RUL band; bound = prediction minus the finite-sample (1 - alpha) quantile. Coverage is measured on held-out engines, not guaranteed. Offline; triage uses the offset."),
      ("Cost-aware maintenance", ORANGE, "r = C_fail / C_waste compares failure with premature maintenance. Policies fire causally from cycle 30. Break-even r shows where predictive triage beats run-to-failure and fixed interval. No monetary values assumed.")]
for i, (t_, col, b_) in enumerate(th):
    tile(s, 0.25 + i * 2.16, BY, 2.1, 1.92, t_, b_, col, fs=6.7)
FX = 8.95
rect(s, FX, BY, 1.95, 1.92, None, BORD)
sec(s, FX + 0.04, BY + 0.01, 1.87, "Triage funnel (engines)")
fun = [("Test engines replayed", NTOT, BLUE), ("Work order issued", N_ORD, TEAL), ("HIGH or CRITICAL", N_HIGH, ORANGE), ("CRITICAL", N_CRIT, RED)]
tops, bots = [1.8, 1.5, 1.2, 0.9], [1.5, 1.2, 0.9, 0.6]
fcx = FX + 0.975
for i, (lab, n, col) in enumerate(fun):
    y = BY + 0.36 + i * 0.385
    fb = s.shapes.build_freeform(Xe(fcx - tops[i] / 2), Ye(y), scale=1.0)
    fb.add_line_segments([(Xe(fcx + tops[i] / 2), Ye(y)), (Xe(fcx + bots[i] / 2), Ye(y + 0.37)), (Xe(fcx - bots[i] / 2), Ye(y + 0.37))], close=True)
    sh = fb.convert_to_shape()
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(col)
    sh.line.fill.background()
    sh.shadow.inherit = False
    put(sh, [(f"{n} ({100 * n / NTOT:.0f}%)  {lab}" if i == 0 else f"{n} ({100 * n / NTOT:.0f}%)", 6.6, True, WHITE)] + ([] if i == 0 else [(lab, 5.2, False, "E3ECF7")]), align=PP_ALIGN.CENTER, m=0.0)
fnd = [f"RMSE {rng(b30)} cycles at true RUL <= 30 versus {rng(bmid)} at RUL 60 to 125.",
       f"90% bound: late rate {lr:.0%} to {lc:.0%}, slack {rng(sl, '.0f')} cycles, coverage {rng(cv, '.2f')}."]
if af and pf:
    fnd.append(f"Anomaly AUROC {rng(af, '.3f')}; persistent FPR {min(pf):.2%} to {max(pf):.2%}.")
if be:
    fnd.append(f"Break-even r {rng(be)} vs run-to-failure (in-sample, simulated).")
tile(s, 10.95, BY, 2.13, 1.92, "Key findings", fnd, NAVY, fs=6.4)

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
    text(S2, 0.25, 7.14, 0.75, 0.26, [("Appendix:", 6.5, True, NAVY)], anchor=MSO_ANCHOR.MIDDLE)
    bw = (12.0 - 0.05 * (n - 1)) / n
    for i, (lab, sl_) in enumerate(APPS):
        b = btn(S2, 1.0 + i * (bw + 0.05), 7.14, bw, 0.26, f"A{i + 1} {lab}", 5.8)
        b.click_action.target_slide = sl_
    TR.update(keep)
    print("navigation built:", n, "appendix slides")


'''

if DST.exists():
    shutil.copy2(DST, DST.with_name(DST.name + f".bak_{time.strftime('%Y%m%d_%H%M%S')}"))
DST.write_text(HEAD + MID + TAIL)
py_compile.compile(str(DST), doraise=True)
print("wrote + compiled scripts/15_deck_v4.py")