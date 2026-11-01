import py_compile, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT / "scripts" / "15_deck_v4.py"
if not P.exists():
    raise SystemExit("scripts/15_deck_v4.py not found; run write_v9.py and write_v10.py first")
A = "# ================================================================== SLIDE 2 (the one-pager, on the template's content page)\n"
B = "# ------------------------------------------------------------------ appendix frame with navigation\n"
src = P.read_text()
if src.count(A) != 1 or src.count(B) != 1:
    raise SystemExit("slide-2 markers not found exactly once")

NEW = r'''# ================================================================== SLIDE 2 (the one-pager, on the template's content page)
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


s = S2
TR.update(ID)
an_, tg_, dcfg = cfg["anomaly"], cfg["triage"], cfg["data"]
nfeat = [len(D[f]["P"]["pre"].sensors) * (len(cfg["features"]["stats"]) + 2) for f in FDS]
nsen = [len(D[f]["P"]["pre"].sensors) for f in FDS]
rc_ = cfg["mlops"]["retrain"]
bi = [ST[f]["bias"] for f in FDS]
NA = lambda v, fm=".3f": rng(v, fm) if v else "n/a"

text(s, 2.7, 0.08, 9.4, 0.42, [(TITLE, 16, True, TB_NAVY)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 2.7, 0.48, 9.4, 0.28, [("Design and architecture: sense, prepare, detect and predict, decide and act, learn | NASA C-MAPSS FD001 to FD004 | benchmark replay, simulated actions", 7.2, False, TB_BLUE)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
rect(s, 2.7, 0.8, 9.4, 0.03, TB_GOLD)

# ---------------- KPI row
kp = [("Engines escalated", f"{N_ORD} / {NTOT}", "simulated work order issued", N_ORD / NTOT, TB_BLUE, "gear", None, None),
      ("RUL error (RMSE)", rng(rm), "cycles; ring = share of 125-cycle cap", nr_, TB_CYAN, "target", None, None)]
if af:
    kp.append(("Anomaly discrimination", rng(af, ".3f"), "AUROC, Isolation Forest on latent", min(af), TB_SKY, "pulse", None, None))
kp.append(("Bound reliability", rng(cv, ".2f"), "90% lower bound, measured coverage", min(cv), TB_NAVY, "shield", None, None))
kp.append(("Late predictions", f"{lr:.0%} to {lc:.0%}", "unsafe side: raw to 90% bound", lr, RED, "clock", lc, GREEN))
if tp:
    kp.append(("Early-warning recall", rng(tp, ".2f"), "persistent alerts, RUL <= 30 proxy", min(tp), TB_GOLD, "bell", None, None))
KW = (12.83 - 0.07 * (len(kp) - 1)) / len(kp)
for i, (lab, val, sub, fr, col, kind, fr2, col2) in enumerate(kp):
    x = 0.25 + i * (KW + 0.07)
    rect(s, x, 0.88, KW, 0.67, TB_BG, TB_LINE)
    rect(s, x, 0.88, 0.05, 0.67, col)
    pic(s, ring_icon(f"k{i}", fr, col, kind, fr2, col2), x + 0.08, 0.91, 0.6, 0.6)
    text(s, x + 0.72, 0.89, KW - 0.74, 0.17, [(lab.upper(), 5.6, True, TB_BLUE)])
    text(s, x + 0.72, 1.04, KW - 0.74, 0.28, [(val, 11, True, TB_NAVY)])
    text(s, x + 0.72, 1.31, KW - 0.74, 0.24, [(sub, 5.6, False, "5B6B80")])

# ---------------- story band: problem, objective, constraints and coverage
story = [("PROBLEM UNDERSTANDING", TB_NAVY, "Failures are caught after breakdown or at fixed service intervals, causing unplanned downtime and avoidable cost, although sensor streams carry early degradation signatures."),
         ("OBJECTIVE", TB_BLUE, "A closed loop without manual steps: ingest, detect degradation, predict remaining useful life, triage by rule, raise a work order, retrain under a gated registry; humans review exceptions only."),
         ("CONSTRAINTS AND DATASET COVERAGE", TB_CYAN, "Lightweight, open-source, no LLM. Rare failures are handled by healthy-only anomaly training and low-RUL weighting. All four subsets, FD001 to FD004 (1 or 6 conditions, 1 or 2 fault modes), are processed independently.")]
SW = (12.83 - 0.12 * 2) / 3
for i, (lab, col, body) in enumerate(story):
    x = 0.25 + i * (SW + 0.12)
    rect(s, x, 1.61, 0.05, 0.52, col)
    put(rect(s, x + 0.05, 1.61, SW - 0.05, 0.52, tint(col, 0.9), TB_LINE), [(lab, 6, True, col), (body, 6.3, False, INK)], align=PP_ALIGN.LEFT, m=0.08)

# ---------------- architecture
AY0 = 2.2
rect(s, 0.25, AY0, 8.8, 2.82, "F7FAFE", TB_LINE)
sec(s, 0.3, AY0 + 0.01, 8.7, "System architecture: closed-loop flow from telemetry to simulated work order", TB_NAVY)
AX0, STEP, CWd = 0.32, 1.74, 1.62
for lab, c0, c1, col in [("DATA ENGINEERING: PREPARE", 0, 1, TB_NAVY), ("MACHINE LEARNING: DETECT AND PREDICT", 2, 2, TB_CYAN), ("AGENT, NON-LLM: DECIDE AND ACT", 3, 4, TB_AMB)]:
    put(rect(s, AX0 + c0 * STEP, AY0 + 0.3, (c1 - c0) * STEP + CWd, 0.18, tint(col, 0.82), col), [(lab, 5.6, True, col)], align=PP_ALIGN.CENTER, m=0.01)
stages = [
    ("Ingest and validate", TB_NAVY, "FFFFFF", TB_NAVY, [
        ("Source", "C-MAPSS FD001-FD004, 26 columns"), ("Quality", "NaN, duplicates, gaps, label checks"),
        ("Regimes", f"KMeans k={dcfg['n_regimes']} z-score, FD002 and FD004"), ("Sensors", f"{min(nsen)}-{max(nsen)} kept, |Spearman| >= {dcfg['min_abs_trend']}")], "data/CMAPSSData/*.txt"),
    ("Causal features", TB_BLUE, "FFFFFF", TB_BLUE, [
        ("Smoothing", f"trailing median, w={dcfg['smooth_window']}, causal"), ("Windows", "30 and 10 cycles, trailing only"),
        ("Statistics", "mean, std, min, max, slope, kurtosis"), ("Target", f"RUL capped at {cap_}; weight 3 below 30")], "artifacts/features/*.joblib"),
    ("Models", TB_CYAN, "FFFFFF", TB_CYAN, [
        ("Anomaly", f"autoencoder + Isolation Forest, k={an_['persist_k']}"), ("RUL", f"XGBoost {cfg['rul']['xgb']['n_estimators']} trees, grouped CV"),
        ("Bound", "offset deployed, conformal 90% offline"), ("Attribution", "exact TreeSHAP by sensor")], "rul_bundle, anomaly_ifl"),
    ("Triage rules", TB_GOLD, TB_NAVY, TB_AMB, [
        ("Severity", "CRITICAL, HIGH, MEDIUM, LOW, REVIEW"), ("Cutoffs", f"RUL {tg_['critical_rul']} / {tg_['high_rul']} / {tg_['medium_rul']} cycles, provisional"),
        ("Anomaly", f"k={an_['persist_k']} persistent flags and score cut"), ("Safety", "invalid or short history: REVIEW")], "configs/config.yaml"),
    ("Work orders", TB_SKY, "FFFFFF", TB_SKY, [
        ("Order", "simulated, status OPEN_SIMULATED"), ("Identity", "deterministic ID, restart-safe"),
        ("Issue rule", "once per severity, on escalation"), ("Human", "CRITICAL flagged for review")], "work_orders_v2/*.json"),
]
for i, (nm, col, tcol, dcol, chips, art_) in enumerate(stages):
    x = AX0 + i * STEP
    rect(s, x, AY0 + 0.51, CWd, 0.3, col, None, MSO_SHAPE.PENTAGON)
    put(rect(s, x + 0.05, AY0 + 0.54, 0.24, 0.24, WHITE, None, MSO_SHAPE.OVAL), [(str(i + 1), 8, True, dcol)], align=PP_ALIGN.CENTER, m=0.0)
    text(s, x + 0.33, AY0 + 0.51, CWd - 0.45, 0.3, [(nm, 7, True, tcol)], anchor=MSO_ANCHOR.MIDDLE)
    for j, (lab, det) in enumerate(chips):
        put(rect(s, x + 0.01, AY0 + 0.85 + j * 0.375, CWd - 0.02, 0.35, tint(col), col, MSO_SHAPE.ROUNDED_RECTANGLE),
            [(lab, 6.2, True, dcol), (det, 5.3, False, INK)], align=PP_ALIGN.LEFT, m=0.05)
    put(rect(s, x + 0.01, AY0 + 2.35, CWd - 0.02, 0.17, "E3E9F0", None, MSO_SHAPE.ROUNDED_RECTANGLE), [("Artifact: " + art_, 5, False, "334155")], align=PP_ALIGN.CENTER, m=0.01)
put(rect(s, 0.32, AY0 + 2.56, 0.85, 0.22, TB_NAVY, None, MSO_SHAPE.ROUNDED_RECTANGLE), [("MLOPS", 6.2, True, WHITE)], align=PP_ALIGN.CENTER, m=0.01)
cp = ["MLflow: params, metrics, models", "Registry: promote if RMSE improves", f"Drift: PSI>{rc_['psi_alarm']}, RMSE>{rc_['rmse_factor']}x, {rc_['days']} d", "Retrain: engine-held-out, rollback", "Monitor: read-only Streamlit"]
CWp = (8.68 - 0.85 - 0.05 * 5) / 5
for j, t_ in enumerate(cp):
    put(rect(s, 1.22 + j * (CWp + 0.05), AY0 + 2.56, CWp, 0.22, tint(TB_NAVY, 0.82), TB_NAVY, MSO_SHAPE.ROUNDED_RECTANGLE), [(t_, 5.4, True, TB_NAVY)], align=PP_ALIGN.CENTER, m=0.02)

# ---------------- native charts (right column)
native_cov(s, 9.12, AY0, 3.96, 1.38)
native_cost2(s, 9.12, AY0 + 1.42, 3.96, 1.4)

# ---------------- bottom row
RY = 5.08
RH = 2.0
rows = [["Design decision", "Reasoning", "Evidence"],
        [f"Capped RUL target ({cap_})", "Early-life degradation is not observable from sensors; the cap stabilises the regression target.", f"RMSE {rng(rm)} cycles"],
        ["Gradient-boosted trees", "Tabular rolling features and exact TreeSHAP; no deep model gave a reliable gain over it.", "7 models compared"],
        ["Healthy-only autoencoder", "Failures are rare: model normal behaviour and score deviation; no failure labels needed.", f"AUROC {NA(af)}"],
        [f"Persistence k = {an_['persist_k']}", "An alert needs consecutive flags, which suppresses isolated sensor noise.", f"FPR {min(pf):.2%} to {max(pf):.2%}" if pf else "n/a"],
        ["Early-biased risk bound", "Over-estimating RUL is the unsafe error (asymmetric NASA score); the bound shifts predictions early.", f"late {lr:.0%} to {lc:.0%}"],
        ["Engine-grouped validation", "Row-level splits leak the same engine into training and validation; grouping prevents it.", f"{cfg['eval']['cv_folds']}-fold GroupKFold"],
        ["Deterministic rule triage", "Auditable and reproducible: each order records the rule that fired.", "no LLM"]]
table(s, 0.25, RY, 4.95, RH, rows, [1.2, 2.55, 1.2], 5.6)
cimg(s, "s2_band", 5.25, RY, 1.95, RH, d_band, "RMSE by true-RUL band")
cimg(s, "s2_hist", 7.25, RY, 1.85, RH, d_hist, "Test error distribution")

rect(s, 9.17, RY, 1.88, RH, "F7FAFE", TB_LINE)
sec(s, 9.2, RY + 0.01, 1.82, "Triage funnel (engines)", TB_NAVY)
fun = [("Test engines replayed", NTOT, TB_NAVY), ("Work order issued", N_ORD, TB_BLUE), ("HIGH or CRITICAL", N_HIGH, TB_GOLD), ("CRITICAL", N_CRIT, RED)]
tops, bots = [1.72, 1.44, 1.16, 0.88], [1.44, 1.16, 0.88, 0.6]
fcx = 9.17 + 0.94
for i, (lab, n, col) in enumerate(fun):
    y = RY + 0.33 + i * 0.415
    fb = s.shapes.build_freeform(Xe(fcx - tops[i] / 2), Ye(y), scale=1.0)
    fb.add_line_segments([(Xe(fcx + tops[i] / 2), Ye(y)), (Xe(fcx + bots[i] / 2), Ye(y + 0.4)), (Xe(fcx - bots[i] / 2), Ye(y + 0.4))], close=True)
    sh = fb.convert_to_shape()
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(col)
    sh.line.fill.background()
    sh.shadow.inherit = False
    put(sh, [(f"{n} ({100 * n / NTOT:.0f}%)", 7.4, True, WHITE if col != TB_GOLD else TB_NAVY), (lab, 5.4, False, "E3ECF7" if col != TB_GOLD else TB_NAVY)], align=PP_ALIGN.CENTER, m=0.0)

lead = [anom[f]["IFl_lead_median"] for f in FA]
fnd = [f"Accuracy: RMSE {rng(b30)} cycles at true RUL <= 30 versus {rng(bmid)} at RUL 60 to 125; mean signed error {rng(bi, '+.1f')} cycles.",
       f"Safety: the 90% lower bound cuts late predictions from {lr:.0%} to {lc:.0%} at {rng(sl, '.0f')} cycles of slack; coverage {rng(cv, '.2f')} below the cap."]
if af and pf:
    fnd.append(f"Detection: AUROC {rng(af, '.3f')}, persistent FPR {min(pf):.2%} to {max(pf):.2%}, median alert lead {rng(lead, '.0f')} cycles before failure.")
fnd.append(f"Action: {sum(sum(CNT[f].values()) for f in FDS)} simulated work orders over {N_ORD} of {NTOT} engines; {N_CRIT} reach CRITICAL.")
if be:
    fnd.append(f"Economics: break-even r = {rng(be)} against run-to-failure (in-sample, simulated, no monetary values).")
tile(s, 11.1, RY, 1.98, RH, "Key findings", fnd, TB_NAVY, fs=6.3)

'''
out = src[:src.index(A)] + NEW + src[src.index(B):]
shutil.copy2(P, P.with_name(P.name + f".bak_{time.strftime('%Y%m%d_%H%M%S')}"))
P.write_text(out)
py_compile.compile(str(P), doraise=True)
print("patched + compiled scripts/15_deck_v4.py")

# ===================== PATCH: title case, cost chart image, bold icons, logo crop =====================
WRAP = r'''# TITLE-CASE WRAPPERS
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
'''

ICONS = r'''from matplotlib.patches import Circle, Polygon


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


'''

_p = ROOT / "scripts" / "15_deck_v4.py"
_s = _p.read_text()
_REPL = [
    ('"System architecture: closed-loop flow from telemetry to simulated work order"', '"System Architecture: Closed-Loop Flow from Telemetry to Simulated Work Order"'),
    ('("Ingest and validate", TB_NAVY,', '("Ingest and Validate", TB_NAVY,'),
    ('("Causal features", TB_BLUE,', '("Causal Features", TB_BLUE,'),
    ('("Triage rules", TB_GOLD,', '("Triage Rules", TB_GOLD,'),
    ('("Work orders", TB_SKY,', '("Work Orders", TB_SKY,'),
    ('"Triage funnel (engines)", TB_NAVY)', '"Triage Funnel (Engines)", TB_NAVY)'),
    ('"Key findings", fnd, TB_NAVY', '"Key Findings", fnd, TB_NAVY'),
    ('["Design decision", "Reasoning", "Evidence"]', '["Design Decision", "Reasoning", "Evidence"]'),
    ('cimg(s, "s2_band", 5.25, RY, 1.95, RH, d_band, "RMSE by true-RUL band")', 'cimg(s, "s2_band", 5.25, RY, 1.95, RH, d_band, "RMSE by True-RUL Band")'),
    ('cimg(s, "s2_hist", 7.25, RY, 1.85, RH, d_hist, "Test error distribution")', 'cimg(s, "s2_hist", 7.25, RY, 1.85, RH, d_hist, "Test Error Distribution")'),
    ('"Lower-bound coverage, engines below cap (mean of 4 subsets)"', '"Lower-Bound Coverage, Engines Below Cap (Mean of 4 Subsets)"'),
    ('native_cost2(s, 9.12, AY0 + 1.42, 3.96, 1.4)', 'cimg(s, "s2_cost", 9.12, AY0 + 1.42, 3.96, 1.4, d_cost, "Cost-Aware Replay, Mean of 4 Subsets (Simulated)")'),
    ('S2_DECOR.append(("pic", sh.image.blob, sh.left, sh.top, sh.width, sh.height, None))',
     'S2_DECOR.append(("pic", sh.image.blob, sh.left, sh.top, sh.width, sh.height, (sh.crop_left, sh.crop_right, sh.crop_top, sh.crop_bottom)))'),
    ('s_.shapes.add_picture(io.BytesIO(a), l, t, w, h)',
     's_pc = s_.shapes.add_picture(io.BytesIO(a), l, t, w, h)\n                if ex_:\n                    s_pc.crop_left, s_pc.crop_right, s_pc.crop_top, s_pc.crop_bottom = ex_'),
    ('s = S2\nTR.update(ID)\nan_, tg_, dcfg', ICONS + 's = S2\nTR.update(ID)\nan_, tg_, dcfg'),
    ('C4 = [0.15 + i * 3.25 for i in range(4)]\n', WRAP + 'C4 = [0.15 + i * 3.25 for i in range(4)]\n'),
]
for _old, _new in _REPL:
    if _new in _s:
        continue
    if _s.count(_old) != 1:
        raise SystemExit(f"patch: expected exactly 1 match, found {_s.count(_old)} for {_old[:60]!r}")
    _s = _s.replace(_old, _new)
_p.write_text(_s)
py_compile.compile(str(_p), doraise=True)
print("title-case / icon / cost-chart / logo-crop patch applied")


# ===================== PATCH: mind-map storytelling slide 2 =====================
MIND = r'''# MINDMAP SLIDE
s = S2
TR.update(ID)
an_, tg_, dcfg = cfg["anomaly"], cfg["triage"], cfg["data"]
nfeat = [len(D[f]["P"]["pre"].sensors) * (len(cfg["features"]["stats"]) + 2) for f in FDS]
nsen = [len(D[f]["P"]["pre"].sensors) for f in FDS]
rc_ = cfg["mlops"]["retrain"]
NA = lambda v, fm=".2f": rng(v, fm) if v else "n/a"
MIDB, DCY = "3B7DD8", "0B7F8A"
CWc, CHc, STEPc, Y0c, LXc, RXc = 4.35, 1.74, 1.82, 1.58, 0.25, 8.73
HX, HY, HR = 6.665, 4.27, 0.82


def d_nfeat(axs):
    ax = axs[0, 0]
    ax.bar(range(4), nfeat, 0.55, color=[H(SUBF[f]) for f in FDS])
    for i, v in enumerate(nfeat):
        ax.text(i, v + 2, str(v), ha="center", fontsize=5.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(FDS)
    ax.set_ylabel("Feature columns")
    ax.set_ylim(0, max(nfeat) * 1.2)


text(s, 2.7, 0.08, 9.4, 0.42, [("SENTINEL: Closed-Loop Predictive Maintenance for a Connected Fleet", 16, True, TB_NAVY)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 2.7, 0.48, 9.4, 0.28, [("Storyline: Sense, Prepare, Engineer, Detect, Predict, Decide, Act, Learn | NASA C-MAPSS FD001 to FD004 | Benchmark Replay, Simulated Actions", 7, False, TB_BLUE)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
rect(s, 2.7, 0.8, 9.4, 0.03, TB_GOLD)

# ---------------- story ribbon
steps = [("1 Sense", "Card 2 | Telemetry In", TB_NAVY, "FFFFFF"), ("2 Prepare", "Card 2 | Validate, Normalise", TB_BLUE, "FFFFFF"),
         ("3 Engineer", "Card 3 | Causal Features", TB_SKY, "FFFFFF"), ("4 Detect", "Card 4 | Anomaly Scoring", TB_CYAN, "FFFFFF"),
         ("5 Predict", "Card 4 | RUL and Bound", DCY, "FFFFFF"), ("6 Decide", "Card 5 | Rule Triage", TB_GOLD, TB_NAVY),
         ("7 Act", "Card 5 | Work Orders", TB_AMB, "FFFFFF"), ("8 Learn", "Card 6 | Gated Retraining", MIDB, "FFFFFF")]
RW = 1.67
for i, (a_, b_, col, tcol) in enumerate(steps):
    put(rect(s, 0.25 + i * 1.595, 0.88, RW, 0.58, col, None, MSO_SHAPE.PENTAGON), [(a_, 8, True, tcol), (b_, 5.6, False, tcol)], align=PP_ALIGN.CENTER, m=0.03)

# ---------------- branches (drawn first so the hub and cards sit on top)
for k in range(3):
    yc = Y0c + CHc / 2 + k * STEPc
    line(s, HX, HY, LXc + CWc, yc, "7FA6D6", False, 1.75)
    line(s, HX, HY, RXc, yc, "7FA6D6", False, 1.75)

# ---------------- KPI grid above the hub
kp = [("Engines Escalated", f"{N_ORD} / {NTOT}", N_ORD / NTOT, TB_BLUE, "gear", None, None),
      ("RUL Error (RMSE)", rng(rm), nr_, TB_CYAN, "target", None, None),
      ("Anomaly AUROC", NA(af), min(af) if af else 0, TB_SKY, "pulse", None, None),
      ("Bound Coverage 90%", rng(cv, ".2f"), min(cv), TB_NAVY, "shield", None, None),
      ("Late Predictions", f"{lr:.0%} to {lc:.0%}", lr, RED, "clock", lc, GREEN),
      ("Early-Warning Recall", NA(tp), min(tp) if tp else 0, TB_GOLD, "bell", None, None)]
for i, (lab, val, fr, col, kind, fr2, col2) in enumerate(kp):
    x, y = 4.7 + (i % 3) * 1.33, 1.58 + (i // 3) * 0.73
    rect(s, x, y, 1.27, 0.67, TB_BG, TB_LINE)
    rect(s, x, y, 0.04, 0.67, col)
    pic(s, ring_icon(f"m{i}", fr, col, kind, fr2, col2), x + 0.06, y + 0.14, 0.4, 0.4)
    text(s, x + 0.48, y + 0.02, 0.78, 0.26, [(lab.upper(), 4.8, True, TB_BLUE)])
    text(s, x + 0.48, y + 0.29, 0.78, 0.36, [(val, 7.6, True, TB_NAVY)], anchor=MSO_ANCHOR.MIDDLE)
text(s, 4.7, 3.02, 3.93, 0.3, [(f"Evidence: Official Test, Last Observed Cycle per Engine ({NTOT} Engines, 4 Subsets)", 5.8, True, TB_BLUE)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# ---------------- hub
rect(s, HX - HR - 0.1, HY - HR - 0.1, 2 * HR + 0.2, 2 * HR + 0.2, None, TB_GOLD, MSO_SHAPE.OVAL)
put(rect(s, HX - HR, HY - HR, 2 * HR, 2 * HR, TB_NAVY, None, MSO_SHAPE.OVAL),
    [("SENTINEL", 11, True, "FFFFFF"), ("Closed-Loop", 6.5, False, "C9D6E6"), ("Predictive Maintenance", 6.5, False, "C9D6E6"), ("Telemetry to Work Order", 5.8, False, "F2A900")],
    align=PP_ALIGN.CENTER, m=0.02)

# ---------------- evidence charts below the hub
cimg(s, "s2_cov", 4.7, 5.2, 1.93, 1.76, d_cov, "Lower-Bound Coverage")
cimg(s, "s2_cost", 6.7, 5.2, 1.93, 1.76, d_cost, "Cost Replay (Simulated)")

# ---------------- retraining table rows (from the stored demo, if present)
rt_rows = None
try:
    rd_ = sorted(art(cfg, "retrain_demo", "x").parent.glob("*.json"))
    R_ = sorted((json.loads(p_.read_text()) for p_ in rd_), key=lambda r: (r["synthetic_drift_std"], r["days_since"]))
    rt_rows = [["Scenario", "v1 RMSE", "v2 RMSE", "Promoted"]]
    for r in R_:
        lab = f"Drift {r['synthetic_drift_std']:g} sd" if r["synthetic_drift_std"] else (f"Scheduled {r['days_since']} d" if r["days_since"] else "No Drift")
        rt_rows.append([lab, f"{r['v1_val']['rmse']:.1f}", f"{r['v2_val']['rmse']:.1f}" if "v2_val" in r else "n/a", ("yes" if r["v2_promoted"] else "no") if "v2_val" in r else "n/a"])
except (OSError, KeyError, ValueError, TypeError):
    rt_rows = None


def card(x, y, col, tcol, num, title, bl, kind, tw):
    rect(s, x, y, CWc, CHc, "FFFFFF", TB_LINE)
    rect(s, x, y, CWc, 0.27, col)
    put(rect(s, x + 0.04, y + 0.025, 0.22, 0.22, "FFFFFF", None, MSO_SHAPE.OVAL), [(str(num), 7, True, TB_AMB if col == TB_GOLD else col)], align=PP_ALIGN.CENTER, m=0.0)
    text(s, x + 0.3, y, CWc - 0.34, 0.27, [(title, 8, True, tcol)], anchor=MSO_ANCHOR.MIDDLE)
    put(rect(s, x, y + 0.27, tw, CHc - 0.27, tint(col, 0.94)), [("\u2022 " + b, 6.1, False, INK) for b in bl], align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, m=0.06)
    cx_, cy_, cw_, ch_ = x + tw + 0.04, y + 0.31, CWc - tw - 0.08, CHc - 0.35
    if kind == "life":
        cimg(s, "mm_life", cx_, cy_, cw_, ch_, d_life, "Engine Life, Cycles")
    elif kind == "sens":
        cimg(s, "mm_sens", cx_, cy_, cw_, ch_, d_sens, "Sensor Selection")
    elif kind == "nfeat":
        cimg(s, "mm_nf", cx_, cy_, cw_, ch_, d_nfeat, "Feature Columns")
    elif kind == "scatter":
        cimg(s, "mm_sc", cx_, cy_, cw_, ch_, d_scatter, "Predicted vs True RUL")
    elif kind == "funnel":
        text(s, cx_, cy_ - 0.02, cw_, 0.16, [("Triage Funnel (Engines)", 6.3, True, TB_NAVY)], align=PP_ALIGN.CENTER)
        fun = [("Replayed", NTOT, TB_NAVY), ("Order Issued", N_ORD, TB_BLUE), ("HIGH+", N_HIGH, TB_GOLD), ("CRITICAL", N_CRIT, RED)]
        tops, bots = [cw_ - 0.04, cw_ - 0.32, cw_ - 0.6, cw_ - 0.88], [cw_ - 0.32, cw_ - 0.6, cw_ - 0.88, cw_ - 1.16]
        fx = cx_ + cw_ / 2
        for i, (lab, n, c_) in enumerate(fun):
            yy = cy_ + 0.16 + i * 0.3
            fb = s.shapes.build_freeform(Xe(fx - tops[i] / 2), Ye(yy), scale=1.0)
            fb.add_line_segments([(Xe(fx + tops[i] / 2), Ye(yy)), (Xe(fx + bots[i] / 2), Ye(yy + 0.285)), (Xe(fx - bots[i] / 2), Ye(yy + 0.285))], close=True)
            sh = fb.convert_to_shape()
            sh.fill.solid()
            sh.fill.fore_color.rgb = rgb(c_)
            sh.line.fill.background()
            sh.shadow.inherit = False
            put(sh, [(f"{n} ({100 * n / NTOT:.0f}%) {lab}" if i < 2 else f"{n} {lab}", 5.4, True, TB_NAVY if c_ == TB_GOLD else "FFFFFF")], align=PP_ALIGN.CENTER, m=0.0)
    elif kind == "table" and rt_rows:
        text(s, cx_, cy_ - 0.02, cw_, 0.3, [("Retraining Demonstration (FD001, Synthetic Drift)", 5.8, True, TB_NAVY)], align=PP_ALIGN.CENTER)
        table(s, cx_, cy_ + 0.3, cw_, 0.9, rt_rows, [1.5, 0.7, 0.7, 0.8], 5)


cards = [
    (LXc, Y0c, TB_NAVY, "FFFFFF", 1, "Problem and Objective", [
        "Problem: failures are found after breakdown or at fixed service intervals, causing unplanned downtime and avoidable maintenance cost.",
        "Opportunity: sensor streams carry degradation signatures well before failure.",
        "Objective: close the loop from telemetry to maintenance work order with no manual step; humans review exceptions only.",
        "Constraints: lightweight, open-source, no LLM; rare failures; end-to-end run from data and configuration alone.",
        f"Dataset coverage: FD001 to FD004 ({NTOT} test engines), 1 or 6 operating conditions, 1 or 2 fault modes, each processed independently."], "life", 2.55),
    (LXc, Y0c + STEPc, TB_BLUE, "FFFFFF", 2, "Data Engineering", [
        "Ingest: 26 columns per cycle (3 settings, 21 sensors) from batch files; the source interface is replaceable by a streaming feed.",
        "Validate: NaN, duplicates, cycle order and gaps; official test RUL labels aligned to engine order.",
        f"Regimes: KMeans k={dcfg['n_regimes']} on operating settings (FD002, FD004); per-regime z-score from training data only.",
        f"Clean: causal trailing median (w={dcfg['smooth_window']}); constant sensors dropped; sensors with |Spearman| < {dcfg['min_abs_trend']} dropped.",
        f"Output: {min(nsen)} to {max(nsen)} informative sensors per subset; raw inputs preserved."], "sens", 2.55),
    (LXc, Y0c + 2 * STEPc, TB_SKY, "FFFFFF", 3, "Causal Feature Engineering", [
        "Windows: trailing 30 and 10 cycles only; never crosses engine boundaries; no future observations.",
        f"Statistics: mean, std, min, max, slope, kurtosis give {min(nfeat)} to {max(nfeat)} feature columns.",
        f"Target: RUL = maximum cycle minus cycle, capped at {cap_}; sample weight 3 below RUL 30.",
        f"Validation: engine-grouped {cfg['eval']['cv_folds']}-fold CV; the official test set is read once.",
        "FD001 ablation: slope features carry the signal; CV RMSE 24.8 (mean only) to 13.9 (full set)."], "nfeat", 2.55),
    (RXc, Y0c, TB_CYAN, "FFFFFF", 4, "Detection and Prediction Models", [
        f"Anomaly: autoencoder on healthy cycles {an_['healthy_cycle_range'][0]} to {an_['healthy_cycle_range'][1]} plus Isolation Forest on latent and error; alert after k={an_['persist_k']} consecutive flags.",
        f"RUL: XGBoost {cfg['rul']['xgb']['n_estimators']} trees, depth {cfg['rul']['xgb']['max_depth']}; seven model families compared, none reliably better.",
        "Risk bound: 0.8-quantile offset deployed; conformal 90% lower bound evaluated offline (measured, not guaranteed).",
        "Attribution: exact TreeSHAP; top-3 sensors attached to each work order.",
        f"Result: RMSE {rng(rm)} cycles; AUROC {NA(af, '.3f')}; late rate {lr:.0%} raw, {lc:.0%} with the bound."], "scatter", 2.55),
    (RXc, Y0c + STEPc, TB_GOLD, TB_NAVY, 5, "Decision Agent (Non-LLM)", [
        "Rules: CRITICAL, HIGH, MEDIUM, LOW, REVIEW; first matching rule wins and is recorded on the order.",
        f"Triggers: conservative RUL <= {tg_['critical_rul']} / {tg_['high_rul']} / {tg_['medium_rul']} cycles (provisional) or persistent anomaly above the score cut.",
        "Safety: invalid input or under 30 cycles of history gives REVIEW, never LOW.",
        "Orders: simulated, OPEN_SIMULATED, deterministic ID, once per severity on escalation, restart-safe.",
        f"Outcome: {N_ORD} of {NTOT} engines escalated; {N_CRIT} CRITICAL for human review. No physical action is triggered."], "funnel", 2.55),
    (RXc, Y0c + 2 * STEPc, MIDB, "FFFFFF", 6, "MLOps and Learning", [
        "Tracking: MLflow parameters, metrics and model artifacts per run (local store).",
        "Registry: champion and challenger aliases; promote only if validation RMSE improves.",
        f"Drift gate: PSI > {rc_['psi_alarm']}, RMSE > {rc_['rmse_factor']}x baseline, or {rc_['days']} days since training.",
        "Retrain: engine-held-out validation; previous champion kept for rollback.",
        "Monitor: read-only dashboard (RUL, bounds, cost policy, anomaly, work orders).",
        "Limit: demonstrated on FD001 with synthetic drift; real labels arrive only after failures."], "table", 2.3 if rt_rows else CWc),
]
for c_ in cards:
    card(*c_)
for k in range(3):
    yc = Y0c + CHc / 2 + k * STEPc
    for xx, cc in ((LXc + CWc, cards[k][2]), (RXc, cards[3 + k][2])):
        rect(s, xx - 0.05, yc - 0.05, 0.1, 0.1, cc, "FFFFFF", MSO_SHAPE.OVAL)

'''
_p = ROOT / "scripts" / "15_deck_v4.py"
_s = _p.read_text()
if "# MINDMAP SLIDE" not in _s:
    _a = "s = S2\nTR.update(ID)\nan_, tg_, dcfg"
    _b = "# ------------------------------------------------------------------ appendix frame with navigation\n"
    if _s.count(_a) != 1 or _s.count(_b) != 1:
        raise SystemExit("mind-map patch: slide-2 markers not found exactly once; run the earlier write_v11 patches first")
    _s = _s[:_s.index(_a)] + MIND + _s[_s.index(_b):]
    _p.write_text(_s)
py_compile.compile(str(_p), doraise=True)
print("mind-map slide 2 patch applied")