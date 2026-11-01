import py_compile, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TS = time.strftime("%Y%m%d_%H%M%S")
P = ROOT / "scripts" / "13_deck_v2.py"
if not P.exists():
    raise SystemExit("scripts/13_deck_v2.py not found; run write_v5.py and write_v6.py first")

NEW = r'''from pptx.enum.dml import MSO_LINE
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

'''

s = P.read_text()
A = "# ------------------------------------------------------------------ slide 1\n"
B = "# ------------------------------------------------------------------ appendix\n"
if s.count(A) != 1 or s.count(B) != 1:
    raise SystemExit("slide markers not found exactly once; file differs from write_v5/v6 output")
i, j = s.index(A), s.index(B)
s = s[:i] + A + NEW + s[j:]
shutil.copy2(P, P.with_name(P.name + f".bak_{TS}"))
P.write_text(s)
py_compile.compile(str(P), doraise=True)
print("patched + compiled scripts/13_deck_v2.py")