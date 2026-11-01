import py_compile, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TS = time.strftime("%Y%m%d_%H%M%S")
P = ROOT / "scripts" / "13_deck_v2.py"
if not P.exists():
    raise SystemExit("scripts/13_deck_v2.py not found; run write_v5.py first")

NEW = r'''FEAT = "2E75B6"
s = prs.slides.add_slide(BL)
rect(s, 0, 0, 13.333, 0.5, NAVY)
text(s, 0.15, 0.04, 8.6, 0.42, [("FleetPM: Predictive Maintenance for a Connected Fleet", 17, True, WHITE)], anchor=MSO_ANCHOR.MIDDLE)
text(s, 8.6, 0.04, 4.6, 0.42, [("NASA C-MAPSS benchmark replay | simulated actions | technical review", 8.5, False, "C9D6E6")],
     align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)


def rng(v, f=".1f"):
    return f"{min(v):{f}} to {max(v):{f}}"


def band_rmse(f, lo_, hi_):
    d = D[f]
    e, tr = d["pred_t"] - d["true_cap"], d["true_raw"]
    m = (tr > lo_) & (tr <= hi_)
    return float(np.sqrt((e[m] ** 2).mean())) if m.any() else float("nan")


rm = [ST[f]["rmse"] for f in FDS]
cv = [conf[f]["test"]["conformal_a0.1"]["cov_lt_cap"] for f in FDS]
lr = np.mean([conf[f]["test"]["raw"]["late_rate"] for f in FDS])
lc = np.mean([conf[f]["test"]["conformal_a0.1"]["late_rate"] for f in FDS])
be = [b for b in (cost[f]["break_even"]["rul_conformal10"]["vs_rtf"]["point"] for f in FDS) if b is not None]
af = [anom[f]["auroc_IF_latent_error"] for f in FDS if anom[f]]
pf = [anom[f]["IFl_fpr_persist"] for f in FDS if anom[f]]
sl = [conf[f]["test"]["conformal_a0.1"]["slack_mean"] for f in FDS]
b30 = [band_rmse(f, -1, 30) for f in FDS]
bmid = [band_rmse(f, 60, D[f]["cap"]) for f in FDS]

kp = [("Engines replayed", f"{NTOT}", f"{len(FDS)} subsets, official test, last cycle", BLUE, MSO_SHAPE.HEXAGON),
      ("RMSE vs capped RUL", rng(rm), "cycles; intervals in appendix", TEAL, MSO_SHAPE.DONUT),
      ("Anomaly AUROC", rng(af, ".3f") if af else "n/a", "Isolation Forest, latent and error", PURPLE, MSO_SHAPE.PENTAGON),
      ("90% bound coverage", rng(cv, ".2f"), "engines below cap; measured", NAVY, MSO_SHAPE.DIAMOND),
      ("Late predictions", f"{lr:.0%} to {lc:.0%}", "raw to 90% bound, mean of subsets", RED, MSO_SHAPE.CHEVRON),
      ("Break-even r", rng(be) if be else "n/a", "vs run-to-failure; in-sample", AMBER, MSO_SHAPE.OVAL)]
for i, (lab, val, sub, col, ic) in enumerate(kp):
    x = 0.15 + i * 2.19
    rect(s, x, 0.58, 2.1, 0.72, LIGHT)
    rect(s, x, 0.58, 0.05, 0.72, col)
    text(s, x + 0.1, 0.6, 1.7, 0.2, [(lab.upper(), 6.5, True, "51627A")])
    text(s, x + 0.1, 0.77, 1.95, 0.32, [(val, 15, True, NAVY)])
    text(s, x + 0.1, 1.08, 2.0, 0.2, [(sub, 6.5, False, "64748B")])
    rect(s, x + 1.82, 0.64, 0.22, 0.22, col, shape=ic)

# ---------------- architecture: five layers
AX, AY, AW, AH = 0.15, 1.38, 8.15, 3.5
rect(s, AX, AY, AW, AH, None, "D6DEE9")
sec(s, AX + 0.05, AY + 0.01, AW - 0.1, "System architecture: data, feature, model, decision and operations layers")
LN, GAP, LY0 = 0.52, 0.12, AY + 0.36
CW, CG, X0 = 1.68, 0.10, 1.1
CXS = [X0 + i * (CW + CG) for i in range(4)]
dcfg, an_, tg_, rc_ = cfg["data"], cfg["anomaly"], cfg["triage"], cfg["mlops"]["retrain"]
nfeat = [len(D[f]["P"]["pre"].sensors) * (len(cfg["features"]["stats"]) + 2) for f in FDS]
cap_ = int(D["FD001"]["cap"])
lanes = [
    ("DATA", BLUE, [
        (0, 1, "C-MAPSS source", "FD001-FD004 train, test, RUL files; 26 columns"),
        (1, 1, "Schema validation", "columns, NaN, duplicates, cycle gaps, label alignment"),
        (2, 1, "Regime normalisation", f"KMeans k={dcfg['n_regimes']} on settings (FD002, FD004); per-regime z-score"),
        (3, 1, "Sensor selection", f"constants dropped; mean |Spearman| with cycle >= {dcfg['min_abs_trend']}")]),
    ("FEATURES", FEAT, [
        (0, 2, "Causal smoothing and rolling statistics",
         f"trailing median w={dcfg['smooth_window']}; window 30: mean, std, min, max, slope, kurtosis; window 10: mean, slope"),
        (2, 1, "Feature matrix", f"{min(nfeat)} to {max(nfeat)} columns; windows never cross engines"),
        (3, 1, "Label construction", f"RUL = max cycle - cycle, cap {cap_}; official test RUL vector")]),
    ("MODELS", TEAL, [
        (0, 1, "Anomaly detector", f"autoencoder on cycles {an_['healthy_cycle_range'][0]}-{an_['healthy_cycle_range'][1]} + Isolation Forest; k={an_['persist_k']}"),
        (1, 1, "RUL regressor", f"XGBoost, {cfg['rul']['xgb']['n_estimators']} trees, depth {cfg['rul']['xgb']['max_depth']}; engine-grouped {cfg['eval']['cv_folds']}-fold CV"),
        (2, 1, "Risk bound", "0.8-quantile offset deployed; conformal 90% offline"),
        (3, 1, "Attribution", "exact TreeSHAP grouped by sensor on work orders")]),
    ("DECISION", ORANGE, [
        (0, 2, "Deterministic triage",
         f"CRITICAL, HIGH, MEDIUM, LOW, REVIEW from conservative RUL, anomaly score and persistence; cutoffs {tg_['critical_rul']}, {tg_['high_rul']}, {tg_['medium_rul']} (provisional)"),
        (2, 1, "Work-order agent", "OPEN_SIMULATED, deterministic IDs, issued on escalation"),
        (3, 1, "Cost policy replay", "break-even r vs run-to-failure and fixed interval")]),
    ("OPERATIONS", NAVY, [
        (0, 1, "MLflow registry", "champion and challenger aliases; promote on validation RMSE"),
        (1, 1, "Drift gate", f"PSI > {rc_['psi_alarm']}, RMSE > {rc_['rmse_factor']}x baseline, or {rc_['days']} days"),
        (2, 2, "Monitoring dashboard", "read-only Streamlit: fleet RUL, bounds, cost policy, anomaly, work orders")]),
]
for li, (lname, col, nodes) in enumerate(lanes):
    y = LY0 + li * (LN + GAP)
    put(rect(s, 0.22, y, 0.8, LN, col), [(lname, 7, True, WHITE)], align=PP_ALIGN.CENTER, m=0.02)
    for c0, span, t1, t2 in nodes:
        x, w = CXS[c0], span * CW + (span - 1) * CG
        put(rect(s, x, y, w, LN, col, shape=MSO_SHAPE.ROUNDED_RECTANGLE),
            [(t1, 7.5, True, WHITE), (t2, 6, False, "E3ECF7")], align=PP_ALIGN.CENTER, m=0.04)
    if li in (0, 1, 3):
        for a, b in zip(nodes[:-1], nodes[1:]):
            xe = CXS[a[0]] + a[1] * CW + (a[1] - 1) * CG
            line(s, xe, y + LN / 2, xe + CG, y + LN / 2, "334155", True, 1.25)
    if li < len(lanes) - 1:
        for c in range(4):
            xc = CXS[c] + CW / 2
            line(s, xc, y + LN, xc, y + LN + GAP, GREY, True, 1.0)

# ---------------- funnel
FX, FW = 8.38, 2.2
rect(s, FX, AY, FW, AH, None, "D6DEE9")
sec(s, FX + 0.05, AY + 0.01, FW - 0.1, "Triage funnel (engines)")
fun = [("Test engines replayed", NTOT, BLUE), ("Work order issued", N_ORD, TEAL), ("HIGH or CRITICAL", N_HIGH, ORANGE), ("CRITICAL", N_CRIT, RED)]
tops, bots = [2.0, 1.66, 1.32, 0.98], [1.66, 1.32, 0.98, 0.64]
fcx = FX + FW / 2
for i, (lab, n, col) in enumerate(fun):
    y = AY + 0.4 + i * 0.76
    fb = s.shapes.build_freeform(E(fcx - tops[i] / 2), E(y), scale=1.0)
    fb.add_line_segments([(E(fcx + tops[i] / 2), E(y)), (E(fcx + bots[i] / 2), E(y + 0.72)), (E(fcx - bots[i] / 2), E(y + 0.72))], close=True)
    sh = fb.convert_to_shape()
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(col)
    sh.line.fill.background()
    sh.shadow.inherit = False
    put(sh, [(f"{n} ({100 * n / NTOT:.0f}%)", 9.5, True, WHITE), (lab, 6.5, False, "E3ECF7")], align=PP_ALIGN.CENTER, m=0.02)

# ---------------- coverage chart
CX2, CW2 = 10.66, 2.52
rect(s, CX2, AY, CW2, AH, None, "D6DEE9")


def ch_cov2(w, h):
    v = np.array([[conf[f]["test"][m]["cov_lt_cap"] for f in FDS] for m in M])
    mu = v.mean(1)
    fig, ax = fig_(w, h)
    ax.bar(range(4), mu, color=[H(MCOL[m]) for m in M], yerr=[mu - v.min(1), v.max(1) - mu], capsize=3, ecolor="#334155")
    for y in (0.8, 0.9):
        ax.axhline(y, ls=":", c=H(GREY), lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Raw", "Offset\n0.8 q", "Conf.\n80%", "Conf.\n90%"], fontsize=6)
    ax.tick_params(labelsize=6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage", fontsize=6.5)
    ax.set_title("Lower-bound coverage, engines below cap", fontsize=7)
    clean(ax)
    return save(fig, "cov2")


pic(s, ch_cov2(2.44, 2.55), CX2 + 0.04, AY + 0.04, 2.44, 2.55)
text(s, CX2 + 0.02, AY + 2.64, CW2 - 0.04, 0.85,
     [("Bars: mean over four subsets. Whiskers: minimum to maximum. Dotted lines: nominal 80% and 90%. Measured on official test engines with true RUL below the cap; not a guarantee.", 6.5, False, "334155")])

# ---------------- bottom row
BY = 4.96
tl = [("Target and features", BLUE,
       f"RUL is capped at {cap_} cycles because sensors carry no observable degradation signal early in life. Features are strictly causal: trailing windows of 30 and 10 cycles over regime-normalised, median-smoothed sensors (mean, standard deviation, minimum, maximum, slope, kurtosis). Sample weight 3 below RUL 30."),
      ("Anomaly detection", TEAL,
       f"An autoencoder trained on healthy cycles {an_['healthy_cycle_range'][0]} to {an_['healthy_cycle_range'][1]} defines a normal operating envelope; an Isolation Forest scores latent state and reconstruction error. An alert requires {an_['persist_k']} consecutive flagged cycles, which suppresses isolated noise. The output is unusual behaviour, not a confirmed fault."),
      ("Conformal lower bound", NAVY,
       "Out-of-fold over-prediction residuals are calibrated per predicted-RUL band; the bound is the prediction minus the finite-sample (1 - alpha) quantile. Coverage is measured on held-out engines, not guaranteed. Evaluated offline; the deployed triage still uses the 0.8-quantile offset."),
      ("Cost-aware maintenance", ORANGE,
       "Failure and premature maintenance are compared through r = C_fail / C_waste. Policies fire causally from cycle 30. The break-even r states when predictive triage beats run-to-failure and fixed-interval maintenance. Replay on training trajectories; no monetary values are assumed.")]
for i, (t_, col, b_) in enumerate(tl):
    tile(s, 0.15 + (i % 2) * 3.47, BY + (i // 2) * 1.1, 3.4, 1.06, t_, b_, col, fs=7.5)


def ch_cost2(w, h):
    fig, ax = fig_(w, h)
    for k in PLAB:
        ax.loglog(RGRID, np.mean([cost[f]["curves"][k] for f in FDS], 0), color=H(PCOL[k]), lw=1.6, label=PLAB[k])
    ax.set_xlabel("r = C_fail / C_waste", fontsize=6.5)
    ax.set_ylabel("Mean cost per engine", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.set_title("Cost-aware replay, mean of 4 subsets (simulated)", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False)
    clean(ax)
    return save(fig, "cost2")


pic(s, ch_cost2(3.2, 2.18), 7.06, BY, 3.2, 2.18)
fnd = [f"Accuracy is highest near end of life: RMSE {rng(b30)} cycles when true RUL is 30 or less, against {rng(bmid)} cycles for RUL 60 to 125.",
       f"The 90% lower bound lowers the late-prediction rate from {lr:.0%} to {lc:.0%} at a mean slack of {rng(sl, '.0f')} cycles; coverage {rng(cv, '.2f')} is measured on engines below the cap."]
if af and pf:
    fnd.append(f"Anomaly scoring reaches AUROC {rng(af, '.3f')}; persistent false-positive rate {min(pf):.2%} to {max(pf):.2%} on healthy rows.")
if be:
    fnd.append(f"Predictive thresholds beat run-to-failure above r = {rng(be)} (90% bound policy); in-sample on training trajectories, simulated.")
tile(s, 10.32, BY, 2.86, 2.18, "Key findings", fnd, NAVY, fs=7.3)
text(s, 0.15, 7.17, 13.0, 0.25, [("Benchmark replay of simulated NASA C-MAPSS data; all actions are simulated; thresholds are provisional. Detailed numbers, intervals and assumptions are in the appendix.", 6.5, False, GREY)])

'''

s = P.read_text()
A = "# ------------------------------------------------------------------ slide 1\n"
B = "# ------------------------------------------------------------------ appendix\n"
if s.count(A) != 1 or s.count(B) != 1:
    raise SystemExit("slide markers not found exactly once; file differs from write_v5 output")
i, j = s.index(A), s.index(B)
s = s[:i] + A + NEW + s[j:]
pairs = [('m=0.06, font="Calibri"):', 'm=0.06, font="Arial"):'),
         ("r.font.size, r.font.bold, r.font.name = Pt(sz), b, font", "r.font.size, r.font.bold, r.font.name = Pt(round(sz * 0.94, 1)), b, font"),
         ('r.font.size, r.font.bold, r.font.name = Pt(fs), i == 0, "Calibri"', 'r.font.size, r.font.bold, r.font.name = Pt(round(fs * 0.94, 1)), i == 0, "Arial"')]
for old, new in pairs:
    if new in s:
        continue
    if s.count(old) != 1:
        raise SystemExit(f"expected exactly 1 match for {old!r}, found {s.count(old)}")
    s = s.replace(old, new)
shutil.copy2(P, P.with_name(P.name + f".bak_{TS}"))
P.write_text(s)
py_compile.compile(str(P), doraise=True)
print("patched + compiled scripts/13_deck_v2.py")