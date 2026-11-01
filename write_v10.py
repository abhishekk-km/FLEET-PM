import py_compile, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT / "scripts" / "15_deck_v4.py"
if not P.exists():
    raise SystemExit("scripts/15_deck_v4.py not found; run write_v9.py first")
A = "# ================================================================== SLIDE 2 (the one-pager, on the template's content page)\n"
B = "# ------------------------------------------------------------------ appendix frame with navigation\n"
src = P.read_text()
if src.count(A) != 1 or src.count(B) != 1:
    raise SystemExit("slide-2 markers not found exactly once")

NEW = r'''# ================================================================== SLIDE 2 (the one-pager, on the template's content page)
TB_NAVY, TB_BLUE, TB_SKY, TB_CYAN, TB_GOLD, TB_GREEN, TB_BG, TB_LINE = "0B2F6B", "1566C0", "2F8FD0", "0E9AA7", "F2A900", "2E8B57", "EEF5FC", "BCD2EA"


def tint(h, a=0.86):
    c = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "".join(f"{int(v + (255 - v) * a):02X}" for v in c)


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

text(s, 2.7, 0.08, 9.4, 0.42, [(TITLE, 18, True, TB_NAVY)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 2.7, 0.48, 9.4, 0.28, [("NASA C-MAPSS benchmark replay | simulated actions | technical review", 8, False, TB_BLUE)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
rect(s, 2.7, 0.8, 9.4, 0.03, TB_GOLD)

gauges = [("Engines with work order", f"{N_ORD} / {NTOT}", f"{len(FDS)} subsets, official test", N_ORD / NTOT, TB_BLUE, f"{N_ORD / NTOT:.0%}", None, None),
          ("RMSE vs capped RUL", rng(rm), "cycles; ring = share of cap", nr_, TB_CYAN, f"{nr_:.0%}", None, None)]
if af:
    gauges.append(("Anomaly AUROC", rng(af, ".3f"), "Isolation Forest, latent + error", min(af), TB_SKY, f"{min(af):.2f}", None, None))
gauges.append(("90% bound coverage", rng(cv, ".2f"), "engines below cap; measured", min(cv), TB_NAVY, f"{min(cv):.2f}", None, None))
gauges.append(("Late predictions", f"{lr:.0%} to {lc:.0%}", "outer raw, inner 90% bound", lr, RED, "", lc, GREEN))
if tp:
    gauges.append(("Persistent TPR", rng(tp, ".2f"), "anomaly, RUL <= 30 proxy", min(tp), TB_GOLD, f"{min(tp):.0%}", None, None))
KW = (12.83 - 0.07 * (len(gauges) - 1)) / len(gauges)
for i, (lab, val, sub, fr, col, ctr, fr2, col2) in enumerate(gauges):
    x = 0.25 + i * (KW + 0.07)
    rect(s, x, 0.9, KW, 0.72, TB_BG, TB_LINE)
    rect(s, x, 0.9, 0.05, 0.72, col)
    pic(s, ring_img(f"k{i}", fr, col, ctr, fr2, col2), x + 0.08, 0.95, 0.62, 0.62)
    text(s, x + 0.72, 0.92, KW - 0.74, 0.17, [(lab.upper(), 5.6, True, TB_BLUE)])
    text(s, x + 0.72, 1.07, KW - 0.74, 0.28, [(val, 11, True, TB_NAVY)])
    text(s, x + 0.72, 1.36, KW - 0.74, 0.25, [(sub, 5.6, False, "5B6B80")])

# ---------------- architecture
rect(s, 0.25, 1.7, 8.55, 2.68, "F7FAFE", TB_LINE)
sec(s, 0.3, 1.71, 8.45, "System architecture: end-to-end data flow (benchmark replay)", TB_NAVY)
GOLDTXT = "7A5200"
stages = [
    ("Ingest and validate", TB_NAVY, "FFFFFF", TB_NAVY, [
        ("Source", "FD001-FD004, 26 columns, official RUL files"), ("Validation", "NaN, duplicates, cycle gaps, label alignment"),
        ("Regimes", f"KMeans k={dcfg['n_regimes']}, per-regime z-score (FD002, FD004)"), ("Sensors", f"{min(nsen)} to {max(nsen)} kept, |Spearman| >= {dcfg['min_abs_trend']}")],
     "data/CMAPSSData/*.txt"),
    ("Causal features", TB_BLUE, "FFFFFF", TB_BLUE, [
        ("Smoothing", f"causal trailing median, w={dcfg['smooth_window']}"), ("Windows", "30 and 10 cycles, trailing only"),
        ("Statistics", "mean, std, min, max, slope, kurtosis"), ("Matrix", f"{min(nfeat)} to {max(nfeat)} columns; RUL cap {cap_}")],
     "artifacts/features/*.joblib"),
    ("Models", TB_CYAN, "FFFFFF", TB_CYAN, [
        ("Anomaly", f"autoencoder + Isolation Forest, k={an_['persist_k']}"), ("RUL", f"XGBoost {cfg['rul']['xgb']['n_estimators']} trees, depth {cfg['rul']['xgb']['max_depth']}, grouped CV"),
        ("Bound", "0.8-q offset deployed; conformal 90% offline"), ("Attribution", "exact TreeSHAP grouped by sensor")],
     "rul_bundle, anomaly_ifl, OOF npz"),
    ("Deterministic triage", TB_GOLD, TB_NAVY, GOLDTXT, [
        ("Severities", "CRITICAL, HIGH, MEDIUM, LOW, REVIEW"), ("RUL cutoffs", f"{tg_['critical_rul']} / {tg_['high_rul']} / {tg_['medium_rul']} cycles (provisional)"),
        ("Anomaly rule", f"persistent k={an_['persist_k']} and score cut"), ("Safety", "invalid input: REVIEW, never LOW")],
     "configs/config.yaml (triage)"),
    ("Work orders, monitor", TB_GREEN, "FFFFFF", TB_GREEN, [
        ("Work orders", "simulated, status OPEN_SIMULATED"), ("Identity", "deterministic ID, restart-safe"),
        ("Issue rule", "once per severity, on escalation"), ("Monitor", "read-only Streamlit; no physical action")],
     "artifacts/work_orders_v2/*.json"),
]
for i, (nm, col, tcol, dcol, chips, art_) in enumerate(stages):
    x = 0.32 + i * 1.68
    rect(s, x, 2.03, 1.66, 0.3, col, None, MSO_SHAPE.PENTAGON)
    put(rect(s, x + 0.05, 2.06, 0.24, 0.24, WHITE, None, MSO_SHAPE.OVAL), [(str(i + 1), 8, True, dcol)], align=PP_ALIGN.CENTER, m=0.0)
    text(s, x + 0.33, 2.03, 1.2, 0.3, [(nm, 7, True, tcol)], anchor=MSO_ANCHOR.MIDDLE)
    for j, (lab, det) in enumerate(chips):
        put(rect(s, x + 0.02, 2.37 + j * 0.37, 1.6, 0.34, tint(col), col, MSO_SHAPE.ROUNDED_RECTANGLE),
            [(lab, 6.2, True, dcol), (det, 5.4, False, INK)], align=PP_ALIGN.LEFT, m=0.05)
    put(rect(s, x + 0.02, 3.87, 1.6, 0.24, "E3E9F0", None, MSO_SHAPE.ROUNDED_RECTANGLE), [("Artifact: " + art_, 5.2, False, "334155")], align=PP_ALIGN.CENTER, m=0.02)
put(rect(s, 0.32, 4.14, 1.0, 0.22, TB_NAVY, None, MSO_SHAPE.ROUNDED_RECTANGLE), [("CONTROL PLANE", 6, True, WHITE)], align=PP_ALIGN.CENTER, m=0.02)
cp = [("MLflow tracking", "params, metrics, models"), ("Registry gate", "promote on validation RMSE"),
      ("Drift gate", f"PSI>{rc_['psi_alarm']}, RMSE>{rc_['rmse_factor']}x, {rc_['days']} d"), ("Retraining", "engine-held-out, rollback kept")]
CWp = (8.4 - 1.0 - 0.05 * 4) / 4
for j, (t1, t2) in enumerate(cp):
    put(rect(s, 1.37 + j * (CWp + 0.05), 4.14, CWp, 0.22, tint(TB_NAVY, 0.82), TB_NAVY, MSO_SHAPE.ROUNDED_RECTANGLE),
        [(f"{t1}: {t2}", 5.6, True, TB_NAVY)], align=PP_ALIGN.CENTER, m=0.02)

# ---------------- native charts
native_cov(s, 8.9, 1.7, 4.18, 1.27)
native_cost2(s, 8.9, 3.0, 4.18, 1.38)

# ---------------- bottom-left: theory tiles above charts
th = [("Target and features", TB_NAVY, f"RUL capped at {cap_} cycles: early-life degradation is not observable. Causal trailing windows (30, 10 cycles) over regime-normalised sensors: mean, std, min, max, slope, kurtosis."),
      ("Anomaly detection", TB_BLUE, f"Autoencoder fitted on healthy cycles {an_['healthy_cycle_range'][0]}-{an_['healthy_cycle_range'][1]} sets a normal envelope; Isolation Forest scores latent state and error. Alert needs {an_['persist_k']} consecutive flags. Signals unusual behaviour, not a confirmed fault."),
      ("Conformal lower bound", TB_CYAN, "Out-of-fold residuals calibrated per predicted-RUL band; bound = prediction minus finite-sample (1 - alpha) quantile. Coverage measured on held-out engines, not guaranteed. Offline; triage uses the offset."),
      ("Cost-aware maintenance", "C98A00", "r = C_fail / C_waste compares failure with premature maintenance. Policies fire causally from cycle 30; break-even r shows where predictive triage beats run-to-failure and fixed interval. No monetary values.")]
for i, (t_, col, b_) in enumerate(th):
    tile(s, 0.25 + i * 2.14, 4.45, 2.08, 1.1, t_, b_, col, fs=6.3)
FA2 = FA if FA else None
row = [("s2_band", d_band, "RMSE by true-RUL band"), ("s2_hist", d_hist, "Test error distribution"),
       ("s2_auc", d_auroc if FA2 else d_hist, "Anomaly AUROC by scorer"), ("s2_tf", d_tprfpr if FA2 else d_hist, "Persistent TPR and FPR")]
for i, (nm, fn, ttl) in enumerate(row):
    cimg(s, nm, 0.25 + i * 2.14, 5.6, 2.08, 1.48, fn, ttl)

# ---------------- bottom-right: funnel and findings
rect(s, 8.9, 4.45, 2.0, 2.63, "F7FAFE", TB_LINE)
sec(s, 8.93, 4.46, 1.94, "Triage funnel (engines)", TB_NAVY)
fun = [("Test engines replayed", NTOT, TB_NAVY), ("Work order issued", N_ORD, TB_BLUE), ("HIGH or CRITICAL", N_HIGH, TB_GOLD), ("CRITICAL", N_CRIT, RED)]
tops, bots = [1.84, 1.52, 1.2, 0.88], [1.52, 1.2, 0.88, 0.56]
fcx = 8.9 + 1.0
for i, (lab, n, col) in enumerate(fun):
    y = 4.82 + i * 0.56
    fb = s.shapes.build_freeform(Xe(fcx - tops[i] / 2), Ye(y), scale=1.0)
    fb.add_line_segments([(Xe(fcx + tops[i] / 2), Ye(y)), (Xe(fcx + bots[i] / 2), Ye(y + 0.53)), (Xe(fcx - bots[i] / 2), Ye(y + 0.53))], close=True)
    sh = fb.convert_to_shape()
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(col)
    sh.line.fill.background()
    sh.shadow.inherit = False
    put(sh, [(f"{n} ({100 * n / NTOT:.0f}%)", 8, True, WHITE if col != TB_GOLD else TB_NAVY), (lab, 5.8, False, "E3ECF7" if col != TB_GOLD else TB_NAVY)], align=PP_ALIGN.CENTER, m=0.0)
nas = [ST[f]["nasa"] / ST[f]["n"] for f in FDS]
bi = [ST[f]["bias"] for f in FDS]
fnd = [f"RMSE {rng(b30)} cycles at true RUL <= 30 versus {rng(bmid)} at RUL 60 to 125.",
       f"NASA score per engine {rng(nas)}; mean signed error {rng(bi, '+.1f')} cycles (positive is late).",
       f"90% bound: late rate {lr:.0%} to {lc:.0%}, slack {rng(sl, '.0f')} cycles, coverage {rng(cv, '.2f')} below cap."]
if af and pf:
    fnd.append(f"Anomaly AUROC {rng(af, '.3f')}; persistent FPR {min(pf):.2%} to {max(pf):.2%}.")
if be:
    fnd.append(f"Break-even r {rng(be)} vs run-to-failure (in-sample, simulated).")
fnd.append(f"{sum(sum(CNT[f].values()) for f in FDS)} simulated work orders over {N_ORD} engines; {N_CRIT} reach CRITICAL.")
fnd.append(f"{min(nsen)} to {max(nsen)} sensors and {min(nfeat)} to {max(nfeat)} features retained per subset.")
tile(s, 10.95, 4.45, 2.13, 2.63, "Key findings", fnd, TB_NAVY, fs=6.6)

'''
out = src[:src.index(A)] + NEW + src[src.index(B):]
shutil.copy2(P, P.with_name(P.name + f".bak_{time.strftime('%Y%m%d_%H%M%S')}"))
P.write_text(out)
py_compile.compile(str(P), doraise=True)
print("patched + compiled scripts/15_deck_v4.py")