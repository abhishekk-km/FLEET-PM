import py_compile, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT / "scripts" / "15_deck_v4.py"
s = P.read_text()
A = "# MINDMAP SLIDE"
B = "# ------------------------------------------------------------------ appendix frame with navigation\n"
if s.count(A) != 1 or s.count(B) != 1:
    raise SystemExit("slide-2 markers not found exactly once; apply write_v11 and write_v12 first")

NEW = r'''# MINDMAP SLIDE
s = S2
TR.update(ID)
MIDB, DCY = "3B7DD8", "0B7F8A"
BASE = "https://abhishekk-km.github.io/FLEET-PM/interactive/"
REPO = "https://github.com/abhishekk-km/FLEET-PM"
an_, tg_, dcfg = cfg["anomaly"], cfg["triage"], cfg["data"]
nfeat = [len(D[f]["P"]["pre"].sensors) * (len(cfg["features"]["stats"]) + 2) for f in FDS]
nsen = [len(D[f]["P"]["pre"].sensors) for f in FDS]
rc_ = cfg["mlops"]["retrain"]
NA = lambda v, fm=".2f": rng(v, fm) if v else "n/a"
lead = [anom[f]["IFl_lead_median"] for f in FA]
sl = [conf[f]["test"]["conformal_a0.1"]["slack_mean"] for f in FDS]
NAV_ON = False
CWc, CHc, STEPc, Y0c, LXc, RXc = 4.35, 1.56, 1.62, 2.1, 0.25, 8.73
HX, HY, HR = 6.665, 4.35, 0.66


def lnk(sh, url):
    sh.click_action.hyperlink.address = url


def lbtn(x, y, w, h, label, url, fill=TB_NAVY, col="FFFFFF", fs=6.4):
    b = rect(s, x, y, w, h, fill, None, MSO_SHAPE.ROUNDED_RECTANGLE)
    put(b, [(label, fs, True, col)], align=PP_ALIGN.CENTER, m=0.03)
    lnk(b, url)


def lpic(name, x, y, w, h, draw, ttl, url=None):
    pic(s, mini(name, w, h, draw, ttl), x, y, w, h)
    rect(s, x, y, w, h, None, BORD)
    if url:
        lnk(s.shapes[-2], url)
        lbtn(x + w - 0.62, y + h - 0.16, 0.58, 0.13, "Interactive", url, TB_GOLD, TB_NAVY, 4.6)


def d_nfeat(axs):
    ax = axs[0, 0]
    ax.bar(range(4), nfeat, 0.55, color=[H(SUBF[f]) for f in FDS])
    for i, v in enumerate(nfeat):
        ax.text(i, v + 2, str(v), ha="center", fontsize=5.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(FDS)
    ax.set_ylabel("Feature columns")
    ax.set_ylim(0, max(nfeat) * 1.2)


text(s, 2.7, 0.08, 9.4, 0.4, [(TITLE, 16, True, TB_NAVY)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 2.7, 0.46, 9.4, 0.28, [("From Telemetry to Work Order: Sense, Prepare, Engineer, Detect, Predict, Decide, Act, Learn | NASA C-MAPSS FD001 to FD004 | Benchmark Replay, Simulated Actions", 6.8, False, TB_BLUE)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
rect(s, 2.7, 0.76, 9.4, 0.03, TB_GOLD)

# ---------------- pipeline architecture band
nodes = [("Sense", "Telemetry Ingest", "26 columns per cycle, batch now, stream-ready", "FD001 to FD004, 3 settings, 21 sensors", TB_NAVY, "data/CMAPSSData"),
         ("Prepare", "Validate, Normalise", "NaN, duplicate, gap, label checks", f"KMeans k={dcfg['n_regimes']} z-score, median w={dcfg['smooth_window']}", TB_BLUE, "validated frames"),
         ("Engineer", "Causal Features", "windows 30 and 10, 6 statistics", f"{min(nfeat)} to {max(nfeat)} columns, RUL cap {cap_}", TB_SKY, "features.joblib"),
         ("Detect", "Anomaly Scoring", f"autoencoder, healthy cycles {an_['healthy_cycle_range'][0]}-{an_['healthy_cycle_range'][1]}", f"Isolation Forest, k={an_['persist_k']} flags", TB_CYAN, "anomaly_ifl.joblib"),
         ("Predict", "RUL and Risk Bound", f"XGBoost {cfg['rul']['xgb']['n_estimators']} trees, grouped CV", "offset deployed, conformal 90% offline", DCY, "rul_bundle.joblib"),
         ("Decide", "Rule Triage", "CRITICAL, HIGH, MEDIUM, LOW, REVIEW", f"RUL cut {tg_['critical_rul']}/{tg_['high_rul']}/{tg_['medium_rul']}; invalid: never LOW", TB_GOLD, "configs/config.yaml"),
         ("Act", "Work Orders", "simulated, OPEN_SIMULATED, restart-safe", "deterministic ID, TreeSHAP top-3 sensors", TB_AMB, "work_orders_v2/*.json"),
         ("Learn", "Gated Retraining", "MLflow champion and challenger", f"PSI>{rc_['psi_alarm']}, RMSE>{rc_['rmse_factor']}x, {rc_['days']} d; rollback", MIDB, "mlflow.db")]
NX = lambda i: 0.25 + i * 1.62
for lab, a0, a1, col in [("Data Engineering", 0, 2, TB_NAVY), ("Machine Learning", 3, 4, TB_CYAN), ("Agent (Non-LLM)", 5, 6, TB_AMB), ("MLOps", 7, 7, MIDB)]:
    put(rect(s, NX(a0), 0.84, NX(a1) + 1.5 - NX(a0), 0.15, tint(col, 0.8), col), [(lab.upper(), 5.2, True, col)], align=PP_ALIGN.CENTER, m=0.01)
for i, (verb, ttl, l1, l2, col, artf) in enumerate(nodes):
    tc_ = TB_NAVY if col == TB_GOLD else "FFFFFF"
    dc_ = TB_AMB if col == TB_GOLD else col
    rect(s, NX(i), 1.01, 1.5, 0.3, col, None, MSO_SHAPE.PENTAGON)
    put(rect(s, NX(i) + 0.04, 1.04, 0.24, 0.24, "FFFFFF", None, MSO_SHAPE.OVAL), [(str(i + 1), 8, True, dc_)], align=PP_ALIGN.CENTER, m=0.0)
    text(s, NX(i) + 0.32, 1.01, 1.0, 0.3, [(verb, 8, True, tc_)], anchor=MSO_ANCHOR.MIDDLE)
    put(rect(s, NX(i), 1.31, 1.5, 0.55, tint(col, 0.9), col), [(ttl, 6.6, True, dc_), (l1, 5.3, False, INK), (l2, 5.3, False, INK)], align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE, m=0.05)
    put(rect(s, NX(i), 1.87, 1.5, 0.13, "E3E9F0", None, MSO_SHAPE.ROUNDED_RECTANGLE), [(artf, 4.6, False, "334155")], align=PP_ALIGN.CENTER, m=0.01)
    if i < 7:
        rect(s, NX(i) + 1.5, 1.1, 0.12, 0.14, "334155", None, MSO_SHAPE.RIGHT_ARROW)
xa, xb = NX(7) + 0.75, NX(4) + 0.75
line(s, xa, 2.0, xa, 2.06, "334155", False, 1.25)
line(s, xa, 2.06, xb, 2.06, "334155", False, 1.25)
line(s, xb, 2.06, xb, 2.0, "334155", True, 1.25)
text(s, NX(5) + 0.1, 2.0, 3.0, 0.1, [("Feedback: new labels, drift gate, retrain, promote on RMSE gain", 4.6, True, "334155")], align=PP_ALIGN.CENTER)

# ---------------- branches (under hub and cards)
for k in range(3):
    yc = Y0c + CHc / 2 + k * STEPc
    line(s, HX, HY, LXc + CWc, yc, "7FA6D6", False, 1.75)
    line(s, HX, HY, RXc, yc, "7FA6D6", False, 1.75)

# ---------------- KPI grid
kp = [("Engines Escalated", f"{N_ORD} / {NTOT}", N_ORD / NTOT, TB_BLUE, "gear", None, None),
      ("RUL Error (RMSE)", rng(rm), nr_, TB_CYAN, "target", None, None),
      ("Anomaly AUROC", NA(af, ".3f"), min(af) if af else 0, TB_SKY, "pulse", None, None),
      ("Bound Coverage 90%", rng(cv, ".2f"), min(cv), TB_NAVY, "shield", None, None),
      ("Late Predictions", f"{lr:.0%} to {lc:.0%}", lr, RED, "clock", lc, GREEN),
      ("Early-Warning Recall", NA(tp), min(tp) if tp else 0, TB_GOLD, "bell", None, None)]
for i, (lab, val, fr, col, kind, fr2, col2) in enumerate(kp):
    x, y = 4.7 + (i % 3) * 1.33, 2.12 + (i // 3) * 0.7
    rect(s, x, y, 1.27, 0.66, TB_BG, TB_LINE)
    rect(s, x, y, 0.04, 0.66, col)
    pic(s, ring_icon(f"v13k{i}", fr, col, kind, fr2, col2), x + 0.06, y + 0.13, 0.4, 0.4)
    text(s, x + 0.48, y + 0.02, 0.78, 0.26, [(lab.upper(), 4.8, True, TB_BLUE)])
    text(s, x + 0.48, y + 0.28, 0.78, 0.36, [(val, 7.4, True, TB_NAVY)], anchor=MSO_ANCHOR.MIDDLE)
text(s, 4.7, 3.55, 3.93, 0.2, [(f"Evidence: Official Test, Last Observed Cycle ({NTOT} Engines, 4 Subsets)", 5.6, True, TB_BLUE)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# ---------------- hub
rect(s, HX - HR - 0.09, HY - HR - 0.09, 2 * HR + 0.18, 2 * HR + 0.18, None, TB_GOLD, MSO_SHAPE.OVAL)
put(rect(s, HX - HR, HY - HR, 2 * HR, 2 * HR, TB_NAVY, None, MSO_SHAPE.OVAL),
    [("SENTINEL", 10.5, True, "FFFFFF"), ("Closed-Loop", 6.2, False, "C9D6E6"), ("Predictive Maintenance", 6.2, False, "C9D6E6"), ("Telemetry to Work Order", 5.4, False, "F2A900")],
    align=PP_ALIGN.CENTER, m=0.02)

# ---------------- centre charts
lpic("v13_cov", 4.7, 5.12, 1.93, 1.78, d_cov, "90% Bound Coverage by Method", BASE + "coverage.html")
lpic("v13_cost", 6.7, 5.12, 1.93, 1.78, d_cost, "Cost vs Failure-to-Waste r", BASE + "cost_replay.html")

# ---------------- retraining table rows
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
    rect(s, x, y, CWc, 0.26, col)
    put(rect(s, x + 0.04, y + 0.02, 0.22, 0.22, "FFFFFF", None, MSO_SHAPE.OVAL), [(str(num), 7, True, TB_AMB if col == TB_GOLD else col)], align=PP_ALIGN.CENTER, m=0.0)
    text(s, x + 0.3, y, CWc - 0.34, 0.26, [(title, 8, True, tcol)], anchor=MSO_ANCHOR.MIDDLE)
    put(rect(s, x, y + 0.26, tw, CHc - 0.26, tint(col, 0.94)), [("\u2022 " + b, 5.9, False, INK) for b in bl], align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, m=0.05)
    cx_, cy_, cw_, ch_ = x + tw + 0.03, y + 0.29, CWc - tw - 0.06, CHc - 0.32
    if kind == "life":
        cimg(s, "v13_life", cx_, cy_, cw_, ch_, d_life, "Engine Life, Cycles")
    elif kind == "sens":
        cimg(s, "v13_sens", cx_, cy_, cw_, ch_, d_sens, "Sensors Kept vs Dropped")
    elif kind == "nfeat":
        cimg(s, "v13_nf", cx_, cy_, cw_, ch_, d_nfeat, "Feature Columns")
    elif kind == "scatter":
        lpic("v13_sc", cx_, cy_, cw_, ch_, d_scatter, "Predicted vs True RUL", BASE + "pred_vs_true.html")
    elif kind == "funnel":
        text(s, cx_, cy_ - 0.02, cw_, 0.15, [("Triage Funnel (Engines)", 6, True, TB_NAVY)], align=PP_ALIGN.CENTER)
        fun = [("Replayed", NTOT, TB_NAVY), ("Orders", N_ORD, TB_BLUE), ("HIGH+", N_HIGH, TB_GOLD), ("CRITICAL", N_CRIT, RED)]
        tops = [cw_ - 0.04, cw_ - 0.3, cw_ - 0.56, cw_ - 0.82]
        bots = [cw_ - 0.3, cw_ - 0.56, cw_ - 0.82, cw_ - 1.08]
        fx = cx_ + cw_ / 2
        for i, (lab, n, c_) in enumerate(fun):
            yy = cy_ + 0.16 + i * 0.27
            fb = s.shapes.build_freeform(Xe(fx - tops[i] / 2), Ye(yy), scale=1.0)
            fb.add_line_segments([(Xe(fx + tops[i] / 2), Ye(yy)), (Xe(fx + bots[i] / 2), Ye(yy + 0.255)), (Xe(fx - bots[i] / 2), Ye(yy + 0.255))], close=True)
            sh = fb.convert_to_shape()
            sh.fill.solid()
            sh.fill.fore_color.rgb = rgb(c_)
            sh.line.fill.background()
            sh.shadow.inherit = False
            lnk(sh, BASE + "funnel.html")
            put(sh, [(f"{n} {lab}", 5.4, True, TB_NAVY if c_ == TB_GOLD else "FFFFFF")], align=PP_ALIGN.CENTER, m=0.0)
    elif kind == "table" and rt_rows:
        text(s, cx_, cy_ - 0.02, cw_, 0.3, [("Retraining Demo (FD001, Synthetic Drift)", 5.6, True, TB_NAVY)], align=PP_ALIGN.CENTER)
        table(s, cx_, cy_ + 0.28, cw_, 0.85, rt_rows, [1.5, 0.7, 0.7, 0.8], 4.8)


cards = [
    (LXc, Y0c, TB_NAVY, "FFFFFF", 1, "Problem and Objective", [
        "Problem: failures found after breakdown or at fixed intervals cause downtime and avoidable cost.",
        "Opportunity: sensor streams show degradation well before failure.",
        "Objective: telemetry to work order with no manual step; humans review exceptions only.",
        "Constraints: lightweight, open-source, no LLM, rare failures.",
        f"Coverage: FD001 to FD004, {NTOT} test engines, 1 or 6 conditions, 1 or 2 fault modes."], "life", 2.65),
    (LXc, Y0c + STEPc, TB_BLUE, "FFFFFF", 2, "Data Engineering", [
        "Ingest: 26 columns per cycle from batch files; stream-ready interface.",
        "Validate: NaN, duplicates, gaps; test RUL labels aligned to engines.",
        f"Regimes: KMeans k={dcfg['n_regimes']} on settings; per-regime z-score from training data only.",
        f"Clean: causal median w={dcfg['smooth_window']}; constant and weak-trend sensors dropped.",
        f"Output: {min(nsen)} to {max(nsen)} informative sensors; raw inputs preserved."], "sens", 2.65),
    (LXc, Y0c + 2 * STEPc, TB_SKY, "FFFFFF", 3, "Causal Feature Engineering", [
        "Trailing windows of 30 and 10 cycles; never cross engines, no future data.",
        f"Six statistics per sensor give {min(nfeat)} to {max(nfeat)} columns.",
        f"Target: RUL capped at {cap_}; weight 3 below RUL 30.",
        f"Engine-grouped {cfg['eval']['cv_folds']}-fold CV; test set read once.",
        "FD001 ablation: slope features carry the signal (CV RMSE 24.8 to 13.9)."], "nfeat", 2.65),
    (RXc, Y0c, TB_CYAN, "FFFFFF", 4, "Detection and Prediction Models", [
        f"Anomaly: autoencoder on healthy cycles + Isolation Forest; alert after k={an_['persist_k']} flags.",
        f"RUL: XGBoost {cfg['rul']['xgb']['n_estimators']} trees, depth {cfg['rul']['xgb']['max_depth']}; 7 families compared.",
        "Bound: offset deployed; conformal 90% offline (measured, not guaranteed).",
        "Attribution: exact TreeSHAP, top-3 sensors per order.",
        f"Result: RMSE {rng(rm)}; AUROC {NA(af, '.3f')}; late {lr:.0%} to {lc:.0%}."], "scatter", 2.65),
    (RXc, Y0c + STEPc, TB_GOLD, TB_NAVY, 5, "Decision Agent (Non-LLM)", [
        "Rules: CRITICAL, HIGH, MEDIUM, LOW, REVIEW; first match wins, recorded on order.",
        f"RUL cutoffs {tg_['critical_rul']}/{tg_['high_rul']}/{tg_['medium_rul']} cycles (provisional) or persistent anomaly.",
        "Safety: invalid input or short history gives REVIEW, never LOW.",
        "Orders: simulated, deterministic ID, once per severity, restart-safe.",
        f"Outcome: {N_ORD} of {NTOT} escalated; {N_CRIT} CRITICAL for human review."], "funnel", 2.65),
    (RXc, Y0c + 2 * STEPc, MIDB, "FFFFFF", 6, "MLOps and Learning", [
        "MLflow tracks parameters, metrics, models per run.",
        "Registry: champion and challenger; promote only on RMSE gain.",
        f"Drift gate: PSI>{rc_['psi_alarm']}, RMSE>{rc_['rmse_factor']}x, or {rc_['days']} days.",
        "Retrain on engine-held-out validation; rollback kept.",
        "Limit: FD001, synthetic drift; real labels arrive after failures."], "table", 2.4 if rt_rows else CWc),
]
for c_ in cards:
    card(*c_)
for k in range(3):
    yc = Y0c + CHc / 2 + k * STEPc
    for xx, cc in ((LXc + CWc, cards[k][2]), (RXc, cards[3 + k][2])):
        rect(s, xx - 0.05, yc - 0.05, 0.1, 0.1, cc, "FFFFFF", MSO_SHAPE.OVAL)

# ---------------- links footer
text(s, 0.25, 7.0, 3.6, 0.3, [("Charts marked Interactive open live, hoverable versions:", 6, True, TB_NAVY)], anchor=MSO_ANCHOR.MIDDLE)
lbtn(3.9, 7.02, 3.0, 0.26, "Live Dashboard (Streamlit, localhost:8501)", "http://localhost:8501")
lbtn(7.0, 7.02, 2.8, 0.26, "All Interactive Charts (Index)", BASE + "index.html", TB_BLUE)
lbtn(9.9, 7.02, 3.18, 0.26, "Source Code: github.com/abhishekk-km/FLEET-PM", REPO, TB_CYAN)

'''
out = s[:s.index(A)] + NEW + s[s.index(B):]
shutil.copy2(P, P.with_name(P.name + f".bak_{time.strftime('%Y%m%d_%H%M%S')}"))
P.write_text(out)
py_compile.compile(str(P), doraise=True)
print("patched + compiled scripts/15_deck_v4.py")