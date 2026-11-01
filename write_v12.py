import py_compile, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TS = time.strftime("%Y%m%d_%H%M%S")

INTERACTIVE = r'''import _bootstrap
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import _ext_common as C
from fleetpm.config import load_config, art, ROOT

cfg = load_config()
FDS = cfg["data"]["subsets"]
OUT = ROOT / "docs" / "interactive"
OUT.mkdir(parents=True, exist_ok=True)
conf = json.loads(art(cfg, "ext_conformal.json").read_text())
cost = json.loads(art(cfg, "ext_cost_policy.json").read_text())
D = {f: C.load_oof(cfg, f) for f in FDS}
COL = dict(zip(FDS, ["#1566C0", "#0E9AA7", "#F2A900", "#0B2F6B"]))
M = ["raw", "offset_q0.8", "conformal_a0.2", "conformal_a0.1"]
ML = {"raw": "Raw", "offset_q0.8": "Offset (0.8 q)", "conformal_a0.2": "Conformal 80%", "conformal_a0.1": "Conformal 90%"}
PL = {"run_to_failure": "Run to failure", "fixed_interval": "Fixed interval", "rul_raw": "RUL threshold, raw",
      "rul_offset": "RUL threshold, offset", "rul_conformal10": "RUL threshold, 90% bound"}
PC = {"run_to_failure": "#8A97A8", "fixed_interval": "#D99A00", "rul_raw": "#1566C0", "rul_offset": "#0E9AA7", "rul_conformal10": "#0B2F6B"}
RG = np.logspace(0, 3, 200)
NOTE = "Benchmark replay of simulated NASA C-MAPSS data. Interactive: hover, zoom, click legend entries."
PAGES = []


def save(fig, name, title, h=560):
    fig.update_layout(title=dict(text=title, x=0.01), template="plotly_white", font=dict(family="Arial", size=12), height=h,
                      margin=dict(l=60, r=20, t=70, b=70), annotations=list(fig.layout.annotations) + [dict(text=NOTE, x=0, y=-0.16, xref="paper", yref="paper", showarrow=False, font=dict(size=10, color="#5B6B80"))])
    fig.write_html(OUT / f"{name}.html", include_plotlyjs="cdn")
    PAGES.append((name, title))
    print("wrote", name)


# 1 predicted vs true
fig = go.Figure()
for f in FDS:
    d = D[f]
    e = d["pred_t"] - d["true_cap"]
    fig.add_trace(go.Scatter(x=d["true_cap"], y=d["pred_t"], mode="markers", name=f, marker=dict(color=COL[f], size=6, opacity=.75),
                             customdata=np.c_[np.arange(1, len(e) + 1), e, d["true_raw"]],
                             hovertemplate=f + " engine %{customdata[0]:.0f}<br>true RUL (capped) %{x:.0f}<br>predicted %{y:.1f}<br>error %{customdata[1]:+.1f}<br>true RUL (raw) %{customdata[2]:.0f}<extra></extra>"))
fig.add_trace(go.Scatter(x=[0, 125], y=[0, 125], mode="lines", line=dict(color="#8A97A8", dash="dash"), name="ideal", hoverinfo="skip"))
fig.update_xaxes(title="True RUL (capped at 125 cycles)")
fig.update_yaxes(title="Predicted RUL (cycles)")
save(fig, "pred_vs_true", "Predicted vs True RUL at the Last Observed Cycle, All Test Engines")

# 2 RMSE by band
fig = go.Figure()
labs = ["<=30", "31-60", "61-cap", ">cap (capped label)"]
for f in FDS:
    d = D[f]
    e, tr, cp = d["pred_t"] - d["true_cap"], d["true_raw"], d["cap"]
    ms = [tr <= 30, (tr > 30) & (tr <= 60), (tr > 60) & (tr <= cp), tr > cp]
    fig.add_trace(go.Bar(x=labs, y=[float(np.sqrt((e[m] ** 2).mean())) if m.any() else 0 for m in ms], name=f, marker_color=COL[f],
                         customdata=[int(m.sum()) for m in ms], hovertemplate="%{x}: RMSE %{y:.1f} cycles, n=%{customdata}<extra>" + f + "</extra>"))
fig.update_layout(barmode="group")
fig.update_xaxes(title="True RUL band (cycles)")
fig.update_yaxes(title="RMSE (cycles)")
save(fig, "rmse_bands", "RUL Error by True-RUL Band: Accuracy Peaks Near End of Life")

# 3 coverage
fig = go.Figure()
for f in FDS:
    t = conf[f]["test"]
    fig.add_trace(go.Bar(name=f, x=[ML[m] for m in M], y=[t[m]["cov_lt_cap"] for m in M], marker_color=COL[f],
                         customdata=np.c_[[t[m]["slack_mean"] for m in M], [t[m]["late_rate"] for m in M], [t[m]["rmse_cap"] for m in M]],
                         hovertemplate="%{x}<br>coverage %{y:.3f}<br>mean slack %{customdata[0]:.1f} cycles<br>late rate %{customdata[1]:.2f}<br>RMSE %{customdata[2]:.1f}<extra>" + f + "</extra>"))
for y in (0.8, 0.9):
    fig.add_hline(y=y, line_dash="dot", line_color="#8A97A8")
fig.update_layout(barmode="group")
fig.update_yaxes(title="Coverage on engines below the cap", range=[0, 1.05])
save(fig, "coverage", "Lower-Bound Coverage by Method (Measured on Official Test Engines, Not a Guarantee)")

# 4 cost replay
fig = go.Figure()
groups = FDS + ["Mean of 4"]
for g in groups:
    for k in PL:
        c = np.mean([cost[f]["curves"][k] for f in FDS], 0) if g == "Mean of 4" else np.array(cost[g]["curves"][k])
        fig.add_trace(go.Scatter(x=RG, y=c, mode="lines", name=PL[k], line=dict(color=PC[k], width=2.5), visible=(g == groups[0]),
                                 hovertemplate="r=%{x:.1f}<br>cost %{y:.1f} (C_waste units)<extra>" + PL[k] + "</extra>"))
n = len(PL)
fig.update_layout(updatemenus=[dict(buttons=[dict(label=g, method="update", args=[{"visible": [(i // n) == gi for i in range(len(groups) * n)]}]) for gi, g in enumerate(groups)],
                                    x=0.01, y=1.14, xanchor="left")])
fig.update_xaxes(title="r = C_fail / C_waste", type="log")
fig.update_yaxes(title="Mean cost per engine (C_waste units)", type="log")
save(fig, "cost_replay", "Cost-Aware Policy Replay: Predictive Triage vs Run-to-Failure and Fixed Interval (Simulated)")

# 5 anomaly
am = {f: json.loads(art(cfg, "models", f, "anomaly_metrics.json").read_text()) for f in FDS if art(cfg, "models", f, "anomaly_metrics.json").exists()}
FA = list(am)
fig = make_subplots(rows=1, cols=3, subplot_titles=("AUROC by scorer", "Persistent TPR and FPR", "Alert lead time (cycles)"))
for k, lab, c in [("auroc_HI", "Health index", "#8A97A8"), ("auroc_IF_error_1d", "IF on error", "#0E9AA7"), ("auroc_IF_latent_error", "IF on latent and error", "#0B2F6B")]:
    fig.add_trace(go.Bar(x=FA, y=[am[f][k] for f in FA], name=lab, marker_color=c), 1, 1)
fig.add_trace(go.Bar(x=FA, y=[am[f]["IFl_tpr_persist"] for f in FA], name="Persistent TPR", marker_color="#0E9AA7"), 1, 2)
fig.add_trace(go.Bar(x=FA, y=[am[f]["IFl_fpr_persist"] for f in FA], name="Persistent FPR", marker_color="#B3202A"), 1, 2)
fig.add_trace(go.Bar(x=FA, y=[am[f]["IFl_lead_median"] for f in FA], name="Median lead", marker_color="#0B2F6B"), 1, 3)
fig.add_trace(go.Scatter(x=FA, y=[am[f]["IFl_lead_p10"] for f in FA], mode="markers", name="10th percentile", marker=dict(color="#E36414", size=11)), 1, 3)
fig.update_yaxes(range=[0.9, 1.0], row=1, col=1)
save(fig, "anomaly", "Anomaly Detection: Isolation Forest on Latent State and Reconstruction Error")

# 6 funnel
RANK = {"REVIEW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
wo = art(cfg, "work_orders_v2", "x").parent
mx = {}
for p in wo.rglob("WO-*.json"):
    d = json.loads(p.read_text())
    s_ = d.get("severity")
    if s_ in RANK:
        k = (p.name.split("-")[1], d.get("engine_id"))
        mx[k] = max(mx.get(k, 0), RANK[s_])
ntot = sum(len(D[f]["true_cap"]) for f in FDS)
vals = [ntot, len(mx), sum(v >= 2 for v in mx.values()), sum(v == 3 for v in mx.values())]
fig = go.Figure(go.Funnel(y=["Test engines replayed", "Simulated work order issued", "HIGH or CRITICAL", "CRITICAL (human review)"], x=vals,
                          textinfo="value+percent initial", marker=dict(color=["#0B2F6B", "#1566C0", "#F2A900", "#B3202A"])))
save(fig, "funnel", "Triage Funnel: From Replayed Engines to Critical Escalations (Highest Severity per Engine)", 520)

# 7 per-engine explorers
for f in FDS:
    P = D[f]["P"]
    T = P["Fte"]
    z = np.clip(np.load(art(cfg, "models", f, C.PRED_FILE))["test_xgb"], 0, None)
    q = conf[f]["q_conf"]["0.1"]
    eng, cyc, tr = T.engine_id.to_numpy(), T.cycle.to_numpy(), T["rul"].to_numpy()
    ids = np.unique(eng)
    fig = go.Figure()
    for k, e in enumerate(ids):
        m = eng == e
        v = (k == 0)
        fig.add_trace(go.Scatter(x=cyc[m], y=np.minimum(tr[m], 125), name="True RUL (capped benchmark label)", line=dict(color="#111827", width=2), visible=v))
        fig.add_trace(go.Scatter(x=cyc[m], y=z[m], name="Predicted RUL", line=dict(color="#1566C0", width=2), visible=v))
        fig.add_trace(go.Scatter(x=cyc[m], y=C.apply_q(z[m], q), name="90% lower bound", line=dict(color="#E36414", width=2, dash="dot"), visible=v))
    nt = 3 * len(ids)
    fig.update_layout(updatemenus=[dict(buttons=[dict(label=f"Engine {int(e)}", method="update", args=[{"visible": [(i // 3) == k for i in range(nt)]}]) for k, e in enumerate(ids)],
                                        x=0.01, y=1.14, xanchor="left")])
    for y in (15, 30, 60):
        fig.add_hline(y=y, line_dash="dash", line_color="#C98A00", opacity=.5)
    fig.update_xaxes(title="Operating cycle")
    fig.update_yaxes(title="RUL (cycles)")
    save(fig, f"engine_explorer_{f}", f"Engine Replay Explorer, {f}: True vs Predicted RUL and 90% Lower Bound (Dashed: Triage Cutoffs)")

rows = "".join(f'<li><a href="{n}.html">{t}</a></li>' for n, t in PAGES)
(OUT / "index.html").write_text(f"<html><head><meta charset=utf-8><title>SENTINEL interactive charts</title></head><body style='font-family:Arial;margin:30px'><h2>SENTINEL: Interactive Charts</h2><p>Benchmark replay of NASA C-MAPSS data, simulated actions only.</p><ul>{rows}</ul></body></html>")
print("wrote index.html in", OUT)
'''

NEW = r'''# MINDMAP SLIDE
s = S2
TR.update(ID)
MIDB, DCY = "3B7DD8", "0B7F8A"
BASE = "https://abhishekk-km.github.io/FLEET-PM/interactive/"
REPO = "https://github.com/abhishekk-km/FLEET-PM"
an_, tg_, dcfg = cfg["anomaly"], cfg["triage"], cfg["data"]
nfeat = [len(D[f]["P"]["pre"].sensors) * (len(cfg["features"]["stats"]) + 2) for f in FDS]
rc_ = cfg["mlops"]["retrain"]
NA = lambda v, fm=".2f": rng(v, fm) if v else "n/a"
lead = [anom[f]["IFl_lead_median"] for f in FA]
sl = [conf[f]["test"]["conformal_a0.1"]["slack_mean"] for f in FDS]


def lnk(sh, url):
    sh.click_action.hyperlink.address = url


def lbtn(x, y, w, h, label, url, fill=TB_NAVY, col="FFFFFF", fs=6.4):
    b = rect(s, x, y, w, h, fill, None, MSO_SHAPE.ROUNDED_RECTANGLE)
    put(b, [(label, fs, True, col)], align=PP_ALIGN.CENTER, m=0.03)
    lnk(b, url)


def lpic(name, x, y, w, h, draw, ttl, url):
    pic(s, mini(name, w, h, draw, ttl), x, y, w, h)
    lnk(s.shapes[-1], url)
    rect(s, x, y, w, h, None, BORD)
    lbtn(x + w - 0.66, y + h - 0.17, 0.62, 0.14, "Interactive", url, TB_GOLD, TB_NAVY, 4.8)


text(s, 2.7, 0.08, 9.4, 0.42, [(TITLE, 16, True, TB_NAVY)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 2.7, 0.48, 9.4, 0.28, [("Pipeline Architecture Map: Sense, Prepare, Engineer, Detect, Predict, Decide, Act, Learn | NASA C-MAPSS FD001 to FD004 | Benchmark Replay, Simulated Actions", 7, False, TB_BLUE)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
rect(s, 2.7, 0.8, 9.4, 0.03, TB_GOLD)

# ---------------- pipeline architecture map
nodes = [("Sense", "Telemetry Ingest", ["26 columns per cycle", "batch now, stream-ready", "FD001 to FD004"], TB_NAVY, "data/CMAPSSData"),
         ("Prepare", "Validate, Normalise", ["NaN, duplicate, gap checks", f"KMeans k={dcfg['n_regimes']} regime z-score", f"causal median, w={dcfg['smooth_window']}"], TB_BLUE, "validated frames"),
         ("Engineer", "Causal Features", ["windows 30 and 10 cycles", "6 statistics per sensor", f"{min(nfeat)} to {max(nfeat)} columns"], TB_SKY, "features.joblib"),
         ("Detect", "Anomaly Scoring", ["autoencoder, cycles 30-70", "Isolation Forest on latent", f"alert after k={an_['persist_k']} flags"], TB_CYAN, "anomaly_ifl.joblib"),
         ("Predict", "RUL and Risk Bound", [f"XGBoost {cfg['rul']['xgb']['n_estimators']} trees, cap {cap_}", "offset bound deployed", "conformal 90% offline"], DCY, "rul_bundle.joblib"),
         ("Decide", "Rule Triage", ["CRITICAL to LOW, REVIEW", f"RUL cut {tg_['critical_rul']}/{tg_['high_rul']}/{tg_['medium_rul']} cycles", "invalid input: never LOW"], TB_GOLD, "configs/config.yaml"),
         ("Act", "Work Orders", ["simulated, OPEN_SIMULATED", "deterministic, restart-safe", "TreeSHAP top-3 sensors"], TB_AMB, "work_orders_v2/*.json"),
         ("Learn", "Gated Retraining", ["MLflow champion/challenger", f"PSI>{rc_['psi_alarm']}, RMSE>{rc_['rmse_factor']}x, {rc_['days']} d", "rollback kept"], MIDB, "mlflow.db")]
NX = lambda i: 0.25 + i * 1.62
for lab, a0, a1, col in [("Data Engineering", 0, 2, TB_NAVY), ("Machine Learning", 3, 4, TB_CYAN), ("Agent (Non-LLM)", 5, 6, TB_AMB), ("MLOps", 7, 7, MIDB)]:
    put(rect(s, NX(a0), 0.88, NX(a1) + 1.5 - NX(a0), 0.17, tint(col, 0.8), col), [(lab.upper(), 5.4, True, col)], align=PP_ALIGN.CENTER, m=0.01)
for i, (verb, ttl, lines_, col, artf) in enumerate(nodes):
    tc_ = TB_NAVY if col == TB_GOLD else "FFFFFF"
    dc_ = TB_AMB if col == TB_GOLD else col
    rect(s, NX(i), 1.08, 1.5, 0.32, col, None, MSO_SHAPE.PENTAGON)
    put(rect(s, NX(i) + 0.04, 1.12, 0.24, 0.24, "FFFFFF", None, MSO_SHAPE.OVAL), [(str(i + 1), 8, True, dc_)], align=PP_ALIGN.CENTER, m=0.0)
    text(s, NX(i) + 0.32, 1.08, 1.0, 0.32, [(verb, 8, True, tc_)], anchor=MSO_ANCHOR.MIDDLE)
    put(rect(s, NX(i), 1.4, 1.5, 0.8, tint(col, 0.9), col), [(ttl, 6.8, True, dc_)] + [(t, 5.5, False, INK) for t in lines_], align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE, m=0.05)
    put(rect(s, NX(i), 2.21, 1.5, 0.15, "E3E9F0", None, MSO_SHAPE.ROUNDED_RECTANGLE), [(artf, 4.8, False, "334155")], align=PP_ALIGN.CENTER, m=0.01)
    if i < 7:
        rect(s, NX(i) + 1.5, 1.72, 0.12, 0.16, "334155", None, MSO_SHAPE.RIGHT_ARROW)
xa, xb = NX(7) + 0.75, NX(4) + 0.75
line(s, xa, 2.37, xa, 2.45, "334155", False, 1.25)
line(s, xa, 2.45, xb, 2.45, "334155", False, 1.25)
line(s, xb, 2.45, xb, 2.37, "334155", True, 1.25)
put(rect(s, 0.25, 2.5, 1.45, 0.22, TB_NAVY, None, MSO_SHAPE.ROUNDED_RECTANGLE), [("MLOPS CONTROL PLANE", 5.8, True, "FFFFFF")], align=PP_ALIGN.CENTER, m=0.01)
chips = ["Feedback: new labels retrain, promote only if RMSE improves", "Config: single YAML, seed 42", "Validation: engine-grouped 5-fold CV", "Tracking: MLflow SQLite", "Monitor: read-only Streamlit", "Source: GitHub repository"]
CW_ = (12.83 - 1.5 - 0.05 * 6) / 6
for j, t_ in enumerate(chips):
    put(rect(s, 1.75 + j * (CW_ + 0.05), 2.5, CW_, 0.22, tint(TB_NAVY, 0.84), TB_NAVY, MSO_SHAPE.ROUNDED_RECTANGLE), [(t_, 5.2, True, TB_NAVY)], align=PP_ALIGN.CENTER, m=0.02)

# ---------------- story band
story = [("Problem Understanding", TB_NAVY, "Failures are found after breakdown or at fixed service intervals, causing unplanned downtime and avoidable cost, although sensor streams carry early degradation signatures."),
         ("Objective", TB_BLUE, "Close the loop without manual steps: ingest, detect, predict remaining useful life, triage by rule, raise a work order, retrain under a gated registry; humans review exceptions only."),
         ("Constraints and Dataset Coverage", TB_CYAN, f"Lightweight, open-source, no LLM; rare failures handled by healthy-only training and low-RUL weighting. All four subsets ({NTOT} test engines; 1 or 6 conditions; 1 or 2 fault modes) processed independently.")]
SW = (12.83 - 0.12 * 2) / 3
for i, (lab, col, body) in enumerate(story):
    x = 0.25 + i * (SW + 0.12)
    rect(s, x, 2.78, 0.05, 0.5, col)
    put(rect(s, x + 0.05, 2.78, SW - 0.05, 0.5, tint(col, 0.9), TB_LINE), [(lab.upper(), 5.8, True, col), (body, 6, False, INK)], align=PP_ALIGN.LEFT, m=0.07)

# ---------------- interactive charts
CWg = (12.83 - 0.08 * 4) / 5
ch5 = [("s3_sc", d_scatter, f"Predicted vs True RUL, {NTOT} Test Engines", BASE + "pred_vs_true.html"),
       ("s3_bd", d_band, "RMSE by True-RUL Band, Cycles", BASE + "rmse_bands.html"),
       ("s3_cv", d_cov, "90% Bound Coverage by Method", BASE + "coverage.html"),
       ("s3_cs", d_cost, "Cost vs Failure-to-Waste Ratio r", BASE + "cost_replay.html"),
       ("s3_an", d_auroc if FA else d_hist, "Anomaly AUROC by Scorer", BASE + "anomaly.html")]
for i, (nm, fn, ttl, url) in enumerate(ch5):
    lpic(nm, 0.25 + i * (CWg + 0.08), 3.34, CWg, 1.7, fn, ttl, url)

# ---------------- KPI rings
kp = [("Engines Escalated", f"{N_ORD} / {NTOT}", N_ORD / NTOT, TB_BLUE, "gear", None, None),
      ("RUL Error (RMSE)", rng(rm), nr_, TB_CYAN, "target", None, None),
      ("Anomaly AUROC", NA(af, ".3f"), min(af) if af else 0, TB_SKY, "pulse", None, None),
      ("Bound Coverage 90%", rng(cv, ".2f"), min(cv), TB_NAVY, "shield", None, None),
      ("Late Predictions", f"{lr:.0%} to {lc:.0%}", lr, RED, "clock", lc, GREEN),
      ("Early-Warning Recall", NA(tp), min(tp) if tp else 0, TB_GOLD, "bell", None, None)]
for i, (lab, val, fr, col, kind, fr2, col2) in enumerate(kp):
    x, y = 0.25 + (i % 3) * 1.32, 5.1 + (i // 3) * 0.99
    rect(s, x, y, 1.27, 0.94, TB_BG, TB_LINE)
    rect(s, x, y, 0.04, 0.94, col)
    pic(s, ring_icon(f"v12k{i}", fr, col, kind, fr2, col2), x + 0.07, y + 0.2, 0.52, 0.52)
    text(s, x + 0.62, y + 0.05, 0.64, 0.34, [(lab.upper(), 4.8, True, TB_BLUE)])
    text(s, x + 0.62, y + 0.4, 0.64, 0.5, [(val, 8, True, TB_NAVY)], anchor=MSO_ANCHOR.MIDDLE)

# ---------------- funnel
FXs, FWs = 4.27, 2.2
rect(s, FXs, 5.1, FWs, 1.95, "F7FAFE", TB_LINE)
text(s, FXs + 0.02, 5.11, FWs - 0.04, 0.24, [("Triage Funnel: Highest Severity per Engine", 6.2, True, TB_NAVY)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
fun = [("Replayed", NTOT, TB_NAVY), ("Work Order Issued", N_ORD, TB_BLUE), ("HIGH or CRITICAL", N_HIGH, TB_GOLD), ("CRITICAL, Human Review", N_CRIT, RED)]
tops, bots = [2.06, 1.7, 1.34, 0.98], [1.7, 1.34, 0.98, 0.62]
fcx = FXs + FWs / 2
for i, (lab, n, col) in enumerate(fun):
    y = 5.38 + i * 0.41
    fb = s.shapes.build_freeform(Xe(fcx - tops[i] / 2), Ye(y), scale=1.0)
    fb.add_line_segments([(Xe(fcx + tops[i] / 2), Ye(y)), (Xe(fcx + bots[i] / 2), Ye(y + 0.39)), (Xe(fcx - bots[i] / 2), Ye(y + 0.39))], close=True)
    sh = fb.convert_to_shape()
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(col)
    sh.line.fill.background()
    sh.shadow.inherit = False
    lnk(sh, BASE + "funnel.html")
    put(sh, [(f"{n} ({100 * n / NTOT:.0f}%)", 7.4, True, TB_NAVY if col == TB_GOLD else "FFFFFF"), (lab, 5.2, False, TB_NAVY if col == TB_GOLD else "E3ECF7")], align=PP_ALIGN.CENTER, m=0.0)

# ---------------- design decisions
rows = [["Design Decision", "Reasoning", "Evidence"],
        [f"Capped RUL Target ({cap_})", "Early-life degradation is not observable; the cap stabilises regression.", f"RMSE {rng(rm)}"],
        ["Gradient-Boosted Trees", "Tabular rolling features, exact TreeSHAP; no deep model was reliably better.", "7 families compared"],
        ["Healthy-Only Autoencoder", "Failures are rare: learn normal behaviour, score deviation.", f"AUROC {NA(af, '.3f')}"],
        [f"Persistence k = {an_['persist_k']}", "Consecutive flags suppress isolated sensor noise.", f"FPR {min(pf):.2%} to {max(pf):.2%}" if pf else "n/a"],
        ["Early-Biased Risk Bound", "Over-estimating RUL is the unsafe error under the asymmetric NASA score.", f"late {lr:.0%} to {lc:.0%}"],
        ["Engine-Grouped Validation", "Row-level splits leak engines across folds; grouping prevents it.", f"{cfg['eval']['cv_folds']}-fold GroupKFold"]]
table(s, 6.56, 5.1, 4.4, 1.95, rows, [1.25, 2.25, 0.95], 5.2)

# ---------------- key findings
fnd = [f"Accuracy: RMSE {rng(b30)} cycles at true RUL <= 30 versus {rng(bmid)} at RUL 60 to 125.",
       f"Safety: the 90% bound cuts late predictions {lr:.0%} to {lc:.0%} at {rng(sl, '.0f')} cycles slack.",
       f"Detection: AUROC {NA(af, '.3f')}; alerts lead failure by {rng(lead, '.0f')} cycles." if lead else "Detection: see anomaly chart.",
       f"Action: {N_ORD} of {NTOT} engines escalated; {N_CRIT} CRITICAL.",
       (f"Economics: break-even r {rng(be)} vs run-to-failure (simulated)." if be else "Economics: see cost chart.")]
tile(s, 11.04, 5.1, 2.04, 1.95, "Key Findings", fnd, TB_NAVY, fs=5.9)

# ---------------- links footer
text(s, 0.25, 7.1, 3.6, 0.3, [("Every chart above opens its live, hoverable version:", 6, True, TB_NAVY)], anchor=MSO_ANCHOR.MIDDLE)
lbtn(3.9, 7.12, 3.0, 0.26, "Live Dashboard (Streamlit, localhost:8501)", "http://localhost:8501")
lbtn(7.0, 7.12, 2.8, 0.26, "All Interactive Charts (Index)", BASE + "index.html", TB_BLUE)
lbtn(9.9, 7.12, 3.18, 0.26, "Source Code: github.com/abhishekk-km/FLEET-PM", REPO, TB_CYAN)
NAV_ON = False

'''

# ------------------------------------------------------------------ write interactive script
p16 = ROOT / "scripts" / "16_interactive.py"
if p16.exists():
    shutil.copy2(p16, p16.with_name(p16.name + f".bak_{TS}"))
p16.write_text(INTERACTIVE)
py_compile.compile(str(p16), doraise=True)
print("wrote scripts/16_interactive.py")

# ------------------------------------------------------------------ patch deck script
P = ROOT / "scripts" / "15_deck_v4.py"
s = P.read_text()
A = "# MINDMAP SLIDE"
B = "# ------------------------------------------------------------------ appendix frame with navigation\n"
if s.count(A) != 1 or s.count(B) != 1:
    raise SystemExit("slide-2 markers not found exactly once; apply the v11 mind-map patch first")
s = s[:s.index(A)] + NEW + s[s.index(B):]
pairs = [
    ('    bw = (12.0 - 0.05 * (n - 1)) / n\n    for i, (lab, sl_) in enumerate(APPS):', '    bw = (12.0 - 0.05 * (n - 1)) / n\n    for i, (lab, sl_) in enumerate(APPS if NAV_ON else []):'),
    ('[("Appendix:", 6.5, True, NAVY)]', '[("Appendix:" if NAV_ON else "", 6.5, True, NAVY)]'),
    ('prs.save(out)\n', 'prs.save(out)\n_q = Presentation(str(out))\n_l = _q.slides._sldIdLst\nfor _sid in list(_l)[2:]:\n    _q.part.drop_rel(_sid.rId)\n    _l.remove(_sid)\n_q.save(str(art(cfg, "figures", "fleetpm_submit_2slides.pptx")))\nprint("saved fleetpm_submit_2slides.pptx (cover + slide 2)")\n'),
]
for old, new in pairs:
    if new in s:
        continue
    if s.count(old) != 1:
        raise SystemExit(f"expected exactly 1 match, found {s.count(old)} for {old[:60]!r}")
    s = s.replace(old, new)
shutil.copy2(P, P.with_name(P.name + f".bak_{TS}"))
P.write_text(s)
py_compile.compile(str(P), doraise=True)
print("patched + compiled scripts/15_deck_v4.py")