import _bootstrap
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
