import py_compile, shutil, time
from datetime import datetime, timezone
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


def write(rel, text):
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.read_text() != text:
        backup(p)
    p.write_text(text)
    if rel.endswith(".py"):
        py_compile.compile(str(p), doraise=True)
    print("wrote", rel)


patch("scripts/_ext_common.py", [(
    "    d[key] = obj\n",
    "    if isinstance(d.get(key), dict) and isinstance(obj, dict):\n        d[key].update(obj)\n    else:\n        d[key] = obj\n")])

patch("scripts/10_conformal.py", [
    ('cov_raw=float((L <= D["true_raw"]).mean()),',
     'cov_raw=float((L <= D["true_raw"]).mean()),\n'
     '                         cov_lt_cap=float(c[D["true_raw"] < D["cap"]].mean()), n_lt_cap=int((D["true_raw"] < D["cap"]).sum()),'),
    ("cov_raw {r['cov_raw']:.3f} \"",
     "cov_raw {r['cov_raw']:.3f} cov_lt_cap {r['cov_lt_cap']:.3f} (n {r['n_lt_cap']}) \""),
])

patch("scripts/12_explain.py", [
    ('ap.add_argument("--skip-faith", action="store_true")',
     'ap.add_argument("--skip-faith", action="store_true")\n'
     'ap.add_argument("--check", action="store_true", help="fit fold 1 at deployed settings (500 trees), compare with saved OOF RMSE, exit")'),
    ('    r["xgb"]["learning_rate"] = o["learning_rate"] * o["n_estimators"] / args.trees\n',
     '    r["xgb"]["learning_rate"] = o["learning_rate"] * o["n_estimators"] / args.trees\n'
     '    if args.check:\n'
     '        r["xgb"]["n_estimators"], r["xgb"]["learning_rate"] = o["n_estimators"], o["learning_rate"]\n'),
    ('    full, imp_all = [], []\n',
     '    full, imp_all, ref_rm = [], [], []\n'
     '    zref = np.clip(np.load(art(cfg, "models", fd, "rul_boost_preds_gru.npz"))["oof_xgb"], 0, None)\n'),
    ('        full.append(r0)\n',
     '        full.append(r0)\n'
     '        ref_rm.append(cut_eval(y[va], zref[va], eng[va], ev["min_cut_len"], ev["n_cut_reps"], seed)["rmse"])\n'
     '        if args.check:\n'
     '            print(f"[{fd}] check fold 1: refit RMSE {r0:.4f} vs saved deployed OOF RMSE {ref_rm[-1]:.4f} (trees {r[\'xgb\'][\'n_estimators\']})", flush=True)\n'
     '            raise SystemExit(0)\n'),
    ("top3 {rank[:3]} ({time.time() - t0:.0f}s)",
     "top3 {rank[:3]} deployed-OOF ref {ref_rm[-1]:.2f} ({time.time() - t0:.0f}s)"),
    ("    null = []\n    for _ in range(30):\n        Xn = np.column_stack([rng.permutation(Xs[:, j]) for j in range(Xs.shape[1])])\n",
     "    null = []\n    cov = np.cov(Xs, rowvar=False)\n    for _ in range(30):\n        Xn = rng.multivariate_normal(np.zeros(Xs.shape[1]), cov, size=len(Xs))\n"),
    ("    out.update(ari_k2_mean=float(np.mean(ari)), ari_k2_ci=C.pct(ari), cluster_sizes_k2=np.bincount(ref).tolist(),",
     '    life = Z.groupby("engine_id", sort=True).cycle.max().to_numpy(float)\n'
     "    out.update(life_spearman_k2=float(stats.spearmanr(ref, life)[0]), ari_k2_mean=float(np.mean(ari)), ari_k2_ci=C.pct(ari), cluster_sizes_k2=np.bincount(ref).tolist(),"),
    ("column-shuffle null silhouette mean {out['null_sil_k2_mean']:.3f} p95 {out['null_sil_k2_p95']:.3f}\")",
     "Gaussian-null silhouette mean {out['null_sil_k2_mean']:.3f} p95 {out['null_sil_k2_p95']:.3f} \"\n"
     "          f\"spearman(cluster,life) {out['life_spearman_k2']:+.2f}\")"),
])

write(".streamlit/config.toml", '''[theme]
base = "light"
primaryColor = "#0B5CAD"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F3F6FA"
textColor = "#14213D"
font = "sans serif"

[browser]
gatherUsageStats = false
''')

APP = r'''import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fleetpm.config import load_config  # noqa: E402
from fleetpm.models.metrics import summarize  # noqa: E402

st.set_page_config(page_title="FleetPM | Predictive Maintenance Monitor", layout="wide")
cfg = load_config()
ART = ROOT / cfg["paths"]["artifacts"]
RAW = ROOT / cfg["paths"]["raw"]
PRED = "rul_boost_preds_gru.npz"
TR = cfg["triage"]
WIN = cfg["features"]["window"]
BANDS = [30, 60]
NAVY, BLUE, TEAL, AMBER, ORANGE, RED, GREEN, GREY, INK = (
    "#0B2A4A", "#0B5CAD", "#14808C", "#D99A00", "#E36414", "#B3202A", "#2E8B57", "#8A97A8", "#111827")
SEV_COL = {"CRITICAL": RED, "HIGH": ORANGE, "MEDIUM": AMBER, "LOW": GREEN, "REVIEW": GREY}
SEV_ORDER = list(SEV_COL)
M_LABEL = {"raw": "Raw prediction", "offset_q0.8": "Offset (0.8 quantile)", "conformal_a0.2": "Conformal 80%",
           "conformal_a0.1": "Conformal 90%"}
M_COL = {"Raw prediction": GREY, "Offset (0.8 quantile)": TEAL, "Conformal 80%": BLUE, "Conformal 90%": NAVY}
P_LABEL = {"run_to_failure": "Run to failure", "fixed_interval": "Fixed interval (best)",
           "rul_raw": "RUL threshold, raw", "rul_offset": "RUL threshold, offset",
           "rul_conformal10": "RUL threshold, 90% bound"}
P_COL = {"Run to failure": GREY, "Fixed interval (best)": AMBER, "RUL threshold, raw": BLUE,
         "RUL threshold, offset": TEAL, "RUL threshold, 90% bound": NAVY}
NAMES = ["T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30", "phi", "NRf", "NRc", "BPR",
         "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"]
R_FINE = np.logspace(0, 3, 200)

st.markdown("""<style>
.block-container{padding-top:1rem;max-width:1450px}
.topbar{background:#0B2A4A;color:#fff;padding:16px 24px;border-radius:4px;margin-bottom:14px;
display:flex;justify-content:space-between;align-items:center}
.topbar .t{font-size:1.45rem;font-weight:700;letter-spacing:.2px}
.topbar .s{font-size:.85rem;color:#C9D6E6;margin-top:2px}
.badge{border:1px solid #7FA6D6;color:#DCE8F7;padding:4px 10px;border-radius:3px;font-size:.75rem}
.kpi{background:#F3F6FA;border-left:4px solid #0B5CAD;padding:10px 14px;border-radius:3px;height:100%}
.kpi.green{border-left-color:#2E8B57}.kpi.amber{border-left-color:#D99A00}
.kpi.red{border-left-color:#B3202A}.kpi.grey{border-left-color:#8A97A8}
.kl{font-size:.72rem;color:#51627A;text-transform:uppercase;letter-spacing:.6px}
.kv{font-size:1.7rem;font-weight:700;color:#0B2A4A;line-height:1.25}
.ks{font-size:.74rem;color:#64748B}
.sec{font-size:1.05rem;font-weight:700;color:#0B2A4A;border-bottom:2px solid #D6DEE9;padding-bottom:3px;margin:18px 0 6px}
.secn{font-size:.82rem;color:#51627A;margin-bottom:8px}
.note{background:#FFF8E5;border-left:4px solid #D99A00;padding:8px 12px;font-size:.82rem;color:#4A3B00;border-radius:3px;margin:8px 0}
</style>""", unsafe_allow_html=True)


def kpi(col, label, value, sub="", tone=""):
    col.markdown(f'<div class="kpi {tone}"><div class="kl">{label}</div><div class="kv">{value}</div>'
                 f'<div class="ks">{sub}</div></div>', unsafe_allow_html=True)


def section(title, note=""):
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="secn">{note}</div>' if note else ""),
                unsafe_allow_html=True)


def warn(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def show(chart, h=320):
    st.altair_chart(chart.properties(height=h).configure_axis(
        gridColor="#E6EBF2", domainColor="#9AA7B8", labelColor="#334155", titleColor="#334155"
    ).configure_view(strokeWidth=0).configure_legend(orient="bottom", labelColor="#334155", titleColor="#334155"))


@st.cache_data(show_spinner=False)
def _jread(path, mtime):
    return json.loads(Path(path).read_text())


def jload(rel):
    p = ART / rel
    return _jread(str(p), p.stat().st_mtime) if p.exists() else None


@st.cache_data(show_spinner=False)
def replay(fd, mtime):
    z = np.load(ART / "models" / fd / PRED)
    pred = np.clip(np.asarray(z["test_xgb"], float), 0, None)
    df = pd.read_csv(RAW / f"test_{fd}.txt", sep=r"\s+", header=None, usecols=[0, 1], names=["engine_id", "cycle"])
    df = df.sort_values(["engine_id", "cycle"], kind="stable").reset_index(drop=True)
    rul = np.loadtxt(RAW / f"RUL_{fd}.txt", dtype=float).ravel()
    ids = np.sort(df.engine_id.unique())
    if len(pred) != len(df) or len(rul) != len(ids) or not (ids == np.arange(1, len(ids) + 1)).all():
        raise ValueError(f"{fd}: predictions, test rows and official RUL vector are not aligned")
    tmax = df.groupby("engine_id").cycle.transform("max").to_numpy()
    df["pred"] = pred
    df["true_raw"] = rul[df.engine_id.to_numpy() - 1] + (tmax - df.cycle.to_numpy())
    cap = float(cfg["data"]["rul_cap"][fd])
    df["true_cap"] = np.minimum(df.true_raw, cap)
    return df, cap


def get(fd):
    try:
        return replay(fd, (ART / "models" / fd / PRED).stat().st_mtime)
    except (ValueError, OSError) as e:
        st.error(str(e))
        return None, None


def bound(pred, q):
    return np.clip(pred - np.asarray(q)[np.digitize(pred, BANDS)], 0, None)


def sev_of(r):
    return np.select([r <= TR["critical_rul"], r <= TR["high_rul"], r <= TR["medium_rul"]],
                     ["CRITICAL", "HIGH", "MEDIUM"], "LOW")


def q90(fd):
    c = jload("ext_conformal.json")
    try:
        return c[fd]["q_conf"]["0.1"]
    except (TypeError, KeyError):
        return None


def last_frame(df, q):
    L = df.groupby("engine_id", sort=True).tail(1).copy()
    L["err"] = L.pred - L.true_cap
    L["direction"] = np.where(L.err > 0, "Late (over-estimate)", "Early (under-estimate)")
    L["bound"] = bound(L.pred.to_numpy(), q) if q else L.pred
    L["severity"] = np.where(L.cycle < WIN, "REVIEW", sev_of(L.bound.to_numpy()))
    return L


def fleet_stats(fd):
    df, cap = get(fd)
    if df is None:
        return None
    L = df.groupby("engine_id", sort=True).tail(1)
    sc = summarize(L.true_cap.to_numpy(), L.pred.to_numpy())
    sr = summarize(L.true_raw.to_numpy(), L.pred.to_numpy())
    return dict(subset=fd, engines=len(L), rmse_cap=sc["rmse"], mae_cap=sc["mae"], nasa_cap=sc["nasa"],
                rmse_raw=sr["rmse"], late_rate=float((L.pred > L.true_cap).mean()),
                bias=float((L.pred - L.true_cap).mean()))


have = [f for f in cfg["data"]["subsets"] if (ART / "models" / f / PRED).exists() and (RAW / f"test_{f}.txt").exists()]
st.markdown('<div class="topbar"><div><div class="t">FleetPM Predictive Maintenance Monitor</div>'
            '<div class="s">NASA C-MAPSS turbofan benchmark replay | Prepared for Tata Elxsi technical review</div></div>'
            '<div class="badge">SIMULATION ONLY | NO PHYSICAL ACTIONS</div></div>', unsafe_allow_html=True)
if not have:
    st.error("No prediction artifacts or raw test files found. Run scripts/01_build_features.py and scripts/03b_rul_boost.py.")
    st.stop()
fd = st.sidebar.selectbox("Engine subset", have)
st.sidebar.markdown("**Evidence coverage by subset**")
for fname in ("ext_conformal.json", "ext_cost_policy.json", "ext_explain.json"):
    d = jload(fname)
    st.sidebar.caption(f"{fname}: " + (", ".join(sorted(d)) if d else "not generated"))
st.sidebar.caption("Replay, not live telemetry. Thresholds are provisional. Not production-ready.")
df, cap = get(fd)
if df is None:
    st.stop()
q = q90(fd)
LF = last_frame(df, q)
tabs = st.tabs(["Overview", "Fleet RUL", "Engine view", "Risk bounds", "Cost policy", "Explainability", "Anomaly",
                "Work orders", "System health"])

with tabs[0]:
    s = fleet_stats(fd)
    c = st.columns(5)
    kpi(c[0], "Engines (test, last cycle)", s["engines"], f"subset {fd}")
    kpi(c[1], f"RMSE vs capped labels", f"{s['rmse_cap']:.2f}", f"cap {cap:.0f} cycles")
    kpi(c[2], "NASA score (capped)", f"{s['nasa_cap']:.0f}", "lower is better", "amber")
    kpi(c[3], "Late-prediction rate", f"{s['late_rate']:.0%}", "pred above true: the unsafe side", "red")
    kpi(c[4], "Engines at HIGH or CRITICAL", int(LF.severity.isin(["HIGH", "CRITICAL"]).sum()),
        "RUL-only, 90% bound" if q else "RUL-only, raw prediction", "green")
    section("All subsets, official test, last observed cycle per engine",
            "Capped labels are the primary yardstick. Raw-label RMSE is inflated where true RUL exceeds the training cap.")
    S = pd.DataFrame([x for x in (fleet_stats(f) for f in have) if x])
    st.dataframe(S.rename(columns={"rmse_cap": "RMSE (capped)", "mae_cap": "MAE (capped)", "nasa_cap": "NASA (capped)",
                                   "rmse_raw": "RMSE (raw)", "late_rate": "late rate", "bias": "bias (cycles)"}).round(3),
                 hide_index=True)
    m = S.melt("subset", ["rmse_cap", "rmse_raw"], var_name="label", value_name="RMSE")
    m["label"] = m.label.map({"rmse_cap": "Capped labels", "rmse_raw": "Raw labels"})
    show(alt.Chart(m).mark_bar().encode(
        x=alt.X("subset:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="label:N",
        y=alt.Y("RMSE:Q", title="RMSE (cycles)"),
        color=alt.Color("label:N", scale=alt.Scale(domain=["Capped labels", "Raw labels"], range=[BLUE, AMBER]), title=None)), 260)
    warn("Scope: historical replay of a public simulation benchmark. No real telemetry, no verified fault-onset labels, "
         "no maintenance-system integration. All work orders and actions are simulated.")

with tabs[1]:
    section(f"{fd}: predicted vs true RUL at the last observed cycle")
    a, b = st.columns(2)
    sc = alt.Chart(LF).mark_circle(size=60, opacity=0.75).encode(
        x=alt.X("true_cap:Q", title="True RUL (capped)"), y=alt.Y("pred:Q", title="Predicted RUL"),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Late (over-estimate)", "Early (under-estimate)"],
                                                       range=[RED, BLUE]), title=None),
        tooltip=["engine_id", "cycle", alt.Tooltip("pred:Q", format=".1f"), "true_raw"])
    diag = alt.Chart(pd.DataFrame({"x": [0, cap], "y": [0, cap]})).mark_line(color=GREY, strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
    with a:
        show(sc + diag, 340)
    with b:
        show(alt.Chart(LF).mark_bar().encode(
            x=alt.X("err:Q", bin=alt.Bin(step=5), title="Error = predicted minus true (cycles); positive is late"),
            y=alt.Y("count():Q", title="Engines"),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["Late (over-estimate)", "Early (under-estimate)"],
                                                           range=[RED, BLUE]), legend=None)), 340)
    section("Error by true-RUL band", "Accuracy is best near end of life and degrades mid-life.")
    bands = pd.cut(LF.true_raw, [-1, 30, 60, cap, 1e9], labels=["<=30", "31-60", f"61-{int(cap)}", f">{int(cap)} (capped)"])
    B = LF.assign(band=bands).groupby("band", observed=True).agg(
        n=("err", "size"), rmse=("err", lambda e: float(np.sqrt((e ** 2).mean()))), bias=("err", "mean")).reset_index()
    B["label"] = "n=" + B.n.astype(str)
    B["band"] = B.band.astype(str)
    bars = alt.Chart(B).mark_bar(color=BLUE).encode(x=alt.X("band:N", sort=list(B.band), title="True RUL band (cycles)"),
                                                    y=alt.Y("rmse:Q", title="RMSE (cycles)"))
    c1, c2 = st.columns([3, 2])
    with c1:
        show(bars + bars.mark_text(dy=-7, color=INK).encode(text="label:N"), 280)
    with c2:
        st.dataframe(B[["band", "n", "rmse", "bias"]].round(2), hide_index=True)
    section("Provisional severity of the fleet",
            "RUL-only, from the 90% lower bound when available. Excludes anomaly evidence and persistence, so it is not the deployed triage. REVIEW means fewer than 30 cycles of history.")
    SV = LF.severity.value_counts().reindex(SEV_ORDER).fillna(0).rename_axis("severity").reset_index(name="engines")
    show(alt.Chart(SV).mark_bar().encode(
        x=alt.X("severity:N", sort=SEV_ORDER, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("engines:Q"),
        color=alt.Color("severity:N", scale=alt.Scale(domain=SEV_ORDER, range=[SEV_COL[k] for k in SEV_ORDER]), legend=None)), 240)
    section("Largest over-estimates against capped labels (late is the dangerous direction)")
    st.dataframe(LF.sort_values("err", ascending=False).head(12)[["engine_id", "cycle", "pred", "true_cap", "true_raw", "err"]].round(1),
                 hide_index=True)

with tabs[2]:
    ids = LF.engine_id.tolist()
    dflt = ids.index(int(LF.loc[LF.pred.idxmin(), "engine_id"]))
    eid = st.selectbox("Engine (default: lowest predicted RUL)", ids, index=dflt, key=f"eng_{fd}")
    E = df[df.engine_id == eid]
    W = pd.DataFrame({"cycle": E.cycle.to_numpy(), "True RUL (capped, benchmark label)": E.true_cap.to_numpy(),
                      "Predicted RUL": E.pred.to_numpy()})
    if q:
        W["Lower bound (90%)"] = bound(E.pred.to_numpy(), q)
    dom = list(W.columns[1:])
    rng_c = [INK, BLUE, ORANGE][:len(dom)]
    Lg = W.melt("cycle", var_name="series", value_name="rul")
    lines = alt.Chart(Lg).mark_line(strokeWidth=2).encode(
        x=alt.X("cycle:Q", title="Operating cycle"), y=alt.Y("rul:Q", title="RUL (cycles)"),
        color=alt.Color("series:N", scale=alt.Scale(domain=dom, range=rng_c), title=None))
    warm = alt.Chart(pd.DataFrame({"a": [0], "b": [WIN]})).mark_rect(opacity=0.12, color=GREY).encode(x="a:Q", x2="b:Q")
    th = pd.DataFrame({"y": [TR["critical_rul"], TR["high_rul"], TR["medium_rul"]], "s": ["CRITICAL", "HIGH", "MEDIUM"]})
    rules = alt.Chart(th).mark_rule(strokeDash=[5, 4]).encode(
        y="y:Q", color=alt.Color("s:N", scale=alt.Scale(domain=["CRITICAL", "HIGH", "MEDIUM"], range=[RED, ORANGE, AMBER]), legend=None))
    show(warm + lines + rules, 380)
    st.caption(f"Grey band: first {WIN} cycles (feature warm-up, treated as REVIEW). Dashed lines: provisional triage thresholds. "
               "The true-RUL curve is the benchmark label and would not exist in deployment.")
    r = LF[LF.engine_id == eid].iloc[0]
    c = st.columns(4)
    kpi(c[0], "Cycles observed", int(r.cycle))
    kpi(c[1], "Predicted RUL", f"{r.pred:.1f}")
    kpi(c[2], "90% lower bound" if q else "Bound unavailable", f"{r.bound:.1f}", "run scripts/10_conformal.py" if not q else "", "amber")
    kpi(c[3], "Indicative severity", r.severity, "RUL-only, not the deployed triage", "red" if r.severity in ("CRITICAL", "HIGH") else "green")

with tabs[3]:
    conf = jload("ext_conformal.json")
    if not conf:
        st.info("ext_conformal.json not generated. Run scripts/10_conformal.py.")
    else:
        rows = []
        for f, rec in conf.items():
            for mname, r in rec["test"].items():
                lo, hi = r["ci"]["marg"]
                rows.append(dict(subset=f, label=M_LABEL.get(mname, mname), coverage=r["cov_cap"], lo=lo, hi=hi,
                                 slack=r["slack_mean"], late_rate=r["late_rate"], rmse=r["rmse_cap"], nasa=r["nasa_cap"],
                                 cov_lt_cap=r.get("cov_lt_cap")))
        D = pd.DataFrame(rows)
        d = D[D.subset == fd]
        order = list(d.label)
        section(f"{fd}: empirical coverage of RUL lower bounds on the official test set",
                "Share of engines whose true RUL is at or above the bound. Bars show 95% engine-bootstrap CIs. Dotted lines mark nominal 80% and 90%.")
        base = alt.Chart(d).encode(x=alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
                                   color=alt.Color("label:N", scale=alt.Scale(domain=list(M_COL), range=list(M_COL.values())), legend=None))
        ci = base.mark_rule(size=4).encode(y=alt.Y("lo:Q", scale=alt.Scale(domain=[0, 1]), title="Coverage"), y2="hi:Q")
        pt = base.mark_point(filled=True, size=130).encode(y="coverage:Q")
        nom = alt.Chart(pd.DataFrame({"y": [0.8, 0.9]})).mark_rule(strokeDash=[2, 3], color=GREY).encode(y="y:Q")
        a, b = st.columns(2)
        with a:
            show(ci + pt + nom, 320)
        with b:
            show(alt.Chart(d).mark_point(filled=True, size=140).encode(
                x=alt.X("slack:Q", title="Mean slack (prediction minus bound, cycles)"),
                y=alt.Y("coverage:Q", scale=alt.Scale(domain=[0, 1]), title="Coverage"),
                color=alt.Color("label:N", scale=alt.Scale(domain=list(M_COL), range=list(M_COL.values())), title=None),
                tooltip=["label", "coverage", "slack", "rmse", "nasa"]), 320)
        st.dataframe(d.drop(columns=["lo", "hi"]).round(3), hide_index=True)
        section("Cross-subset coverage")
        st.dataframe(D.pivot(index="label", columns="subset", values="coverage").round(3))
        nested = conf[fd].get("nested", {})
        if nested:
            section("Calibration check on training engines only (nested engine split)",
                    f"Selected variant: {conf[fd].get('variant')}. Nominal coverage is 1 minus alpha.")
            N = pd.DataFrame([dict(variant=k.split("|")[0], nominal=1 - float(k.split("|")[1]), held_out_coverage=v["cut_cov_mean"],
                                   sd=v["cut_cov_sd"]) for k, v in nested.items()])
            st.dataframe(N.round(3), hide_index=True)
        warn("Coverage is measured, not guaranteed. Calibration cuts are uniform in life while test cuts sit about 60% into life, and "
             "engines whose true RUL exceeds the cap are almost trivially covered. Compare cov_lt_cap with coverage to see the effect.")

with tabs[4]:
    cp = jload("ext_cost_policy.json")
    if not cp or fd not in cp:
        st.info(f"No cost-policy results for {fd}. Run scripts/11_cost_policy.py.")
    else:
        rec = cp[fd]
        section(f"{fd}: cost-aware policy replay on {rec['n_engines']} training trajectories",
                "Cost per engine in units of C_waste. r = C_fail / C_waste. No monetary values are assumed.")
        use_cv = st.toggle("Select policy parameters by 5-fold engine CV (less optimistic)", value=False)
        src = rec["cv_curves"] if use_cv else rec["curves"]
        rows = [dict(r=float(rr), cost=float(cc), policy=P_LABEL.get(k, k)) for k, cur in src.items() for rr, cc in zip(R_FINE, cur)]
        C = pd.DataFrame(rows)
        a, b = st.columns(2)
        lg = alt.Color("policy:N", scale=alt.Scale(domain=list(P_COL), range=list(P_COL.values())), title=None)
        with a:
            show(alt.Chart(C).mark_line(strokeWidth=2).encode(
                x=alt.X("r:Q", scale=alt.Scale(type="log"), title="r = C_fail / C_waste"),
                y=alt.Y("cost:Q", scale=alt.Scale(type="log"), title="Mean cost per engine"), color=lg), 340)
        C["ratio"] = C.cost / C.r
        with b:
            show(alt.Chart(C[C.policy != "Run to failure"]).mark_line(strokeWidth=2).encode(
                x=alt.X("r:Q", scale=alt.Scale(type="log"), title="r = C_fail / C_waste"),
                y=alt.Y("ratio:Q", title="Cost relative to run-to-failure"), color=lg), 340)
        ropts = [x["r"] for x in rec["at_r"]["rul_raw"]]
        rsel = st.select_slider("Failure-to-waste cost ratio r", options=ropts, value=100.0 if 100.0 in ropts else ropts[0])
        T = []
        for k, lst in rec["at_r"].items():
            x = [y for y in lst if y["r"] == rsel][0]
            T.append({"policy": P_LABEL.get(k, k), "parameter": x["param"], "cost": x["cost"], "failures avoided": x["failures_avoided"],
                      "mean wasted life (all)": x["mean_wasted_all"], "mean wasted life (maintained)": x["mean_wasted_when_maintained"]})
        st.dataframe(pd.DataFrame(T).round(3), hide_index=True)
        section("Break-even r (policy beats the comparator above this value)")
        be = []
        for k, v in rec["break_even"].items():
            def fmt(x):
                if x["point"] is None:
                    return "never"
                return f"{x['point']:.1f}" + (f" ({x['ci'][0]:.1f} to {x['ci'][1]:.1f})" if x.get("ci") else "")
            be.append({"policy": P_LABEL.get(k, k), "vs run-to-failure": fmt(v["vs_rtf"]), "vs best fixed interval": fmt(v["vs_fixed"])})
        st.dataframe(pd.DataFrame(be), hide_index=True)
        warn(rec["assumptions"] + ". Because the threshold is swept, raw, offset and bound policies are near-reparametrisations; "
             "compare them with the CV toggle before drawing conclusions.")

with tabs[5]:
    ex = jload("ext_explain.json")
    if not ex or fd not in ex:
        st.info(f"No explanation results for {fd}. Run scripts/12_explain.py.")
    else:
        rec = ex[fd]
        fa = rec.get("faithfulness")
        if fa:
            sh = pd.DataFrame({"sensor": list(fa["mean_abs_shap_by_sensor"]), "importance": list(fa["mean_abs_shap_by_sensor"].values())})
            sh["name"] = sh.sensor.map(lambda s: f"{NAMES[int(s.split('_')[1]) - 1]} ({s})")
            sh = sh.sort_values("importance", ascending=False)
            section(f"{fd}: mean absolute TreeSHAP contribution by sensor", "Descriptive of the fitted model, not of physical cause.")
            show(alt.Chart(sh).mark_bar(color=BLUE).encode(x=alt.X("importance:Q", title="Mean |SHAP| (cycles)"),
                                                           y=alt.Y("name:N", sort="-x", title=None)), 320)
            rows = []
            for k, v in fa["k"].items():
                for kind, key in (("Remove top-k minus remove random-k", "remove_diff"), ("Keep top-k minus keep random-k", "keep_diff")):
                    mean, lo, hi = v[key]
                    rows.append(dict(k=int(k), kind=kind, mean=mean, lo=lo, hi=hi))
            Fd = pd.DataFrame(rows)
            section("Retraining faithfulness test (RMSE difference; 95% t-interval over folds)",
                    "Remove: positive means top-attributed sensors matter more than random ones. Keep: negative means they are more informative than random ones.")
            cols = st.columns(2)
            for col, kind in zip(cols, Fd.kind.unique()):
                sub = Fd[Fd.kind == kind]
                base = alt.Chart(sub).encode(x=alt.X("k:O", title="k sensors"))
                with col:
                    st.caption(kind)
                    show(base.mark_rule(size=4, color=BLUE).encode(y=alt.Y("lo:Q", title="RMSE difference"), y2="hi:Q")
                         + base.mark_point(filled=True, size=120, color=NAVY).encode(y="mean:Q")
                         + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=GREY).encode(y="y:Q"), 260)
            rem = Fd[Fd.kind.str.startswith("Remove")]
            if ((rem.lo <= 0) & (rem.hi >= 0)).all():
                warn("All remove-top intervals contain zero: no retraining evidence that top-attributed sensors are uniquely necessary. "
                     "This is consistent with redundant sensors (HYPOTHESIS). Treat work-order sensor lists as model attributions.")
            else:
                sig = rem[rem.lo > 0]
                warn("Remove-top is worse than remove-random for k = " + ", ".join(str(int(x)) for x in sig.k) + ".")
        cl = rec.get("clustering")
        section("Unsupervised degradation-signature clustering (no fault labels)")
        rows = []
        for f, r in ex.items():
            c = r.get("clustering")
            if c:
                rows.append(dict(subset=f, engines=c["n_engines"], silhouette_k2=c["silhouette"]["2"], ARI_k2=c["ari_k2_mean"],
                                 ARI_ci_lo=c["ari_k2_ci"][0], null_p95=c["null_sil_k2_p95"], life_spearman=c.get("life_spearman_k2")))
        if rows:
            st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
        else:
            st.info("No clustering results stored. Run scripts/12_explain.py --skip-faith.")
        img = ART / "figures" / "ext_clusters.png"
        if img.exists():
            st.image(str(img))
        warn("FD001 and FD002 have one fault mode and act as negative controls. Any mapping of clusters to fan or compressor degradation "
             "is a HYPOTHESIS. readme.txt lists two fault modes for FD003 and FD004 while the paper text describes HPC degradation only.")

with tabs[6]:
    rows = []
    for f in cfg["data"]["subsets"]:
        m = jload(f"models/{f}/anomaly_metrics.json")
        if m:
            rows.append(dict(subset=f, auroc_HI=m.get("auroc_HI"), auroc_IFe=m.get("auroc_IF_error_1d"), auroc_IFl=m.get("auroc_IF_latent_error"),
                             persist_TPR=m.get("IFl_tpr_persist"), persist_FPR=m.get("IFl_fpr_persist"),
                             lead_median=m.get("IFl_lead_median"), lead_p10=m.get("IFl_lead_p10"),
                             engine_false_alarm=m.get("IFl_eng_fa_rate"), unassessable=m.get("eng_unassessable")))
    if not rows:
        st.info("No anomaly_metrics.json found.")
    else:
        A = pd.DataFrame(rows)
        section("Deployed Isolation-Forest detector on latent and error: held-out test engines",
                "Positives are rows with RUL <= 30 and negatives are rows at the RUL cap. Both are benchmark proxies: no failure-onset labels exist.")
        m = A.melt("subset", ["auroc_HI", "auroc_IFe", "auroc_IFl"], var_name="scorer", value_name="AUROC")
        m["scorer"] = m.scorer.map({"auroc_HI": "Health index", "auroc_IFe": "IF on error", "auroc_IFl": "IF on latent and error (deployed)"})
        show(alt.Chart(m).mark_bar().encode(x=alt.X("subset:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="scorer:N",
                                            y=alt.Y("AUROC:Q", scale=alt.Scale(domain=[0.8, 1.0])),
                                            color=alt.Color("scorer:N", scale=alt.Scale(range=[GREY, TEAL, NAVY]), title=None)), 300)
        st.dataframe(A.round(4), hide_index=True)
        warn("Rows before cycle 30 are false-alarm heavy and are routed to REVIEW. Persistent true-positive rate differs materially across subsets.")

with tabs[7]:
    @st.cache_data(ttl=60, show_spinner=False)
    def wo_files(root):
        return sorted(str(p) for p in Path(root).rglob("WO-*.json") if "mlartifacts" not in p.parts)
    files = wo_files(str(ART))
    rows = []
    for p in files:
        try:
            rows.append(json.loads(Path(p).read_text()))
        except json.JSONDecodeError:
            st.warning(f"Unreadable work order: {Path(p).name}")
    if not rows:
        st.info(f"No simulated work orders found under {ART.relative_to(ROOT)}. Locate them with:  find artifacts -iname 'WO-*' | head")
    else:
        W = pd.DataFrame(rows)
        section(f"{len(W)} simulated work orders on disk (all subsets)", "Status OPEN_SIMULATED. Issued once per engine per severity level.")
        if "severity" in W.columns:
            vc = W.severity.value_counts().reindex(SEV_ORDER).dropna().rename_axis("severity").reset_index(name="orders")
            show(alt.Chart(vc).mark_bar().encode(x=alt.X("severity:N", sort=SEV_ORDER, title=None, axis=alt.Axis(labelAngle=0)), y="orders:Q",
                                                 color=alt.Color("severity:N", scale=alt.Scale(domain=SEV_ORDER, range=[SEV_COL[k] for k in SEV_ORDER]), legend=None)), 240)
        keep = [c for c in ["work_order_id", "subset", "engine_id", "severity", "rule_fired", "rul_conservative", "health_index",
                            "top_degradation_sensors", "status", "model_version"] if c in W.columns]
        flt = st.selectbox("Severity filter", ["All"] + SEV_ORDER) if "severity" in W.columns else "All"
        V = W if flt == "All" else W[W.severity == flt]
        st.dataframe(V[keep].head(200), hide_index=True)

with tabs[8]:
    section("Artifact inventory and freshness")
    items = [ART / "models" / fd / PRED, ART / "models" / fd / "rul_bundle_v2_gru.joblib", ART / "models" / fd / "anomaly_ifl.joblib",
             ART / "models" / fd / "anomaly_metrics.json", ART / "ext_conformal.json", ART / "ext_cost_policy.json",
             ART / "ext_explain.json", ART / "mlflow.db"]
    inv = pd.DataFrame([dict(file=str(p.relative_to(ROOT)), exists=p.exists(),
                             modified=pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M") if p.exists() else None,
                             size_mb=round(p.stat().st_size / 1e6, 2) if p.exists() else None) for p in items])
    st.dataframe(inv, hide_index=True)
    pr, bd = ART / "models" / fd / PRED, ART / "models" / fd / "rul_bundle_v2_gru.joblib"
    if pr.exists() and bd.exists() and abs(pr.stat().st_mtime - bd.stat().st_mtime) > 3600:
        warn("Predictions and the model bundle differ in age by more than one hour; confirm they come from the same training run.")
    section("Consistency check: live recomputation vs stored results",
            "Recomputes RMSE and NASA from the raw files and the prediction artifact, then compares with ext_conformal.json (computed by a separate script).")
    conf = jload("ext_conformal.json")
    if conf and fd in conf and "raw" in conf[fd]["test"]:
        s = fleet_stats(fd)
        stored = conf[fd]["test"]["raw"]
        ok_r = abs(s["rmse_cap"] - stored["rmse_cap"]) < 1e-3
        ok_n = abs(s["nasa_cap"] - stored["nasa_cap"]) < 1e-2 * max(1.0, abs(stored["nasa_cap"])) * 0.01 + 1e-3
        st.dataframe(pd.DataFrame([dict(metric="RMSE (capped)", live=s["rmse_cap"], stored=stored["rmse_cap"], status="consistent" if ok_r else "MISMATCH"),
                                   dict(metric="NASA (capped)", live=s["nasa_cap"], stored=stored["nasa_cap"], status="consistent" if ok_n else "MISMATCH")]).round(4),
                     hide_index=True)
    else:
        st.info("No stored reference for this subset. Run scripts/10_conformal.py.")
    st.caption("Retraining history is not shown: the retrain demo is FD001 with synthetic drift and writes to MLflow only. "
               "Nothing on this page triggers training or inference.")
'''
write("dashboard/app.py", APP)

LOGP = ROOT / "docs" / "SYSTEM_LOG.md"
LOGP.parent.mkdir(parents=True, exist_ok=True)
cur = LOGP.read_text() if LOGP.exists() else ""
if "## 15. Extensions audit" not in cur:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with LOGP.open("a") as f:
        f.write(f"""
## 15. Extensions audit, Tasks 1-3 ({stamp})
Status words as in section 0. Outputs inspected by the author.
- No deterioration: raw and offset test RMSE and NASA and offset bins reproduce section 6.4 and 7 exactly (VERIFIED).
- Task 1, conformal one-sided lower bounds (VERIFIED): nested training-only calibration selected one random cut per engine on all subsets; held-out coverage 0.895/0.903/0.897/0.900 at alpha 0.1 and 0.792/0.802/0.799/0.801 at alpha 0.2. Official test, conformal 90%: 0.920/0.915/0.880/0.927, all engine-bootstrap CIs contain 0.90. Conformal 80% over-covers on FD002 (0.869, CI excludes 0.80). The 0.8-quantile offset covers 0.72/0.826/0.70/0.77 and under-covers on FD003 (CI [0.61, 0.78]). Cost: mean slack 6-22 cycles; on FD002/FD004 NASA of the 90% bound is about 3.4x raw. Coverage is empirical, not guaranteed; cap-saturated engines inflate marginal coverage (HYPOTHESIS, cov_lt_cap added to measure it).
- Task 2, cost-aware replay (VERIFIED run, in-sample parameters, simulated, no anomaly signal, relative cost units): at r=100 threshold policies cost 7.7-14.8 per engine vs 100 for run-to-failure and 61-89 for the best fixed interval. Break-even r vs run-to-failure 1.6-1.9 (raw), 3.0-6.5 (offset), 4.1-9.2 (90% bound). Raw, offset and bound policies are near-reparametrisations under a swept threshold, so Task 2 does not demonstrate a benefit of the bound. Raw policy avoids 1.000/0.981/0.980/0.980 of failures at r=100.
- Task 3 clustering (VERIFIED run): k=2 silhouette 0.285/0.200/0.434/0.397, ARI 0.85/0.62/0.98/0.99 for FD001-FD004. The negative control did not come out clean (FD001 is far above the weak column-shuffle null of 0.08). Two-fault-mode reading remains a HYPOTHESIS; Gaussian null and Spearman(cluster, life) added.
- Task 3 faithfulness, FD001 only (VERIFIED run): remove-top minus remove-random RMSE -0.25/-0.37/-0.03/-0.02 for k=1,2,3,5, all CIs contain 0; keep-top vs keep-random mixed sign, CIs contain 0. No evidence top-SHAP sensors are uniquely necessary; redundancy is a HYPOTHESIS. Work-order sensor lists are attributions only.
- UNVERIFIED: faithfulness full-model CV RMSE 14.51 vs 13.85 deployed OOF (300 vs 500 trees or design() mismatch); resolve with scripts/12_explain.py --fd FD001 --skip-cluster --check.
- Defect fixed: save_merge overwrote whole subset entries, erasing FD001 clustering from ext_explain.json.
""")
    print("appended section 15 to docs/SYSTEM_LOG.md")
print("done")