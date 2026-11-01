import py_compile, re, shutil, time
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


OLD_LOG = '''def log(title, lines):
    p = ROOT / "docs" / "SYSTEM_LOG.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with p.open("a") as f:
        f.write(f"\\n## {title} ({stamp})\\nAuto-appended from script output. Audit pending until reviewed.\\n")
        for ln in lines:
            f.write(f"- {ln}\\n")
'''
NEW_LOG = '''def log(title, lines):
    import re
    p = ROOT / "docs" / "SYSTEM_LOG.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    txt = p.read_text() if p.exists() else ""
    parts = [x for x in re.split(r"(?m)^(?=## )", txt) if not x.startswith(f"## {title} (")]
    body = "".join(parts).rstrip("\\n") + "\\n"
    body += f"\\n## {title} ({stamp})\\nAuto-generated from script output. Audit pending until reviewed.\\n"
    body += "".join(f"- {ln}\\n" for ln in lines)
    p.write_text(body)
'''
patch("scripts/_ext_common.py", [(OLD_LOG, NEW_LOG)])

FIXED = '''    rec["fixed_thr"] = {}
    print(" fixed thresholds chosen in advance (no sweep): failures avoided / mean wasted life (all engines) / cost at r=10, r=100")
    for k in TRI:
        _, Fa_k, Wa_k = fam[k]
        rec["fixed_thr"][k] = {}
        for t in (15, 30, 60):
            i = int(t)
            fa, wa = float(1 - Fa_k[i].mean()), float(Wa_k[i].mean())
            cst = {str(int(r)): float(r * Fa_k[i].mean() + wa) for r in R_SEL}
            rec["fixed_thr"][k][str(t)] = dict(failures_avoided=fa, mean_wasted=wa, cost=cst)
            print(f"  {k:<16} thr {t:<3} avoided {fa:.3f} wasted {wa:5.1f} cost r=10 {cst['10']:.1f} r=100 {cst['100']:.1f}")
'''
patch("scripts/11_cost_policy.py", [("    be = breakevens(fam, R_FINE)\n", FIXED + "    be = breakevens(fam, R_FINE)\n")])

DASH_ANCHOR = '        warn(rec["assumptions"] + ". Because the threshold is swept'
DASH_NEW = '''        if rec.get("fixed_thr"):
            section("Fixed thresholds chosen in advance (no sweep)",
                    "Thresholds 15, 30 and 60 cycles on each signal, not tuned on data. Cost in C_waste units per engine.")
            FT = [{"signal": P_LABEL.get(k, k), "threshold": int(t), "failures avoided": v["failures_avoided"],
                   "mean wasted life": v["mean_wasted"], "cost at r=10": v["cost"]["10"], "cost at r=100": v["cost"]["100"]}
                  for k, d in rec["fixed_thr"].items() for t, v in d.items()]
            st.dataframe(pd.DataFrame(FT).round(3), hide_index=True)
'''
patch("dashboard/app.py", [(DASH_ANCHOR, DASH_NEW + DASH_ANCHOR)])

rs = ROOT / "scripts" / "run.sh"
if not rs.exists():
    raise SystemExit("scripts/run.sh not found; not creating it")
t = rs.read_text()
if "10_conformal.py" not in t:
    backup(rs)
    with rs.open("a") as f:
        f.write(("" if t.endswith("\n") else "\n") +
                "python scripts/10_conformal.py\npython scripts/11_cost_policy.py\npython scripts/12_explain.py --skip-faith\n"
                "# heavy, XGBoost only, run per subset: python scripts/12_explain.py --fd FD001 --skip-cluster --draws 2\n")
    print("appended 3 stages to scripts/run.sh")

LOGP = ROOT / "docs" / "SYSTEM_LOG.md"
txt = LOGP.read_text() if LOGP.exists() else ""
parts = re.split(r"(?m)^(?=## )", txt)
key = lambda x: re.sub(r" \(\d{4}-\d\d-\d\d.*\)\s*$", "", x.split("\n", 1)[0])
seen, keep = set(), []
for x in reversed(parts):
    if x.startswith("## Task"):
        if key(x) in seen:
            continue
        seen.add(key(x))
    keep.append(x)
txt = "".join(reversed(keep))
if not txt.lstrip().startswith("# "):
    txt = ("# FleetPM system log\n\nNOTE: the consolidated sections 0-14 are NOT present in this file. It holds only auto-generated "
           "extension entries and audit sections. Reconstruct sections 0-14 from the earlier consolidated log text before relying on this file.\n\n" + txt)
if "## 16. Extensions audit 2" not in txt:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    txt = txt.rstrip("\n") + f"""

## 16. Extensions audit 2 ({stamp})
- RESOLVED (VERIFIED): faithfulness refit at 500 trees reproduces saved deployed OOF RMSE for fold 1 exactly (15.4576), so design() and column order match. 300 trees cost about 0.4 RMSE on fold 1 (15.84 vs 15.46). The 14.51 vs 13.85 aggregate gap is UNVERIFIED (probably fold-mean vs pooled averaging).
- Conformal coverage on engines with true RUL below the cap (VERIFIED): 90% bound 0.910/0.891/0.859/0.901; 80% bound 0.742/0.832/0.776/0.779; offset 0.8 0.685/0.777/0.647/0.685; raw 0.315/0.436/0.376/0.392 (FD001-FD004). Cap-saturated engines inflated marginal coverage by about 0.01-0.03. No CIs computed for this subset.
- Clustering with Gaussian-covariance null (VERIFIED run): k=2 silhouette 0.285/0.200/0.434/0.397 vs null p95 0.307/0.198/0.321/0.304. Negative control FD001 inside the null, FD002 borderline, FD003 and FD004 clearly above. Spearman(cluster, life) -0.18/-0.43/-0.63/-0.51: clusters track engine life, so degradation speed is a confound. Fan/HPC mapping remains a HYPOTHESIS; readme-vs-paper discrepancy unresolved.
- Work orders live in artifacts/work_orders_v2/FD00X/. Dashboard v2 searches recursively.
- Defect: docs/SYSTEM_LOG.md lacked sections 0-14 (header added, duplicate auto entries removed, log() now replaces its own section on rerun).
- Added fixed-threshold (15/30/60) cost comparison to Task 2, because swept thresholds make the three signals near-reparametrisations. Result pending.
"""
backup(LOGP)
LOGP.write_text(txt)
print("SYSTEM_LOG.md deduplicated, headed, section 16 appended")