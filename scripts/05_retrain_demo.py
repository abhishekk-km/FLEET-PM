import _bootstrap
import argparse, json, numpy as np, mlflow
from fleetpm.config import load_config, art
from fleetpm.pipeline import load_prepared, feat_cols
from fleetpm.models import rul_tabular as T
from fleetpm.models.metrics import cut_eval
from fleetpm.mlops import tracking as mt
from fleetpm.mlops.drift import psi

ap = argparse.ArgumentParser()
ap.add_argument("--fd", default="FD001"); ap.add_argument("--days-since", type=int, default=0)
ap.add_argument("--inject-drift", type=float, default=0.0, help="SYNTHETIC: shift ALL features of arriving and validation engines by this many std")
a = ap.parse_args()
cfg = load_config(); r = cfg["rul"]; ev = cfg["eval"]; seed = cfg["seed"]; mt.setup(cfg)
P = load_prepared(cfg, a.fd); F = P["Ftr"]; cols = feat_cols(F)
X = F[cols].to_numpy(float); y = F["rul_cap"].to_numpy(float); eng = F.engine_id.to_numpy()
ids = np.unique(eng); np.random.default_rng(seed).shuffle(ids)
n = len(ids); G = {"init": ids[:int(.5 * n)], "arr": ids[int(.5 * n):int(.8 * n)], "val": ids[int(.8 * n):]}
M = {k: np.isin(eng, v) for k, v in G.items()}
if a.inject_drift:
    sh = a.inject_drift * X.std(0)
    for k in ("arr", "val"):
        X[M[k]] = X[M[k]] + sh
name = r["primary"]
tag = f"d{str(a.inject_drift).replace('.', 'p')}_t{a.days_since}"
reg = f"{cfg['mlops']['model_prefix']}_{a.fd}_retrain_demo_{tag}"


def train_log(mask, run_tag):
    with mlflow.start_run(run_name=f"{a.fd}_{tag}_{run_tag}") as run:
        m = T.fit(T.make(name, r, seed), name, X[mask], y[mask], T.weights(y[mask], r))
        v = cut_eval(y[M["val"]], T.predict(m, X[M["val"]]), eng[M["val"]], ev["min_cut_len"], ev["n_cut_reps"], seed)
        mlflow.log_metrics({"val_rmse": v["rmse"], "val_nasa": v["nasa"]}); mlflow.log_param("train_engines", int(len(np.unique(eng[mask]))))
        uri = mt.log_sk_model(m)
        ok, ver = mt.register_and_maybe_promote(uri, run.info.run_id, reg, "val_rmse", cfg["mlops"]["promote_margin"])
    return m, v, ok, ver


m1, v1, ok, ver = train_log(M["init"], "v1_initial"); print("v1", v1, "champion:", ok, "ver", ver)
cur = cut_eval(y[M["arr"]], T.predict(m1, X[M["arr"]]), eng[M["arr"]], ev["min_cut_len"], ev["n_cut_reps"], seed)
ps_all = {cols[i]: psi(X[M["init"], i], X[M["arr"], i]) for i in range(len(cols))}
ps = {k: round(v, 3) for k, v in sorted(ps_all.items(), key=lambda kv: -kv[1])[:5]}
rc = cfg["mlops"]["retrain"]
max_psi = max(ps_all.values())
n_alarm = int(sum(v > rc["psi_alarm"] for v in ps_all.values()))
print("v1 on arrivals:", cur, "| PSI top5 of all", len(cols), "features:", ps, "| features above alarm:", n_alarm)
t_rmse = bool(cur["rmse"] > rc["rmse_factor"] * v1["rmse"])
t_days = bool(a.days_since >= rc["days"])
t_psi = bool(max_psi > rc["psi_alarm"])
trig = t_rmse or t_days or t_psi
print("retrain triggered:", trig, {"rmse": t_rmse, "days": t_days, "psi": t_psi})
res = {"fd": a.fd, "model": name, "synthetic_drift_std": a.inject_drift, "days_since": a.days_since,
       "engines": {k: int(len(v)) for k, v in G.items()}, "v1_val": v1, "v1_arrivals": cur,
       "psi_top5": ps, "max_psi": float(max_psi), "n_features_above_alarm": n_alarm,
       "trigger": {"rmse": t_rmse, "days": t_days, "psi": t_psi},
       "retrain_triggered": bool(trig), "v1_promoted": bool(ok)}
if trig:
    _, v2, ok2, ver2 = train_log(M["init"] | M["arr"], "v2_retrained")
    print("v2", v2, "| promoted to champion:", ok2, "ver", ver2)
    res.update({"v2_val": v2, "v2_promoted": bool(ok2), "v2_version": str(ver2)})
out = art(cfg, "retrain_demo", f"{a.fd}_{tag}.json")
out.write_text(json.dumps(res, indent=2))
print("saved", out.name)
