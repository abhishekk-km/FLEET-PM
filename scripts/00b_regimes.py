import _bootstrap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from fleetpm.config import load_config, ROOT
from fleetpm.data.loader import load_subset, SET

cfg = load_config()
for fd in ("FD002", "FD004"):
    tr, _, _ = load_subset(ROOT / cfg["paths"]["raw"], fd)
    print(f"\n== {fd} ==")
    print(tr[SET].round(2).value_counts().to_string())
    for c in SET:
        print(c, "round(1):", tr[c].round(1).nunique(), "| round(2):", tr[c].round(2).nunique())
    Z = StandardScaler().fit_transform(tr[SET].to_numpy())
    for k in range(4, 9):
        km = KMeans(k, n_init=10, random_state=cfg["seed"]).fit(Z)
        print(f"k={k} inertia={km.inertia_:.1f} sizes={sorted(np.bincount(km.labels_).tolist())}")