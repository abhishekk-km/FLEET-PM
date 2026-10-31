import _bootstrap
import numpy as np, pandas as pd
from fleetpm.config import load_config, ROOT
from fleetpm.data.loader import load_subset, SET
from fleetpm.data.validate import validate
from fleetpm.data.preprocess import Preprocessor, add_rul

cfg = load_config()
pd.set_option("display.width", 200)
for fd in cfg["data"]["subsets"]:
    tr, te, rul = load_subset(ROOT / cfg["paths"]["raw"], fd)
    print(f"\n===== {fd} =====\n{validate(tr, te, rul, fd)}")
    L = tr.groupby("engine_id").cycle.max()
    Lt = te.groupby("engine_id").cycle.max()
    print("train life min/mean/max:", L.min(), round(L.mean(), 1), L.max())
    print("test len  min/mean/max:", Lt.min(), round(Lt.mean(), 1), Lt.max())
    print("test RUL  min/mean/max:", rul.min(), round(rul.mean(), 1), rul.max())
    print("test cut fraction len/(len+RUL) mean/min/max:",
          *np.round([(Lt.to_numpy() / (Lt.to_numpy() + rul)).mean(),
                     (Lt.to_numpy() / (Lt.to_numpy() + rul)).min(),
                     (Lt.to_numpy() / (Lt.to_numpy() + rul)).max()], 3))
    print("unique rounded(1) setting combos:", len(tr[SET].round(1).drop_duplicates()))
    pre = Preprocessor(cfg, fd).fit(add_rul(tr.copy(), cfg["data"]["rul_cap"][fd]))
    print("exactly-constant sensors:", pre.const)
    print("mean |spearman(sensor,cycle)| per engine (after regime-norm+smooth):")
    print(pre.trend.round(3).sort_values(ascending=False).to_string())