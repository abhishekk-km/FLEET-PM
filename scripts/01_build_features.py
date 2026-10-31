import _bootstrap
from fleetpm.config import load_config
from fleetpm.pipeline import prepare, save_prepared, feat_cols

cfg = load_config()
for fd in cfg["data"]["subsets"]:
    P = prepare(cfg, fd)
    save_prepared(cfg, P)
    print(fd, "sensors kept:", len(P["pre"].sensors), "| features:", len(feat_cols(P["Ftr"])),
          "| train rows:", len(P["Ftr"]), "| test rows:", len(P["Fte"]))