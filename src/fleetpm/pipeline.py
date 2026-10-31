import joblib
from .config import ROOT, art
from .data.loader import load_subset
from .data.validate import validate
from .data.preprocess import Preprocessor, add_rul, add_rul_test
from .features.rolling import build_features


def prepare(cfg, fd):
    tr, te, rul = load_subset(ROOT / cfg["paths"]["raw"], fd)
    rep = validate(tr, te, rul, fd)
    cap = cfg["data"]["rul_cap"][fd]
    add_rul(tr, cap)
    add_rul_test(te, rul, cap)
    pre = Preprocessor(cfg, fd).fit(tr)
    Ztr, Zte = pre.transform(tr), pre.transform(te)
    W, st = cfg["features"]["window"], cfg["features"]["stats"]
    Ftr = build_features(Ztr, pre.sensors, W, st)
    Fte = build_features(Zte, pre.sensors, W, st)
    for F, d in ((Ftr, tr), (Fte, te)):
        F["rul"] = d["rul"].to_numpy()
        F["rul_cap"] = d["rul_cap"].to_numpy()
    return dict(fd=fd, cap=cap, report=rep, pre=pre, rul=rul,
                Ztr=Ztr, Zte=Zte, Ftr=Ftr, Fte=Fte)


def save_prepared(cfg, P):
    joblib.dump(P, art(cfg, "features", f"{P['fd']}.joblib"))


def load_prepared(cfg, fd):
    return joblib.load(art(cfg, "features", f"{fd}.joblib"))


def feat_cols(F):
    return [c for c in F.columns if "__" in c]