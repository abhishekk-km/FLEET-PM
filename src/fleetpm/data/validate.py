def validate(tr, te, rul, fd):
    rep = {}
    for n, d in (("train", tr), ("test", te)):
        if d.isna().any().any():
            raise ValueError(f"{fd} {n}: NaN present")
        if d.duplicated(["engine_id", "cycle"]).any():
            raise ValueError(f"{fd} {n}: duplicate (engine_id,cycle)")
        g = d.groupby("engine_id")["cycle"]
        if not (g.min() == 1).all():
            raise ValueError(f"{fd} {n}: cycle does not start at 1")
        if not ((g.max() - g.min() + 1) == g.count()).all():
            raise ValueError(f"{fd} {n}: gaps in cycles")
        rep[f"{n}_engines"] = int(g.ngroups)
    ids = sorted(te.engine_id.unique())
    if ids != list(range(1, len(rul) + 1)):
        raise ValueError(f"{fd}: test ids != 1..len(RUL) ({len(ids)} vs {len(rul)})")
    return rep