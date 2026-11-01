import pandas as pd
from pathlib import Path

SET = [f"setting_{i}" for i in range(1, 4)]
SEN = [f"sensor_{i}" for i in range(1, 22)]
COLS = ["engine_id", "cycle"] + SET + SEN


def _read(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, sep=r"\s+", header=None)
    if df.shape[1] != 26:
        raise ValueError(f"{p}: {df.shape[1]} columns, expected 26")
    df.columns = COLS
    df[["engine_id", "cycle"]] = df[["engine_id", "cycle"]].astype(int)
    return df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


def load_subset(raw: Path, fd: str):
    tr = _read(raw / f"train_{fd}.txt")
    te = _read(raw / f"test_{fd}.txt")
    rul = pd.read_csv(raw / f"RUL_{fd}.txt", header=None, sep=r"\s+")[0].to_numpy(float)
    return tr, te, rul