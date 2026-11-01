import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .loader import SET, SEN

EPS = 1e-12


def add_rul(df, cap):
    T = df.groupby("engine_id")["cycle"].transform("max")
    df["rul"] = (T - df["cycle"]).astype(float)
    df["rul_cap"] = df["rul"].clip(upper=cap)
    return df


def add_rul_test(df, rul, cap):  # ASSUMPTION[A2][A11]
    T = df.groupby("engine_id")["cycle"].transform("max")
    df["rul"] = rul[df["engine_id"].to_numpy() - 1] + (T - df["cycle"]).to_numpy()
    df["rul_cap"] = df["rul"].clip(upper=cap)
    return df


class Preprocessor:
    """Regime z-score (train-fitted) -> trailing median smoothing -> sensor selection."""

    def __init__(self, cfg, fd):
        d = cfg["data"]
        self.seed = cfg["seed"]
        self.k = d["n_regimes"] if d["multi_condition"][fd] else 1
        self.win = d["smooth_window"]
        self.min_trend = d["min_abs_trend"]

    def _reg(self, df):
        if self.k == 1:
            return np.zeros(len(df), int)
        return self.km.predict(self.sc.transform(df[SET].to_numpy()))

    def _nz(self, df):
        reg = self._reg(df)
        X = (df[SEN].to_numpy() - self.mu.loc[reg].to_numpy()) / np.maximum(
            self.sd.loc[reg].to_numpy(), EPS)
        Z = pd.DataFrame(X, columns=SEN, index=df.index)
        Z = Z.groupby(df["engine_id"].to_numpy()).transform(
            lambda s: s.rolling(self.win, min_periods=1).median())
        Z.insert(0, "regime_id", reg)
        Z.insert(0, "cycle", df["cycle"].to_numpy())
        Z.insert(0, "engine_id", df["engine_id"].to_numpy())
        return Z

    def fit(self, tr):
        if self.k > 1:
            self.sc = StandardScaler().fit(tr[SET].to_numpy())
            self.km = KMeans(self.k, n_init=10, random_state=self.seed).fit(
                self.sc.transform(tr[SET].to_numpy()))
        reg = self._reg(tr)
        g = tr[SEN].groupby(reg)
        self.mu, self.sd = g.mean(), g.std(ddof=0)
        self.const = [c for c in SEN if (self.sd[c] <= EPS).all()]
        Z = self._nz(tr)
        rho = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for c in SEN:
                if c in self.const:
                    continue
                rho[c] = np.nanmean([abs(spearmanr(g_["cycle"], g_[c])[0])
                                     for _, g_ in Z.groupby("engine_id")])
        self.trend = pd.Series(rho)
        self.sensors = [c for c in SEN if c not in self.const and rho[c] >= self.min_trend]
        return self

    def transform(self, df):
        return self._nz(df)[["engine_id", "cycle", "regime_id"] + self.sensors]