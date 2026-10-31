import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class _AE(nn.Module):
    def __init__(s, d, h, z):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(d, h[0]), nn.ReLU(), nn.Linear(h[0], h[1]),
                              nn.ReLU(), nn.Linear(h[1], z))
        s.dec = nn.Sequential(nn.Linear(z, h[1]), nn.ReLU(), nn.Linear(h[1], h[0]),
                              nn.ReLU(), nn.Linear(h[0], d))

    def forward(s, x):
        z = s.enc(x)
        return s.dec(z), z


class AEDetector:
    """HI = reconstruction MSE / q(healthy calibration MSE). HI>1 => beyond healthy envelope."""

    def __init__(self, cfg, seed):
        self.c, self.seed = cfg["anomaly"], seed
        self.dev = cfg.get("device", "cpu")

    def _t(self, X):
        z = np.clip(self.sc.transform(X), -10, 10)           # bound outlier features
        return torch.tensor(z, dtype=torch.float32, device=self.dev)

    def fit(self, Xtr, Xes):
        c = self.c
        torch.manual_seed(self.seed)
        self.sc = StandardScaler().fit(Xtr)
        self.sc.scale_ = np.maximum(self.sc.scale_, 1e-2)   # floor near-constant features
        Xt, Xv = self._t(Xtr), self._t(Xes)
        self.net = _AE(Xt.shape[1], c["hidden"], c["latent"]).to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        best, wait, state = np.inf, 0, None
        for _ in range(c["epochs"]):
            self.net.train()
            perm = torch.randperm(len(Xt), device=self.dev)
            for i in range(0, len(Xt), c["batch_size"]):
                b = Xt[perm[i:i + c["batch_size"]]]
                opt.zero_grad()
                loss = ((self.net(b)[0] - b) ** 2).mean()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = ((self.net(Xv)[0] - Xv) ** 2).mean().item()
            if v < best - 1e-6:
                best, wait, state = v, 0, copy.deepcopy(self.net.state_dict())
            else:
                wait += 1
                if wait >= c["patience"]:
                    break
        self.net.load_state_dict(state)
        self.net.eval()
        return self

    def errors(self, X):
        with torch.no_grad():
            x = self._t(X)
            return ((self.net(x)[0] - x) ** 2).mean(1).cpu().numpy()

    def latent(self, X):
        with torch.no_grad():
            return self.net(self._t(X))[1].cpu().numpy()

    def calibrate(self, Xcal):
        self.tau = float(np.quantile(self.errors(Xcal), self.c["tau_quantile"]))
        return self

    def index(self, X):
        return self.errors(X) / self.tau


def persistent(flag, eng, k):
    """True where the flag has held for >=k consecutive cycles within an engine."""
    flag = np.asarray(flag, bool)
    reset = pd.Series(~flag).groupby(eng).cumsum().to_numpy()
    run = pd.Series(flag.astype(int)).groupby([eng, reset]).cumsum().to_numpy()
    return run >= k