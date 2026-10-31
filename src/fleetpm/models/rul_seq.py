import copy
import numpy as np
import torch
import torch.nn as nn


class _Net(nn.Module):
    def __init__(s, kind, f, c):
        super().__init__()
        s.kind, h = kind, c["hidden"]
        if kind in ("lstm", "gru"):
            cls = nn.LSTM if kind == "lstm" else nn.GRU
            s.rnn = cls(f, h, num_layers=c["layers"], batch_first=True, dropout=c["dropout"])
        elif kind == "cnn1d":
            s.cnn = nn.Sequential(nn.Conv1d(f, h, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(h, h, 3, padding=1), nn.ReLU(),
                                  nn.AdaptiveMaxPool1d(1))
        elif kind == "cnnlstm":
            # Same architecture as the CMAPSS notebook (Keras): Conv(32,k5) -> Conv(64,k3)
            # -> MaxPool(3) -> LSTM(50) -> Dropout -> LSTM(50) -> Dropout -> Dense(50) -> Dense(1)
            s.cnn = nn.Sequential(nn.Conv1d(f, 32, 5), nn.ReLU(),
                                  nn.Conv1d(32, 64, 3), nn.ReLU(),
                                  nn.MaxPool1d(3))
            s.rnn = nn.LSTM(64, 50, num_layers=2, batch_first=True, dropout=0.2)
        else:
            raise ValueError(kind)
        if kind == "cnnlstm":
            s.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 1))
        else:
            s.head = nn.Sequential(nn.Dropout(c["dropout"]), nn.Linear(h, 1))

    def forward(s, x):
        if s.kind == "cnn1d":
            z = s.cnn(x.transpose(1, 2)).squeeze(-1)
        elif s.kind == "cnnlstm":
            z = s.rnn(s.cnn(x.transpose(1, 2)).transpose(1, 2))[0][:, -1]
        else:
            z = s.rnn(x)[0][:, -1]
        return s.head(z).squeeze(-1)


class SeqRegressor:
    def __init__(self, kind, cfg, cap, seed):
        self.kind, self.cap, self.seed = kind, cap, seed
        self.c, self.r = cfg["rul"]["seq"], cfg["rul"]
        self.vf, self.dev = cfg["eval"]["seq_inner_val_frac"], cfg.get("device", "cpu")

    def fit(self, X, y, eng):
        if self.kind == "cnnlstm" and X.shape[1] < 9:
            raise ValueError("cnnlstm needs features.window >= 9 (two valid convs + MaxPool(3)).")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        ids = np.unique(eng)
        vid = rng.choice(ids, max(1, int(self.vf * len(ids))), replace=False)
        v = np.isin(eng, vid)
        w = np.where(y < self.r["low_rul_threshold"], self.r["low_rul_weight"], 1.0)
        T = lambda a: torch.tensor(a, dtype=torch.float32)
        Xt, yt, wt = T(X[~v]), T(y[~v] / self.cap), T(w[~v])
        Xv, yv, wv = T(X[v]).to(self.dev), T(y[v] / self.cap).to(self.dev), T(w[v]).to(self.dev)
        self.net = _Net(self.kind, X.shape[2], self.c).to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.c["lr"])
        best, wait, state, bs = np.inf, 0, None, self.c["batch_size"]
        for _ in range(self.c["epochs"]):
            self.net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), bs):
                j = perm[i:i + bs]
                opt.zero_grad()
                loss = (wt[j].to(self.dev) * (self.net(Xt[j].to(self.dev)) - yt[j].to(self.dev)) ** 2).mean()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = (wv * (self.net(Xv) - yv) ** 2).mean().item()
            if vl < best - 1e-7:
                best, wait, state = vl, 0, copy.deepcopy(self.net.state_dict())
            else:
                wait += 1
                if wait >= self.c["patience"]:
                    break
        self.net.load_state_dict(state)
        self.net.eval()
        return self

    def predict(self, X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                xb = torch.tensor(X[i:i + 2048], dtype=torch.float32, device=self.dev)
                out.append(self.net(xb).cpu().numpy())
        return np.clip(np.concatenate(out) * self.cap, 0, None)