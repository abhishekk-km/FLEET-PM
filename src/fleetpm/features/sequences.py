import numpy as np


def build_sequences(Z, sensors, W):  # ASSUMPTION[A15] left-pad with first row
    out = []
    for _, g in Z.groupby("engine_id", sort=True):
        a = g[sensors].to_numpy(np.float32)
        p = np.concatenate([np.repeat(a[:1], W - 1, 0), a])
        out.append(np.lib.stride_tricks.sliding_window_view(p, W, axis=0).transpose(0, 2, 1))
    return np.concatenate(out)