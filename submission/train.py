#!/usr/bin/env python3
"""Train a signal/background classifier on the pool and export it as model.json (text) for fit.py.

    python3 train.py [pool.npz] [--out model.json] [--aug] [--epochs 60]

A small multilayer perceptron (two hidden layers of 64) trained with weighted binary
cross-entropy by Adam, on the EVEN-indexed pool events only: fit.py builds its templates from
the ODD-indexed events, so the templates are never made from the events the network saw.
Numpy only, so the same network is evaluated under the contract (numpy/scipy/torch).

The event weights are used: each class's weights are clipped at their 99th percentile (a few
huge weights must not dominate) and the two classes are balanced, so the score is tuned to the
mixture that is measured.  --aug morphs each training signal event to a random grid mass, so
the score is trained across the mass hypotheses the fit scans.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hww as H  # noqa: E402
from features import FEATURES, features  # noqa: E402

TRAIN_MASS_GRID = np.arange(120.0, 130.01, 1.0)   # the masses fit.py scans for templates


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _forward(Ws, bs, X):
    a = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        a = a @ W + b
        if i < len(Ws) - 1:
            a = np.maximum(a, 0.0)
    return a


def augment(sig, rng, grid):
    """Morph each signal event to a random grid mass (array indexed as sig)."""
    ms = rng.choice(grid, size=len(sig["weight"]))
    out = {k: v.copy() for k, v in sig.items()}
    for m in np.unique(ms):
        sel = ms == m
        mm = H.morph(H.select(sig, sel), m)
        for c in H.SCALED:
            out[c][sel] = mm[c]
    return out


def train_mlp(X, y, w, Xv, yv, wv, hidden=(64, 64), epochs=60, batch=2048, lr=2e-3, l2=1e-5,
              patience=8, seed=1):
    """Weighted binary cross-entropy, Adam, early stopping on the (weighted) validation loss."""
    rng = np.random.default_rng(seed)
    sizes = [X.shape[1], *hidden, 1]
    Ws = [rng.normal(0.0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
    bs = [np.zeros(b) for b in sizes[1:]]
    m = [np.zeros_like(p) for p in Ws + bs]
    v = [np.zeros_like(p) for p in Ws + bs]
    t, best = 0, (np.inf, None, 0)

    def val_loss():
        p = np.clip(_sigmoid(_forward(Ws, bs, Xv))[:, 0], 1e-7, 1 - 1e-7)
        return float(-np.sum(wv * (yv * np.log(p) + (1.0 - yv) * np.log(1.0 - p))) / wv.sum())

    ep = 0
    for ep in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            j = order[s:s + batch]
            xb, yb, wb = X[j], y[j], w[j]
            acts = [xb]
            for i, (W, b) in enumerate(zip(Ws, bs)):
                z = acts[-1] @ W + b
                acts.append(np.maximum(z, 0.0) if i < len(Ws) - 1 else z)
            d = ((_sigmoid(acts[-1][:, 0]) - yb) * (wb / wb.sum()))[:, None]
            gW, gb = [None] * len(Ws), [None] * len(Ws)
            for i in range(len(Ws) - 1, -1, -1):
                gW[i] = acts[i].T @ d + l2 * Ws[i]
                gb[i] = d.sum(0)
                if i > 0:
                    d = (d @ Ws[i].T) * (acts[i] > 0)
            t += 1
            params, grads = Ws + bs, gW + gb
            for k in range(len(params)):
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                params[k] -= lr * (m[k] / (1.0 - 0.9 ** t)) / (np.sqrt(v[k] / (1.0 - 0.999 ** t)) + 1e-8)
        loss = val_loss()
        if loss < best[0] - 1e-5:
            best = (loss, ([W.copy() for W in Ws], [b.copy() for b in bs]), ep)
        elif ep - best[2] >= patience:
            break
    return best[1][0], best[1][1], ep + 1, best[0]


def weighted_auc(y, s, w):
    o = np.argsort(s, kind="mergesort")
    y, w = y[o], w[o]
    neg = np.cumsum(w * (1.0 - y))
    return float(np.sum(w * y * neg) / ((w * y).sum() * (w * (1.0 - y)).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", nargs="?", default=str(HERE.parent.parent / "datasets" / "public" / "seed_000" / "mc" / "pool.npz"))
    ap.add_argument("--out", default=str(HERE / "model.json"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--aug", action="store_true", help="train on mass-morphed signal")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    pool = H.load(a.pool)
    N = len(pool["label"])
    parity = np.arange(N) % 2
    train = H.select(pool, parity == 0)       # fit.py builds its templates from parity == 1
    held = H.select(pool, parity == 1)
    rng = np.random.default_rng(a.seed)
    if a.aug:
        sig_i = np.where(train["label"] == 1)[0]
        aug = augment(H.select(train, sig_i), rng, TRAIN_MASS_GRID)
        train = {k: v.copy() for k, v in train.items()}
        for k in train:
            train[k][sig_i] = aug[k]

    y = train["label"].astype(float)
    w = train["weight"].astype(float)
    n = len(y)
    perm = rng.permutation(n)
    va, tr = perm[:int(0.15 * n)], perm[int(0.15 * n):]

    wt = w.copy()
    for c in (0, 1):
        sel = y == c
        wt[sel] = np.minimum(wt[sel], np.quantile(w[sel], 0.99))
    is_tr = np.zeros(n, bool)
    is_tr[tr] = True
    for c in (0, 1):
        wt[y == c] *= 0.5 / wt[(y == c) & is_tr].sum()
    wt *= n

    X = features(train)
    mean, std = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-9
    Xs = (X - mean) / std
    Ws, bs, ne, loss = train_mlp(Xs[tr], y[tr], wt[tr], Xs[va], y[va], wt[va], epochs=a.epochs, seed=a.seed)

    # held-out check on the odd events: AUC and the Asimov Z of a (score, m_T) template at 125 GeV
    p = _sigmoid(_forward(Ws, bs, (features(held) - mean) / std)[:, 0])
    p_tr = _sigmoid(_forward(Ws, bs, Xs[tr])[:, 0])
    yl, wl = held["label"].astype(int), held["weight"].astype(float) * 2.0
    print(f"trained {ne} epochs (aug={a.aug}, val loss {loss:.5f}); weighted AUC: "
          f"train {weighted_auc(y[tr], p_tr, wt[tr]):.4f}, held-out {weighted_auc(yl.astype(float), p, wl):.4f}")
    edges = [np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), H.MT_EDGES]
    mt = H.kinematics(held)["mt"]
    hs, _, _ = np.histogram2d(p[yl == 1], mt[yl == 1], edges, weights=wl[yl == 1])
    hb, _, _ = np.histogram2d(p[yl == 0], mt[yl == 0], edges, weights=wl[yl == 0])
    print(f"expected Asimov Z at 125 GeV, (score, m_T) bins: {H.asimov_significance(hs, hb):.2f}")

    model = {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "activation": "relu",
             "augmented": bool(a.aug), "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in zip(Ws, bs)]}
    Path(a.out).write_text(json.dumps(model))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
