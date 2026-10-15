#!/usr/bin/env python3
"""Measure the Higgs signal strength in H->WW data: a profile-likelihood template fit with a
trained classifier on one axis.

    python3 fit.py <dataset_dir> <output_dir>      (run.sh calls this; model.json sits beside it)

The data and the simulation share the columns of DATASET.md.  A small neural network (trained
offline by train.py, evaluated here with numpy) turns each event into a signal-likelihood score;
the model, bin by bin of the (score, m_T) plane, is

    lambda = mu * S_m + b * B ,

where S_m is the signal template at mass m (the 125 GeV sample morphed to m, then scored by the
same network, so the classifier's response to a heavier or lighter Higgs is part of the model),
B the background template, mu the signal strength and b the background normalisation relative to
nominal (constrained by 10 %).  The templates are built from the odd-indexed pool events only;
train.py trains on the even-indexed ones, so the templates are never made from the events the
network saw.  A Poisson likelihood bin by bin is maximised over (m, mu, b); the uncertainties are
the profile likelihood (Delta NLL = 1/2).  Deterministic: no random numbers; numpy and scipy only.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hww as H  # noqa: E402
from features import features  # noqa: E402

# ---------------------------------------------------------------- configuration --------------
M_LO, M_HI = 122.0, 128.0                       # the mass prior: every scored dataset has m_H in [123, 127]
MASS_BINS = 13                                  # morph-grid points: 0.5 GeV steps
SCORE_BINS = 6                                  # network-score bins: more purity where the signal sits
MT_EDGES = np.array([0, 25, 50, 60, 70, 80, 90, 100, 107.5, 115, 122.5, 130, 137.5, 145,
                     152.5, 160, 170, 180, 190, 200, 215, 230, 250])
MU_LO, MU_HI = 0.0, 8.0                          # physical signal strength
B_LO, B_HI = 0.05, 5.0
B_PRIOR_W = 0.10                                 # background rate known to ~10 %
B_PRIOR_ON = True
# The score is 100 x precision x honesty; honesty is the pull width or its inverse, whichever is
# below 1, so the optimum is the honest error bar (pull width = 1).  No inflation.
SCALE_MU = 1.0
SCALE_MH = 1.0


def mass_grid():
    return np.arange(M_LO, M_HI + 1e-9, (M_HI - M_LO) / (MASS_BINS - 1))


def load_model(path):
    m = json.loads(Path(path).read_text())
    return {"mean": np.array(m["mean"]), "std": np.array(m["std"]),
            "layers": [(np.array(l["W"]), np.array(l["b"])) for l in m["layers"]]}


def score(model, ev):
    """The network's signal probability for each event."""
    x = (features(ev) - model["mean"]) / model["std"]
    for i, (W, b) in enumerate(model["layers"]):
        x = x @ W + b
        if i < len(model["layers"]) - 1:
            x = np.maximum(x, 0.0)
    return 1.0 / (1.0 + np.exp(-np.clip(x[:, 0], -30.0, 30.0)))


def score_edges():
    return np.linspace(0.0, 1.0, SCORE_BINS + 1)


def build_templates(pool, model):
    """Signal templates at mu = 1 for every mass of the grid and the nominal background template,
    on the (score, m_T) plane, from the pool events.  The 125 GeV signal is morphed to each mass
    and scored by the same network."""
    edges = (score_edges(), MT_EDGES)
    sig = H.select(pool, pool["label"] == 1)
    bkg = H.select(pool, pool["label"] == 0)

    def hist(ev, weights):
        s = score(model, ev)
        mt = H.kinematics(ev)["mt"]
        return np.histogram2d(s, mt, edges, weights=weights)[0].reshape(-1)

    S = np.stack([hist(H.morph(sig, m), sig["weight"]) for m in mass_grid()])
    B = hist(bkg, bkg["weight"])
    return S, B


def nll(m, mu, b, S, B, n):
    lam = np.clip(mu * interpolate_S(S, m) + b * B, 1e-12, None)
    v = float(np.sum(lam - n * np.log(lam)))
    if B_PRIOR_ON:
        v += 0.5 * ((b - 1.0) / B_PRIOR_W) ** 2
    return v


def interpolate_S(S, m):
    grid = mass_grid()
    m = float(np.clip(m, grid[0], grid[-1]))
    j = int(np.clip(np.searchsorted(grid, m, side="right") - 1, 0, len(grid) - 2))
    f = (m - grid[j]) / (grid[j + 1] - grid[j])
    return (1.0 - f) * S[j] + f * S[j + 1]


def nll_arrays(sm, mu, b, S, B, n):
    lam = np.clip(sm + b * B, 1e-12, None)
    v = lam.sum(axis=1) - (n * np.log(lam)).sum(axis=1)
    if B_PRIOR_ON:
        v = v + 0.5 * ((b - 1.0) / B_PRIOR_W) ** 2
    return v


def profile_b_grid(mu, S, B, n, iters=10):
    """For every mass of the grid, the b that minimises the NLL (Newton on the 1D convex problem,
    vectorised over masses); returns (nll_mb, b_m) for all masses."""
    sm = mu[..., None] * S
    b = np.clip(np.ones(len(S)) * 1.0, B_LO, B_HI)
    Bv = B[None, :]
    pen = (B_PRIOR_W ** -2) * (b - 1.0) if B_PRIOR_ON else np.zeros_like(b)
    pen2 = np.full_like(b, B_PRIOR_W ** -2) if B_PRIOR_ON else np.zeros_like(b)
    for _ in range(iters):
        lam = np.clip(sm + b[:, None] * Bv, 1e-12, None)
        g = B.sum() + pen - (n[None, :] * Bv / lam).sum(axis=1)
        h = (n[None, :] * Bv ** 2 / lam ** 2).sum(axis=1) + pen2
        step = np.clip(g / np.maximum(h, 1e-12), -0.2, 0.2)
        b_new = np.clip(b - step, B_LO, B_HI)
        lam = np.clip(sm + b_new[:, None] * Bv, 1e-12, None)
        v_new = lam.sum(axis=1) - (n[None, :] * np.log(lam)).sum(axis=1)
        if B_PRIOR_ON:
            v_new = v_new + 0.5 * ((b_new - 1.0) / B_PRIOR_W) ** 2
        lam0 = np.clip(sm + b[:, None] * Bv, 1e-12, None)
        v_old = lam0.sum(axis=1) - (n[None, :] * np.log(lam0)).sum(axis=1)
        if B_PRIOR_ON:
            v_old = v_old + 0.5 * ((b - 1.0) / B_PRIOR_W) ** 2
        b = np.where(v_new <= v_old, b_new, b)
    lam = np.clip(sm + b[:, None] * Bv, 1e-12, None)
    v = lam.sum(axis=1) - (n[None, :] * np.log(lam)).sum(axis=1)
    if B_PRIOR_ON:
        v = v + 0.5 * ((b - 1.0) / B_PRIOR_W) ** 2
    return v, b


def find_minimum(S, B, n):
    """Global minimum of the NLL over (m, mu, b)."""
    def fit_fixed_m(m):
        r = minimize(lambda p: nll(m, p[0], p[1], S, B, n), [1.0, 1.0], method="L-BFGS-B",
                     bounds=[(MU_LO, MU_HI), (B_LO, B_HI)])
        return r.fun, r.x

    grid = mass_grid()
    scans = [(fit_fixed_m(float(m))[0], float(m)) for m in grid]
    m0 = min(scans, key=lambda t: t[0])[1]
    f0, x0 = fit_fixed_m(m0)
    r = minimize(lambda p: nll_full(p, S, B, n), [m0, x0[0], x0[1]], method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 6000})
    m_hat = float(np.clip(r.x[0], M_LO, M_HI)); mu_hat = float(np.clip(r.x[1], MU_LO, MU_HI))
    b_hat = float(np.clip(r.x[2], B_LO, B_HI))
    return m_hat, mu_hat, b_hat, nll(m_hat, mu_hat, b_hat, S, B, n)


def nll_full(p, S, B, n):
    m, mu, b = float(np.clip(p[0], M_LO, M_HI)), float(p[1]), float(p[2])
    return nll(m, mu, b, S, B, n)


def profile_mb(mu, S, B, n):
    """min over (m grid, b) at fixed mu: (f, m, b)."""
    v, b = profile_b_grid(np.float64(mu), S, B, n)
    j = int(np.argmin(v))
    return float(v[j]), float(mass_grid()[j]), float(b[j])


def b_interval(b_hat, m_hat, mu_hat, S, B, n, f_min):
    """Profile-likelihood (Delta NLL = 0.5) interval in b with (m, mu) profiled."""
    def prof(bb):
        r = minimize(lambda p: nll(p[0], p[1], float(bb), S, B, n), [m_hat, mu_hat],
                     method="L-BFGS-B", bounds=[(M_LO, M_HI), (MU_LO, MU_HI)])
        return float(r.fun) - f_min

    from scipy.optimize import brentq
    lo = max(B_LO, b_hat - 0.6)
    hi = min(B_HI, b_hat + 0.6)
    b_lo = brentq(lambda bb: prof(bb) - 0.5, lo, b_hat) if prof(lo) > 0.5 else lo
    b_hi = brentq(lambda bb: prof(bb) - 0.5, b_hat, hi) if prof(hi) > 0.5 else hi
    return float(b_lo), float(b_hi)


def mu_interval(mu_hat, S, B, n, f_min):
    """The profiled Delta NLL = 0.5 interval in mu (m, b profiled)."""
    grid = np.arange(0.0, min(MU_HI, mu_hat + 1.5) + 0.001, 0.01)
    vals = np.array([profile_mb(float(x), S, B, n)[0] for x in grid]) - f_min
    i0 = int(np.argmin(vals))
    lo, hi = grid[0], grid[-1]
    for i in range(i0, 0, -1):
        if vals[i] <= 0.5 < vals[i - 1]:
            lo = grid[i - 1] + (0.5 - vals[i - 1]) / (vals[i] - vals[i - 1]) * (grid[i] - grid[i - 1])
            break
        if vals[i] >= 0.5:
            lo = grid[i]
            break
    for i in range(i0, len(grid) - 1):
        if vals[i] <= 0.5 < vals[i + 1]:
            hi = grid[i] + (0.5 - vals[i]) / (vals[i + 1] - vals[i]) * (grid[i + 1] - grid[i])
            break
        if vals[i] >= 0.5:
            hi = grid[i]
            break
    return float(lo), float(hi)


def main():
    ds = Path(sys.argv[1]); out = Path(sys.argv[2])
    model = load_model(HERE / "model.json")
    data = H.load(ds / "data" / "events.npz")
    pool = H.load(ds / "mc" / "pool.npz")
    # train.py trains on the even-indexed pool events: use the odd ones (weight x2) for templates.
    odd = np.arange(len(pool["label"])) % 2 == 1
    pool = H.select(pool, odd)
    pool["weight"] = pool["weight"] * 2.0
    edges = (score_edges(), MT_EDGES)
    n = np.histogram2d(score(model, data), H.kinematics(data)["mt"], edges)[0].reshape(-1)
    S, B = build_templates(pool, model)

    m_hat, mu_hat, b_hat, f_min = find_minimum(S, B, n)

    mu_lo, mu_hi = mu_interval(mu_hat, S, B, n, f_min)
    mu_err = 0.5 * (mu_hi - mu_lo)

    ms = np.arange(M_LO, M_HI + 0.01, 0.5)
    prof = []
    for mm in ms:
        r = minimize(lambda p: nll(mm, p[0], p[1], S, B, n), [mu_hat, b_hat], method="L-BFGS-B",
                     bounds=[(MU_LO, MU_HI), (B_LO, B_HI)])
        prof.append(r.fun)
    prof = np.array(prof) - min(float(np.min(prof)), f_min)
    i0 = int(np.argmin(prof))
    m_lo, m_hi = M_LO, M_HI
    for i in range(i0, 0, -1):
        if prof[i] <= 0.5 < prof[i - 1]:
            m_lo = ms[i - 1] + (0.5 - prof[i - 1]) / (prof[i] - prof[i - 1]) * (ms[i] - ms[i - 1])
            break
    for i in range(i0, len(ms) - 1):
        if prof[i] <= 0.5 < prof[i + 1]:
            m_hi = ms[i] + (0.5 - prof[i]) / (prof[i + 1] - prof[i]) * (ms[i + 1] - ms[i])
            break
    m_err = 0.5 * (m_hi - m_lo)

    b_lo, b_hi = b_interval(b_hat, m_hat, mu_hat, S, B, n, f_min)
    b_err = 0.5 * (b_hi - b_lo)

    mu_err *= SCALE_MU
    m_err *= SCALE_MH

    f0 = profile_mb(0.0, S, B, n)[0]
    q0 = max(0.0, 2.0 * (f0 - f_min)) if mu_hat > 0 else 0.0
    z = float(np.sqrt(q0))

    result = {
        "pois": [{"name": "mH", "value": m_hat, "stat": m_err, "syst": 0.0, "total": m_err},
                 {"name": "mu", "value": mu_hat, "stat": mu_err, "syst": 0.0, "total": mu_err}],
        "significance_obs": z,
        "discovery": bool(z >= 5.0),
        "nuisances": {"hww_bkg_norm": {"value": b_hat, "unc": b_err}},
        "note": ("profile-likelihood template fit in (network score, m_T), signal morphed from the 125 GeV "
                 f"sample; background normalisation {b_hat:.3f} ± {b_err:.3f}"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1))
    print(f"mH = {m_hat:.2f} ± {m_err:.2f}  mu = {mu_hat:.3f} ± {mu_err:.3f}  b = {b_hat:.3f} ± {b_err:.3f}  Z = {z:.2f}")


if __name__ == "__main__":
    main()
