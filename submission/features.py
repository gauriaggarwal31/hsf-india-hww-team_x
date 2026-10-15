"""The event features for the signal/background classifier, in one place so that the offline
trainer (train.py) and the run-time pipeline (fit.py) build exactly the same matrix.

Numpy only: fit.py imports this at run time, where the allowed packages are numpy, scipy and
torch.  The derived kinematics (m_ll, m_T, p_T^ll, delta phi) do most of the separation; the
lepton flavours and charges (the same-flavour, opposite-sign pairs carry the ZZ part of the
background), the jets and the per-lepton transverse masses add a few per cent.
"""
import numpy as np

import hww as H

FEATURES = [
    "lep_pt_0", "lep_pt_1", "met_et", "dphi", "mll", "ptll", "mt", "jet_n", "deta",
    "same_flavour", "opp_sign", "dphi_met_lep", "pt_sum", "jet_pt_0", "mll_over_mt",
    "ptll_over_mt", "mt_l0", "mt_l1", "dphi_ll_met", "abs_eta_0", "abs_eta_1",
]


def _dphi(a, b):
    return np.abs(np.mod(a - b + 3.0 * np.pi, 2.0 * np.pi) - np.pi)


def features(ev):
    """The feature matrix of a set of events, in the order of FEATURES."""
    k = H.kinematics(ev)
    f = lambda c: np.asarray(ev[c], float)
    pt0, pt1, phi0, phi1 = f("lep_pt_0"), f("lep_pt_1"), f("lep_phi_0"), f("lep_phi_1")
    eta0, eta1 = f("lep_eta_0"), f("lep_eta_1")
    met, mphi = f("met_et"), f("met_phi")
    d0, d1 = _dphi(mphi, phi0), _dphi(mphi, phi1)
    px = pt0 * np.cos(phi0) + pt1 * np.cos(phi1)
    py = pt0 * np.sin(phi0) + pt1 * np.sin(phi1)
    cols = {
        "lep_pt_0": pt0, "lep_pt_1": pt1, "met_et": met,
        "dphi": k["dphi"], "mll": k["mll"], "ptll": k["ptll"], "mt": k["mt"],
        "jet_n": f("jet_n"),
        "deta": np.abs(eta0 - eta1),
        "same_flavour": (ev["lep_type_0"] == ev["lep_type_1"]).astype(float),
        "opp_sign": (f("lep_charge_0") * f("lep_charge_1") < 0).astype(float),
        "dphi_met_lep": np.minimum(d0, d1),
        "pt_sum": pt0 + pt1,
        "jet_pt_0": f("jet_pt_0"),
        "mll_over_mt": k["mll"] / np.maximum(k["mt"], 1.0),
        "ptll_over_mt": k["ptll"] / np.maximum(k["mt"], 1.0),
        "mt_l0": np.sqrt(np.clip(2.0 * pt0 * met * (1.0 - np.cos(d0)), 0.0, None)),
        "mt_l1": np.sqrt(np.clip(2.0 * pt1 * met * (1.0 - np.cos(d1)), 0.0, None)),
        "dphi_ll_met": _dphi(np.arctan2(py, px), mphi),
        "abs_eta_0": np.abs(eta0), "abs_eta_1": np.abs(eta1),
    }
    return np.column_stack([cols[n] for n in FEATURES])
