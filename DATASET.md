# The dataset

Every dataset, the public one and each sealed one, is a directory with this layout:

```
<dataset_dir>/
  data/events.npz        the data: one row per event, no labels, no weights
  mc/pool.npz            the simulation: signal and background events with labels and weights
  mc/pool_meta.json      what the pool is and what it is normalised to
```

Read an `.npz` with `numpy.load`; it is a dictionary of equal-length arrays, one per column.

## The events

Each event is a proton-proton collision at 13 TeV in which exactly two leptons (electrons or muons) were
reconstructed, with the missing transverse energy and up to two jets. The columns, the same in `data/events.npz`
and `mc/pool.npz`:

| column | type | meaning |
|---|---|---|
| `lep_pt_0`, `lep_eta_0`, `lep_phi_0` | float32 | the leading lepton: transverse momentum [GeV], pseudorapidity, azimuth |
| `lep_pt_1`, `lep_eta_1`, `lep_phi_1` | float32 | the second lepton |
| `lep_charge_0`, `lep_charge_1` | int16 | ±1 |
| `lep_type_0`, `lep_type_1` | int16 | 11 for an electron, 13 for a muon |
| `met_et`, `met_phi` | float32 | missing transverse energy [GeV] and its azimuth |
| `jet_n` | int16 | number of jets |
| `jet_pt_0`, `jet_eta_0`, `jet_phi_0`, `jet_pt_1`, `jet_eta_1`, `jet_phi_1` | float32 | the two leading jets, zero when absent |

Leptons can be taken as massless. The simulation is the ATLAS Open Data release for 13 TeV, reduced to this
table for the HSF-India machine-learning tutorial; the same events are in that tutorial's file. `hww.py`
computes the usual derived quantities (dilepton mass, transverse mass, dilepton pT, opening angle).

## The pool (`mc/pool.npz`)

Three further columns: `label` (1 for signal, 0 for background), `process` (the simulated process, see
`pool_meta.json`), and `weight`. The weights are normalised so that **the sum of weights of a process is the number
of its events expected in the data** at the dataset's luminosity (`pool_meta.json`: `lumi_fb`, 1000 fb⁻¹) for the
nominal rate: at signal strength μ = 1 for the signal, at the nominal normalisation for the background. The
signal is a Higgs boson of mass 125 GeV decaying to two W bosons and then two leptons and two neutrinos
(gluon fusion and vector-boson fusion production); the background is the continuum production of two W
bosons (and ZZ) decaying to two leptons and two neutrinos.

## How a signal of another mass is made

The simulation exists at 125 GeV only. A signal of mass m is defined, for this task, as the 125 GeV sample
with **every lepton transverse momentum and the missing transverse energy multiplied by m / 125**, angles and
jets unchanged. The data were made with this rule; `hww.py` (`morph`) implements it, and templates built with it
are exact. The number of signal events expected at μ = 1 does not depend on the mass.

## The public dataset

`datasets/public/seed_000` has a Higgs of mass **125 GeV** at **μ = 1** and the background at its **nominal
normalisation** (`truth.json`). The sealed datasets have their own planted values, see `PROMPT.md`; they are not
published.
