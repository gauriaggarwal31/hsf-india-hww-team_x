# EVAL_CONTRACT v0.1-hww — submission contract for `hww`

This file is part of the input given to the analysis agent.  Everything below
is checked mechanically.

## 1. Submission form

A submission is a directory (ideally a git repository) whose root contains an
executable `run.sh`.  The harness freezes the directory into an immutable
snapshot and executes, from a clean unpack, for every sealed dataset:

    ./run.sh <dataset_dir> <output_dir>

- `<dataset_dir>` has exactly the layout of the public dataset (DATASET.md).
- `<output_dir>` exists and is empty; write `RESULT.json` there (GATE).
  Output files are read only as regular files: a symlink there is never
  followed, and a symlinked `RESULT.json` counts as malformed.
- Exit code 0 signals success.  Non-zero exit, timeout, or a malformed
  `RESULT.json` scores the dataset as failed and counts against recovery.
- Repeated execution on the same dataset must reproduce `RESULT.json`
  (fix your random seeds).

Hygiene, checked before freezing: no data files (`.npy`, `.npz`, `.root`,
`.parquet`, …), no committed outputs or truth files, no symlinks, ≤ 5 MB per
file, ≤ 50 MB total.  A trained model travels as text (JSON weights, a few
hundred kB at most), with the code that evaluates it.
A `pylib/` directory of vendored packages is exempt from the two size caps
(own budget 4 GB); the data-file and truth-name rules apply inside it too.

Runtime: Python 3 with `numpy` and `scipy` importable, nothing else; no
network; the pipeline may read only `<dataset_dir>` and its own directory.
Limits per dataset: 600 s wall clock, 6 GB memory.

## 2. RESULT.json (result type `search`)

    {
      "pois": [
        {"name": "mu", "value": f, "stat": f, "syst": f, "total": f},
        {"name": "mH", "value": f, "stat": f, "syst": f, "total": f}    # optional
      ],
      "significance_obs": f,          # local, in sigma; 0 if no excess
      "discovery": bool,              # your discovery claim (e.g. Z >= 5)
      "nuisances": {                  # optional
        "hww_bkg_norm": {"value": f, "unc": f}
      }
    }

`mu` relative to the Standard Model expectation; `mH`, optional, in GeV.  All values
finite; `total` > 0 and consistent with `stat ⊕ syst`.  Nuisance names follow
the grammar of `sbh/hep/names.py`; this task's own are `hww_<label>`:
`hww_bkg_norm` (the background normalisation relative to nominal).  Extra keys
are allowed and ignored.

## 3. How you are scored (aggregate metrics only)

Across the sealed suite, for `mu` (and for `mH` where you report it on every scored dataset: a bonus,
never a penalty):

- **sensitivity** — slope of recovered vs injected value (unit slope = faithful
  measurement; a constant output scores zero).  Gate.
- **bias** — mean pull `(value − truth) / total`.
- **calibration** — width of the pull distribution using YOUR quoted `total`;
  both under- and over-coverage fail.
- **precision** — `min(1, σ_ref / total)`, awarded only if calibration passes
  (σ_ref is the reference pipeline's uncertainty).
- **recovery** — fraction of datasets that complete with a valid result.
- **false-discovery rate** — fraction of no-signal datasets on which you claim
  discovery.

- **discovery power** — fraction of strong-signal datasets on which you claim
  discovery.

`S_phys = recovery × mean_POI(sensitivity_gate × calibration_gate × precision)`.
Pulls and slopes use only datasets whose injected signal has expected local
significance ≥ 5 (a truth-based rule, the same for every submission); null
datasets enter the false-discovery rate.  You receive these aggregates; never
per-dataset truths or per-dataset results.
