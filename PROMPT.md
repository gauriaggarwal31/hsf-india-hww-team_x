# The task

Measure how much Higgs boson there is in the data.

The dataset (`DATASET.md`) holds events with two leptons, missing transverse energy and jets. Most are the
continuum production of two W bosons. Some are a Higgs boson decaying to two W bosons and then two leptons and
two neutrinos: the signal, of unknown strength μ, the number of signal events relative to the Standard Model
expectation (μ = 1 is the Standard Model, μ = 0 is no Higgs at all). The background rate is known to about
10 % only.

Beside the data is the simulation: the signal and the background, with labels, and weights normalised to the
expected events in the data. Train on it, build your templates from it, do with it what a real analysis does.

Deliver a pipeline, `run.sh <dataset_dir> <output_dir>`, that reads one dataset and writes
`<output_dir>/RESULT.json` with:

- `mu`, the signal strength, with its uncertainty;
- optionally `mH`, the Higgs mass in GeV, with its uncertainty;
- `significance_obs`, how incompatible the data are with no signal, in standard deviations;
- `discovery`, whether you claim the signal is there.

The format, the rules and how it is scored: `EVAL_CONTRACT.md`. The public dataset has μ = 1 and the nominal
background; the sealed datasets, on which the pipeline is run unchanged, have a signal strength between 0 and 3
or no signal at all and a background normalisation within 10 % of nominal, and nothing else about them is
promised. A pipeline that assumes the public values will be wrong on them.
