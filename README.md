# CWoLa anomaly detection with mass decorrelation

An undergraduate ML-for-HEP research project exploring whether distance
correlation (DisCo) can reduce background mass sculpting while preserving
signal sensitivity in weakly supervised CWoLa hunting. It is a reduced EFP
study inspired by [Kitchen Sink Anomaly Detection](https://arxiv.org/abs/2604.20965),
not a full reproduction. The [DisCo method](https://arxiv.org/abs/2001.05310)
is an established technique; its evaluation here is the project application.

This repository starts from the supplied Drive study and its reported 105-fit
exploratory scan. That scan observed a trade-off: the tested regularization
strengths frequently reduced sensitivity. It does not demonstrate an improved
detector, discovery significance, or a fake bump. See
[historical results](docs/EXPLORATORY_RESULTS.md) and
[research protocol](docs/PROTOCOL.md).

## Setup

Use Python 3.11 and install the packages in `requirements.txt` in a virtual
environment. The CPU PyTorch distribution is sufficient for the current
study script. FastJet, EnergyFlow and hdf5plugin are needed when extracting
features from the original LHCO HDF5 file; the feature cache is excluded from
Git. The dataset and generated predictions, models and plots are not shipped.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m unittest test_study
```

`requirements.lock.txt` records the full Linux/Python 3.11 CPU environment used
for the checks; its CPU-specific PyTorch build requires the PyTorch CPU index.
On Windows, activate with `.venv\Scripts\activate`. The locally supplied
feature cache can be used directly:

```bash
python study.py --feature-file data/exploratory_200k.npz --output results/development
python report_study.py --output results/development
```

The default development grid uses five seeds, signal fractions 0, 0.005,
0.01, and DisCo strengths 0.001, 0.003, 0.01 and 0.03 for oracle and
sideband penalties, plus three baselines. This is 165 fits, not a quick run.
For an engineering check, use a separate output directory:

```bash
python study.py --feature-file data/exploratory_200k.npz --events 5000 --epochs 2 --seeds 11 --fractions 0.01 --lambdas 0.01 --permutations 9 --output results/smoke
```

Alternatively provide `--data /path/to/events_anomalydetection.h5`; feature
extraction is sequential and can be slow. Runs bind configuration, source-file
SHA-256, training/extraction source hashes and package versions; changed
settings require a new output directory. A cached feature file's own hash does
not prove the identity or correctness of the original raw LHCO file.

## Scientific boundaries

- The signal region is 3300–3700 GeV; the adjoining sidebands span
  3100–3300 and 3700–3900 GeV. Training targets region membership.
- Oracle DisCo uses simulation truth for its background mask; sideband DisCo
  does not. Sideband decorrelation does not establish closure in the SR.
- All development runs are restricted to original source rows below 200,000,
  which were already explored. Later rows are reserved for a future confirmation
  protocol; this repository does not yet implement a confirmation run.
- Cuts target validation sideband acceptance. Signal and background efficiencies
  and SIC are measured in the held-out SR; actual background acceptance is used.
  SIC is a sensitivity proxy, not discovery significance.
- Seeds share a test sample, so their spread is not an independent-data
  confidence interval. A convincing final analysis needs a fixed selection
  rule, fresh data, uncertainty estimates and a background-model study.

The preexisting methods description is retained as [historical reference](README_study.md);
where commands or test-set labels differ, this README and the protocol describe
the current implementation.
