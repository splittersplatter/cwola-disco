# Historical Drive study description

This file documents the original Drive scan. Its Windows commands, lambda
defaults, paths and description of its test set are historical; see README.md
and docs/PROTOCOL.md for the current development-only implementation.

# CWoLa + DisCo study

Run ` .\.venv\Scripts\python.exe study.py ` from the project terminal.
The default experiment processes 200,000 raw rows and compares seven configurations
across five seeds and three training SR signal fractions: 105 fits total.
Feature extraction is sequential and cached; completed fits resume automatically.
Use a different `--output` directory when changing settings.

## Scientific question
Can score/mJJ decorrelation reduce background sculpting while retaining sensitivity
to a small resonant signal? A visible bump is not an acceptance criterion.

Configurations: EFP-only MLP; EFP + two jet masses MLP; the same mass/EFP inputs
with HistGradientBoostingClassifier; oracle DisCo and sideband DisCo, each at
lambda=0.1 and 1.0. Lambda=0 is the ordinary mass/EFP MLP. Jet-mass inputs are
m_light and m_heavy-m_light, NOT dijet mass. Degree <=4 remains a computationally
tractable reduced EFP basis; this is not full Kitchen Sink reproduction.

All models train against SR=1 versus SB=0. MLP: 128/64/32 hidden units, ReLU,
Adam lr=0.001, batch size 512, at most 15 epochs, patience 5. Each selects its
checkpoint by validation BCE + lambda times its own validation DisCo penalty.
GBDT: 150 iterations, 15 leaves, L2=1; fixed settings (no test tuning).
Balanced BCE weights depend only on training class counts. DisCo uses uniform
weights on the applicable events, not those classification weights.

DisCo implementation: center both pairwise absolute-distance matrices A and B,
then dCor^2 = mean(A B)/sqrt(mean(A^2) mean(B^2)). This is the biased empirical
estimator; finite-sample values are not expected to vanish under independence.
Training penalty uses up to 128 applicable events per shuffled minibatch.
Validation penalty uses a fixed random subset up to 512 events. Mass is in TeV;
the penalized score is sigmoid(logit). Constant scores have zero penalty, so
signal sensitivity and score standard deviation must also be inspected.

Oracle: penalize truth-background events across the complete window. This uses
simulation labels to define the penalty and is NOT a fully data-driven method.
Sideband: penalize SB events from BOTH sidebands jointly. No truth is used to
mask its loss. This does not guarantee independence inside the SR.

## Events, leakage and injection

Anti-kt R=1; two highest-pt jets; leading pt>1200 GeV; reorder by increasing
jet mass. SR [3300,3700) GeV; SB [3100,3300) and [3700,3900) GeV.
Feature cache version 2 additionally stores both jet-mass features.
Log transforms, standardization and constant removal are fit on training only.

Final test is a fixed stratified 25% subset of eligible events whose original row
IDs are >=50,000, excluding the previously explored development prefix. Remaining
events form development pools; each seed changes train/validation splits and
initialization, but models within a seed share exactly the same samples.
For tiny software smoke tests, the eligible pool falls back to all rows.

Training and checkpoint-validation mixtures contain all background and a random
SR-only signal subset. Fractions 0, 0.005, 0.01 mean S/(B_SR+S), rounded to an
integer count separately for each split. Subsets are nested for the same seed.
Signal outside the SR is intentionally excluded: an idealized injection protocol,
not a model of realistic resonance tails or sideband contamination.
All held-out SR signal is used to estimate efficiencies, regardless of injection
fraction. Therefore evaluation class balance is NOT the training signal abundance.
Zero-injection runs still evaluate on unseen signal as a diagnostic.

Validation sideband score quantiles determine 20% and 5% cuts. These are fixed
before scoring the test set. They target SB acceptance, not exactly 5% SR
background efficiency; actual SR efficiency is always reported. The validation
ROC/signal metrics use simulation truth for evaluation only, not checkpointing.
The predefined lambda grid is reported in full; no best model is picked on test.
Future tuning requires validation or a new final test set.

## Metrics and outputs

`summary.csv`: every seed, fraction, model, cut and evaluation split.
Each run folder: metrics, test/validation predictions and source row IDs,
model weights, preprocessing and MLP loss history.
`tradeoff.png`: test SIC vs JS divergence minus its permutation-null median,
with seed standard deviations. These seeds share a test set and are correlated;
error bars are training/split variability, NOT independent-sample confidence bounds.

Signal AUC and SIC use ONLY the SR. SIC=eps_signal/sqrt(eps_background) is a
sensitivity proxy, not discovery significance. Zero surviving background yields
undefined SIC, not infinity. Very small counts should not support a conclusion.
Sculpting: background score/mass dCor^2 on up to 1000 test events, 16-bin JS
between selected/all background, and a 199-shuffle conditional permutation
reference. A permutation preserves the selected count and randomizes which masses
are selected. The p-value describes a fixed-score cut; multiple tested models/cuts
are exploratory and are NOT corrected discovery claims. Raw dCor^2 is biased.
The subtracted JS metric can be negative from fluctuations; it is a display aid.

Run ` .\.venv\Scripts\python.exe report_study.py ` after study completion for
aggregate tables, feature diagnostics and mass-efficiency comparison plots.

## Validation and boundaries

`python -m unittest test_study`: identical and constant-variable dCor, nonlinear
dependence, gradients, scale/translation invariance, disjoint/fixed test sets and
nested signal injections. A 2000-row smoke study covers every model family.
The experimental run is an initial comparison, not exhaustive optimization.
The first 200k raw rows remain a development subset: assess ordering before a
final publication-style analysis. No formal background fit, bump p-value or
look-elsewhere correction is implemented. No claim of a new particle is made.

References:
- https://arxiv.org/html/2604.20965v1 (Kitchen Sink feature sets and CWoLa setup)
- https://arxiv.org/abs/2001.05310 (DisCo)
- https://energyflow.network/docs/efp/
- https://fastjet.readthedocs.io/en/latest/

The report also writes fixed_background_efficiency.csv: a separate simulation benchmark whose SR background cut is calibrated using validation truth. Actual test background acceptance is reported alongside signal efficiency. This is not the data-driven sideband calibration.

The representative closure plot uses the first configured seed and displays lambda=0.1 for readability; the trade-off plot and tables include every tested strength. ROC/SIC curves are diagnostic test curves, not a basis for choosing thresholds. paired_differences.csv compares each DisCo run to its matched mass/EFP MLP baseline.
