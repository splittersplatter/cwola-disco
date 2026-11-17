# Study protocol and next steps

## Objective

Measure the trade-off between signal sensitivity and background mass distortion
for an EFP-based CWoLa classifier. A visually prominent bump is not a target.

## Historical exploratory scan

The supplied study trained EFP-only MLP, EFP plus jet-mass MLP, matched-feature
gradient-boosted trees, and oracle/sideband DisCo MLPs for five seeds and three
injections. It used a fixed test set within the first 200,000 source rows; the
test results have been inspected and can no longer be confirmation data.
See [the saved summary](EXPLORATORY_RESULTS.md).

## Current development experiment

Use the existing prefix as development material, including its already inspected
test set. The default lambda grid explores smaller strengths 0.001, 0.003, 0.01,
0.03, with ordinary MLP (lambda zero), EFP-only MLP and boosted trees retained
as references. Compare matched event splits and report all grid points. The
code forbids rows 200,000 or above. Tiny `--events` runs are engineering checks,
not adequate physics evidence. All runs need a separate output directory if
configuration, inputs, training code or environment changes.

The saved cache is useful for development but its recorded source provenance is
path, size and modification time, not a verified original-file checksum. Before
final use, verify the original LHCO data and recompute or independently validate
features and row identity. Sampling the first rows can introduce ordering bias.

## Confirmation is not yet implemented

Before touching additional raw rows, specify and record a criterion that balances
sensitivity and distortion; choose a single sideband penalty strength using only
development/validation information. Fix the background calibration, sample
definitions, injection protocol, metric and treatment of low counts. Reserve
fresh source rows not used by this or prior studies, document their provenance
and check source ordering and overlap. Preserve a separate test pool throughout
training and selection. Independently estimate uncertainty, including shared
events and threshold calibration; one seed's result is insufficient.

Evaluate a null background control and nonzero injection conditions; compare
the fixed model with the matched zero-penalty MLP, EFP ablation and boosted-tree
reference. Oracle penalties provide a truth-assisted benchmark only. Retain
realistic sideband signal leakage as a separately specified robustness study.
Use mass-binned background efficiency and a finite-sample reference, as well as
signal efficiency and SIC with survivor counts. The current idealized injection
excludes signal from sidebands and lacks a formal bump fit, systematic
uncertainties and discovery significance.
