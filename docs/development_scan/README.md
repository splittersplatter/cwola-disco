# Smaller-penalty exploratory scan

This scan ran on 2026-11-17 with the supplied 200,000-row feature cache. It
completed 165/165 fits: five seeds, three injected SR fractions, three baseline
models and eight oracle/sideband DisCo configurations. All 660 expected
seed/fraction/model/cut/split records are present and unique. The report script
checked saved train, validation and test row sets for overlap, and checked the
test rows are fixed across the run. These checks do not verify the raw source or
the original feature extraction.

See [the complete table](REPORT.md), [trade-off plot](tradeoff.png), per-run
[summary](summary.csv), [paired differences](paired_differences.csv) and
[run metadata](run_metadata.json). Mass-efficiency plots are shown for the
first seed and the first configured regularization strength, not all seeds.
The `fixed_background_efficiency.csv` table uses truth-assisted SR calibration
and is a separate simulation diagnostic, not the data-driven sideband cut.

At 1% injection, sideband DisCo 0.03 had mean SR SIC 2.96 ± 0.48 and JS 0.0073,
compared with 2.83 ± 0.21 and 0.0197 for the unregularized mass/EFP MLP. At
0.5% injection the same regularized model had SIC 0.62 ± 1.08 versus 1.79 ± 1.02
for the MLP. These are mean ± seed SD on a shared, previously inspected test set;
they are exploratory, not independent-data confidence intervals or a basis for
declaring a winner. Sideband cuts target 5% sideband, not fixed SR, acceptance.

There is a material reproducibility question: the earlier Drive report gave a
0.5% mass/EFP MLP mean SIC of 2.857 ± 0.334, while this scan gives 1.792 ±
1.024; the GBDT mean at that condition agrees to the displayed precision. The
original and current runs used different recorded NumPy/SciPy versions and
environments, but the cause of the MLP discrepancy has not been established.
For seed 11 at 0.5%, the saved training, checkpoint-validation, test and
validation row arrays match in order, as do preprocessing means, scales and
active-feature masks. The original selected epoch 14 of 15 and had test SIC
2.756; the current run selected epoch 3 and stopped after epoch 8, with SIC
0.021. Their first training losses were 0.69363512 and 0.69363519 respectively.
Thus the discrepancy occurs despite matched splits and preprocessing; it does
not by itself establish whether differences in numerical libraries, hardware,
training stability or another factor caused the divergent training. Compare
weights and other seeds before claiming reproduction.

These outputs contain no individual event predictions or source dataset. The
source-cache hash identifies the supplied feature file, not the original LHCO
HDF5 data. No confirmation model has been selected, and no fresh-data test,
background fit, discovery significance or systematic uncertainty is claimed.
