# Results supplied with the starting study

The Drive report states that 105/105 fits completed using the first 200,000 raw
rows, yielding 69,698 selected events. Five training seeds shared one test set;
the reported standard deviations are not independent-data confidence intervals.
This is a historical summary, not a new result produced by this repository.

At 1% SR training injection and the validation-sideband 5% cut:

| Model | SR SIC, mean ± seed SD | Background JS |
|---|---:|---:|
| EFP-only MLP | 3.088 ± 0.265 | 0.0069 |
| EFP + jet-mass MLP | 2.937 ± 0.171 | 0.0263 |
| EFP + jet-mass GBDT | 2.676 ± 0.112 | 0.0257 |
| Oracle DisCo, lambda 0.1 | 0.881 ± 1.022 | 0.0058 |
| Sideband DisCo, lambda 0.1 | 1.175 ± 0.879 | 0.0037 |

The reported JS permutation-null medians were typically around 0.0045–0.0052.
Actual SR background efficiencies differ because cuts target sideband acceptance.
At 0.5% injection EFP-only MLP was unstable across seeds (SIC 1.189 ± 1.427),
while the mass/EFP MLP gave 2.857 ± 0.334. A low JS can accompany low
sensitivity; sideband decorrelation does not guarantee signal-region closure.

A spot check of seed 55 at 1% injection recomputed SIC from saved predictions:
2.6350 for mass/EFP MLP and 2.1103 for sideband DisCo 0.1. The row sets were
disjoint and matched across these two fits. All 105 expected run-folder names
were present, but every output and the raw data were not independently audited.

[Source report](https://drive.google.com/file/d/1uPK3_giBIBfl6I24duuxeJnUrCqkIiQP/view)
and [source findings](https://drive.google.com/file/d/1GNCpJ8tThuJC5I_d6jNC7Ne2O5p1l4Kq/view).
