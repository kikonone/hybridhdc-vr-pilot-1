# Phase 09 Statistical Appendix

- Statistical unit: subject_id (n=35).
- Test: paired Wilcoxon signed-rank for each missing condition versus its own Full Primary reference.
- Multiplicity: Holm correction separately within each of four model-task families (five comparisons per family).
- Effect size: signed rank-biserial; positive values indicate worse performance under modality removal after harmonizing metric direction.
- Uncertainty: 2,000 deterministic paired subject bootstrap resamples, 95% percentile CI.
- All comparisons, including non-significant ones, are present in `phase09_pairwise_statistics.csv`.
- LOSO summaries use 2,000 subject bootstraps and are descriptive stability estimates; LOSO splits are not independent samples for a second inferential test.

Flight comparisons:
- hdc_classification: degradation=0.518092, Holm p=2.91038e-10, effect=1.000
- traditional_classification: degradation=0.510262, Holm p=2.91038e-10, effect=1.000
- hdc_regression: degradation=0.590627, Holm p=2.91038e-10, effect=1.000
- traditional_regression: degradation=0.718438, Holm p=2.91038e-10, effect=1.000
