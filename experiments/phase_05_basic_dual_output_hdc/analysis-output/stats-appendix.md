# Phase 05 Statistical Appendix

## Descriptive design
- 20 preregistered configurations: four dimensions × five seeds.
- Each configuration has exactly 419 subject-wise OOF predictions.
- Dimension summaries report mean, sample SD (`ddof=1`), minimum, and maximum over five seed-level OOF metrics.
- OOF metrics were recomputed directly from 8,380 aligned configuration-run rows and cross-checked against 100 fold-metric blocks.

## Inferential boundary
The frozen plan did not preregister a significance test for dimension, seed, or comparison with Phase 04. No p-values, confidence intervals, effect-size tests, or multiple-comparison procedures were added post hoc. Baseline comparisons are descriptive and do not support “significantly better” wording.

## Repeated-measure caution
The same 419 runs appear once per preregistered configuration. These repeated predictions are not treated as 8,380 independent experimental observations. Seeds are summarized as a stability distribution, not averaged into an ensemble prediction.
