# Phase 05 Strict Analysis Report

## Analysis question
How do the four preregistered Vanilla HDC dimensions behave across five preregistered seeds for classification, similarity-based bounded regression, Ridge-readout bounded regression, and measured efficiency under the frozen 419-run subject-wise OOF protocol?

## Evidence boundary
- Unit of stability summary: seed-level OOF metric (`n=5` seeds per dimension).
- Error bars: sample SD across seeds.
- No seed ensemble, post-hoc canonical configuration, inferential significance test, or outer-test tuning.
- Phase 04 comparisons are descriptive; compatibility audit: `PASS`.

## Key findings
- Highest observed mean classification Macro-F1 in the preregistered matrix: D=5,000, 0.779764 ± 0.011622 SD.
- Lowest observed mean similarity-regression bounded MAE: D=2,000, 0.699930 ± 0.008131 SD.
- Lowest observed mean Ridge-readout bounded MAE: D=10,000, 0.276390 ± 0.006419 SD.
- These are descriptive observations across the complete preregistered matrix, not new model-selection decisions.

## Claim Candidates
- Claim: Vanilla HDC performance varies with dimension but is observable across all five preregistered seeds.
  - Source evidence: dimension-level mean, sample SD, min, and max tables.
  - Allowed wording: descriptive differences and stability across the preregistered matrix.
  - Forbidden stronger wording: statistically significant superiority or a selected canonical dimension.
  - Uncertainty: five seeds; no preregistered inferential contrast.
  - Next check: use the complete matrix in the next planned analysis without selecting from outer-test performance.
  - Decision: keep
- Claim: The two regression heads estimate a bounded difficulty-induced workload proxy.
  - Source evidence: 419-run OOF predictions for 20 configurations per head.
  - Allowed wording: bounded difficulty-induced workload proxy regression.
  - Forbidden stronger wording: directly measured continuous cognitive workload.
  - Uncertainty: target has four difficulty-derived values.
  - Next check: preserve this interpretation in Phase 06.
  - Decision: keep
