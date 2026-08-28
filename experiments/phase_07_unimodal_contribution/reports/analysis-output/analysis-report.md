# Phase 07 analysis report

## Analysis question
Which frozen unimodal feature group contributes the strongest out-of-fold predictive performance, separately for classification and bounded difficulty-induced workload proxy regression, and how does each compare with the frozen multimodal reference?

## Data and comparison unit
The primary analysis retains all 419 OOF runs for each model. Inferential analyses use `subject_id` as the paired statistical unit (n=35); runs, folds, and seeds are not treated as independent samples.

## Classification modality ranking
| modality                  |   macro_f1 |   balanced_accuracy |   accuracy |   severe_error_rate |   seed_macro_f1_mean |   seed_macro_f1_sample_sd |   canonical_rows |   rank |
|:--------------------------|-----------:|--------------------:|-----------:|--------------------:|---------------------:|--------------------------:|-----------------:|-------:|
| flight_parameter_features |   0.863458 |            0.864093 |   0.863962 |           0.0119332 |             0.823377 |                 0.0127511 |              419 |      1 |
| head_movement_features    |   0.373296 |            0.375151 |   0.374702 |           0.23389   |             0.336912 |                 0.0246709 |              419 |      2 |
| eye_tracking_features     |   0.346833 |            0.351635 |   0.350835 |           0.238663  |             0.345778 |                 0.0129414 |              419 |      3 |
| physiological_features    |   0.33219  |            0.340745 |   0.341289 |           0.312649  |             0.279611 |                 0.0185982 |              419 |      4 |
| body_movement             |   0.228379 |            0.245077 |   0.245823 |           0.331742  |             0.262964 |                 0.0158431 |              419 |      5 |

## Regression modality ranking
| modality                  |   bounded_mae |   raw_mae |   bounded_rmse |   bounded_r2 |   bounded_spearman |   clipping_count |   clipping_rate |   seed_bounded_mae_mean |   seed_bounded_mae_sample_sd |   canonical_rows |   rank |
|:--------------------------|--------------:|----------:|---------------:|-------------:|-------------------:|-----------------:|----------------:|------------------------:|-----------------------------:|-----------------:|-------:|
| flight_parameter_features |      0.26111  |  0.279536 |       0.389104 |    0.878647  |           0.931247 |               54 |       0.128878  |                0.273894 |                    0.0080325 |              419 |      1 |
| eye_tracking_features     |      0.847353 |  0.861517 |       1.057    |    0.104487  |           0.426285 |               26 |       0.0620525 |                0.866261 |                    0.0103395 |              419 |      2 |
| head_movement_features    |      0.890777 |  0.905127 |       1.09876  |    0.0323322 |           0.354278 |               19 |       0.0453461 |                0.918626 |                    0.0198197 |              419 |      3 |
| physiological_features    |      1.01079  |  1.0586   |       1.29195  |   -0.337857  |           0.175152 |               52 |       0.124105  |                1.04833  |                    0.0262184 |              419 |      4 |
| body_movement             |      1.02607  |  1.0578   |       1.26688  |   -0.286446  |           0.127774 |               43 |       0.102625  |                1.06448  |                    0.0236965 |              419 |      5 |

## Multimodal deltas
| task           | modality                  |   unimodal_metric |   multimodal_metric | metric      |       delta |
|:---------------|:--------------------------|------------------:|--------------------:|:------------|------------:|
| classification | physiological_features    |          0.33219  |            0.871057 | macro_f1    | -0.538867   |
| classification | eye_tracking_features     |          0.346833 |            0.871057 | macro_f1    | -0.524224   |
| classification | head_movement_features    |          0.373296 |            0.871057 | macro_f1    | -0.497762   |
| classification | flight_parameter_features |          0.863458 |            0.871057 | macro_f1    | -0.00759957 |
| classification | body_movement             |          0.228379 |            0.871057 | macro_f1    | -0.642678   |
| regression     | physiological_features    |          1.01079  |            0.265727 | bounded_mae |  0.745061   |
| regression     | eye_tracking_features     |          0.847353 |            0.265727 | bounded_mae |  0.581626   |
| regression     | head_movement_features    |          0.890777 |            0.265727 | bounded_mae |  0.625049   |
| regression     | flight_parameter_features |          0.26111  |            0.265727 | bounded_mae | -0.00461743 |
| regression     | body_movement             |          1.02607  |            0.265727 | bounded_mae |  0.760342   |

## Stability and uncertainty
Seed stability is reported separately from canonical OOF metrics. Canonical metrics are recomputed after prediction aggregation. Subject-level percentile bootstrap intervals use 2,000 shared resamples (seed 42).

## Missing-modality diagnostics
Availability-stratified results retain the same frozen predictions and do not replace the 419-row main ranking. Eye tracking has 14 fully missing rows and body movement has 29; other modalities have none.

## Strongest evidence
The top classification modality is **flight_parameter_features** (Macro-F1 0.863458); the top regression modality is **flight_parameter_features** (bounded MAE 0.261110). These are predictive comparisons under the frozen evaluation, not causal physiological effects.

## Limitations
Subject-level estimates are based on 35 subjects; the study uses a bounded proxy target, and missingness strata can be small. Multiple-comparison-corrected tests and effect sizes must be read together with uncertainty intervals.

## Claim candidates

### Separate task leaders
- Source evidence: canonical OOF rankings and subject-bootstrap intervals.
- Allowed wording: “Under the frozen Phase 07 protocol, flight_parameter_features ranked first for classification and flight_parameter_features ranked first for bounded difficulty-induced workload proxy regression.”
- Forbidden stronger wording: “One modality is universally best” or any causal interpretation.
- Uncertainty: n=35 subjects; rankings are task-specific.
- Decision: Allowed with the stated scope.

### Multimodal comparison
- Source evidence: read-only canonical references, paired Wilcoxon-Holm tests, and rank-biserial effects.
- Allowed wording: describe the measured direction and magnitude, with corrected p-value and effect size.
- Forbidden stronger wording: “significantly best” without jointly supportive preregistered corrected test and effect evidence.
- Uncertainty: paired subject sample and bounded target.
- Decision: Conditional; cite the statistical appendix.
