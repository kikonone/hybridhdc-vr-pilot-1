# Phase 03 Notes

## Verified Phase 02 prerequisites
- Corrected source: `full_multimodal_run_level_features_corrected_v1.csv`.
- Corrected table: 487 rows, 35 subjects, 487 unique run keys, and 1,247 canonical feature columns.
- Verified metadata: 42 body-movement features, 59 performance features, zero explicit control-input features, and 12 structurally unusable all-NaN features.
- Expected task cohort after excluding `difficulty_level = 0`: 419 rows across difficulty levels 1 to 4.

## Scope boundary
- This phase performs cohort construction, target definition, audit, and frozen-fold creation only. It does not train models, impute, scale, or fit preprocessing.
