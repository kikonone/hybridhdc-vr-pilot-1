# Phase 09 Figure Catalog

## phase09_missing_modality_classification_curve
- Purpose: Compare classification robustness across five missing-modality conditions against Full Primary.
- Data source: subject-level paired metrics
- Uncertainty: 95% subject bootstrap CI, n=35
- Reader should notice: Axes start at zero; metric direction is explicit.
- Caveat: Modality removal is predictive dependence evidence, not causal evidence.

## phase09_missing_modality_regression_curve
- Purpose: Compare regression robustness across five missing-modality conditions against Full Primary.
- Data source: subject-level paired metrics
- Uncertainty: 95% subject bootstrap CI, n=35
- Reader should notice: Axes start at zero; metric direction is explicit.
- Caveat: Modality removal is predictive dependence evidence, not causal evidence.

## phase09_missing_modality_model_comparison
- Purpose: Compare HDC and traditional subject-paired degradation.
- Data source: phase09_missing_modality_robustness.csv
- Uncertainty: 95% paired subject bootstrap CI, n=35
- Reader should notice: Positive values mean worse performance for both task directions.
- Caveat: Between-model contrast is descriptive; preregistered inference compares each condition to its own Full Primary reference.

## phase09_loso_subject_classification
- Purpose: Show held-out-subject classification heterogeneity.
- Data source: phase09_loso_subject_metrics.csv
- Uncertainty: No per-subject CI; each point is one held-out subject, n=35
- Reader should notice: All subjects are retained, including high-error subjects.
- Caveat: LOSO supports subject generalization only, not unseen-scenario generalization.

## phase09_loso_subject_regression
- Purpose: Show held-out-subject regression heterogeneity.
- Data source: phase09_loso_subject_metrics.csv
- Uncertainty: No per-subject CI; each point is one held-out subject, n=35
- Reader should notice: All subjects are retained, including high-error subjects.
- Caveat: LOSO supports subject generalization only, not unseen-scenario generalization.

## phase09_loso_stability_distribution
- Purpose: Summarize the distribution of subject-level primary metrics.
- Data source: phase09_loso_subject_metrics.csv
- Uncertainty: Box/IQR distribution across n=35 subjects
- Reader should notice: Classification and regression have different metric directions, stated in the companion report.
- Caveat: The panel is descriptive and should not be read as cross-task scale equivalence.
