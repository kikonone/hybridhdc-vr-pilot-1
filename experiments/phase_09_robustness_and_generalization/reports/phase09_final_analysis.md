---
type: results-report
date: 2026-08-21
experiment_line: phase09-robustness-generalization
round: 9
purpose: robustness-check
status: complete-pending-freeze
source_artifacts:
  - reports/analysis-output/analysis-report.md
  - reports/analysis-output/stats-appendix.md
linked_experiments: []
linked_results: []
---

# Phase 09 Robustness and Generalization / Round 9 / Robustness Check / 2026-08-21

## Executive Summary
Phase 09 consolidated 720 frozen runs into 10,056 canonical OOF rows and evaluated missing-modality robustness and held-out-subject stability without retraining. The strongest allowable conclusion is predictive dependence on specific feature families, bounded by subject-level uncertainty and missing scenario/session metadata.

## Experiment Identity and Decision Context
This round tests whether frozen HDC and traditional interfaces remain useful after removal of one modality and when evaluated on a wholly held-out subject.

## Setup and Evaluation Protocol
Five missing-modality conditions, four model-task interfaces, 419 runs, five HDC seeds, and 35 subjects were analyzed. Classification uses Macro-F1; the bounded difficulty-induced workload proxy regression uses bounded MAE. Inference uses subjects only.

## Main Findings
- hdc_classification: MISSING_FLIGHT_PARAMETER (mean degradation 0.518092)
- hdc_regression: MISSING_FLIGHT_PARAMETER (mean degradation 0.590627)
- traditional_classification: MISSING_FLIGHT_PARAMETER (mean degradation 0.510262)
- traditional_regression: MISSING_FLIGHT_PARAMETER (mean degradation 0.718438)

## Statistical Validation
Paired Wilcoxon tests, Holm correction, rank-biserial effect sizes, and 2,000 paired subject bootstrap CIs are saved in the statistical artifacts. Non-significance is not treated as equivalence.

## Figure-by-Figure Interpretation
- `phase09_missing_modality_classification_curve.pdf/.png`: Compare classification robustness across five missing-modality conditions against Full Primary. Axes start at zero; metric direction is explicit. Caveat: Modality removal is predictive dependence evidence, not causal evidence.
- `phase09_missing_modality_regression_curve.pdf/.png`: Compare regression robustness across five missing-modality conditions against Full Primary. Axes start at zero; metric direction is explicit. Caveat: Modality removal is predictive dependence evidence, not causal evidence.
- `phase09_missing_modality_model_comparison.pdf/.png`: Compare HDC and traditional subject-paired degradation. Positive values mean worse performance for both task directions. Caveat: Between-model contrast is descriptive; preregistered inference compares each condition to its own Full Primary reference.
- `phase09_loso_subject_classification.pdf/.png`: Show held-out-subject classification heterogeneity. All subjects are retained, including high-error subjects. Caveat: LOSO supports subject generalization only, not unseen-scenario generalization.
- `phase09_loso_subject_regression.pdf/.png`: Show held-out-subject regression heterogeneity. All subjects are retained, including high-error subjects. Caveat: LOSO supports subject generalization only, not unseen-scenario generalization.
- `phase09_loso_stability_distribution.pdf/.png`: Summarize the distribution of subject-level primary metrics. Classification and regression have different metric directions, stated in the companion report. Caveat: The panel is descriptive and should not be read as cross-task scale equivalence.

## Failure Cases / Negative Results / Limitations
High-error and seed-unstable subjects remain in the analysis. Feature-count differences can confound the apparent importance of small modalities. No unseen-session/scenario/task-template/route test is feasible from current metadata.

## What Changed Our Belief
The evidence can update beliefs about model dependence and held-out-subject stability, but not about physiological causality or unseen-scenario flight behavior.

## Next Actions
Freeze Phase 09 only after independent review of this bundle. A future phase should add explicit session, scenario, task-template, and route metadata before claiming broader generalization.

## Artifact and Reproducibility Index
- Canonical OOF: `results/oof/`
- Metrics/statistics: `results/summaries/`
- Figures: `figures/`
- Strict analysis bundle: `reports/analysis-output/`
- Audits: `audits/`
