# Phase 09 Strict Analysis Report

## Analysis question
How do five frozen modality-removal retraining conditions alter classification Macro-F1 and bounded MAE relative to frozen Full Primary references, and how stable are the four selected model-task interfaces across 35 held-out subjects?

## Evidence contract
- Unit of inference: 35 subjects.
- Primary metrics: Macro-F1 (higher is better) and bounded MAE (lower is better).
- Missing-modality inference: paired Wilcoxon, Holm within each model-task family of five comparisons, rank-biserial effect size, and 2,000 paired subject bootstraps.
- LOSO is a subject-generalization estimate, not scenario generalization.

## Key findings
- hdc_classification: MISSING_FLIGHT_PARAMETER (mean degradation 0.518092)
- hdc_regression: MISSING_FLIGHT_PARAMETER (mean degradation 0.590627)
- traditional_classification: MISSING_FLIGHT_PARAMETER (mean degradation 0.510262)
- traditional_regression: MISSING_FLIGHT_PARAMETER (mean degradation 0.718438)

Flight-feature evidence:
- hdc_classification: degradation=0.518092, Holm p=2.91038e-10, effect=1.000
- traditional_classification: degradation=0.510262, Holm p=2.91038e-10, effect=1.000
- hdc_regression: degradation=0.590627, Holm p=2.91038e-10, effect=1.000
- traditional_regression: degradation=0.718438, Holm p=2.91038e-10, effect=1.000

## Claim Candidates
- Claim: modality removal changes predictive performance relative to the frozen Full Primary reference.
  - Source evidence: `phase09_missing_modality_robustness.csv` and `phase09_pairwise_statistics.csv`.
  - Allowed wording: model dependence on the removed feature family, with direction and uncertainty stated.
  - Forbidden stronger wording: physiological causality or universal behavioral importance.
  - Uncertainty: n=35 subjects; no unseen-session/scenario metadata.
  - Next check: a metadata-supported scenario/session study.
  - Decision: keep with boundary.
- Claim: flight-feature dependence may generalize across held-out subjects.
  - Source evidence: missing-flight paired analysis plus LOSO subject stability.
  - Allowed wording: `SUBJECT_GENERALIZATION_OF_FLIGHT_DEPENDENCE` when descriptive stability supports it.
  - Forbidden stronger wording: `GENERALIZABLE_FLIGHT_BEHAVIOR` or unseen-scenario generalization.
  - Uncertainty: session, scenario, task-template, and route identifiers are unavailable.
  - Next check: collect the missing metadata and pre-register grouped generalization splits.
  - Decision: weaken; flight generalizable-behavior remains inconclusive.

## Limitations
Small modality effects may partly reflect feature-count differences; this analysis cannot isolate causal information content from dimensionality. Non-significant Wilcoxon results are not equivalence evidence. No subject was removed or used for reselection.
