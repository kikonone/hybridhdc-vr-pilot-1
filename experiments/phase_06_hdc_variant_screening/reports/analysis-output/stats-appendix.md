# Phase 06 Statistics Appendix

## Units and sample structure

- Inferential unit: subject (35).
- Prediction unit: run (419 per configuration).
- Outer folds: 5 partition units; not treated as independent samples.
- Seeds: 5 registered repeats; not treated as independent subjects.
- Dimensions: 1000, 2000, 5000, 10000.

## Descriptive statistics

For every variant × dimension, the bundle reports count, mean, sample SD, median, minimum, and maximum across the five registered seeds for classification and regression metrics. No seed predictions are averaged into a new ensemble, no seed is deleted, and no single seed is selected.

## Inferential analysis

The frozen `phase06_model_selection_rules_v1` does not preregister a subject-level paired bootstrap, permutation test, effect-size estimator, confidence-interval method, or multiple-comparison correction for the final four-model comparison. No inferential test, effect size, CI, or multiplicity-adjusted p-value was added post hoc. Inferential analysis is deferred until a future statistical protocol is frozen.

## Metric audit

Metrics were independently recomputed from OOF prediction rows. Phase 05 OOF metrics and Phase 06 saved fold metrics were cross-checked at maximum absolute tolerance 1e-12.

## Limitations

- Seed SD describes algorithmic variability, not subject-level uncertainty.
- OOF predictions support aligned descriptive comparison but a future selection on these scores induces optimism.
- Peak memory is incomplete across variants, limiting Pareto analysis to performance and time.
- Runtime instrumentation differs in field granularity between Phase 05 and the new variants; all missing fields remain `NOT_AVAILABLE`.

<!-- PHASE06_SELECTION_RESOLUTION_V2 -->
## Post-freeze selection amendment scope

The Phase 06 canonical model-selection amendment was defined after final-confirmation artifacts existed, but its executable selector was restricted to previously saved inner-CV and unlabeled efficiency evidence. Outer-OOF artifacts were hash-sealed before selection and were not read by the selector.

Selection summaries aggregate three inner folds within each outer-training task, then weight the five outer-training task means equally. Folds and seeds are not treated as independent subjects. The regression RMSE tie-break was unavailable for Phase 05 saved inner records and was not invoked because the winner was already unique under mean bounded MAE and its across-outer-fold sample SD. Runtime has heterogeneous recorded scope between Phase 05 and Phase 06 and is only a late deterministic tie-break. Outer-OOF values are descriptive; no new inferential test or claim of significant superiority is made. Selection-induced optimism and the post-freeze timing require confirmation by a future fixed-procedure LOSO analysis.
