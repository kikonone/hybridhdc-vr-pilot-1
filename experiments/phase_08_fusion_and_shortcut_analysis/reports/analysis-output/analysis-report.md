# Phase 08 Strict Analysis Report

## Analysis question
Do frozen early-fusion, performance-feature, and flight-provenance conditions change subject-wise performance under the preregistered comparison families?

## Evidence contract
- Unit: 35 paired subjects.
- Classification primary metric: subject-level Macro-F1 (higher is better).
- Regression primary metric: subject-level bounded MAE (lower is better).
- Inference: Wilcoxon signed-rank, Holm within family/model/task, rank-biserial effect size, 2,000 paired-subject bootstrap samples (seed 42).


Classification metrics:

| condition                             | model_family   | source_status           |   macro_f1 |   balanced_accuracy |   accuracy |   severe_error_rate |
|:--------------------------------------|:---------------|:------------------------|-----------:|--------------------:|-----------:|--------------------:|
| BEST_SINGLE_FLIGHT_REFERENCE          | HDC            | REUSED_FROZEN_REFERENCE |   0.863458 |            0.864093 |   0.863962 |          0.0119332  |
| FLIGHT_BEHAVIORAL_ONLY                | HDC            | NEW_PHASE08_RUN         |   0.854351 |            0.854387 |   0.854415 |          0.0143198  |
| FLIGHT_BEHAVIORAL_ONLY                | TRADITIONAL    | NEW_PHASE08_RUN         |   0.94508  |            0.945302 |   0.945107 |          0.00477327 |
| FLIGHT_FULL                           | HDC            | REUSED_FROZEN_REFERENCE |   0.863458 |            0.864093 |   0.863962 |          0.0119332  |
| FLIGHT_FULL                           | TRADITIONAL    | NEW_PHASE08_RUN         |   0.952313 |            0.952423 |   0.952267 |          0.00477327 |
| FULL_PRIMARY_REFERENCE                | HDC            | REUSED_FROZEN_REFERENCE |   0.871057 |            0.871236 |   0.871122 |          0.0190931  |
| FULL_PRIMARY_REFERENCE                | TRADITIONAL    | REUSED_FROZEN_REFERENCE |   0.935608 |            0.935686 |   0.935561 |          0.0143198  |
| FUSION_PE                             | HDC            | NEW_PHASE08_RUN         |   0.355564 |            0.358732 |   0.357995 |          0.250597   |
| FUSION_PE                             | TRADITIONAL    | NEW_PHASE08_RUN         |   0.430086 |            0.43511  |   0.434368 |          0.214797   |
| FUSION_PEH                            | HDC            | NEW_PHASE08_RUN         |   0.41833  |            0.423115 |   0.422434 |          0.221957   |
| FUSION_PEH                            | TRADITIONAL    | NEW_PHASE08_RUN         |   0.449249 |            0.451414 |   0.451074 |          0.21957    |
| FUSION_PEHF                           | HDC            | NEW_PHASE08_RUN         |   0.861271 |            0.861666 |   0.861575 |          0.0190931  |
| FUSION_PEHF                           | TRADITIONAL    | NEW_PHASE08_RUN         |   0.937978 |            0.938045 |   0.937947 |          0.0143198  |
| PERFORMANCE_ONLY_AUXILIARY            | HDC            | NEW_PHASE08_RUN         |   0.557493 |            0.558537 |   0.558473 |          0.102625   |
| PERFORMANCE_ONLY_AUXILIARY            | TRADITIONAL    | NEW_PHASE08_RUN         |   0.716667 |            0.718664 |   0.718377 |          0.107399   |
| WITHOUT_PERFORMANCE_PRIMARY_REFERENCE | HDC            | REUSED_FROZEN_REFERENCE |   0.871057 |            0.871236 |   0.871122 |          0.0190931  |
| WITHOUT_PERFORMANCE_PRIMARY_REFERENCE | TRADITIONAL    | REUSED_FROZEN_REFERENCE |   0.935608 |            0.935686 |   0.935561 |          0.0143198  |
| WITH_PERFORMANCE_AUXILIARY            | HDC            | NEW_PHASE08_RUN         |   0.841979 |            0.842436 |   0.842482 |          0.0238663  |
| WITH_PERFORMANCE_AUXILIARY            | TRADITIONAL    | NEW_PHASE08_RUN         |   0.940243 |            0.940449 |   0.940334 |          0.0119332  |

Regression metrics:

| condition                             | model_family   | source_status           |   bounded_mae |   bounded_rmse |   bounded_r2 |   bounded_spearman |   clipping_rate |
|:--------------------------------------|:---------------|:------------------------|--------------:|---------------:|-------------:|-------------------:|----------------:|
| BEST_SINGLE_FLIGHT_REFERENCE          | HDC            | REUSED_FROZEN_REFERENCE |     0.26111   |       0.389104 |    0.878647  |           0.931247 |      0.128878   |
| FLIGHT_BEHAVIORAL_ONLY                | HDC            | NEW_PHASE08_RUN         |     0.26111   |       0.389104 |    0.878647  |           0.931247 |      0.128878   |
| FLIGHT_BEHAVIORAL_ONLY                | TRADITIONAL    | NEW_PHASE08_RUN         |     0.0922351 |       0.235353 |    0.955602  |           0.954107 |      0.193317   |
| FLIGHT_FULL                           | HDC            | REUSED_FROZEN_REFERENCE |     0.26111   |       0.389104 |    0.878647  |           0.931247 |      0.128878   |
| FLIGHT_FULL                           | TRADITIONAL    | NEW_PHASE08_RUN         |     0.0929993 |       0.236368 |    0.955219  |           0.953841 |      0.188544   |
| FULL_PRIMARY_REFERENCE                | HDC            | REUSED_FROZEN_REFERENCE |     0.265727  |       0.394327 |    0.875367  |           0.928379 |      0.124105   |
| FULL_PRIMARY_REFERENCE                | TRADITIONAL    | REUSED_FROZEN_REFERENCE |     0.107486  |       0.255084 |    0.947846  |           0.952984 |      0.176611   |
| FUSION_PE                             | HDC            | NEW_PHASE08_RUN         |     0.892153  |       1.13781  |   -0.0376777 |           0.364008 |      0.133652   |
| FUSION_PE                             | TRADITIONAL    | NEW_PHASE08_RUN         |     0.818992  |       1.0198   |    0.166412  |           0.445289 |      0.0286396  |
| FUSION_PEH                            | HDC            | NEW_PHASE08_RUN         |     0.861677  |       1.09766  |    0.0342714 |           0.39513  |      0.0859189  |
| FUSION_PEH                            | TRADITIONAL    | NEW_PHASE08_RUN         |     0.817035  |       1.00763  |    0.186186  |           0.455734 |      0.00954654 |
| FUSION_PEHF                           | HDC            | NEW_PHASE08_RUN         |     0.263814  |       0.39312  |    0.876129  |           0.928776 |      0.124105   |
| FUSION_PEHF                           | TRADITIONAL    | NEW_PHASE08_RUN         |     0.106652  |       0.254157 |    0.948224  |           0.952996 |      0.169451   |
| PERFORMANCE_ONLY_AUXILIARY            | HDC            | NEW_PHASE08_RUN         |     0.544883  |       0.700483 |    0.606707  |           0.784078 |      0.0739857  |
| PERFORMANCE_ONLY_AUXILIARY            | TRADITIONAL    | NEW_PHASE08_RUN         |     0.526922  |       0.718379 |    0.586355  |           0.758366 |      0.0859189  |
| WITHOUT_PERFORMANCE_PRIMARY_REFERENCE | HDC            | REUSED_FROZEN_REFERENCE |     0.265727  |       0.394327 |    0.875367  |           0.928379 |      0.124105   |
| WITHOUT_PERFORMANCE_PRIMARY_REFERENCE | TRADITIONAL    | REUSED_FROZEN_REFERENCE |     0.107486  |       0.255084 |    0.947846  |           0.952984 |      0.176611   |
| WITH_PERFORMANCE_AUXILIARY            | HDC            | NEW_PHASE08_RUN         |     0.250266  |       0.387497 |    0.879647  |           0.929031 |      0.138425   |
| WITH_PERFORMANCE_AUXILIARY            | TRADITIONAL    | NEW_PHASE08_RUN         |     0.104831  |       0.246773 |    0.951189  |           0.953602 |      0.183771   |

## Statistical summary
9 of 26 registered comparisons pass Holm-adjusted alpha 0.05. Non-significance is not treated as equivalence.

## Registered-question answers

1. **Adding head to PE:** PE→PEH was not Holm-significant for either model or task; it is not supported as a reliable increment.
2. **Adding flight to PEH:** the increment was large and Holm-significant in both models and tasks (classification subject Macro-F1 Δ HDC 0.507, traditional 0.544; regression bounded-MAE Δ HDC -0.598, traditional -0.710, where negative is better).
3. **Performance features:** with-performance did not create a universal anomalous gain. HDC classification moved downward and traditional classification changed little; HDC bounded MAE improved modestly with Holm support, while the traditional regression change was not significant. Performance-only retained substantial signal but was significantly worse than the without-performance reference in both models/tasks. This is shortcut-risk evidence, not proof of direct leakage.
4. **Model/task direction:** HDC and traditional agree strongly on the flight increment and on performance-only being weaker than the full reference. They do not show a universal performance-feature gain. Classification and regression therefore support the same central flight-increment pattern.
5. **Fusion versus frozen references:** FUSION_PEHF was not Holm-significantly better than the frozen best-flight or full-primary references. Numerical proximity is not evidence of superiority.
6. **Behavioral-only sensitivity:** removing 3 ambiguous flight features left high performance. FLIGHT_BEHAVIORAL_ONLY−FLIGHT_FULL classification Δ was -0.011 (HDC) and -0.008 (traditional), both non-significant; bounded-MAE Δ was 0.000000 and -0.000766, also non-significant. HDC regression predictions were identical under the two conditions. Non-significance is not equivalence.
7. **Generalization boundary:** these results show that the observed flight advantage persists in the 323 provenance-labeled behavioral-response features after excluding the 3 ambiguous acquisition features. They cannot establish that the advantage is generalizable flight behavior rather than difficulty-adjacent task structure because repeated-session, scenario, task-template, and route/configuration identifiers are absent. Phase 09 unseen-condition validation remains necessary after appropriate metadata exist.

## Claim Candidates
- Claim: Performance-feature conditions are shortcut-risk diagnostics, not causal physiological evidence.
  - Source evidence: performance comparison family and static inventory.
  - Allowed wording: predictive information changes under auxiliary performance features.
  - Forbidden stronger wording: leakage is proven by high performance alone.
  - Uncertainty: unseen-condition metadata are unavailable.
  - Next check: Phase 09 unseen-condition validation after metadata collection.
  - Decision: keep
