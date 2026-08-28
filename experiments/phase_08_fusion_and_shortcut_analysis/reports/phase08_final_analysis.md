# Phase 08 Final Analysis

## Executive Summary
All 370 frozen runs were consolidated without retraining into 10,894 canonical OOF rows. Metrics were independently recalculated and all inference used 35 paired subjects. Results support condition-specific predictive comparisons, not causal or unseen-condition generalization claims.

## Experiment Identity and Decision Context
Phase 08 / canonical OOF and shortcut analysis / analysis complete pending freeze.

## Setup and Evaluation Protocol
Five frozen outer folds; five HDC seeds aggregated by class-score mean or raw-regression mean; traditional predictions concatenated once per run key. Regression is bounded difficulty-induced workload proxy regression.

## Main Findings

## Registered-question answers

1. **Adding head to PE:** PE→PEH was not Holm-significant for either model or task; it is not supported as a reliable increment.
2. **Adding flight to PEH:** the increment was large and Holm-significant in both models and tasks (classification subject Macro-F1 Δ HDC 0.507, traditional 0.544; regression bounded-MAE Δ HDC -0.598, traditional -0.710, where negative is better).
3. **Performance features:** with-performance did not create a universal anomalous gain. HDC classification moved downward and traditional classification changed little; HDC bounded MAE improved modestly with Holm support, while the traditional regression change was not significant. Performance-only retained substantial signal but was significantly worse than the without-performance reference in both models/tasks. This is shortcut-risk evidence, not proof of direct leakage.
4. **Model/task direction:** HDC and traditional agree strongly on the flight increment and on performance-only being weaker than the full reference. They do not show a universal performance-feature gain. Classification and regression therefore support the same central flight-increment pattern.
5. **Fusion versus frozen references:** FUSION_PEHF was not Holm-significantly better than the frozen best-flight or full-primary references. Numerical proximity is not evidence of superiority.
6. **Behavioral-only sensitivity:** removing 3 ambiguous flight features left high performance. FLIGHT_BEHAVIORAL_ONLY−FLIGHT_FULL classification Δ was -0.011 (HDC) and -0.008 (traditional), both non-significant; bounded-MAE Δ was 0.000000 and -0.000766, also non-significant. HDC regression predictions were identical under the two conditions. Non-significance is not equivalence.
7. **Generalization boundary:** these results show that the observed flight advantage persists in the 323 provenance-labeled behavioral-response features after excluding the 3 ambiguous acquisition features. They cannot establish that the advantage is generalizable flight behavior rather than difficulty-adjacent task structure because repeated-session, scenario, task-template, and route/configuration identifiers are absent. Phase 09 unseen-condition validation remains necessary after appropriate metadata exist.


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

## Statistical Validation
| comparison_family               | model_family   | task           | metric      | metric_direction   | left_condition                        | right_condition            |   n_subjects |   left_mean |   right_mean |   right_minus_left |   wilcoxon_statistic |     p_value |   rank_biserial_right_minus_left |   bootstrap_ci_low |   bootstrap_ci_high |      p_holm | significant_holm_0_05   |
|:--------------------------------|:---------------|:---------------|:------------|:-------------------|:--------------------------------------|:---------------------------|-------------:|------------:|-------------:|-------------------:|---------------------:|------------:|---------------------------------:|-------------------:|--------------------:|------------:|:------------------------|
| A_EARLY_FUSION                  | HDC            | classification | macro_f1    | higher_is_better   | FUSION_PE                             | FUSION_PEH                 |           35 |   0.292299  |    0.344576  |        0.0522769   |                136   | 0.028177    |                       0.451613   |         0.0114332  |         0.0941489   | 0.084531    | False                   |
| A_EARLY_FUSION                  | TRADITIONAL    | classification | macro_f1    | higher_is_better   | FUSION_PE                             | FUSION_PEH                 |           35 |   0.378135  |    0.39269   |        0.0145553   |                218   | 0.389706    |                       0.169355   |        -0.0271528  |         0.0569234   | 0.634621    | False                   |
| A_EARLY_FUSION                  | HDC            | classification | macro_f1    | higher_is_better   | FUSION_PEH                            | FUSION_PEHF                |           35 |   0.344576  |    0.8517    |        0.507123    |                  0   | 3.65104e-07 |                       1          |         0.430358   |         0.580674    | 1.46042e-06 | True                    |
| A_EARLY_FUSION                  | TRADITIONAL    | classification | macro_f1    | higher_is_better   | FUSION_PEH                            | FUSION_PEHF                |           35 |   0.39269   |    0.937018  |        0.544328    |                  0   | 5.82077e-11 |                       1          |         0.494351   |         0.594734    | 1.74623e-10 | True                    |
| A_EARLY_FUSION                  | HDC            | classification | macro_f1    | higher_is_better   | BEST_SINGLE_FLIGHT_REFERENCE          | FUSION_PEHF                |           35 |   0.852925  |    0.8517    |       -0.00122556  |                 59.5 | 0.977294    |                       0.00833333 |        -0.019052   |         0.0184642   | 1           | False                   |
| A_EARLY_FUSION                  | HDC            | classification | macro_f1    | higher_is_better   | FULL_PRIMARY_REFERENCE                | FUSION_PEHF                |           35 |   0.861651  |    0.8517    |       -0.0099516   |                 21   | 0.858832    |                      -0.0666667  |        -0.0326554  |         0.0082967   | 1           | False                   |
| A_EARLY_FUSION                  | TRADITIONAL    | classification | macro_f1    | higher_is_better   | FULL_PRIMARY_REFERENCE                | FUSION_PEHF                |           35 |   0.934637  |    0.937018  |        0.00238095  |                  0   | 0.317311    |                       1          |         0          |         0.00714286  | 0.634621    | False                   |
| B_PERFORMANCE_SHORTCUT          | HDC            | classification | macro_f1    | higher_is_better   | WITHOUT_PERFORMANCE_PRIMARY_REFERENCE | WITH_PERFORMANCE_AUXILIARY |           35 |   0.861651  |    0.828771  |       -0.0328803   |                 36.5 | 0.058253    |                      -0.522876   |        -0.0667566  |        -0.00190751  | 0.058253    | False                   |
| B_PERFORMANCE_SHORTCUT          | TRADITIONAL    | classification | macro_f1    | higher_is_better   | WITHOUT_PERFORMANCE_PRIMARY_REFERENCE | WITH_PERFORMANCE_AUXILIARY |           35 |   0.934637  |    0.939739  |        0.00510204  |                  5.5 | 0.287787    |                       0.47619    |        -0.0070085  |         0.017483    | 0.287787    | False                   |
| B_PERFORMANCE_SHORTCUT          | HDC            | classification | macro_f1    | higher_is_better   | WITHOUT_PERFORMANCE_PRIMARY_REFERENCE | PERFORMANCE_ONLY_AUXILIARY |           35 |   0.861651  |    0.523306  |       -0.338345    |                  1   | 1.16415e-10 |                      -0.996825   |        -0.397695   |        -0.281314    | 2.32831e-10 | True                    |
| B_PERFORMANCE_SHORTCUT          | TRADITIONAL    | classification | macro_f1    | higher_is_better   | WITHOUT_PERFORMANCE_PRIMARY_REFERENCE | PERFORMANCE_ONLY_AUXILIARY |           35 |   0.934637  |    0.703165  |       -0.231472    |                  1   | 8.71771e-07 |                      -0.996212   |        -0.290962   |        -0.172568    | 1.74354e-06 | True                    |
| C_FLIGHT_PROVENANCE_SENSITIVITY | HDC            | classification | macro_f1    | higher_is_better   | FLIGHT_FULL                           | FLIGHT_BEHAVIORAL_ONLY     |           35 |   0.852925  |    0.841545  |       -0.0113802   |                 13   | 0.136838    |                      -0.527273   |        -0.0290919  |         0.00462493  | 0.136838    | False                   |
| C_FLIGHT_PROVENANCE_SENSITIVITY | TRADITIONAL    | classification | macro_f1    | higher_is_better   | FLIGHT_FULL                           | FLIGHT_BEHAVIORAL_ONLY     |           35 |   0.950784  |    0.94291   |       -0.00787415  |                  0   | 0.108809    |                      -1          |        -0.0177555  |         0           | 0.108809    | False                   |
| A_EARLY_FUSION                  | HDC            | regression     | bounded_mae | lower_is_better    | FUSION_PE                             | FUSION_PEH                 |           35 |   0.892992  |    0.862048  |       -0.0309431   |                263   | 0.403589    |                      -0.165079   |        -0.0969294  |         0.0345372   | 0.807177    | False                   |
| A_EARLY_FUSION                  | TRADITIONAL    | regression     | bounded_mae | lower_is_better    | FUSION_PE                             | FUSION_PEH                 |           35 |   0.818938  |    0.816961  |       -0.00197708  |                313   | 0.980649    |                      -0.00634921 |        -0.0464411  |         0.0436636   | 0.980649    | False                   |
| A_EARLY_FUSION                  | HDC            | regression     | bounded_mae | lower_is_better    | FUSION_PEH                            | FUSION_PEHF                |           35 |   0.862048  |    0.26382   |       -0.598228    |                  0   | 5.82077e-11 |                      -1          |        -0.688922   |        -0.512154    | 2.32831e-10 | True                    |
| A_EARLY_FUSION                  | TRADITIONAL    | regression     | bounded_mae | lower_is_better    | FUSION_PEH                            | FUSION_PEHF                |           35 |   0.816961  |    0.106633  |       -0.710328    |                  0   | 5.82077e-11 |                      -1          |        -0.76592    |        -0.651144    | 1.74623e-10 | True                    |
| A_EARLY_FUSION                  | HDC            | regression     | bounded_mae | lower_is_better    | BEST_SINGLE_FLIGHT_REFERENCE          | FUSION_PEHF                |           35 |   0.260941  |    0.26382   |        0.00287886  |                171   | 0.466194    |                       0.157635   |        -0.00731951 |         0.0126202   | 0.807177    | False                   |
| A_EARLY_FUSION                  | HDC            | regression     | bounded_mae | lower_is_better    | FULL_PRIMARY_REFERENCE                | FUSION_PEHF                |           35 |   0.265742  |    0.26382   |       -0.00192188  |                194   | 0.0475187   |                      -0.384127   |        -0.0036707  |        -0.000165134 | 0.142556    | False                   |
| A_EARLY_FUSION                  | TRADITIONAL    | regression     | bounded_mae | lower_is_better    | FULL_PRIMARY_REFERENCE                | FUSION_PEHF                |           35 |   0.107441  |    0.106633  |       -0.000807647 |                 91   | 0.394457    |                      -0.212121   |        -0.00259998 |         0.000779456 | 0.788914    | False                   |
| B_PERFORMANCE_SHORTCUT          | HDC            | regression     | bounded_mae | lower_is_better    | WITHOUT_PERFORMANCE_PRIMARY_REFERENCE | WITH_PERFORMANCE_AUXILIARY |           35 |   0.265742  |    0.25017   |       -0.0155715   |                179   | 0.0251547   |                      -0.431746   |        -0.0299999  |        -6.10282e-05 | 0.0251547   | True                    |
| B_PERFORMANCE_SHORTCUT          | TRADITIONAL    | regression     | bounded_mae | lower_is_better    | WITHOUT_PERFORMANCE_PRIMARY_REFERENCE | WITH_PERFORMANCE_AUXILIARY |           35 |   0.107441  |    0.104782  |       -0.00265845  |                256   | 0.342327    |                      -0.187302   |        -0.00757922 |         0.00136263  | 0.342327    | False                   |
| B_PERFORMANCE_SHORTCUT          | HDC            | regression     | bounded_mae | lower_is_better    | WITHOUT_PERFORMANCE_PRIMARY_REFERENCE | PERFORMANCE_ONLY_AUXILIARY |           35 |   0.265742  |    0.544536  |        0.278795    |                  0   | 5.82077e-11 |                       1          |         0.225897   |         0.335205    | 1.16415e-10 | True                    |
| B_PERFORMANCE_SHORTCUT          | TRADITIONAL    | regression     | bounded_mae | lower_is_better    | WITHOUT_PERFORMANCE_PRIMARY_REFERENCE | PERFORMANCE_ONLY_AUXILIARY |           35 |   0.107441  |    0.526785  |        0.419344    |                  0   | 5.82077e-11 |                       1          |         0.364033   |         0.481819    | 1.16415e-10 | True                    |
| C_FLIGHT_PROVENANCE_SENSITIVITY | HDC            | regression     | bounded_mae | lower_is_better    | FLIGHT_FULL                           | FLIGHT_BEHAVIORAL_ONLY     |           35 |   0.260941  |    0.260941  |        0           |                  0   | 1           |                       0          |         0          |         0           | 1           | False                   |
| C_FLIGHT_PROVENANCE_SENSITIVITY | TRADITIONAL    | regression     | bounded_mae | lower_is_better    | FLIGHT_FULL                           | FLIGHT_BEHAVIORAL_ONLY     |           35 |   0.0929446 |    0.0921784 |       -0.000766165 |                 74   | 0.149178    |                      -0.359307   |        -0.00180917 |         0.000166877 | 0.149178    | False                   |

## Figure-by-Figure Interpretation
See `reports/analysis-output/figure-catalog.md`; each figure separates visual observation, registered statistical support, and evidence boundary.

## Failure Cases / Negative Results / Limitations
# Phase 08 Generalization Limitations

- Unseen-session holdout: `NOT_FEASIBLE_DUE_TO_METADATA`; session is perfectly nested within subject.
- Unseen-scenario holdout: `NOT_FEASIBLE_DUE_TO_METADATA`; no explicit scenario identifier exists.
- Task-template holdout: `NOT_FEASIBLE_DUE_TO_METADATA`; only the common task-ils task is identified.
- Flight task-setting-only: `NOT_FEASIBLE_EMPTY_PROVENANCE_GROUP`; zero verified task-setting features exist.
- Absence of a significant FLIGHT_FULL versus behavioral-only difference must not be read as equivalence.
- Current evidence cannot distinguish generalizable flight behavior from task structure close to the difficulty label. Phase 09 still requires explicit repeated-session/scenario/task-template/route metadata and unseen-condition validation.

## What Changed Our Belief
The completed analysis quantifies fusion increments and shortcut sensitivity while preserving uncertainty about unseen task conditions. Numeric superiority alone is not promoted to significance.

## Next Actions
Run a separate Phase 08 freeze step after review. Do not enter Phase 09 until explicit metadata support unseen-condition splits.

## Artifact and Reproducibility Index
OOF: `results/oof/`; summaries: `results/summaries/`; figures: `figures/`; audits: `audits/`; scripts: `scripts/`.
