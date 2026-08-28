# Phase 06 Figure Catalog

## phase06_classification_macro_f1_by_variant_dimension.png

- Purpose: Compare classification performance and seed variability across variants and dimensions.
- Data source: phase06_classification_seed_aggregate.csv
- Reader should notice: Points are five-seed means; bars are sample SD, not confidence intervals.
- What this changes: Describes performance/stability trade-offs but cannot select a final family without a preregistered final rule.
- Caveat: Seeds are repeated runs, not independent subjects.

## phase06_classification_seed_stability.png

- Purpose: Expose seed-specific classification trajectories rather than only means.
- Data source: phase06_classification_metrics_by_config.csv
- Reader should notice: Each line is one registered seed across dimensions.
- What this changes: Reveals whether dimension trends are consistent across seeds.
- Caveat: No seed is selected individually.

## phase06_similarity_regression_mae_by_variant_dimension.png

- Purpose: Compare similarity-regression error and seed variability.
- Data source: phase06_similarity_regression_seed_aggregate.csv
- Reader should notice: Lower MAE is better; bars are sample SD.
- What this changes: Shows the descriptive regression trade-off by representation size.
- Caveat: The estimand is a bounded difficulty-induced workload proxy.

## phase06_regression_head_comparison.png

- Purpose: Compare four similarity heads with the single common Ridge baseline.
- Data source: phase06_similarity_regression_seed_aggregate.csv and phase06_common_ridge_seed_aggregate.csv
- Reader should notice: Common Ridge is plotted once, not copied to four variants.
- What this changes: Shows whether a regularized sample-HV readout changes the descriptive error range.
- Caveat: No final regression head can be selected because the frozen final ranking rule is absent.

## phase06_performance_time_pareto.png

- Purpose: Identify nondominated performance–time configurations.
- Data source: phase06_performance_efficiency_pareto.csv
- Reader should notice: Larger markers are on the two-dimensional Pareto front.
- What this changes: Separates efficient trade-offs from dominated configurations.
- Caveat: Runtime protocols differ in granularity; peak memory is incomplete, so this is not a three-objective Pareto analysis.

## Figures intentionally not created

- `phase06_performance_memory_pareto.png`: peak memory is incomplete across variants; creating it would imply a false three-objective comparison.
- Best-classification confusion matrix and best-regression prediction/residual figures: final selection is blocked by the incomplete frozen rule, so no best model may be declared.

<!-- PHASE06_SELECTION_RESOLUTION_V2 -->
## Selection-resolution note

No new figure was required for model selection. Existing figures remain descriptive outer-OOF views and must not be interpreted as selector inputs. The auditable inner-only ranking is provided in the two selection trace CSVs and the Pareto CSV; the outer-evidence seal proves those displayed outer results did not change across selection.
