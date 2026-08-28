# Latest Classification + Regression Plan

## Status

Classification and regression are parallel primary tasks. The canonical complete Chinese execution plan is `最新完整实验计划_分类回归双任务.md`; `p1_aligned_latest_plan.md` is the shorter P1-alignment version.

## Dual-Task Contract

- Classification target: `target_class = difficulty_level - 1`, values 0-3.
- Regression target: `target_score = difficulty_level`, observed values 1-4.
- Both tasks use the same multimodal inputs, saved subject folds, and fold-local preprocessing.
- Classification predicts an operational level; regression predicts a bounded workload-proxy score.
- The regression target is not direct continuous psychological workload ground truth.

## HDC Design

- Shared hypervector encoder.
- Classification head: nearest class prototype.
- Regression head 1: similarity-weighted prototype decoding.
- Regression head 2: regularized linear/ridge readout on encoded hypervectors.

## Primary Evaluation

- Classification: Macro-F1 and balanced accuracy.
- Regression: MAE and RMSE.
- Supporting: Spearman correlation, severe-error rate, adjacent accuracy, quadratic weighted kappa, timing, and memory.
- Consistency: agreement between the classifier output and rounded/clipped regression output.

## Immediate Next Step

Freeze one reusable five-fold subject split and run the first paired benchmark: Logistic Regression and SVM; Ridge and SVR; Vanilla HDC classification and similarity-weighted HDC regression.
