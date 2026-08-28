# Phase 04B Final Summary: Traditional Regression Baselines

## Executive Summary

Phase 04B evaluated eight traditional variants for **bounded difficulty-induced workload proxy regression** using the Primary without-performance dataset. Gradient Boosting Regressor achieved the lowest canonical bounded OOF MAE (0.10748622346925181) under the frozen evaluation protocol. This is a descriptive baseline comparison; no statistical-significance or causal claim is made.

## Experiment Identity and Evaluation Protocol

- Target: `target_score = difficulty_level`
- Target values: `1.0, 2.0, 3.0, 4.0`
- Modeling rows: 419
- Subjects: 35
- Primary predictive features: 1,176
- Input: Primary without-performance
- Outer CV: frozen five-fold subject-wise split
- Inner CV: three-fold `GroupKFold` on outer-training subjects for tuned models
- Primary metric: bounded OOF MAE (lower is better)
- Prediction bounding: clip to `[1.0, 4.0]` without rounding
- Frozen fold SHA-256: `e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f`

## Final Model Comparison

| rank_by_mae_bounded | model | mae_bounded | rmse_bounded | r2_bounded | spearman_bounded | difference_from_best_mae |
|---|---|---|---|---|---|---|
| 1 | Gradient Boosting Regressor | 0.10748622346925181 | 0.2550837953674904 | 0.947846168738032 | 0.9529838128466465 | 0.0 |
| 2 | Random Forest Regressor | 0.14373508353221956 | 0.29507239918996686 | 0.9302124980596442 | 0.9444375073511405 | 0.03624886006296775 |
| 3 | Elastic Net | 0.2747300064787801 | 0.42723402089682117 | 0.8536973696776151 | 0.9147649428003137 | 0.1672437830095283 |
| 4 | RBF SVR | 0.31085924739107956 | 0.4773068147090191 | 0.8173937040407635 | 0.9014207457906958 | 0.20337302392182777 |
| 5 | Linear SVR | 0.31759267122550844 | 0.4981974544527872 | 0.8010593715584978 | 0.8840047199364387 | 0.21010644775625664 |
| 6 | Ridge | 0.3194087375117215 | 0.5179330179580228 | 0.7849855373746046 | 0.8772756373953986 | 0.2119225140424697 |
| 7 | Dummy Regressor Median | 0.9988066825775657 | 1.2043576408064898 | -0.16260295299317007 | -0.0032212583039886496 | 0.8913204591083138 |
| 8 | Dummy Regressor Mean | 0.9988137856574609 | 1.116974097074507 | -1.5383298377136256e-05 | -0.005085561464023763 | 0.8913275621882091 |

## Best Traditional Regression Model

Gradient Boosting Regressor is the best traditional regression baseline for the current Primary without-performance data and frozen evaluation protocol, with bounded OOF MAE `0.10748622346925181`. This conclusion must not be generalized to HDC or to other feature/data settings.

## Validation and Evidence Boundary

- Final OOF coverage audit: `PASS`
- Final leakage audit: `PASS`
- All metrics are descriptive canonical OOF results over 419 runs.
- No inferential statistical tests, confidence intervals, or effect-size claims are included in this freeze summary.
- Phase 04B contains traditional regression baselines only.
- HDC, modality ablation, with-performance, performance-only, and other performance-feature experiments have not been executed in this phase.
- The target is a difficulty-induced workload proxy, not directly measured continuous cognitive workload.

## Limitations

The target has four observed values and should be interpreted as an ordered bounded proxy. Model ranking is specific to 35 subjects, 419 runs, 1,176 without-performance features, and the frozen subject-wise protocol. Later subject-level uncertainty analysis may change the strength—but not the recorded value—of descriptive comparisons.

## Next Action

Phase 04B is frozen. The next planned phase may begin only by reading these artifacts; frozen Phase 04B results must not be silently modified.

## Artifact and Reproducibility Index

- Final comparison: `results/summaries/phase04b_final_regressor_comparison.csv`
- Final audits: `audits/phase04b_final_*_audit.json`
- Manifest: `manifests/phase04b_final_artifact_manifest.json`
- Freeze record: `configs/phase04b_freeze.json`
- Notebook: `Phase_04B_Regression_Baselines.ipynb`
