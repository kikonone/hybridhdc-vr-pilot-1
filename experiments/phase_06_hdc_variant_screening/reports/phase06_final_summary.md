# Phase 06 Final Summary

OOF consolidation, four-family alignment, independent metric recalculation, seed/dimension stability summaries, actual-record efficiency tables, two-dimensional Pareto analysis, scientific figures, and the strict analysis bundle are complete.

Historically, Phase 06 was **NOT_FROZEN** because final model selection was `MODEL_SELECTION_BLOCKED`: the original frozen rules covered inner-CV Quick Screen classification only and did not define final four-family classification or regression selection. That FAIL remains preserved for provenance. The audited amendment v2 has now resolved the gate without using outer-OOF for selection, and Phase 06 is **FROZEN**; Phase 07 is ready but has not been started.

Key audits: final OOF coverage PASS; alignment PASS; leakage PASS; metric recalculation PASS; original model-selection audit FAIL preserved; amendment/isolation/seal/resolution audits PASS.

<!-- PHASE06_SELECTION_RESOLUTION_V2 -->
## Model-selection resolution (post-freeze amendment v2)

The original `MODEL_SELECTION_BLOCKED` result is retained as provenance. Its cause was that the frozen rule covered only per-outer-fold, new-variant Quick Screen classification (`classification_only=true`, `regression_heads_executed=false`); it did not define the final four-family classification comparison, regression-head comparison, or efficiency/Pareto handling. The original rule and FAIL audit were not modified.

Amendment v2 was necessary to define an executable final selection without consulting outer-test performance. It evaluated 8 classification families and 20 regression families from saved inner-CV evidence, using equal weighting of the five outer-training tasks and no selection of a single seed. The unique selections are **HDC+OnlineHD Hybrid at d=5000** for classification and **COMMON_ENCODER_READOUT_BASELINE at d=10000** for regression. The preselection outer-evidence seal contains 72 artifacts and has SHA-256 `7f78fe0e66f6c020142cc7fe099b7844e84cd34af5c39df5dbf5f48e528a42e9`; its post-selection integrity audit passed.

The Phase 06 canonical model-selection amendment was defined after final-confirmation artifacts existed, but its executable selector was restricted to previously saved inner-CV and unlabeled efficiency evidence. Outer-OOF artifacts were hash-sealed before selection and were not read by the selector.

After selection was fixed, descriptive outer-OOF summaries were read. Across the five frozen seeds, the selected classifier had Macro-F1 0.822309 (sample SD 0.032325), balanced accuracy 0.821894, and severe-error rate 0.020048. The selected regression head had bounded MAE 0.276390 (sample SD 0.006419), bounded RMSE 0.407517, R² 0.866858, and Spearman correlation 0.924752. These are descriptive results, not evidence of statistically significant superiority.

Because the amendment was defined after final-confirmation artifacts existed, selection-induced optimism cannot be ruled out even with executable isolation. Phase 07 must use this now-fixed procedure and the frozen selected families without revisiting outer-OOF ranking. A later final LOSO evaluation is responsible for the more independent confirmation.

Phase 06 is eligible for freezing only after all resolution gates pass; the final gate audit records that outcome.
