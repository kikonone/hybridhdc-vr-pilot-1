# Phase 06 Strict Analysis Report

## Comparison question

Across four P1 HDC families and four registered dimensions, how do OOF classification performance, bounded difficulty-induced workload proxy regression, five-seed stability, and recorded efficiency compare, and can the frozen rules select final working models?

## Comparability verification

All four classification and similarity-regression variants contain 20 complete dimension × seed configurations, each aligned on the same 419 run keys, 35 subjects, and five frozen outer folds. Vanilla predictions are read directly from the frozen Phase 05 interface. The common Ridge readout is referenced once and is not duplicated by variant.

## Descriptive observations

- The highest observed five-seed mean classification Macro-F1 is `0.828433` for HDC+OnlineHD Hybrid at dimension 10000 (sample SD `0.019116`). This is a descriptive maximum, not a frozen final selection.
- The lowest observed five-seed mean bounded MAE is `0.276390` for Common Ridge readout at dimension 10000 (sample SD `0.006419`). This is a descriptive minimum, not a frozen final selection.
- The valid Pareto analysis is performance–time two-dimensional because peak memory is unavailable for all three new variants. No three-objective memory claim is made.

## Model-selection outcome

Historical gate: `MODEL_SELECTION_BLOCKED`. The original frozen file governs inner-CV Quick Screen selection only, is classification-only, and explicitly says regression heads were not executed. It does not preregister final outer-OOF classification or regression ranking/tie-breaking. The original FAIL is preserved. A disclosed post-freeze amendment subsequently defined an inner-evidence-only selector, with outer evidence sealed and inaccessible to the selector; its successful resolution audit permits the selected configurations to be reported and Phase 06 to be frozen.

## Statistical boundary

Subject is the inferential unit; run is the prediction unit; fold is a partition; seed is a repeated algorithmic run. No subject-level paired bootstrap, permutation test, effect-size definition, multiplicity procedure, or final-selection CI was preregistered. The analysis therefore reports descriptive five-seed summaries only and does not use “significantly better.”

## Selection-induced optimism

Any future selection using these complete outer-OOF results would make the selected OOF score selection-conditioned, not an independent confirmation estimate. A later preregistered LOSO or robustness phase is required for independent confirmation wording.

## Claim candidates

- Claim: Four P1 HDC families were compared on exactly aligned OOF prediction sets.
  - Source evidence: final OOF coverage/alignment audits.
  - Allowed wording: “All compared configurations covered the same 419 runs.”
  - Forbidden stronger wording: “One method significantly outperformed the others.”
  - Uncertainty: No preregistered subject-level inference was run.
  - Next check: Freeze a final selection rule before ranking models.
  - Decision: keep.
- Claim: The common Ridge readout is a single encoder-level baseline.
  - Source evidence: frozen Phase 05 OOF and Phase 06 Ridge contract.
  - Allowed wording: “Common Ridge was referenced once across the comparison.”
  - Forbidden stronger wording: “Each prototype variant has a separate Ridge head.”
  - Uncertainty: Runtime attribution uses the frozen Vanilla interface.
  - Next check: none for Phase 06.
  - Decision: keep.

<!-- PHASE06_SELECTION_RESOLUTION_V2 -->
## Model-selection resolution (post-freeze amendment v2)

The original `MODEL_SELECTION_BLOCKED` result is retained as provenance. Its cause was that the frozen rule covered only per-outer-fold, new-variant Quick Screen classification (`classification_only=true`, `regression_heads_executed=false`); it did not define the final four-family classification comparison, regression-head comparison, or efficiency/Pareto handling. The original rule and FAIL audit were not modified.

Amendment v2 was necessary to define an executable final selection without consulting outer-test performance. It evaluated 8 classification families and 20 regression families from saved inner-CV evidence, using equal weighting of the five outer-training tasks and no selection of a single seed. The unique selections are **HDC+OnlineHD Hybrid at d=5000** for classification and **COMMON_ENCODER_READOUT_BASELINE at d=10000** for regression. The preselection outer-evidence seal contains 72 artifacts and has SHA-256 `7f78fe0e66f6c020142cc7fe099b7844e84cd34af5c39df5dbf5f48e528a42e9`; its post-selection integrity audit passed.

The Phase 06 canonical model-selection amendment was defined after final-confirmation artifacts existed, but its executable selector was restricted to previously saved inner-CV and unlabeled efficiency evidence. Outer-OOF artifacts were hash-sealed before selection and were not read by the selector.

After selection was fixed, descriptive outer-OOF summaries were read. Across the five frozen seeds, the selected classifier had Macro-F1 0.822309 (sample SD 0.032325), balanced accuracy 0.821894, and severe-error rate 0.020048. The selected regression head had bounded MAE 0.276390 (sample SD 0.006419), bounded RMSE 0.407517, R² 0.866858, and Spearman correlation 0.924752. These are descriptive results, not evidence of statistically significant superiority.

Because the amendment was defined after final-confirmation artifacts existed, selection-induced optimism cannot be ruled out even with executable isolation. Phase 07 must use this now-fixed procedure and the frozen selected families without revisiting outer-OOF ranking. A later final LOSO evaluation is responsible for the more independent confirmation.

Phase 06 is eligible for freezing only after all resolution gates pass; the final gate audit records that outcome.
