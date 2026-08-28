# Phase 05 No-Retraining Compliance Completion

## Scope

This amendment closes the diagnostic and inference-efficiency evidence gaps identified after the original Phase 05 freeze. It uses only the frozen OOF table and saved fitted artifacts. It does not fit or refit a model, change a checkpoint, replace a prediction, select a canonical configuration, or start Phase 06.

## Immutable evidence

- The pre-amendment snapshot records 383 historical artifacts and all 35 existing Notebook cells.
- The original long OOF SHA-256 remained `bbc3d2044e9e1e527eed7e0f40d41ba569033073cb437e6d9aec3ccce1d837ad` before and after diagnostic derivation.
- All 8,380 frozen OOF rows and all 20 dimension-seed configurations were retained.
- Probabilities reconstructed from the four saved similarities sum to one with maximum absolute error `3.33e-16`.
- Reconstructed class predictions have zero mismatches; reconstructed similarity-regression predictions differ by at most `2.66e-15`.

## Added diagnostics

The amendment adds class-wise recall, top-two similarity margin, class probabilities, raw and bounded similarity prediction, per-true-level regression summaries, prediction-range and mean-collapse diagnostics, rounded regression classification diagnostics, and cross-head consistency measures.

Across the complete preregistered matrix, class recall values range from `0.605769` to `1.000000`, and mean top-two similarity margins range from `0.018235` to `0.021318`. Similarity-regression prediction ranges span `1.705962` to `1.881912` within each configuration and round only to levels 2 and 3. Ridge prediction ranges span the full bounded interval (`3.0`) and round to all four levels. The classification/similarity rounded agreement ranges from `0.257757` to `0.331742`; classification/Ridge rounded agreement ranges from `0.670644` to `0.747017`.

These are descriptive diagnostics of the frozen matrix. They do not authorize post-hoc selection or a stronger inferential claim.

## Completed inference-efficiency protocol

Each of the 100 saved fold-seed-dimension artifacts was loaded read-only. Codebooks were prepared outside the timed repetitions. The complete frozen preprocessing, record encoding, classification head, similarity-regression head, and Ridge head were then run with five warm-ups and thirty measured repetitions using `time.perf_counter_ns`. Repeated outputs were discarded after comparison with frozen predictions.

- Frozen-prediction maximum absolute difference: `8.88e-16`.
- Sum of fold-median complete OOF inference time across configurations: `63.5962–786.3345 ms` for 419 rows.
- Complete inference: `151,780.91–1,876,693.20 ns/sample`.
- Encoding throughput: `575.45–8,717.58 rows/s`.
- Maximum saved model-component memory across folds: `181,224–1,315,224 bytes`.
- Maximum Python allocation observed during a full inference pass: `4,703,200–10,941,744 bytes`.

The allocation measure is a Python `tracemalloc` peak, not total process RSS or accelerator memory.

## Training-time boundary

Training timing was not remeasured. Exact protocol-compliant training timing would require refitting, which this amendment expressly prohibits. Its status is therefore `NOT_PERFORMED_RETRAINING_PROHIBITED`; the original historical training-time fields remain unchanged and are not represented as satisfying the new repeated-timing protocol.

## Evidence index

- Amendment authorization: `configs/phase05_no_retraining_completion_amendment.json`
- Immutable snapshot: `audits/phase05_no_retraining_pre_amendment_snapshot.json`
- Diagnostic audit: `audits/phase05_no_retraining_diagnostic_completion_audit.json`
- Efficiency audit: `audits/phase05_no_retraining_efficiency_protocol_completion_audit.json`
- Row-level derived diagnostics: `results/oof/vanilla_hdc_final_confirmation_diagnostics.csv`
- Configuration diagnostics: `results/summaries/vanilla_hdc_*_diagnostics_by_config.csv`
- Cross-task diagnostics: `results/summaries/vanilla_hdc_cross_task_consistency_by_config.csv`
- Protocol-complete inference measurements: `results/efficiency/vanilla_hdc_final_confirmation_protocol_completion_by_fold_config.csv`

## Boundary

The scientific result remains the complete preregistered four-dimension by five-seed matrix. No canonical seed or dimension is selected, no ensemble or inferential test is added, and Phase 06 is not executed.
