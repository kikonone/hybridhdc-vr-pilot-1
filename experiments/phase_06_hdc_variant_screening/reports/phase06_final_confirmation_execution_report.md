# Phase 06 Final Confirmation Execution Report

Status: FINAL_CONFIRMATION_COMPLETE

All 300 fold-config runs completed with strict nested inner-CV temperature selection. The reported regression target is the bounded difficulty-induced workload proxy regression.

Ridge handling: `COMMON_ENCODER_READOUT_BASELINE`. Ridge consumes sample hypervectors, while these variants change only prototype/centroid learning; repeating an identical Ridge fit per variant would be pseudo-replication.

Outer-test features were not loaded until inner selections were fixed. Outer-test labels were joined only after predictions were generated. Outer-test data were not used for tuning.

| Variant | Runs | Mean Macro-F1 | Mean bounded MAE |
|---|---:|---:|---:|
| onlinehd | 100 | 0.625592 | 0.433010 |
| multicentroid | 100 | 0.793325 | 0.720639 |
| hybrid | 100 | 0.816172 | 0.571199 |

No final HDC variant, dimension, or seed was selected. Final OOF Consolidation has not been executed.
