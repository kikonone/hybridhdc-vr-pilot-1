# Notes: Phase 02 Full Multimodal Feature Extraction

## Phase 01 Inputs
- File inventory: `experiments/phase_01_raw_data_modality_audit/results/raw_file_inventory.csv`
- Run availability: `experiments/phase_01_raw_data_modality_audit/results/run_modality_availability.csv`
- Modality summary: `experiments/phase_01_raw_data_modality_audit/results/modality_summary.csv`
- Unknown review list: `experiments/phase_01_raw_data_modality_audit/results/unknown_files_for_review.csv`

## Initial Findings
- Phase 01 reports 487 run-level records.
- Strict complete multimodal runs were zero, so Phase 02 must keep incomplete runs and represent missing modalities with NaN.
- `PerfMetrics.csv` is a performance file at `task-ils/PerfMetrics.csv` and needs special handling because it is not stored inside per-run folders.
- StarterCode is not used as the source dataset for Phase 02 extraction.

## Extraction Decisions
- Extract numeric aggregate features for known data modalities and keep unknown files out of model features unless they can be safely reclassified by file/column evidence.
- Record unreadable files and skipped documentation/metadata files in extraction logs.
- Store performance metrics in the full table but group them separately as `performance_features` for Phase 03 leakage handling.

## Final Results
- Processed feature files: 4,945
- Skipped header/non-feature files: 4,057
- Runs in output: 487
- Feature columns excluding identifiers: 1,247
- Long-table rows: 599,569
- Failed files: 0
- Modalities extracted: eye_tracking, ecg, eda, emg, respiration, head_movement, xplane, performance, unknown
- Control input streams were unavailable.
- Phase 03 must create with-performance and without-performance modeling datasets.
