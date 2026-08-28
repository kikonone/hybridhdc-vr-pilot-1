# Notes: Phase 01 Raw Data Modality Audit

## Source Data
- Main dataset source: `vrdataset/dataPackage`
- Data safety rule: no delete, rename, overwrite, or modification inside `vrdataset`.

## Required Outputs
- Notebook: `notebooks/01_raw_data_modality_audit.ipynb`
- File inventory, run modality availability, modality summaries, unknown/unreadable review files, JSON report, figures, log, README, and updated `EXPERIMENT_STATUS.md`.

## Constraints
- No feature extraction.
- No final modeling dataset.
- No ML or HDC training.


## Final Findings
- Files audited: 9003
- Readable files: 9003
- Subjects: 35
- Sessions: 35
- Runs: 487
- Found modalities: ecg, eda, emg, eye_tracking, head_movement, performance, respiration, xplane
- Missing/incomplete modalities: eda, control_input, performance
- Unknown files requiring manual review: 908 (`lslshimmertorsoacc` files)
- Unreadable files: 0
- Strict full multimodal feature extraction feasible: False
- Core multimodal feature extraction without control input feasible: True
