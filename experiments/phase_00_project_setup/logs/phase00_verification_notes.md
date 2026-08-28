# Notes: Phase 00 Verification

## Safety Boundary
- No writes are permitted under `vrdataset`.
- No feature extraction or model training is permitted in this task.

## Findings
- Raw source root: `vrdataset` (9,022 files), including `dataPackage` (9,003), `starterCode` (12), `starterCode/data_feats` (1), and `referenceDocuments` (5).
- Processed feature data exists under Phase 02; labeled modeling datasets exist under Phase 03.
- The starter-code notebook points to `starterCode/data_feats/devSubjsFeatMat.csv`, so running it in place can overwrite a file inside the immutable source tree.
- `compute_flight_performance.m` writes performance CSVs into source-data directories; `extractSaccFix.py` writes generated CSVs and deletes a temporary file in its selected output tree.
- Active experiment scripts read from `vrdataset/dataPackage` and write to experiment phase directories; no direct raw-data write target was found in them.
- Phase scripts use overwrite-style writes and do not implement backup handling for completed outputs.
- Actual experiment folders stop at Phase 03; `experiments/shared` and Phases 04-10 are absent despite README/status references.
- Root `reports` and `experiments/shared/reports` are absent, so the report belongs in the existing Phase 00 results directory.
- Python is CPython 3.12.7 from `D:\Computer\anaconda3\python.exe`; all requirements are installed except `pyxdf`.
- Final report: `experiments/phase_00_project_setup/results/phase00_verification.md`.
- Final raw safety check: 9,022 files; latest source-file timestamp remains 2022-08-25 11:32:34.
- Existing Phase 00 inventory/report/phase-structure files retain their original 2026-07-02 timestamps.
