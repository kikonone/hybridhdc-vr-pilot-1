# Phase 01: Raw Data Audit and Modality Inventory

## Scope
This phase audited `vrdataset/dataPackage` as read-only source data. Generated artifacts were saved only under this experiment folder.

No feature extraction, final modeling dataset creation, ML training, or HDC training was performed.

## Modalities Found

- `ecg`: present in 487 runs
- `eda`: present in 475 runs
- `emg`: present in 487 runs
- `eye_tracking`: present in 487 runs
- `head_movement`: present in 487 runs
- `performance`: present in 419 runs
- `respiration`: present in 487 runs
- `xplane`: present in 487 runs

## Missing or Incomplete Modalities

- `eda` is missing in 12 runs
- `control_input` is missing in 487 runs
- `performance` is missing in 68 runs

## Manual Confirmation Needed

- Unknown files requiring review: 908
- Failed or unreadable files: 0
- `sub-cp004_ses-20210330_task-ils_stream-lslshimmertorsoacc_feat-chunk_level-01B_run-001_dat.csv`
- `sub-cp004_ses-20210330_task-ils_stream-lslshimmertorsoacc_feat-chunk_level-01B_run-001_hea.csv`
- `sub-cp004_ses-20210330_task-ils_stream-lslshimmertorsoacc_feat-chunk_level-01B_run-007_dat.csv`
- `sub-cp004_ses-20210330_task-ils_stream-lslshimmertorsoacc_feat-chunk_level-01B_run-007_hea.csv`
- `sub-cp004_ses-20210330_task-ils_stream-lslshimmertorsoacc_feat-chunk_level-01B_run-012_dat.csv`
- `sub-cp004_ses-20210330_task-ils_stream-lslshimmertorsoacc_feat-chunk_level-01B_run-012_hea.csv`
- `sub-cp004_ses-20210330_task-ils_stream-lslshimmertorsoacc_feat-chunk_level-02B_run-003_dat.csv`
- `sub-cp004_ses-20210330_task-ils_stream-lslshimmertorsoacc_feat-chunk_level-02B_run-003_hea.csv`
- `sub-cp004_ses-20210330_task-ils_stream-lslshimmertorsoacc_feat-chunk_level-02B_run-008_dat.csv`
- `sub-cp004_ses-20210330_task-ils_stream-lslshimmertorsoacc_feat-chunk_level-02B_run-008_hea.csv`
- Manual modality mapping is required before Phase 02 because some files remain unknown or unreadable.

## Feasibility

- Strict full multimodal feature extraction across all listed modalities is feasible: False
- Core multimodal feature extraction without `control_input` is feasible: True
- Strict complete runs: 0 of 487
- Core complete runs: 408 of 487

## Phase 02 Recommendation

Phase 02 should extract synchronized run-level features for the confidently detected modalities: eye tracking, ECG, EDA, EMG, respiration, head movement, X-Plane aircraft/performance streams, and per-run performance metrics. Control input should only be included if manual review identifies a reliable source for it.

## Limitations

- File-level modality detection is conservative: unclear files are labeled `unknown` rather than guessed.
- Row counts were calculated from file lines and CSV headers were previewed safely; no feature matrices or modeling datasets were produced.
- MAT support is implemented, but this dataPackage audit found no `.mat` files among the requested extensions.
- Rest-task runs use `level-000`; ILS workload-class modeling should focus on the ILS difficulty levels unless rest is intentionally included as a separate baseline condition.

## Metadata and Legal Files

- LICENSE, README, metadata, and description files are metadata/legal documents, not modeling data. Count found in this dataPackage audit: 0.
