# Notes: Phase 00 Project Setup

## Initial Observations
- Project root: `E:\hdc-vr-pilot`
- Source dataset root: `vrdataset`
- Required source subfolders: `dataPackage`, `referenceDocuments`, `starterCode`
- Safety rule: do not delete, rename, overwrite, or modify anything under `vrdataset`.

## Git Context
- Existing unrelated changes were observed before this task:
  - Deleted tracked files under `data/`
  - Untracked `vrdataset/`
  - Untracked Word documents in the project root
- These are left untouched.

## Phase 00 Outputs
- Executed notebook: `experiments/phase_00_project_setup/notebooks/00_project_setup_and_data_inventory.ipynb`
- Inventory CSV: `experiments/phase_00_project_setup/results/data_inventory.csv`
- README: `experiments/phase_00_project_setup/README.md`

## Inventory Findings
- Total directories under `vrdataset`: 634
- Total files under `vrdataset`: 9,022
- Total file size: 45,422.94 MiB
- Selected-format inventory rows: 9,017
- Selected extensions found: csv 9,004; ipynb 6; m 1; py 4; txt 2
- Candidate patterns: `devSubjsFeatMat` 1; `perfmetric` 420; `ocuevts` 469; `feat-chunk` 8,114

## Next Phase
- Audit labels and schemas before training.
- Confirm subject-aware split strategy before ML baseline and HDC phases.
