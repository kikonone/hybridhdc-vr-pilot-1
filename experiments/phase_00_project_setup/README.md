# Phase 00 Project Setup and Revised Data Inventory

## What was created

Phase 00 created and updated the experiment workspace for the revised full multimodal thesis plan. All generated files were saved under `experiments`; `vrdataset` was treated as read-only source data.

Key outputs:

- `experiments/phase_00_project_setup/notebooks/00_revised_project_setup_and_inventory.ipynb`
- `experiments/phase_00_project_setup/results/data_inventory.csv`
- `experiments/phase_00_project_setup/results/revised_phase_structure.csv`
- `experiments/phase_00_project_setup/results/project_setup_report.json`
- `experiments/phase_00_project_setup/tables/file_type_summary.csv`
- `experiments/phase_00_project_setup/tables/folder_summary.csv`
- `experiments/phase_00_project_setup/logs/phase_00_log.txt`
- `experiments/shared/reports/REVISED_EXPERIMENT_PLAN.md`

The revised phase folders from Phase 00 through Phase 10 were created with standard `notebooks`, `scripts`, `results`, `figures`, `tables`, and `logs` subfolders.

## Source data roles

`vrdataset/starterCode` is not the final full multimodal dataset. It is mainly an eye-tracking and oculomotor pilot reference pipeline, so it can help interpret example processing logic but should not define the final thesis dataset by itself.

`vrdataset/dataPackage` is the main source for the full multimodal experiment. Later phases should audit this folder for available modalities, subjects, sessions, difficulty levels, runs, streams, and quality constraints before any feature extraction or modeling.

## Inventory summary

The revised notebook saved a full recursive inventory of all files under `vrdataset`.

| Item | Count |
|---|---:|
| Total files under `vrdataset` | 9,022 |
| `dataPackage` files | 9,003 |
| `starterCode` files | 12 |
| `starterCode/data_feats` files | 1 |
| `referenceDocuments` files | 5 |

Requested extension summary:

| Extension | Count |
|---|---:|
| csv | 9,004 |
| mat | 0 |
| xdf | 0 |
| txt | 2 |
| npy | 0 |
| npz | 0 |
| pkl | 0 |
| parquet | 0 |
| py | 4 |
| m | 1 |
| ipynb | 6 |

## What Phase 01 should do next

Phase 01 should perform a raw data audit and modality inventory using `vrdataset/dataPackage` as the primary source. It should identify sensor streams, task folders, subjects, sessions, runs, difficulty levels, missing files, and quality constraints. It should also define the manifest that Phase 02 will use for full multimodal feature extraction.

No feature extraction, labeling, ML training, HDC training, or model evaluation was performed in Phase 00.
