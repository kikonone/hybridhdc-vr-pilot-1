# Phase 00 Project Structure and Data Safety Verification

Generated: 2026-08-19 11:20:38 +08:00  
Repository: `E:\hdc-vr-pilot`  
Scope: read-only repository, data-safety, and environment audit. No feature extraction or modeling was performed.

## Rules File Resolution

The requested `CODEX_RULES.md` does not exist. The repository contains `CODEX_RULES.md.txt`, which was read in full and treated as the applicable rules file. Its requirements concerning anti-hallucination, raw-data safety, completed-output backups, dependency inspection, subject grouping, and phase validation were applied to this verification.

## A. Actual Repository Tree

The tree below is the verified on-disk structure. Deep raw-data contents, Git internals, rendered PDF pages, and office runtime files are collapsed to avoid flooding the report.

```text
E:\hdc-vr-pilot
|-- .docx_qa/                              [temporary document QA; 0 files]
|-- .docx_tmp/                             [temporary office runtime; 1 file, 6 dirs]
|-- .git/
|-- .gitignore
|-- CODEX_RULES.md.txt
|-- EXPERIMENT_STATUS.md
|-- README.md
|-- requirements.txt
|-- classification_regression_latest_plan.md
|-- master_thesis_project_blueprint.md
|-- notes.md
|-- p1_aligned_latest_plan.md
|-- research-question-card.md
|-- task_plan.md
|-- latest-plan and thesis-plan documents (.md/.docx)
|-- experiments/
|   |-- phase_00_project_setup/
|   |   |-- README.md
|   |   |-- figures/                       [empty]
|   |   |-- logs/                          [Phase 00 log, original notes/plan, verification notes/plan]
|   |   |-- notebooks/
|   |   |   `-- 00_revised_project_setup_and_inventory.ipynb
|   |   |-- results/
|   |   |   |-- data_inventory.csv
|   |   |   |-- project_setup_report.json
|   |   |   |-- revised_phase_structure.csv
|   |   |   `-- phase00_verification.md     [this report]
|   |   |-- scripts/
|   |   |   |-- create_phase00_notebook.py
|   |   |   `-- create_revised_phase00_notebook.py
|   |   `-- tables/
|   |       |-- file_type_summary.csv
|   |       `-- folder_summary.csv
|   |-- phase_01_raw_data_modality_audit/
|   |   |-- README.md
|   |   |-- figures/                       [2 PNG files]
|   |   |-- logs/                          [phase_01_log.txt]
|   |   |-- notebooks/                     [01_raw_data_modality_audit.ipynb]
|   |   |-- results/                       [6 CSV/JSON audit artifacts]
|   |   |-- runtime/
|   |   |-- scripts/                       [2 Python scripts plus cache]
|   |   `-- tables/                        [3 CSV summaries]
|   |-- phase_02_full_multimodal_feature_extraction/
|   |   |-- README.md
|   |   |-- figures/                       [2 PNG files]
|   |   |-- ipython/ and jupyter_runtime/  [runtime state]
|   |   |-- logs/                          [5 text logs]
|   |   |-- notebooks/                     [02_full_multimodal_feature_extraction.ipynb]
|   |   |-- results/                       [7 CSV/JSON extraction artifacts]
|   |   |-- scripts/                       [phase02_extract.py plus cache]
|   |   `-- tables/                        [3 CSV summaries]
|   `-- phase_03_multimodal_dataset_labeling/
|       |-- README.md
|       |-- figures/                       [4 PNG files]
|       |-- logs/                          [phase_03_log.txt plus notebook runtime]
|       |-- notebooks/                     [03_multimodal_dataset_labeling_four_class.ipynb]
|       |-- results/                       [13 CSV/JSON dataset artifacts]
|       |-- scripts/                       [2 Python scripts plus cache]
|       `-- tables/                        [5 CSV summaries]
|-- output/
|   `-- figures/                           [12 PNG/SVG HDC plan figures]
|-- tmp/
|   `-- pdfs/                              [36 review/render files, including review_p1.py]
`-- vrdataset/                             [READ-ONLY SOURCE; 9,022 files]
    |-- LICENSE.txt
    |-- SHA256SUMS.txt
    |-- dataPackage/                       [9,003 files; 629 subdirectories]
    |   |-- task-ils/                      [deep subject/session/run data omitted]
    |   `-- task-rest/                     [35 subject directories; deep contents omitted]
    |-- referenceDocuments/                [5 PDF files]
    `-- starterCode/                       [12 files; reference pipeline]
        |-- Step1_LoadExplore_TimeSeriesSignals.ipynb
        |-- Step2_OculomotorAggregateFeatures.ipynb
        |-- Step3_ExplorePredictiveModeling.ipynb
        |-- assembleFeatureMatrix.py
        |-- compute_flight_performance.m
        |-- extractSaccFix.py
        |-- plotData.py
        |-- trainModel.py
        `-- data_feats/devSubjsFeatMat.csv  [338,790 bytes]
```

Verified absences from the actual tree:

- `reports/`
- `experiments/shared/`
- `experiments/phase_04_ml_baseline_four_class/` through `experiments/phase_10_onlinehd_lsl_simulation/`
- `environment.yml`, `environment.yaml`, and `pyproject.toml`

## B. Verified Raw-Data Locations

| Location | Role | Files | Safety status |
|---|---|---:|---|
| `vrdataset/` | Entire downloaded/source package | 9,022 | Must be treated as read-only |
| `vrdataset/dataPackage/` | Main source data for the multimodal experiment | 9,003 | Must be treated as read-only |
| `vrdataset/referenceDocuments/` | Dataset documentation and agreements | 5 | Immutable reference material |
| `vrdataset/starterCode/` | Dataset-provided pilot/reference code and notebooks | 12 | Immutable source package; do not run in place when writes are possible |
| `vrdataset/starterCode/data_feats/` | Dataset-provided eye/oculomotor feature example | 1 | Feature dataset, but still immutable because it is inside `vrdataset` |

The latest file modification timestamp found under `vrdataset` is `2022-08-25 11:32:34` for `SHA256SUMS.txt`. No phase output filename was found under `vrdataset`.

Important limitation: these paths are not protected by filesystem immutability. Their directory attributes are ordinary `Directory`, and inherited Windows ACLs grant `Authenticated Users` modify access. Read-only status is therefore a project policy, not an OS-enforced control. This audit did not perform a write test because doing so would violate the safety rule.

## C. Verified Processed-Data Locations

| Location | Verified content |
|---|---|
| `experiments/phase_01_raw_data_modality_audit/results/` | Raw file inventory, run/modality availability, unknown/unreadable lists, and JSON audit report |
| `experiments/phase_02_full_multimodal_feature_extraction/results/` | Run-level extracted features, long feature table, feature groups, processed/skipped/failed file records, and extraction report |
| `experiments/phase_03_multimodal_dataset_labeling/results/` | Four-class datasets with and without performance features, feature lists/groups, target mapping, leakage report, and construction checks |
| `experiments/*/tables/` | Machine-readable summary tables |
| `experiments/*/figures/` | Phase-generated figures |
| `experiments/*/logs/` | Execution and notebook logs |

Verified feature datasets:

- Immutable reference feature dataset: `vrdataset/starterCode/data_feats/devSubjsFeatMat.csv`.
- Phase 02 processed feature dataset: `experiments/phase_02_full_multimodal_feature_extraction/results/full_multimodal_run_level_features.csv` (8,848,511 bytes).
- Phase 02 long feature dataset: `experiments/phase_02_full_multimodal_feature_extraction/results/feature_extraction_long_table.csv` (present; ignored by Git because of size).
- Phase 03 primary four-class dataset: `experiments/phase_03_multimodal_dataset_labeling/results/cleaned_multimodal_four_class_without_performance.csv` (7,442,223 bytes).
- Phase 03 auxiliary dataset: `experiments/phase_03_multimodal_dataset_labeling/results/cleaned_multimodal_four_class_with_performance.csv` (7,752,788 bytes).

The existing Phase 03 output check reports 419 rows, 1,176 features without performance, 1,235 features with performance, 35 subjects, and verified difficulty-to-target mapping `1..4 -> 0..3`. These are cited from existing machine-readable reports; they were not recomputed during this Phase 00 verification.

## D. Existing Scripts and Notebooks

Experiment notebooks, all verified as JSON-readable and containing execution counts or visible outputs:

- `experiments/phase_00_project_setup/notebooks/00_revised_project_setup_and_inventory.ipynb`
- `experiments/phase_01_raw_data_modality_audit/notebooks/01_raw_data_modality_audit.ipynb`
- `experiments/phase_02_full_multimodal_feature_extraction/notebooks/02_full_multimodal_feature_extraction.ipynb`
- `experiments/phase_03_multimodal_dataset_labeling/notebooks/03_multimodal_dataset_labeling_four_class.ipynb`

Experiment Python scripts:

- `experiments/phase_00_project_setup/scripts/create_phase00_notebook.py`
- `experiments/phase_00_project_setup/scripts/create_revised_phase00_notebook.py`
- `experiments/phase_01_raw_data_modality_audit/scripts/create_phase01_notebook.py`
- `experiments/phase_01_raw_data_modality_audit/scripts/phase01_audit.py`
- `experiments/phase_02_full_multimodal_feature_extraction/scripts/phase02_extract.py`
- `experiments/phase_03_multimodal_dataset_labeling/scripts/create_phase03_notebook.py`
- `experiments/phase_03_multimodal_dataset_labeling/scripts/phase03_dataset.py`

Dataset-provided reference code is listed in section A and remains part of immutable `vrdataset`.

Previous Phase 00 outputs verified present and left unchanged:

- Executed inventory notebook
- `data_inventory.csv`
- `revised_phase_structure.csv`
- `project_setup_report.json`
- `file_type_summary.csv`
- `folder_summary.csv`
- `phase_00_log.txt`
- Phase 00 `README.md`

No prior `phase00_verification.md` was found, so no backup was required. The original completed Phase 00 outputs above were not overwritten.

## E. Environment Information

| Item | Verified value |
|---|---|
| Python | CPython 3.12.7 |
| Executable | `D:\Computer\anaconda3\python.exe` |
| Platform | Windows 11 (`Windows-11-10.0.26200-SP0`) |
| `requirements.txt` | Present; package names are unpinned |
| `environment.yml` / `environment.yaml` | Not present |
| `pyproject.toml` | Not present |

Core package status:

| Package | Installed version/status |
|---|---|
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| matplotlib | 3.9.2 |
| scikit-learn | 1.5.2 |
| scipy | 1.13.1 |
| PyYAML | 6.0.1 |
| jupyter | 1.0.0 |
| nbformat | 5.10.4 |
| nbclient | 0.8.0 |
| ipykernel | 6.28.0 |
| pyxdf | NOT INSTALLED |

No package was installed or upgraded during this verification.

## F. Potential Unsafe Write Operations

### Raw-data write risk

1. `vrdataset/starterCode/compute_flight_performance.m` sets `isExportPerfCsv = true`, writes per-run performance CSVs into `files(f).folder`, and writes `PerfMetrics.csv` into its configured data path. Running an adapted copy against `vrdataset/dataPackage` without redirecting outputs would modify raw source directories.
2. `vrdataset/starterCode/Step2_OculomotorAggregateFeatures.ipynb` uses `./data_feats/devSubjsFeatMat.csv`. If regeneration is triggered while the notebook runs from `starterCode`, it can overwrite the immutable reference feature CSV.
3. `vrdataset/starterCode/extractSaccFix.py` writes generated CSV/text files into a caller-selected output tree and calls `os.remove` on a temporary file there. It is safe only when `outFilePathRoot` points outside `vrdataset`.
4. `vrdataset/starterCode/assembleFeatureMatrix.py` overwrites a caller-provided `filePath` using `to_csv`. It is safe only when that path is outside `vrdataset`.

### Active experiment code

- Static search found reads from `vrdataset/dataPackage` in Phase 00-02 code but no experiment-script write, delete, rename, or move operation whose target is under `vrdataset`.
- Phase outputs are built from phase-owned `results`, `tables`, `figures`, and `logs` paths.
- Phase 01 and Phase 03 scripts can update root-level `EXPERIMENT_STATUS.md`, which is outside `experiments` but not raw data.
- Phase 00-03 scripts use overwrite-style `to_csv`, `write_text`, and figure-save calls. No backup logic was found. Re-running them can overwrite completed outputs, contrary to the completed-output backup rule unless the operator first makes backups or adds guards.

This is a static code audit, not proof of all historical execution. No file-system journal or full checksum comparison was available to prove that raw files were never changed in the past.

## G. Workflow Discrepancies

1. `README.md` and `EXPERIMENT_STATUS.md` describe Phases 04-10 and `experiments/shared/`, but the actual repository contains only Phases 00-03 and no `experiments/shared` directory.
2. `EXPERIMENT_STATUS.md` references `experiments/shared/reports/REVISED_EXPERIMENT_PLAN.md`, `PHASE_00_02_AUDIT_REPORT.json`, and `PHASE_00_02_AUDIT_NOTES.md`; those paths do not exist.
3. The expected shared config `experiments/shared/configs/experiment_config.yaml` does not exist. No project YAML config was found.
4. No `fold_assignments.csv` exists. Before any Phase 04+ modeling, subject-level outer folds must be created once, validated for zero subject overlap, saved, and reused by every classifier/regressor as required by `CODEX_RULES.md.txt`.
5. `requirements.txt` is unpinned, so the exact environment cannot be reproduced from it alone. `pyxdf` is listed but is not installed in the active environment.
6. The rules file has a double extension (`CODEX_RULES.md.txt`), causing exact-name discovery to fail.
7. Twelve generated HDC planning figures exist under root `output/figures`, outside the stated rule that generated figures and reports belong under `experiments`.
8. The Git working tree is not clean: tracked files under the former root `data/` are deleted, and most current project files are untracked. This verification did not alter or restore those unrelated changes.
9. Phase 01 is already marked complete, and Phases 02-03 outputs also exist. A request to begin Phase 01 should therefore be treated as a verification/re-audit decision, not as a blank new phase.

## NEXT PHASE REQUIREMENTS

- Preserve `vrdataset` as read-only and never execute `starterCode` in place when its output points into the source tree.
- Decide whether Phase 01 should be accepted as complete or rerun as a controlled audit; do not silently overwrite its existing outputs.
- Back up any Phase 01 outputs before rerunning the audit.
- Resolve the 908 files already listed for manual review before changing modality assignments.
- Do not create a control-input modality unless source evidence verifies it.
- Before Phase 04 modeling, create and validate one subject-grouped `fold_assignments.csv` and reuse it across all later experiments.

## VERIFIED

- Repository structure and prior Phase 00 outputs were inspected.
- Raw source locations and processed-data locations were identified.
- Existing notebooks, scripts, results, figures, tables, logs, and environment files were inventoried.
- Active experiment scripts have no statically detected writes into `vrdataset`.
- Starter reference code contains conditional or direct write operations and must not be executed in place against raw paths.
- Python and core package versions were recorded without installation or upgrade.
- No feature extraction or modeling was performed.

## NOT VERIFIED

- Historical proof that no process ever modified raw data; no file-system journal audit was available.
- Bit-for-bit validation of all 9,022 source files against `SHA256SUMS.txt`; this expensive checksum pass was outside the requested inventory audit.
- Scientific validity of existing Phase 01-03 findings; this task checked presence, placement, and recorded reports rather than recomputing analyses.
- Provenance of the root `output/figures` artifacts and temporary document-render directories.

## WARNINGS

- `vrdataset` is writable at the OS permission level despite the project read-only policy.
- Dataset-provided starter scripts can write into or delete files from their selected output/data directories.
- Existing phase scripts can overwrite completed outputs without automatic backup.
- `CODEX_RULES.md` is misnamed as `CODEX_RULES.md.txt`.
- Documented shared/config/report paths and future phase folders are absent.
- `pyxdf` is required but not installed; no installation was attempted.
- No saved subject-level fold assignment exists for later leakage-safe modeling.

## OUTPUT FILES

- `experiments/phase_00_project_setup/results/phase00_verification.md`
- `experiments/phase_00_project_setup/logs/phase00_verification_task_plan.md`
- `experiments/phase_00_project_setup/logs/phase00_verification_notes.md`

## PHASE 01 READINESS

CONDITIONALLY READY. The raw package, Phase 00 inventory, Phase 01 scripts/notebook, and prior Phase 01 outputs are present. However, Phase 01 is already recorded as complete, raw directories are not OS-protected, starter code has unsafe in-place write paths, and existing outputs lack automatic backup guards. Any Phase 01 rerun should first back up its current outputs and enforce an output-root check that rejects every path under `vrdataset`.
