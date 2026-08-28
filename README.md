# HDC VR Pilot

A reproducible research repository for multimodal physiological and behavioral data collected during VR piloting tasks. The project uses subject-isolated evaluation protocols to compare conventional machine-learning methods with Hyperdimensional Computing (HDC), focusing on:

- four-level task-difficulty proxy classification;
- bounded task-difficulty proxy regression on a 1–4 scale;
- unimodal contribution, multimodal fusion, shortcut sensitivity, missing-modality robustness, and cross-subject generalization;
- final evidence synthesis, statistical summaries, and a local demonstration UI.

Task difficulty is treated as a **workload proxy** and must not be interpreted as a direct measurement of psychological workload.

## Data and Sharing Restrictions

The source data come from the PhysioNet VR Piloting dataset. The complete dataset is not included in this repository:

- `vrdataset/dataPackage/` is a local, read-only source-data directory and is excluded from Git;
- the oversized `vrdataset/referenceDocuments/DataQualityReport.pdf` is excluded;
- the oversized `feature_extraction_long_table.csv` output is excluded;
- manuscript files, Word documents, rendered PDFs, and presentation-production materials are outside the scope of this code repository.

To reproduce the experiments, obtain the source data under the applicable dataset license and place them in `vrdataset/dataPackage/`.

## Repository Structure

```text
.
├── README.md
├── EXPERIMENT_STATUS.md
├── requirements.txt
├── experiments/
│   ├── project_setup/
│   ├── raw_data_modality_audit/
│   ├── full_multimodal_feature_extraction/
│   ├── multimodal_dataset_labeling/
│   ├── traditional_classification_baselines/
│   ├── traditional_regression_baselines/
│   ├── basic_dual_output_hdc/
│   ├── hdc_variant_screening/
│   ├── unimodal_contribution/
│   ├── fusion_and_shortcut_analysis/
│   ├── robustness_and_generalization/
│   └── demo/
└── vrdataset/
    ├── referenceDocuments/
    └── starterCode/
```

Each experimental phase retains its own README, scripts, notebooks, configurations, audits, results, and figures. Phase-local freeze records and audit files are the authoritative sources for the status of each phase.

## Core Data Protocol

Phase 03 produces frozen subject-level splits and three modeling datasets:

- primary dataset: `experiments/multimodal_dataset_labeling/data/primary_without_performance.csv`
- auxiliary dataset with performance metrics: `experiments/multimodal_dataset_labeling/data/auxiliary_with_performance.csv`
- performance-only dataset: `experiments/multimodal_dataset_labeling/data/performance_only.csv`
- frozen outer-fold assignments: `experiments/multimodal_dataset_labeling/data/fold_assignments.csv`

The primary dataset contains 419 runs from 35 subjects and 1,176 predictor features. Missing-value handling, scaling, feature selection, and model fitting must be performed within the training folds only.

## Environment Setup

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Some phases have additional dependencies. Review the corresponding phase README or phase-specific `requirements.txt` before execution.

## Usage

1. Place the licensed source data in `vrdataset/dataPackage/`.
2. Review `EXPERIMENT_STATUS.md` and the README for the target phase.
3. Use the frozen configurations, execution scripts, and verification scripts provided within that phase.
4. Do not reselect models, dimensions, or random seeds using outer-test performance.
5. Store generated artifacts in the corresponding phase directory; do not write outputs into `vrdataset/`.


## Methodological Principles

- Source data remain read-only
- The primary analysis excludes performance metrics; auxiliary analyses assess shortcut-learning risk separately.
- Evaluation uses subject-isolated cross-validation, with leave-one-subject-out analysis as supplementary evidence.
- Model selection relies only on training-side evidence.
- Splits, configurations, predictions, statistical results, audits, and hashes are retained to support reproducibility.
- Classification and regression targets are task-difficulty proxies; conclusions must not overstate them as direct psychological measurements.

## License and Data Use

The code and included experimental artifacts are intended for research reproduction and method development. The source dataset remains subject to its original license. Users are responsible for confirming all access, use, and redistribution requirements.
