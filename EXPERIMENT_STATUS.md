# Experiment Status

## Current repository state

The repository contains the Phase 00–10 experiment structure, including data auditing, feature extraction, frozen subject-wise datasets and folds, traditional baselines, dual-output HDC experiments, variant screening, modality/fusion analyses, robustness and generalization analyses, final synthesis, reproducibility artifacts, and a local demo UI.

Phase-local README files, manifests, freeze records, and audit outputs are authoritative. Some phase READMEs preserve historical gate text to document the state at the time of that gate; later freeze artifacts in the same phase supersede those interim statements.

## Phase inventory

| Phase | Folder | Repository role |
|---:|---|---|
| 00 | `phase_00_project_setup` | Project inventory and setup |
| 01 | `phase_01_raw_data_modality_audit` | Raw-file and modality audit |
| 02 | `phase_02_full_multimodal_feature_extraction` | Run-level multimodal feature extraction |
| 03 | `phase_03_multimodal_dataset_labeling` | Leakage-controlled datasets and frozen subject folds |
| 04A | `phase_04a_traditional_classification_baselines` | Traditional classification baselines |
| 04B | `phase_04b_traditional_regression_baselines` | Traditional regression baselines |
| 05 | `phase_05_basic_dual_output_hdc` | Basic dual-output HDC evaluation |
| 06 | `phase_06_hdc_variant_screening` | HDC family screening and frozen selection |
| 07 | `phase_07_unimodal_contribution` | Unimodal contribution analysis |
| 08 | `phase_08_fusion_and_shortcut_analysis` | Fusion and shortcut-sensitivity analysis |
| 09 | `phase_09_robustness_and_generalization` | Missing-modality and LOSO evaluation |
| 10 | `phase_10_final_synthesis_and_demo_ui` | Final synthesis, reproducibility package, and demo UI |

## Canonical dataset facts

- Modeling runs: 419
- Subjects: 35
- Difficulty classes: 4
- Primary predictors without performance metrics: 1,176
- Auxiliary predictors with performance metrics: 1,235
- Frozen evaluation design: subject-isolated folds, with LOSO as a supplementary analysis

Canonical Phase 03 files:

- `experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv`
- `experiments/phase_03_multimodal_dataset_labeling/data/auxiliary_with_performance.csv`
- `experiments/phase_03_multimodal_dataset_labeling/data/performance_only.csv`
- `experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv`

## Final synthesis entry points

- Notebook: `experiments/phase_10_final_synthesis_and_demo_ui/Phase_10_Final_Synthesis_and_Demo_UI.ipynb`
- Final synthesis report: `experiments/phase_10_final_synthesis_and_demo_ui/reports/phase10_final_synthesis_report.md`
- Freeze summary: `experiments/phase_10_final_synthesis_and_demo_ui/reports/phase10_final_freeze_summary.md`
- Reproducibility package: `experiments/phase_10_final_synthesis_and_demo_ui/reproducibility_package/`
- Demo UI: `experiments/phase_10_final_synthesis_and_demo_ui/ui/`

## GitHub scope

The restricted raw dataset, oversized generated tables, local tool dependencies, caches, manuscript files, document-rendering artifacts, and presentation files are excluded from Git. Experiment code, configurations, compact result artifacts, figures, audits, and reproduction documentation remain in scope.
