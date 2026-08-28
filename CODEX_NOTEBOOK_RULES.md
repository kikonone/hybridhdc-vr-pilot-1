# Notebook-First Experimental Rules

This file is the authoritative workflow policy for this Master's thesis repository. It applies to every data-verification, feature-engineering, classification, regression, HDC, robustness, and final-analysis phase.

## Repository-Specific Boundaries

- The project root must be detected and validated from repository evidence before paths are resolved.
- `vrdataset/` contains raw/source data and reference material and is read-only.
- Existing phase-local `notebooks/` and `results/` directories under `experiments/` must be reused.
- Do not create `reports/`, `experiments/shared/`, or other planned directories merely to match a proposed structure. Continue using the verified existing project structure and create only a directory required by an actual operation.
- No model training may begin until the relevant validation gates in this document pass.

## 1. Evidence Before Assumption

Never assume that any file, directory, CSV, notebook, column, modality, feature, subject count, run count, sampling rate, model, dependency, metric, or result exists simply because it is mentioned in the research plan.

Always inspect the actual repository and actual data first. If something cannot be verified, label it `NOT VERIFIED`. Never invent missing information.

## 2. Actual Data Overrides Planned Counts

Research-plan values are expectations for validation, not targets that the data must be forced to match.

If actual verified data contradicts the plan:

- preserve the actual data;
- report the discrepancy;
- do not modify data to make the numbers match.

Never fabricate rows, features, modality assignments, labels, metrics, or results.

## 3. Raw Data Safety

Treat all raw/source data as `READ ONLY`.

Never overwrite, rename, delete, or edit raw source data. Processed outputs must go to existing processed/results directories. If no appropriate directory exists, create only the minimum required directory.

This protection must be enforced at the application level even when Windows permissions allow writes. Every future phase notebook or script must resolve its input and output paths and reject any write, delete, rename, move, or overwrite target located inside `vrdataset/`. Do not use a test write to verify this boundary.

Dataset-provided starter-code functions that contain in-place write or delete operations must never be executed against raw data or with an output path inside `vrdataset/`. If such logic is required, use a working copy outside `vrdataset/` and direct every generated artifact to an existing experiment-owned output directory.

## 4. Notebook-First Workflow

All experiment phases must use Jupyter Notebook (`.ipynb`) as the main execution interface.

Standalone `.py` files may be created only for reusable implementation code when necessary, for example `src/hdc/`, `src/evaluation/`, or `src/utils/`. Experiments must still be launched from notebooks, important steps must remain visible in notebook cells, metrics and plots must be visible in notebook outputs, and the notebook remains the experiment record.

## 5. Notebook Execution

Creating a notebook is not sufficient. Before a phase is considered complete:

- restart the kernel;
- execute all cells from top to bottom;
- fix every execution error;
- save the notebook with outputs;
- verify that output files actually exist.

Do not clear outputs and do not depend on hidden notebook state.

## 6. Cell Design

Prefer small logical cells:

- Markdown cell: what the section does;
- Code cell: one logical operation;
- Output: concise evidence.

Avoid one giant cell containing an entire experiment. Do not print huge arrays or entire large datasets.

## 7. Path Handling

Use `pathlib` where possible. Do not blindly assume that `Path.cwd()` is the project root. Detect or explicitly validate the project root first. Avoid machine-specific absolute paths unless already required by the project.

## 8. Machine-Readable Evidence

Notebook output is not enough. Every experiment must also save appropriate CSV, JSON, Parquet where appropriate, PNG, and/or PDF artifacts.

All plots must be generated from saved real results. Never manually type experimental metric values into plotting code.

## 9. Data Unit

The intended modeling unit is one `subject-session-run-difficulty` sample. Verify this from actual data; do not simply assume it is true.

## 10. Subject-Wise Generalization

`subject_id` is the grouping variable. Runs belonging to the same subject must never appear simultaneously in outer training and outer test data.

## 11. Classification Label

After verifying `difficulty_level`, define:

```text
target_class = difficulty_level - 1
```

Expected mapping:

- Difficulty 1 -> Class 0
- Difficulty 2 -> Class 1
- Difficulty 3 -> Class 2
- Difficulty 4 -> Class 3

Do not invent or interpolate labels.

## 12. Regression Label

Define:

```text
target_score = float(difficulty_level)
```

Expected observed targets are `1.0`, `2.0`, `3.0`, and `4.0`. Regression models may output continuous predictions.

Always preserve:

```text
prediction_raw
prediction_bounded = clip(prediction_raw, 1, 4)
```

Do not round predictions before computing primary regression metrics.

## 13. Interpretation Boundary

`target_score` is a bounded task-difficulty-induced workload proxy score. It must not be described as directly measured continuous cognitive workload, stress, clinical state, or psychological state unless independent evidence exists.

## 14. Primary Dataset

Primary thesis results must use the dataset without performance features. With-performance and performance-only datasets are auxiliary shortcut-learning analyses and must never silently replace the primary experiment.

## 15. Unknown Features

Do not assign unknown/torso accelerometer features to body movement without actual evidence. Allowed provenance status values are:

- `VERIFIED_BODY_MOVEMENT`
- `VERIFIED_OTHER`
- `UNVERIFIED`

`UNVERIFIED` features must remain outside the primary dataset.

## 16. Control Inputs

Do not invent a control-input modality. Create it only if actual extracted features and provenance verify that control-input data exists.

## 17. Outer Folds

Phase 03 must generate one frozen `fold_assignments.csv` using subject-wise `GroupKFold(n_splits=5)` if actual data supports the intended design.

After Phase 03, all later phases must reuse the exact same outer folds. Never regenerate outer folds independently for different models.

## 17A. Output Protection

- Never overwrite, mutate, or silently regenerate a frozen `fold_assignments.csv`. If the experimental design must change, preserve the frozen file and create a separately named version only after explicit intent is verified and documented.
- Do not overwrite an important existing experiment output until the operator's intent and the reproducibility impact have been verified.
- Before replacing an existing non-reproducible result, create a backup in the same phase-owned output area and record why replacement is necessary.
- Reproducible outputs may be regenerated only when the intended replacement is clear and raw-data and fold protections remain satisfied.

## 18. Inner Validation

Inside every outer training fold, use subject-wise inner validation where required. The intended design is `GroupKFold(n_splits=3)` using `subject_id`.

The outer test fold must not participate in hyperparameter selection, feature selection, threshold selection, temperature selection, or centroid-number selection.

## 19. Leakage-Safe Preprocessing

Any fitted operation must be fitted only using outer/inner training data as appropriate. This includes median imputation, missing indicators, variance filtering, scaling, feature selection, class weighting, and target transformation.

Do not fit preprocessing globally before cross-validation.

## 20. Feature Selection

Where feature selection is used, planned candidate values are `50`, `100`, `200`, and `all`. Feature selection must occur inside training folds.

## 21. Randomness

Record every random seed. Do not change seeds because results look poor.

Final stochastic HDC evaluations are intended to use predefined seeds such as `42`, `43`, `44`, `45`, and `46`, but first verify whether the project already defines the official seed list.

## 22. Model Results

Never generate synthetic experimental results. Never write expected Accuracy, F1, MAE, or other result values into Markdown before the experiment runs. Never write conclusions before real results are available.

## 23. Timing and Memory

Only report training time, inference time, peak memory, or model size if actually measured. If not measured, report `NOT MEASURED`. Do not estimate values.

## 24. HDC Algorithm Safety

Never invent mathematical HDC algorithms. For OnlineHD-style, Multi-centroid HDC, Hybrid HDC, or any other algorithm, first search local source code, documents, papers, notes, and existing implementations.

If the exact update rule cannot be verified, do not silently invent one. Report `ALGORITHM DEFINITION NOT VERIFIED` and identify the missing definition.

## 25. Experiment Record

Every experiment result must include, when applicable:

- phase;
- timestamp;
- dataset version;
- input path;
- number of samples;
- number of subjects;
- number of features;
- feature group;
- outer fold;
- inner CV;
- model;
- hyperparameters;
- seed;
- target;
- preprocessing;
- metrics;
- training time;
- inference time;
- output path.

## 26. OOF Predictions

Save out-of-fold predictions whenever a model is evaluated using outer cross-validation. OOF predictions are the primary evidence for later statistical analysis.

## 27. Smoke Test First

Before expensive experiments:

- verify inputs;
- verify splits;
- run a minimal smoke test;
- inspect results;
- then run the full experiment.

## 27A. Dependency Restraint

Do not install or use `pyxdf` preemptively. Only consider it during a later Phase 01 or Phase 02 operation when an actual verified input or processing step requires XDF support. Inspect the active environment first and do not install or upgrade packages merely to match a planned dependency list.

## 28. Validation Summary

Every notebook must end with:

```markdown
## Phase Validation Summary

VERIFIED
NOT VERIFIED
WARNINGS
KEY RESULTS
OUTPUT FILES
NEXT PHASE REQUIREMENTS
```

Only include information supported by executed notebook outputs.

## 29. Self-Audit

Before completing every phase, check:

- subject leakage;
- duplicated samples;
- label validity;
- train/test overlap;
- preprocessing leakage;
- missing outputs;
- overwritten source data;
- unverified assumptions;
- notebook execution status.

## 30. Stop Conditions

If a critical prerequisite is absent or scientifically ambiguous, do not invent a workaround merely to make the experiment run. Complete all safe verification steps and clearly report the blocker.

## Notebook Naming

Prefer the existing phase-specific notebook directory. Do not create duplicate notebooks when an equivalent notebook already exists.

If no equivalent notebook exists, use:

```text
Phase_00_Project_Verification.ipynb
Phase_01_Data_Audit_Verification.ipynb
Phase_02_Feature_Verification.ipynb
Phase_03_Dataset_and_Folds.ipynb
Phase_04A_Classification_Baselines.ipynb
Phase_04B_Regression_Baselines.ipynb
Phase_05_Vanilla_Dual_Output_HDC.ipynb
Phase_06_HDC_Variants.ipynb
Phase_07_Single_Modality.ipynb
Phase_08_Fusion_and_Shortcut.ipynb
Phase_09_Robustness_and_LOSO.ipynb
Phase_10_Final_Analysis.ipynb
```

## Phase Completion Gate

A phase is complete only when its notebook has been cleanly executed from top to bottom, its outputs and saved artifacts are verified, its validation summary and self-audit are complete, and no unresolved stop condition invalidates the phase.
