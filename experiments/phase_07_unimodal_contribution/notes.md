# Notes: Phase 07 Initialization

## Scope
Initialization and read-only upstream auditing only. No model training, predictions, OOF generation, or fitted preprocessing is permitted.

## Initial Findings
- Phase 07 target directory had no pre-existing artifacts at conflict-check time.
- Repository notebook policy requires clean top-to-bottom execution with saved outputs and machine-readable evidence.
- Phase 03 and Phase 06 inputs will be referenced in place and never copied into Phase 07.

## Verified Paths and Checksums
- Experiment plan: `E:\hdc-vr-pilot\最新完整实验计划_分类回归双任务.md` — `1fde8fca7cb413bc49e5ab694eda12e5a3bdf6a960fb0114e6eafe4ced18559c`
- Primary data: `E:\hdc-vr-pilot\experiments\phase_03_multimodal_dataset_labeling\data\primary_without_performance.csv` — `0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44`
- Frozen folds: `E:\hdc-vr-pilot\experiments\phase_03_multimodal_dataset_labeling\data\fold_assignments.csv` — `e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f`
- Feature-group manifest: `E:\hdc-vr-pilot\experiments\phase_03_multimodal_dataset_labeling\manifests\feature_group_manifest.json` — `86a22464dc4725407d577db73649a5c4137482b1a5cc8836cbdffba842a8c456`
- Primary-feature manifest: `E:\hdc-vr-pilot\experiments\phase_03_multimodal_dataset_labeling\manifests\primary_feature_manifest.json` — `6808013449eb5bd6e278525ddc7c8231084a9b12a545170f387b669bde910995`
- Phase 06 freeze: `E:\hdc-vr-pilot\experiments\phase_06_hdc_variant_screening\configs\phase06_freeze.json` — `cd94acdf0bcb463b5f48a7c84110a23059899b6b5bf156131694827d7d7bde66`
- Phase 06 classification: `E:\hdc-vr-pilot\experiments\phase_06_hdc_variant_screening\configs\phase06_best_classification_hdc.json` — `174a99de2d993acdea49fdebc9647b28db4648ada2bea7a33f620f4677f031a4`
- Phase 06 regression: `E:\hdc-vr-pilot\experiments\phase_06_hdc_variant_screening\configs\phase06_best_regression_hdc.json` — `acde51709971d57c76eefaffcf1ecd571a4d4c5c36f8d76edf39841c5e7065b8`
- Phase 06 final artifact manifest: `E:\hdc-vr-pilot\experiments\phase_06_hdc_variant_screening\manifests\phase06_final_artifact_manifest.json` — `25df4d2754075061c4000c9260ad0b5b4182203f988375078e2f4156dcd580a5`
- Phase 06 selection-resolution audit: `E:\hdc-vr-pilot\experiments\phase_06_hdc_variant_screening\audits\phase06_model_selection_resolution_audit.json` — `d5095a3ec70490f248c5e4ecfc1584bdabfca2d4d7dd40dfff484868ea12e6da`

## Dataset and Fold Findings
- 419 modeling rows, 35 subjects, 419 unique run keys, and 1176 Primary predictive features.
- `target_class = [0, 1, 2, 3]`; `target_score = [1.0, 2.0, 3.0, 4.0]`; both targets have zero missing values.
- Frozen folds contain 419 rows and 419 unique run keys, align one-to-one with Primary data, and contain outer folds 1–5.
- Every outer fold has zero train/test subject overlap. Every outer-training set supports subject-isolated `GroupKFold(n_splits=3)`.

## Modality Findings
- Physiological: 233 features; 0 fully missing rows; 35 subjects with available data.
- Eye tracking: 416 features; 14 fully missing rows; 34 subjects with available data.
- Head movement: 159 features; 0 fully missing rows; 35 subjects with available data.
- Flight parameter: 326 features; 0 fully missing rows; 35 subjects with available data.
- Body movement: 42 features; 29 fully missing rows; 33 subjects with available data; frozen status honored as `VERIFIED_BODY_MOVEMENT` even though feature names contain `unknown`.
- The five modalities are disjoint and their union is exactly all 1176 Primary features. Every outer train/test partition retains available data for every modality.
- Performance intersection is 0; control-input count is 0; unverified count is 0.

## Phase 06 Interface Findings
- Freeze status is `FROZEN`; next-planned-phase readiness is true; selection evidence is `INNER_CV_ONLY`; outer OOF was not read for selection; no single seed was selected.
- Frozen evaluation seeds remain `[42, 43, 44, 45, 46]`.
- Classification: `hybrid` / `HDC+OnlineHD Hybrid`, dimension 5000, levels 51, feature_k 50, fold-local inner-CV structure selection.
- Regression: `common_ridge` / `COMMON_ENCODER_READOUT_BASELINE`, dimension 10000, levels 51, feature_k 50, fold-local inner-CV Ridge-alpha policy.
- Regression target wording is fixed as `bounded difficulty-induced workload proxy regression`.

## Notebook Persistence
- `Phase_07_Unimodal_Contribution.ipynb` executed cleanly and was saved with outputs.
- 18/18 code cells executed, 18/18 have outputs, and error output count is 0.
- Notebook SHA-256 after execution: `421d314aa2e02998e5f8981722ef637b26c4deb6b2b44e6c1c3c7402c6fde798`.
- All required JSON artifacts exist and parse successfully. `data/` and all `results/` subdirectories contain no files.
- HDC training, prediction, and OOF generation were not executed.

## Exceptions and Warnings
- Repository-wide recursive PowerShell discovery encountered inaccessible LibreOffice temporary directories under `.docx_tmp`; scoped inspection remains available and sufficient.
- The first execution attempt used Codex's bundled Python, which lacks `scikit-learn` and `nbformat`. The scripts passed syntax compilation, but execution stopped at imports before creating audit JSON or the notebook. An existing compatible environment will be reused; no package installation is planned.

## Contract Freeze Evidence
- Phase 05 encoding contract SHA-256: `8bffcbdcad5ef73778a5daf8eb64dd9dd1d8d90c675d10f9cbd72a1360f133ef`.
- Phase 06 variant contract SHA-256: `f9d0bd0f304678bbd00e3fdeaaf9b619511a2e54f9d426e21cc130691cca365b`.
- The full-cohort policy is frozen at 419 rows with fold-local missingness handling and required missing indicators; fully missing rows remain 0/14/0/0/29 across the five modalities.
- Preprocessing is frozen as median `SimpleImputer(add_indicator=True, keep_empty_features=True)`, zero-threshold variance filtering, training-fold scaling, and `SelectKBest(f_classif)` with `effective_feature_k = min(50, post_variance_feature_count)` for both tasks.
- Classification reuses the five Phase 06 fold-specific Hybrid structures at dimension 5000; regression reuses Common Encoder Ridge at dimension 10000 with alpha 0.01 for every fold and seed.
- Seeds are frozen at `[42, 43, 44, 45, 46]`; expected runs are 125 classification, 125 regression, and 250 total.
- Canonical seed aggregation, task-specific modality rankings, subject-level bootstrap/Friedman/Wilcoxon/Holm rules, multimodal reference deltas, and error-analysis requirements are frozen in machine-readable contracts.
- No checkpoint, prediction, or OOF result existed at contract creation. Training and prediction flags remain false.
- Static contract tests: 11 passed in 2.24 seconds.
- Contract artifact audit: PASS. Contract Notebook persistence: PASS with 12/12 appended code cells executed with outputs and 0 error outputs.
- Executed Contract Freeze Notebook SHA-256: `436e957387c35378996d0315b87681b72365e59fd5685b0f6465d4c5980da690`.
- Execution manifest remains at 0 completed runs, `training_executed=false`, and `executor_invoked=false`; checkpoint, prediction, and OOF directories remain empty.

## Unimodal Batch Execution
- Pre-training gates: Python compilation PASS, import check PASS, 18 static/unit tests PASS, and dry-run enumeration PASS with 125 classification + 125 regression = 250 unique runs.
- Core execution completed in one monitored sequential session. All five modalities completed 50/50 runs; every fold and every seed completed 50 runs.
- Final counts: 125 classification runs, 125 regression runs, 250 total runs, and 0 duplicate checkpoint IDs.
- Prediction coverage: 125 classification CSVs with 10,475 rows and 125 regression CSVs with 10,475 rows; total 20,950 seed-level rows.
- For every modality/task/seed, five outer folds combine to exactly 419 rows and 419 unique `run_key` values.
- All checkpoint integrity, artifact hashes, leakage audits, outer subject isolation, finite prediction, legal-label, and bounded-regression checks passed.
- Fully missing modality rows were retained. Eye tracking preserved 14 fully missing rows and body movement preserved 29; `modality_available` is saved per prediction for later stratified diagnostics.
- Body movement produced 78 post-variance columns in each observed fold due to frozen missing indicators; `effective_feature_k=50` was applied without dropping cohort rows.
- scikit-learn emitted its documented Windows/MKL KMeans memory warning during Hybrid initialization; execution remained stable and all 250 post-write checkpoint validations passed.
- No other HDC variant, inner-CV search, parameter reselection, best-seed selection, canonical OOF aggregation, ranking, or statistical test was executed.
- The first post-execution Notebook persistence attempt stopped on a stale historical Contract Freeze assertion requiring `completed_runs=0`. The actual manifest correctly reported 250. The assertion was lifecycle-corrected to allow 0 at freeze time or 250 after execution; no model artifact was changed or rerun.
- The corresponding historical contract unit test was also lifecycle-corrected after its first post-execution run: it now verifies either 0 runs with empty execution outputs or 250 runs with checkpoint/prediction outputs, while requiring canonical OOF to remain absent in both cases.
- Executed batch Notebook persistence: PASS, 6/6 execution code cells with outputs, 0 errors; SHA-256 `d4ed3d39304859122ae91967813d59acaaed49333dac06a8c328177ca7984282`.
