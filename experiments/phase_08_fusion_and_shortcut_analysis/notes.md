# Notes: Phase 08 Initialization

## Scope
Initialization and static/read-only audits only. No training, prediction, OOF generation, or fusion execution is authorized.

## Actual Paths
- Phase 08: `E:\hdc-vr-pilot\experiments\phase_08_fusion_and_shortcut_analysis`
- Phase 03 inputs: `E:\hdc-vr-pilot\experiments\phase_03_multimodal_dataset_labeling`
- Phase 04A interface: `E:\hdc-vr-pilot\experiments\phase_04a_traditional_classification_baselines`
- Phase 04B interface: `E:\hdc-vr-pilot\experiments\phase_04b_traditional_regression_baselines`
- Phase 06 interface: `E:\hdc-vr-pilot\experiments\phase_06_hdc_variant_screening`
- Phase 07 interface: `E:\hdc-vr-pilot\experiments\phase_07_unimodal_contribution`

## Findings

### Actual SHA-256

- Primary without-performance: `0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44` — PASS
- Auxiliary with-performance: `72977a2119e30e37996fb9f0e3404988c4977fb7d2b33992f87bf54bfe5decba` — PASS
- Performance-only: `d602282ae41153886d1306494515f2e41a5e7e89a2cec5c192d44b9ca87a07a4` — PASS
- Frozen fold assignments: `e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f` — PASS
- With-performance manifest: `91cc99688b0b5dca74f6ebabfe1447548cab7f86a4919725f8eff80987a492b3` — PASS
- Performance-only manifest: `80c216dd6ece3f553d9a297dbda62b0505f0fa646c1ba4352abe9af553cb8b81` — PASS
- Phase 04A freeze: `34ea8100d9406f9701750a441aa6537323c28bcdb194cb3fd3645c4f7de4a2e1` — PASS
- Phase 04B freeze: `e2c88b1139a50aab6d47b6477c7bceff74f8443095f9d039ea9af84b715ee790` — PASS
- Phase 06 freeze: `cd94acdf0bcb463b5f48a7c84110a23059899b6b5bf156131694827d7d7bde66` — PASS
- Phase 07 freeze: `8569b48a8210f0ca1316d5a140d292edb892e2a556ea13ee299e9b97699af492` — PASS

### Data Scale and Alignment

- All three datasets contain 419 rows, 35 subjects, and 419 unique `run_key` values.
- Predictive feature counts are 1176 primary, 1235 with-performance, and 59 performance-only.
- Run keys, `subject_id`, targets, and frozen outer folds align one-to-one across all three datasets.
- `target_class = [0, 1, 2, 3]`; `target_score = [1.0, 2.0, 3.0, 4.0]`; targets have no missing values.
- Five frozen outer folds have zero train/test subject overlap; each outer-training partition supports `GroupKFold(n_splits=3, groups=subject_id)`.

### Feature-Set Relations

- With-performance equals the union of primary 1176 and performance 59.
- Primary/performance intersection size is 0.
- Performance-only exactly equals the frozen `performance_features` group.
- Manifest-derived Early Fusion counts pass: 649, 808, 1134, and full-primary 1176.
- All fusion sets exclude performance, target/identifier, control-input, and unverified features and contain no duplicate columns.

### Upstream Model Interfaces

- Phase 04A frozen best classifier: `Gradient Boosting`.
- Phase 04B frozen best regressor: `Gradient Boosting Regressor`.
- Phase 06 frozen classifier: `HDC+OnlineHD Hybrid`, dimension 5000.
- Phase 06 frozen regression head: `COMMON_ENCODER_READOUT_BASELINE`, dimension 10000.
- Phase 07 is frozen; best classification and regression modality is `flight_parameter_features`; all final audits pass and it is ready for the next planned phase.

### Static Performance-Feature Risk Inventory

- All 59 names, source groups, and missing rates were persisted.
- Reserved-field name collisions: 0.
- Label-adjacent name markers (`target`, `difficulty`, `level`, `class`, `score`): 0.
- Exact numeric target copies: 0.
- Deterministic target transforms found by the static mapping check: 0.
- These are static risk-screening results only; they do not prove absence of leakage, and later predictive ability must not be interpreted as physiological causal evidence.

## Contract Freeze Findings

### Flight Provenance

- All 326 frozen `flight_parameter_features` matched Phase 02 corrected provenance rows with `VERIFIED_OTHER` status.
- Raw source stream: `lslxp11xpcac / xplane_flight_state`.
- Raw schema evidence includes aircraft velocity, elevation/AGL altitude, trim, latitude/longitude, pitch/roll/yaw, indicated airspeed, groundspeed, climb rate, ILS deflections, and landing gear.
- Phase 02 documentation explicitly describes the stream as X-Plane aircraft-state summaries and reports no explicit joystick/yoke/throttle/rudder control-input stream.
- Frozen category counts: 323 `BEHAVIORAL_RESPONSE`, 0 `TASK_SETTING_OR_SCENARIO`, 3 `AMBIGUOUS`.
- The 3 ambiguous features are acquisition duration, sample count, and estimated sampling rate. They are excluded from behavioral-only and task-setting-only subsets but retained in `FLIGHT_FULL`.
- `FLIGHT_BEHAVIORAL_ONLY` is an authorized unique 323-feature sensitivity condition; `FLIGHT_TASK_SETTING_ONLY` is not feasible because the provenance group is empty.

### Frozen Run Matrix

- Five core conditions × 60 runs = 300 required model-runs.
- No reusable traditional flight-only artifact was found; HDC flight-only remains a Phase 07 frozen reference, while traditional flight-full requires 10 new fold runs.
- One unique nonempty flight sensitivity condition requires 60 runs.
- Dynamic expected total: `300 + 10 + 60 = 370` model-runs.
- Run manifest identifiers: 370 unique, 0 duplicates.

### Statistics and Shortcut Evidence

- Statistical unit is `subject_id` (`n=35`), with paired subject bootstrap (2000 repetitions, seed 42, percentile 95% CI), Wilcoxon signed-rank, Holm correction within family/model/task, and rank-biserial effect size.
- Comparison families remain separate: Early Fusion, performance shortcut, and flight provenance sensitivity.
- No single threshold can declare leakage. Direct leakage, added performance information, performance-only signal, and flight label-proximity risk have separate frozen evidence wording.

### Phase 09 Metadata Feasibility

- Unseen-session: `NOT_FEASIBLE_DUE_TO_METADATA`; 35 sessions are each perfectly nested in one of 35 subjects, so session and subject generalization cannot be separated.
- Unseen-scenario: `NOT_FEASIBLE_DUE_TO_METADATA`; no explicit scenario identifier exists.
- Task-template: `NOT_FEASIBLE_DUE_TO_METADATA`; only the common `task-ils` task is identifiable.
- Route/configuration: `NOT_FEASIBLE_DUE_TO_METADATA`; no explicit route or configuration identifier exists.
- Difficulty metadata is target-defining and is not an admissible scenario proxy.

### Contract Validation

- Static unit tests: 11/11 PASS.
- Contract Notebook: 16/16 code cells executed with persisted outputs; 0 error outputs.
- Contract artifact audit: PASS.
- Training artifacts added: 0; model training and outer-test prediction executed: NO.

## Risks or Anomalies
- No target-directory artifact conflict was present at initialization.
- A repository-wide recursive directory walk hit unreadable LibreOffice temporary directories; the scoped file search completed without finding `AGENTS.md`.
- The first audit implementation used a negative boolean inside an aggregate pass check; it was corrected and rerun successfully.
- Bundled workspace Python lacked Notebook packages. The existing Anaconda Python (`D:\Computer\anaconda3\python.exe`) supplied `nbformat`, `nbclient`, and the kernel; Notebook execution passed with 8/8 code cells persisted and zero error outputs.
- The execution request names `phase08_fusion_conditions.json` and `phase08_shortcut_conditions.json`, but Contract Freeze stored those objects only inside `phase08_frozen_contract.json`. Two read-only projections were materialized from frozen contract SHA-256 `93121979400646bcc9adffb15c069e8ec9a5f0e95e3dfb0cdcc99e3d3666031c`; no frozen decision or existing contract artifact was changed.

## Frozen Batch Execution

- Completed exactly 370/370 authorized runs: HDC classification 150, HDC regression 150, traditional classification 35, and traditional regression 35.
- Completed all five core fusion/shortcut conditions (60 runs each), `FLIGHT_BEHAVIORAL_ONLY` (60), and traditional-only `FLIGHT_FULL` (10).
- Persisted 31,006 raw outer-test prediction rows plus per-run metrics and hash-bound checkpoints; no failed-run events or recovered checkpoints were recorded.
- Independent executor, checkpoint-integrity, execution-coverage, leakage, artifact, and notebook-persistence audits pass.
- The outer test was never used for tuning. Canonical OOF consolidation and the registered subject-level analysis completed; Phase 09 was not executed. Status is `FROZEN`.

## Canonical OOF Consolidation and Final Analysis

- Reverified 370/370 frozen runs, 31,006 raw prediction rows, and all prediction/checkpoint SHA-256 bindings without modifying predictions.
- Consolidated 5,447 classification and 5,447 regression rows (10,894 total) under the frozen aggregation contract; HDC uses exactly five seeds per run key.
- Verified canonical coverage, cross-condition alignment, subject isolation, leakage invariants, and six upstream frozen-reference interfaces.
- Recomputed all registered metrics and subject-level inference independently. Wilcoxon, Holm correction within registered families, rank-biserial effects, and 2,000 paired-subject bootstrap intervals use `subject_id` (`n=35`).
- Persisted fusion, shortcut, flight behavioral-sensitivity, statistical, limitations, figure, report, and Phase 09 metadata-handoff artifacts.
- Safely appended and executed four read-only final-analysis Notebook cells; all four outputs persisted with zero errors, and prior cells were not rerun.
- Final artifact and reproducibility audits pass. No retraining, prediction regeneration, outer-test tuning, or Phase 09 execution occurred.
- Current status: `FROZEN`; ready for a separately authorized Phase 09 entry under the saved metadata limitations.

## Final Phase 08 Freeze

- Freeze preflight independently revalidated 370/370 runs, 31,006 raw prediction rows, 10,894 canonical OOF rows, all registered checksums, subject isolation, statistical audits, formal reports, five PDF/PNG figure pairs, and a zero-error Notebook.
- `manifests/phase08_final_manifest.json` inventories prediction, fold-metric, checkpoint, OOF, summary, report, figure, audit, Notebook, handoff, contract, dataset, and upstream-freeze SHA-256 values without embedding its own hash.
- `configs/phase08_freeze.json` records the final manifest hash and the registered interpretation/generalization guardrails.
- Raw predictions, canonical OOF, metrics, statistical results, reports, and Phase 03–07 freeze files remained byte-identical; all freeze audits pass.
- One isolated freeze-summary code cell was executed and persisted after 53 preserved historical Notebook cells, with no error output.
- Phase 09 was neither initialized nor executed. Phase 08 is `FROZEN` and the lifecycle gate reports ready to proceed to Phase 09 when separately authorized.
