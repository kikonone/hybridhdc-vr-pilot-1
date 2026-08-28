# Phase 01 Raw Data and Modality Audit Verification

Verified: 2026-08-19

## Method

The existing Phase 01 outputs were preserved and used only as the comparison baseline. The independent verification scanned the 487 run directories in `vrdataset/dataPackage`, parsed identifiers from directory structure, checked identifiers embedded in filenames, inspected CSV headers and first records, and checked paired `_hea.csv` metadata for sample count and effective rate. A modality is marked available only when content columns identify the signal and the paired metadata does not report zero samples.

The full 44 GB signal corpus was not loaded and no features were extracted. `vrdataset` was treated as read-only. `CODEX_RULES.md` was not present in the project root or the immediate `E:\` location after targeted searches.

## VERIFIED DATA COUNTS

| Measure | Verified result | Comparison with existing Phase 01 |
|---|---:|---|
| Files in `dataPackage` | 9,003 | Match |
| Files inside run directories | 9,002 | Newly separated from `task-ils/PerfMetrics.csv` |
| Subjects | 35 | Match |
| Subject-session pairs | 35 | Match |
| Raw run directories | 487 | Match |
| Unique composite `run_key` values | 487 | Match; no duplicates |
| `level-000` runs | 68 | Match |
| `level-01B` runs | 104 | Match |
| `level-02B` runs | 106 | Match |
| `level-03B` runs | 104 | Match |
| `level-04B` runs | 105 | Match |
| ILS performance-summary rows | 419 | Exact match to 419 per-run performance files; no duplicate or orphan keys |

Three run directories are physically absent relative to the repeated raw-data schedule:

- `sub-cp009_ses-20210129_level-000_run-002`
- `sub-cp030_ses-20211025_level-03B_run-005`
- `sub-cp036_ses-20211218_level-000_run-002`

There is also one schedule mapping variance for `sub-cp031`: the combination `level-01B_run-012` is absent and `level-02B_run-012` is present. The latter label is consistent across its directory, filenames, per-run performance file, and `PerfMetrics.csv`, so it is not a filename-only contradiction.

## VERIFIED MODALITIES

| Modality | Existing run count | Content-verified usable run count | Direct evidence |
|---|---:|---:|---|
| Any physiological signal | not separately reported | 487 | At least one usable ECG, EDA, EMG, or respiration stream |
| Eye tracking | 487 | 469 | Gaze, pupil, validity, fixation, or saccade columns; zero-sample placeholders excluded |
| ECG | 487 | 482 | `ecg_*` columns; zero-sample placeholders excluded |
| EDA/GSR | 475 | 470 | `eda_hand_l_kOhms` and related EDA columns; zero-sample placeholders excluded |
| EMG | 487 | 484 | `emg_*` columns; zero-sample placeholders excluded |
| Respiration | 487 | 482 | Shimmer or Respitrace `respiration_*` columns; zero-sample placeholders excluded |
| Head movement | 487 | 486 | X-Plane pilot head position and angle columns |
| X-Plane / flight state | 487 | 486 | Aircraft state, speed, attitude, altitude, and ILS columns |
| Performance | 419 | 419 | Glideslope, localizer, airspeed, and total-error columns |
| Torso/body accelerometer | labelled unknown | 454 | `accelerometry_torso_x/y/z_mps2` plus Data Dictionary Stream ACC definition |
| Explicit control input | 0 | 0 | No joystick, yoke, throttle, rudder, pedal, stick-position, or command stream found |

Torso accelerometry is no longer an unresolved modality label. The prior 908 unknown files are 454 data/header pairs for a documented torso accelerometer stream.

## UNVERIFIED MODALITIES

Explicit control input remains unverified and must be treated as unavailable. X-Plane trim and aircraft-state fields describe aircraft state, not direct pilot command measurements. The documentation notes that the left hand controlled the throttle and the right forearm controlled the joystick, but EDA and EMG are physiological measurements and are not explicit control-input streams.

No other unresolved modality labels remain after consulting content columns and the supplied Data Dictionary. Full signal quality, synchronization quality, and scientific usability beyond safe previews and metadata checks remain outside this verification.

## DISCREPANCIES

The discrepancy table contains 44 rows:

- 38 run-modality availability corrections caused by zero-sample placeholder streams: eye tracking 18, ECG 5, EDA 5, EMG 3, respiration 5, head movement 1, and X-Plane 1.
- 1 resolved modality-label discrepancy: 908 prior `unknown` files are 454 torso accelerometer data/header pairs.
- 3 physically absent run directories.
- 1 missing expected schedule combination for `cp031` and 1 corresponding unexpected substitute (`level-02B_run-012`).

The existing subject count, raw run count, difficulty distribution, unique run-key count, and performance-run count are verified as correct.

## RAW DATA ISSUES

- 27 runs contain 117 obvious metadata/content warnings.
- 38 stream pairs report `sampleCount=0` and contain placeholder data rows. These are files present on disk but signals unavailable for extraction.
- The 38 zero-sample streams comprise 18 eye, 5 ECG, 5 EDA, 5 Shimmer respiration, 3 EMG, 1 X-Plane aircraft-state, and 1 X-Plane pilot/head stream.
- One torso accelerometer stream is obviously short: `sub-cp017_ses-20210521_level-000_run-002`, approximately 8.919 seconds from header metadata.
- No unreadable run CSVs, identifier/path contradictions, duplicate composite run keys, missing data/header pairs, or unresolved file labels were found.
- The 40 invalid effective-rate warnings are retained in the abnormal-file table; most accompany zero-sample streams and should not be interpreted as usable sampling-rate estimates.

## OUTPUT FILES

- Verified run-level table: `results/run_modality_availability_verified.csv`
- Discrepancy table: `results/phase01_verification_discrepancies.csv`
- Verification report: `phase01_verification.md`
- Abnormal-file detail: `results/abnormal_files_verified.csv`
- Missing schedule entries: `results/missing_run_directories_verified.csv`
- Unexpected schedule identifiers: `results/unexpected_schedule_identifiers_verified.csv`
- Machine-readable summary: `verification/verification_summary.json`

The existing Phase 01 inventory, notebook, report, and run-level table were not overwritten.

## PHASE 02 READINESS

Do not treat the prior filename-based availability flags as extraction masks. A future Phase 02 rerun should treat zero-sample placeholders as missing, include torso accelerometry only as an explicitly named modality, and keep explicit control input unavailable. The `cp031` run-012 difficulty variance should be accepted or corrected only after domain confirmation because all available source evidence agrees on difficulty 2.

Core multimodal extraction remains feasible with per-run missingness, but full all-modality extraction is not feasible because explicit control input is absent and several streams are missing or zero-sample. Existing Phase 02/03 artifacts were not modified in this verification and should be reconciled with the verified availability masks before they are reused for modeling.
