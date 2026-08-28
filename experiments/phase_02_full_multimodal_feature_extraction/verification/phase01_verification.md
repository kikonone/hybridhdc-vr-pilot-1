# Phase 01 Verification

This verification reuses the independent content-aware Phase 01 audit and compares it with the existing Phase 02 extraction outputs. Original raw data and completed outputs were not modified.

## VERIFIED DATA COUNTS

- Subjects: 35
- Raw run directories / Phase 02 rows: 487
- Difficulty distribution: 0=68, 1=104, 2=106, 3=104, 4=105
- Duplicate run keys: 0
- Missing identifiers in existing Phase 02 rows: 0

## VERIFIED MODALITIES

- Physiological evidence: ECG, EDA/GSR, EMG, respiration; PPG is also present inside the EDA stream.
- Eye tracking, head movement, X-Plane flight-state, performance, and torso/body accelerometry are content-verified where their run-level flags are true.
- Torso/body accelerometry is verified from DataDictionary.pdf section 6 and explicit accelerometry_torso_x/y/z_mps2 columns.

## UNVERIFIED MODALITIES

- Explicit control input: NOT VERIFIED and unavailable (0 runs, 0 extracted features). References to a hand controlling a joystick or throttle describe the experimental setup, not a recorded control-input stream.
- Unresolved raw modality labels after the content-aware Phase 01 audit: 0 runs.

## DISCREPANCIES

- Existing Phase 02 modality availability overstates content-verified availability in 38 run-modality pairs across 26 runs.
- Mismatch counts: {"ecg": 5, "eda": 5, "emg": 3, "eye_tracking": 18, "head_movement": 1, "respiration": 5, "xplane": 1}.
- The earlier unknown torso stream is resolvable as torso/body accelerometry; it should not remain an unknown modality in verified metadata.

## RAW DATA ISSUES

- Runs with obvious abnormal/missing file observations: 27.
- Three physically absent scheduled run directories and one schedule-mapping hole are recorded in the Phase 01 discrepancy files.
- One internally consistent but unexpected difficulty/run mapping exists for cp031 run-012.

## OUTPUT FILES

- verified_run_level_modality_availability.csv
- discrepancy_table.csv
- verification_summary.json
- feature_provenance.csv

## PHASE 02 READINESS

CONDITIONAL. The raw inventory is sufficient for extraction, but availability must use content/metadata validation. Existing Phase 02 values from placeholder or zero-sample streams require masking or correction before downstream modeling datasets are accepted.
