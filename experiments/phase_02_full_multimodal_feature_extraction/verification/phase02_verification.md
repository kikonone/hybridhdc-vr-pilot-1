# Phase 02 Multimodal Run-Level Feature Extraction Verification

The existing extraction was audited in place. No extraction rerun, source overwrite, or model training was performed.

## VERIFIED

- Raw extracted table: 487 rows x 1252 columns (1247 features plus 5 identifiers).
- Subjects: 35; unique run keys: 487; duplicate run-key rows: 0.
- Every Phase 03 modeling row is unique for subject-session-run-difficulty: duplicate rows=0.
- Modeling-eligible four-class rows: 419 with distribution 1=104, 2=106, 3=104, 4=105.
- Phase 02 feature NaN rate: 0.041185; infinite values: 0.

## NOT VERIFIED

- Explicit joystick/yoke/throttle/rudder control-input features: NOT VERIFIED; the existing control-input group is empty.
- Current Phase 03 scientific readiness is not verified because placeholder/zero-sample streams were treated as extracted in 38 run-modality pairs.

## EXCLUDED RUNS

- Raw extracted rows: 487.
- Final modeling rows: 419.
- Excluded: 68, all level-000 rest runs removed by the explicit Phase 03 four-class filter. No duplicate-key rows were removed.

## FEATURE GROUPS

- Original Phase 02 group counts: {"control_input_features": 0, "eye_tracking_features": 426, "flight_parameter_features": 328, "head_movement_features": 159, "identifier_columns": 5, "performance_features": 59, "physiological_features": 233, "unknown_features": 42}.
- Phase 03 with performance: 1235 features after removing 12 all-missing/high-missing columns.
- Phase 03 without performance: 1176 features.
- Constant/all-NaN columns are listed in constant_columns.csv and must be handled inside training folds where applicable.

## UNKNOWN FEATURES

- Decision A: all 42 formerly unknown features are VERIFIED body/torso movement aggregates.
- Evidence: DataDictionary.pdf section 6, explicit accelerometry_torso_x/y/z_mps2 raw columns, and per-feature extraction provenance.
- Unverified feature count after provenance audit: 0.

## PERFORMANCE FEATURES

- Verified count: 59; the number 59 is supported by the actual Phase 02 feature group and table, not forced.
- These comprise 4 cumulative PerfMetrics.csv values and 55 per-run performance-stream aggregates/metadata features.
- Performance features remain excluded from the primary Phase 03 list and included only in auxiliary lists.

## OUTPUT FILES

- phase02_verification.md
- phase01_verification.md
- verification_summary.json
- feature_provenance.csv
- feature_group_metadata.json
- feature_lists.json
- excluded_runs.csv
- verified_run_level_modality_availability.csv
- discrepancy_table.csv
- constant_columns.csv

## PHASE 03 READINESS

NOT READY. Preserve the original outputs, mask or minimally regenerate the affected placeholder/zero-sample modality features using the verified availability table, rebuild the Phase 03 datasets from that corrected input, and re-run this validation before any model training.
