# Phase 02: Full Multimodal Run-Level Feature Extraction

## Status
Complete. The notebook was executed and saved with visible outputs.

## Source Data Rule
`vrdataset/dataPackage` was treated as read-only. All generated artifacts are inside this Phase 02 experiment folder.

## Extracted Modalities
Successfully extracted modalities: eye_tracking, ecg, eda, emg, respiration, head_movement, xplane, performance, unknown.
Unavailable or not extracted as known modalities: control_input.

## Is The Table Truly Multimodal?
Yes. The output has 487 run rows and 1247 extracted feature columns. Missing modalities are preserved with NaN values.

## Feature Families
Physiological features include ECG, EDA/GSR, EMG, and respiration aggregate statistics plus approximate peak or zero-crossing summaries where feasible.
Eye-tracking features include gaze, pupil diameter, eye openness, eye position, missingness, and available fixation/saccade sequence summaries.
Head movement features include pilot head position/attitude summaries and first-difference summaries when explicit velocity columns were unavailable.
Flight/control features include X-Plane aircraft state summaries. No explicit joystick, yoke, throttle, rudder, or control-input streams were available after extraction.
Performance features include per-run performance metric streams and cumulative glideslope, localizer, airspeed, and total error fields from `PerfMetrics.csv`.
Unknown features contain torso accelerometer streams that Phase 01 marked unknown; they were not forced into a known modality.

## Performance Feature Caution
Performance metrics may encode task outcome information. Phase 03 must build two dataset versions: with performance features and without performance features.

## Why NaN Values Remain
Phase 02 does not impute missing values because missingness must be handled inside Phase 03/model pipelines to avoid leakage.

## Failed Or Skipped Files
Failed files: 0. Header metadata and non-run files were skipped as non-model-feature sources.

## Phase 03 Next Step
Construct the final four-class modeling dataset, create with-performance and without-performance versions, perform controlled imputation/scaling inside modeling pipelines, and then proceed to baseline ML/HDC experiments.
