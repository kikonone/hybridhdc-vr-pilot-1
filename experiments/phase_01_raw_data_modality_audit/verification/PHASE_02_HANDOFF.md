# Phase 02 Handoff

Phase 01 is frozen at 35 subjects and 487 unique raw runs. The 68 `level-000` rest runs are separate from the 419 Difficulty 1-4 modeling candidates. Explicit control input is unavailable. Torso/body accelerometry is a verified raw modality. Modality-specific missingness, 27 warning runs, and 38 zero-sample stream pairs must remain visible rather than being converted silently into valid observations.

Before Phase 02 outputs are accepted, Phase 02 must verify and record:

1. Why the extracted-run count differs from the modeling-run count, with exact run keys in each scope.
2. Whether extracted `unknown` feature columns originate from the now-verified torso/body accelerometer stream; provenance must be column-level, not name-only inference.
3. Whether performance features occur only for the 419 Difficulty 1-4 task runs and are absent from the 68 rest runs.
4. How all 27 warning runs and all 38 zero-sample streams appear after extraction: missing, NaN, placeholder-derived, failed, or excluded.
5. Whether `sub-cp030_ses-20211025_level-03B_run-005` appears in extracted outputs and can be supported by source provenance, because its raw run directory is physically absent.
6. Whether any runs were excluded, with the exact run key, source evidence, exclusion stage, and reason for every exclusion.

Retain the source label `sub-cp031_ses-20211216_level-02B_run-012`; do not manually relabel it. Do not create a control-input feature group without direct extracted-source evidence. Do not begin modeling until these checks pass.

Primary verification inputs:

- `results/run_modality_availability_verified.csv`
- `results/abnormal_files_verified.csv`
- `results/missing_run_directories_verified.csv`
- `results/unexpected_schedule_identifiers_verified.csv`
- `verification/verification_summary.json`
