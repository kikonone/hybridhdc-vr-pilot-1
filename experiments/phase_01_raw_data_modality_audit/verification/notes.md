# Notes: Phase 01 Verification

## Safety
- Raw source: `vrdataset/dataPackage`
- Raw files are read-only for this task.
- Existing Phase 01 outputs are preserved.

## Baseline Claims To Verify
- Subjects: 35
- Sessions: 35
- Runs: 487
- Unknown files: 908 torso accelerometer streams
- Explicit control input: unavailable
- Strict complete runs across the original requested modalities: 0

## Repository Rules
- `CODEX_RULES.md` could not be located at the project root or immediate drive root.

## Verified Findings
- 35 subjects, 35 subject-session pairs, 487 unique raw run directories, and 9,003 dataPackage files.
- Difficulty counts match the prior audit: 68 rest, then 104/106/104/105 for levels 01B/02B/03B/04B.
- Three physically absent run directories: cp009 rest-002, cp030 level-03B run-005, and cp036 rest-002.
- cp031 run-012 is consistently labelled level-02B rather than the level-01B mapping used elsewhere.
- Content-verified usable run counts: eye 469, ECG 482, EDA 470, EMG 484, respiration 482, head movement 486, X-Plane 486, performance 419, torso accelerometer 454, explicit control input 0.
- The 908 prior unknown files are 454 documented torso accelerometer data/header pairs.
- 27 runs have 117 obvious warnings, including 38 zero-sample stream pairs and one 8.919-second torso accelerometer stream.
- Original Phase 01 outputs were preserved.

## Preservation Evidence
- Existing `run_modality_availability.csv` SHA-256: `3BBAE859BE4D413509DBC217343BF6D416E36670A7E9B8F307E25A6AC9A19CBE`.
- Existing `raw_file_inventory.csv` SHA-256: `04B527D8EEE93AB64A80801169B98EE59C8C3596E05F984EA290DE305132794E`.
- Existing Phase 01 notebook and baseline CSV timestamps remained from July 2026.

## Output Validation
- Verified run-level rows: 487 with 39 columns.
- Discrepancy rows: 44.
- Abnormal observation rows: 117.
- Missing schedule rows: 4, comprising 3 physical absences and 1 mapping hole.
- Unexpected schedule rows: 1.
