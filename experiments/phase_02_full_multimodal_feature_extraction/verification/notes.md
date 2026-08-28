# Notes: Phase 02 Verification

## Scope
- Verify existing outputs before considering any extraction rerun.
- Cross-check Phase 01 run/modality evidence against Phase 02 and Phase 03 artifacts.
- Do not train models.

## Evidence Log
- `CODEX_RULES.md` is absent; `CODEX_RULES.md.txt` was read in full before the research audit.
- Inspected Phase 01 verified availability/discrepancy files and verification script.
- Inspected Phase 02 extraction code, notebook outputs, wide table, long provenance table, feature groups, processed/skipped/failed files, logs, and report.
- Inspected Phase 03 construction code, cleaned tables, feature lists/groups, high-missing removals, leakage report, and construction report.
- Inspected DataDictionary.pdf sections for EMG, EDA, ECG, respiration, ACC, X-Plane, and performance.
- Inspected representative raw torso accelerometry and performance CSV contents.

## Findings
- Phase 02 table: 487 rows, 1,252 columns, 35 subjects, 487 unique run keys, no duplicate composite keys, no missing identifiers, 4.1185% feature-cell NaN rate, and no infinities.
- Difficulty distribution is 68 rest rows plus task levels 104/106/104/105. Phase 03 excludes exactly the 68 `level-000` rest rows; no duplicate rows were removed.
- Phase 03 tables contain 419 rows. With performance: 1,235 features. Without performance: 1,176 features.
- Phase 02 has 12 all-NaN features and 214 other single-value features. The Phase 03 primary table still has 213 single-value features.
- Feature counts by prefix: ECG 57, EDA-stream 44, EMG-stream 84, respiration 48, eye 426, head 159, X-Plane 328, performance 59, torso/body 42.
- The 42 former unknown features are verified torso/body accelerometry (Decision A). All remain in the current primary without-performance feature list.
- Performance count 59 is verified: 4 cumulative summary values plus 55 per-run performance aggregates/metadata features. None appear in the primary without-performance list.
- Explicit control-input features are not verified: 0 source runs and 0 extracted features.
- Phase 02 overstates content-verified modality availability in 38 pairs across 26 runs: eye 18, ECG 5, EDA 5, respiration 5, EMG 3, head 1, X-Plane 1.
- Phase 01 content verification reports 27 runs with obvious abnormal/missing file observations and 0 unresolved raw modality labels.
- Phase 03 readiness is NOT READY until affected placeholder/zero-sample modality cells are corrected and the Phase 03 datasets are rebuilt and reverified.

## Placeholder Repair Execution
- Created and executed `notebooks/Phase_02_Feature_Verification.ipynb` from a fresh `python3` kernel; all 12 code cells executed in order with zero error outputs.
- Recalculated exactly 38 Phase 01 unavailable / Phase 02 present run-modality pairs across 26 runs from the verified evidence.
- Masked only modality-dependent cells for those pairs, including signal-derived and invalid placeholder stream metadata; every changed value became NaN.
- Changed-cell assertions prove that no identifier, run metadata, unaffected run, or valid modality cell changed.
- No minimal regeneration was needed and the full extraction pipeline was not run.
- Corrected quality: 487 rows, 1,247 features, 35 subjects, 487 unique run keys, 0 duplicates, 0 missing identifiers, 0 infinities, and 5.1711788% feature-cell NaN rate.
- All-NaN features remain 12 and are marked `STRUCTURALLY_UNUSABLE` in the manifest while remaining present in the corrected table.
- Global single-value columns increase from 214 to 242 after masking; they are recorded and retained for later fold-local filtering.
- The prior 214-to-213 difference is exactly eight excluded constant performance metadata features plus seven eye/X-Plane features made constant by filtering out rest rows.
- Canonical body-movement group contains 42 `VERIFIED_BODY_MOVEMENT` features; performance contains 59 features; control input contains 0 features.
- Primary manifest contains 1,176 features and has an empty intersection with the 59 performance features.
- All 487 existing Phase 02 rows remain: 68 level-000/rest and 419 Difficulty 1-4 rows.
- cp030 level-03B run-005 is absent from both original and corrected Phase 02 tables and is independently verified as a physically absent raw run, so it is not a modeling candidate.
- Raw-data metadata fingerprint and original Phase 02 table SHA-256 match before/after values.
- Corrected Phase 02 outputs pass all 19 critical assertions. Phase 03 was not executed and no model was trained.
