# Notes: Pre-Phase-02 Verification

## Rule File
- Canonical path: `E:/hdc-vr-pilot/CODEX_NOTEBOOK_RULES.md`.
- It is the only repository file matching the Codex/notebook rules search.
- It contains Notebook-first, evidence-before-assumption, raw-data safety, anti-hallucination, leakage, output-protection, and phase-gate rules.
- `Test-Path` from the project root returned true.

## File Count Reconciliation
- Phase 00 scope: every file recursively under `vrdataset` = 9,022.
- Phase 01 scope: every target file under `vrdataset/dataPackage` = 9,003.
- Difference = 19 files outside `dataPackage`:
  - 2 root legal/integrity files;
  - 12 starter-code files;
  - 5 reference-document PDFs.

## Raw Integrity
- Phase 00 inventory versus current source tree: 9,022 versus 9,022.
- Missing paths: 0; extra paths: 0; size changes: 0; timestamp changes: 0.
- Dataset `SHA256SUMS.txt`: 9,021 entries checked; missing: 0; hash mismatches: 0.
- The manifest itself is the 9,022nd source file.

## Frozen Phase 01 Evidence
- Subjects: 35; unique raw runs: 487.
- Difficulty counts: 68, 104, 106, 104, 105 for level-000 through level-04B.
- Difficulty 1-4 candidates: 419.
- Explicit control input: unavailable.
- Torso/body accelerometer: verified raw modality.
- Warning runs: 27; zero-sample stream pairs: 38.
- cp031 run-012 remains source-labelled level-02B and was not manually corrected.
