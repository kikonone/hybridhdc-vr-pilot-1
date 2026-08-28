# Task Plan: Phase 02 Feature Extraction Verification

## Goal
Repair verified Phase 02 placeholder/zero-sample feature inconsistencies by conservative masking, preserve all source outputs, and produce an executed Phase 02 verification notebook plus modeling-ready Phase 02 artifacts without starting Phase 03 or modeling.

## Phases
- [x] Phase 1: Read repository rules and establish a non-destructive verification workspace.
- [x] Phase 2: Inventory and inspect Phase 01/02/03 outputs, scripts, logs, reports, and source documentation.
- [x] Phase 3: Audit run keys, counts, labels, missingness, infinities, constants, feature groups, provenance, and exclusions.
- [x] Phase 4: Generate machine-readable metadata and Phase 01/02 verification reports.
- [x] Phase 5: Re-run verification checks, self-audit deliverables, and report readiness.
- [x] Phase 6: Read `CODEX_NOTEBOOK_RULES.md` and inspect the requested notebook target and existing Phase 02 evidence.
- [x] Phase 7: Implement the provenance-driven placeholder repair and canonical body-movement metadata in `Phase_02_Feature_Verification.ipynb`.
- [x] Phase 8: Restart the kernel, execute all notebook cells, and fix all execution errors.
- [x] Phase 9: Validate corrected artifacts, critical assertions, run integrity, and before/after evidence.
- [x] Phase 10: Complete the Phase 02 repair record and handoff without running Phase 03.

## Key Questions
1. What are the verified raw-extracted and modeling-eligible run counts, and what evidence explains the difference?
2. Does every modeling row uniquely represent subject-session-run-difficulty?
3. Which feature modalities and performance features are directly verifiable?
4. What are the approximately 42 unknown/body-movement candidates, and what evidence supports their status?
5. Do current Phase 01 findings agree with actual source files and Phase 02 outputs?

## Decisions Made
- Treat `CODEX_RULES.md.txt` as the intended rules file because `CODEX_RULES.md` is absent.
- Preserve existing completed outputs; place new verification artifacts under this `verification` directory.
- Interpret the appended Phase 01 requirements as a mandatory prerequisite/cross-check for Phase 02 verification; do not run extraction or modeling unless corruption is proven.
- Classify all 42 former unknown features as Decision A, verified torso/body accelerometry, based on the Data Dictionary, raw columns, and extraction provenance.
- Do not accept existing Phase 03 datasets for modeling until 38 placeholder/zero-sample run-modality pairs are masked or minimally regenerated and the datasets are rebuilt.
- Create a new notebook named `Phase_02_Feature_Verification.ipynb` because no equivalent verification notebook exists; preserve the original extraction notebook.
- Mask modality-dependent signal and stream-metadata values for content-verified unavailable streams; never mask identifiers, run metadata, or valid modalities.
- Preserve all 487 existing Phase 02 rows and the original Phase 02 table; create a corrected versioned derived table and separate modeling manifest.
- Explain the 214-to-213 single-value difference by exact set comparison: eight constant performance fields are excluded from the primary list and seven eye/X-Plane fields become constant after the task-only row filter.

## Errors Encountered
- `CODEX_RULES.md` was not found; the repository contains `CODEX_RULES.md.txt`, which was read in full.
- No existing `Phase_02_Feature_Verification.ipynb` was found; creation is required by the notebook naming policy and current request.
- The bundled document-runtime Python lacks `nbformat`; the existing Anaconda project environment provides `nbformat 5.10.4`, `nbclient 0.8.0`, Jupyter, and the `python3` kernel, so no dependency installation is required.
- The first generated notebook had an escaped-newline syntax error in the raw fingerprint cell; the builder was corrected, the notebook regenerated, and all code cells passed `ast.parse` before execution.
- The first clean-kernel launch failed before cell execution because Windows denied Jupyter `SetFileSecurity` on its default connection file. The Phase 02 executor now uses a phase-owned runtime directory and Jupyter's insecure-write fallback for this sandbox.

## Status
**Complete** - the notebook executed from a fresh kernel with 12/12 code cells and zero errors; all critical assertions passed, corrected Phase 02 artifacts were verified, and Phase 03 was not started.
