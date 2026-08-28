# Task Plan: Phase 03 Dataset and Folds

## Goal
Create and execute the canonical Phase 03 notebook using only the corrected Phase 02 table, then freeze validated subject-wise outer folds and manifests.

## Phases
- [x] Phase 1: Read notebook rules and inspect Phase 02 corrected artifacts.
- [x] Phase 2: Build the Phase 03 notebook and output layout.
- [x] Phase 3: Execute from a fresh kernel and create audits/manifests/folds.
- [x] Phase 4: Verify artifacts, update status, and report readiness.

## Decisions Made
- Canonical input is the Phase 02 corrected v1 table verified by its corrected validation summary and modeling manifest.
- The primary manifest excludes performance, structurally unusable, and unverified features.
- The outer split is frozen as subject-wise GroupKFold with five folds.

## Status
**Complete** - extended Phase 03 verification executed from a fresh kernel; repaired-modality, missingness, leakage, outer-fold, and inner-CV evidence were added without changing the frozen fold assignment.
