# Phase 06 Final Confirmation Execution Plan

- [x] Preflight all frozen contracts, Quick Screen outputs, audits, notebook, and source checksums.
- [x] Snapshot every Quick Screen manifest artifact before training.
- [x] Run resumable OnlineHD Final Confirmation, outer folds 1 through 5.
- [x] Run resumable Multi-centroid Final Confirmation, outer folds 1 through 5.
- [x] Run resumable Hybrid Final Confirmation, outer folds 1 through 5.
- [x] Consolidate 300 checkpoints into fold artifacts and execution summaries.
- [x] Verify leakage, coverage, checkpoint integrity, and Quick Screen preservation.
- [x] Append and execute the Final Confirmation notebook section; audit persistence.
- [x] Stop before Final OOF Consolidation and final HDC selection.

## Guardrails

- Outer-test features are materialized only after inner-CV temperature selection is fixed.
- Outer-test labels are joined only after predictions have been generated.
- Each fold reuses only its own Quick Screen structural selection.
- Ridge is recorded once as `COMMON_ENCODER_READOUT_BASELINE`; no variant-specific Ridge is trained.
- Existing Quick Screen artifacts are read-only, except the explicitly authorized notebook append.
