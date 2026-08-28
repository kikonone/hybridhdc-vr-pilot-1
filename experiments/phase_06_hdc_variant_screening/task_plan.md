# Task Plan: Phase 06 Contract Freeze and Quick Screening

## Goal
Complete the append-only Phase 05 amendment gate, freeze the Phase 06 variant contract, implement and test three new HDC variants, run all frozen five-fold inner-CV quick screens, persist the Notebook, and stop before outer-test access or Final Confirmation.

## Phases
- [x] Phase 1: Snapshot immutable upstream artifacts and validate/create the Phase 05 append-only amendment
- [x] Phase 2: Revalidate Phase 06 initialization and freeze all variant/search/selection contracts
- [x] Phase 3: Implement shared, OnlineHD, Multi-centroid, and Hybrid code by reusing the Phase 05 encoder
- [x] Phase 4: Run the complete unit-test and guard suite
- [x] Phase 5: Run OnlineHD Fold 1→5 quick screening with resumable checkpoints
- [x] Phase 6: Run Multi-centroid Fold 1→5 quick screening with resumable checkpoints
- [x] Phase 7: Run Hybrid Fold 1→5 quick screening with resumable checkpoints
- [x] Phase 8: Consolidate all quick-screen results and independently reproduce every selected configuration
- [x] Phase 9: Append and execute the Phase 06 Notebook; build final manifests and audits
- [x] Phase 10: Verify all artifacts, upstream immutability, zero outer-test access/predictions, and stop

## Frozen Boundaries
- Outer-test features and labels remain sealed; quick screening may load only outer-training rows.
- No outer-test prediction, OOF output, similarity regression, Ridge readout, Final Confirmation, Phase 07, modality ablation, or performance-feature experiment.
- Vanilla HDC is reused read-only from Phase 05 and is not retrained.
- Existing Phase 03–05 files are immutable; Phase 05 authorization is limited to new append-only amendment files when absent.
- Search candidates and tie-breaking rules cannot change after contract freeze.

## Deliverable
- `reports/phase06_quick_screen_completion.md`

## Errors Encountered
- During the first continuous run, Hybrid initialization was found to derive its KMeans stream from the full candidate identifier, which would vary initialization across learning-rate/epoch/margin candidates. Execution was stopped after all OnlineHD and Multi-centroid folds completed and before any Hybrid candidate checkpoint was saved. The implementation was corrected to share a stable outer-fold/inner-split/dimension/centroid initialization while keeping candidate-specific update-order streams; verified checkpoints will be resumed.
- The first consolidation used full-dictionary equality between JSON and CSV round-tripped best rows; sub-ULP decimal serialization differences in timing and metric fields caused false failures in all folds. Independent selection verification was corrected to require the same candidate/config identity and 1e-12 agreement on every ranking metric.

## Status
**Complete** — all frozen quick-screen candidates, fold audits, best-config reproductions, Notebook persistence, and upstream immutability checks pass. Execution stopped before outer-test access, Final Confirmation, and Phase 07.
