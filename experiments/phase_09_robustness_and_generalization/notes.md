# Phase 09 Initialization Notes

## Frozen evidence boundary

- Main input: Phase 03 Primary without-performance only.
- The original five-fold outer assignments are read-only and are not regenerated.
- Best model families and fold-specific configurations are parsed from Phase 04A, 04B, and 06 freezes without reselection.
- Phase 07 provides the frozen five-modality feature partition and unimodal evidence interface.
- Phase 08 provides frozen shortcut conclusions and the metadata-limited Phase 09 handoff.

## Missing-modality contract questions

Two protocols must remain distinct at Contract Freeze:

1. Retraining with one modality removed before fold-local preprocessing.
2. Sudden test-time absence after a full-input model has been trained, using an explicitly frozen masking or neutralization rule.

Initialization verifies feature counts and pipeline capacity only. It does not choose between protocols or execute either one.

## LOSO boundary

The 35 deterministic leave-one-subject-out splits are feasibility objects only. They must not replace the Phase 03 five-fold outer CV, tune parameters, select models, or produce predictions during initialization.

## Planned subject-level stability evidence

Classification Macro-F1, Balanced Accuracy, Severe Error Rate, bounded MAE, bounded RMSE, between-subject distributions, worst-subject diagnostics, subject-ranking stability, and subject-level bootstrap confidence intervals are planned for later execution after Contract Freeze.

## Contract Freeze decisions

- Primary missing-modality protocol: `RETRAIN_WITHOUT_MODALITY` for five new conditions; `FULL_PRIMARY_REFERENCE` is reused and never counted as new training.
- Optional test-time missingness: `NOT_FEASIBLE_DUE_TO_CHECKPOINT_INTERFACE`. The selected traditional checkpoints have no serialized fitted pipeline, and the selected HDC records have no complete model, preprocessing, feature-order, and encoder state bundle. No reproduction inference was attempted.
- LOSO configuration mapping: each held-out subject uses the upstream configuration associated with that subject's original frozen outer fold; that original configuration-selection training partition excludes the subject.
- Dynamic authorization: 300 missing-modality training records plus 420 LOSO records, totaling 720 unique identifiers; all remain `AUTHORIZED_NOT_EXECUTED`.

## Contract Freeze terminal verification

- Six static Contract Freeze tests passed.
- All 720 records contain the required execution fields and exact expected test `run_key` sets.
- LOSO CSV contains 419 rows, 419 unique run keys, and 35 held-out subjects.
- Authorized totals: HDC classification 300, HDC regression 300, traditional classification 60, traditional regression 60.
- Created checkpoint paths: 0; created prediction paths: 0; files in all Phase 09 modeling-result directories: 0.
- Upstream files modified: 0; contract artifact manifest hash mismatches: 0.
- Notebook Contract Freeze cells: 21 total, zero error outputs; initialization history fingerprint audit: PASS.
- Final status: `CONTRACT_FROZEN_NOT_TRAINED`; ready for the separately requested execution step only.

## Final analysis freeze

- Verified 720/720 completed runs, 30,168 raw prediction rows, 10,056 canonical OOF rows, and 35/35 LOSO subjects.
- Missing-modality robustness, LOSO stability, subject-level Wilcoxon/Holm analysis, and 2,000-resample confidence intervals are complete.
- Final manifest and freeze record were generated with zero missing artifacts, duplicate paths, hash mismatches, protected-result changes, or upstream Phase 03-08 changes.
- Generalization boundary remains unchanged: subject generalization was evaluated via LOSO; unseen session/scenario/task-template/route claims remain not feasible due to metadata; the flight generalizable-behavior claim remains inconclusive.
- Terminal status: `FROZEN`. Phase 10 executed: `NO`. Ready for a separately authorized Phase 10 step: `YES`.
