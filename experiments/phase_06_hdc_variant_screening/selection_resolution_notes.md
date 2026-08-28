# Phase 06 Selection Resolution Notes

## Required provenance

- Original selection rule remains unchanged.
- Original model-selection audit FAIL remains unchanged.
- Amendment is defined after Final Confirmation artifacts and outer OOF exist.
- Outer OOF is hash-sealed before selector execution.

## Evidence boundary

- Classification: Quick Screen inner-CV candidate tables/inner metrics and unlabeled efficiency only.
- Regression: saved Final Confirmation inner-selection records and unlabeled efficiency only.
- Selector must not read `results/oof/`, final prediction CSVs, or outer metrics.

## Findings

- Root cause matches the requested diagnosis; original rule and FAIL audit hashes are preserved.
- Preselection seal covers 72 outer-OOF, prediction, and outer-metric artifacts.
- Selector allowlist/isolation tests pass (4/4); two complete runs produced identical output hashes.
- Unique inner-only selections: Hybrid classification at d=5000 and COMMON_ENCODER_READOUT_BASELINE at d=10000.
- Phase 05 inner-selection records do not persist RMSE; the RMSE tie-break was not invoked because mean bounded MAE and its across-fold sample SD uniquely selected the regression winner.
- Post-selection outer-OOF values are descriptive only and cannot change the selected families.
