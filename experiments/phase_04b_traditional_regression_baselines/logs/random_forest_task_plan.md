# Task Plan: Random Forest Regressor

## Goal

Run the frozen nested subject-wise Random Forest regression contract, preserve checkpoints, and complete the required audits without executing any later model.

## Phases

- [x] Phase 1: Verify frozen inputs, completed prerequisites, and model state.
- [ ] Phase 2: Run or reuse one audited outer-fold checkpoint at a time.
- [ ] Phase 3: Build canonical and all-seed OOF results and audits.
- [ ] Phase 4: Persist results in the notebook, update status, and report.

## Decisions Made

- Tuning uses only seed 42 and the fixed 64-candidate grid.
- Evaluation uses seeds 42, 43, 44, 45, and 46; seed 42 is canonical.
- Gradient Boosting Regressor and HDC are out of scope.

## Errors Encountered

- Fold 1 initial launch stopped before any checkpoint was written because the sandbox denied creation of the worker resources required by the contractually fixed `RandomForestRegressor(n_jobs=-1)`. The recovery is to run the unchanged script in the permitted local execution environment.

## Status

Currently in Phase 2: execute outer fold 1 only, then validate its checkpoint before moving on.
