# Random Forest Regressor Notes

## Frozen Contract

- Input: Phase 03 primary without-performance dataset and frozen subject-wise folds.
- Search: 64 candidates per outer fold using bounded-MAE inner GroupKFold selection.
- Tuning seed: 42. Evaluation seeds: 42, 43, 44, 45, 46.
- Canonical OOF: seed 42 only.

## Findings

- Preflight passed: 419 rows, 35 subjects, 1,176 predictive features, 5 frozen folds.
- Frozen fold SHA-256 matched `e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f`.
- All RBF SVR prerequisite audits passed; Random Forest and Gradient Boosting were both NOT_STARTED before execution.
- Fold 1 had no partial result: the initial attempt stopped during the first inner candidate because the sandbox blocked worker creation for `n_jobs=-1`.
- Fold 1 checkpoint audit passed after the permitted local execution: 84 canonical rows, 420 all-seed rows, 64 candidates, seeds 42–46 each covering 84 rows, no missing predictions, bounded predictions, and zero outer/inner subject overlap.
