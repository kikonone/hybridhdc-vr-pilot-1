# Phase 09: Robustness and Generalization

Phase 09 evaluates robustness and generalization using only the frozen Primary dataset (`primary_without_performance.csv`) and upstream-selected model interfaces. The regression task is described as **bounded difficulty-induced workload proxy regression**.

## Core deliverables

- Missing-modality robustness curves
- Selected-model LOSO results
- Subject-level stability analysis

These deliverables are complete and frozen. This directory contains read-only upstream audits, frozen protocols, 720 verified completed run records, 30,168 raw prediction rows, 10,056 canonical OOF rows, subject-level analysis artifacts, and an executed final-freeze notebook summary.

## Scientific boundary

LOSO evaluates generalization to an unseen subject only. It does not establish unseen-scenario, route/configuration, session, or task-template generalization. In the current metadata, each session maps to one subject, while `scenario_id`, `task_template_id`, `route_id`, and `configuration_id` are absent. Consequently, whether flight-parameter advantages reflect generalizable behavior or difficulty-adjacent task structure remains `INCONCLUSIVE_DUE_TO_METADATA`.

`difficulty_level_raw`, `target_class`, `target_score`, run order, and clusters derived from feature values are not admissible scenario identifiers.

## Contract status

The contract status is `CONTRACT_FROZEN_NOT_TRAINED`. Exactly 720 future training runs are authorized: 300 missing-modality retraining runs and 420 LOSO runs. No model training, LOSO prediction, missing-modality prediction, OOF generation, hyperparameter search, or formal statistical analysis was executed during Contract Freeze. Phase 03 frozen five-fold outer assignments remain unchanged and continue to serve as the primary evaluation protocol; LOSO is supplementary.

The optional `SUDDEN_TEST_TIME_MISSINGNESS` protocol is `NOT_FEASIBLE_DUE_TO_CHECKPOINT_INTERFACE` because the upstream selected interfaces do not preserve complete loadable fitted models together with fold preprocessing, feature order, and HDC encoder state. It is not included in the 720 authorized training runs.

## Reproduce the initialization audit

Run `scripts/initialize_phase09.py` and `scripts/build_phase09_notebook.py` for initialization. Contract Freeze is reproduced with `scripts/freeze_phase09_contract.py` followed by `scripts/append_phase09_contract_notebook.py`. These scripts are audit/contract-only and do not import or instantiate model estimators.

## Final Freeze status

Phase 09 is `FROZEN`. The final manifest is `manifests/phase09_final_manifest.json`, the freeze record is `configs/phase09_freeze.json`, and all six final freeze audits pass. Phase 10 was not initialized or executed during this freeze. The project is ready for a separately authorized Phase 10 step.

Reproduce the fail-closed freeze gate with `python scripts/freeze_phase09.py` (read-only dry-run before freezing) and independently verify an existing freeze with `python scripts/verify_phase09_freeze.py`.
