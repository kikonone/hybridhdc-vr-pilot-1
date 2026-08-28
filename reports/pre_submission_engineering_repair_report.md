# Pre-submission Engineering Repair Report

## Decision

**NOT READY FOR FINAL SUBMISSION.** Engineering isolation defects were repaired, but strict final readiness is blocked because original bytes for overwritten Phase 06 and Phase 09 lifecycle/freeze records are unavailable. No production final manifest was rewritten without proof.

## Baseline and final test state

- Baseline command: `python -m pytest -vv`
- Baseline result: 119 passed, 3 failed, 3 errors, 17 warnings (125 collected).
- Final command: `python -m pytest -vv`
- Final result: 125 passed, 1 failed, 0 errors, 18 warnings (126 collected).
- Remaining failure: `experiments/phase_09_robustness_and_generalization/tests/test_phase09_freeze.py::test_read_only_preflight_passes_before_freeze`.
- Remaining failure type: assertion failure from strict final-manifest artifact hash verification.
- Remaining failure is intentionally not skipped, xfailed, deleted, or weakened.

## Failure diagnosis and side effects

### Phase 06

- `test_phase06_preflight.py::test_preflight_all_required_gates_pass` called `run_preflight(PHASE_DIR)`, whose writer targets were hard-wired to the production Phase 06 directory.
- The test overwrote initialization, environment, input/fold, and upstream interface JSON evidence. This was a filesystem-isolation defect, not a scientific audit failure.
- `test_phase06_contains_no_copied_upstream_csv` asserted that `results/oof` was empty. That assertion described the initialization state but was being run against a completed, frozen Phase 06 containing six legitimate final OOF files.
- Running directory, current working directory, fixtures, and order mattered: real production paths were derived from `__file__`; no `tmp_path` fixture was used; the writes occurred before later integrity checks.

### Phase 09

- `test_phase09_contract.py` called `run_freeze()` from `setUpClass`, overwriting production initialization/contract records after predictions already existed. The resulting contract audit correctly observed predictions and became `FAIL`.
- The contract mutation rewrote the 720-run execution manifest back to `AUTHORIZED_NOT_EXECUTED`, causing three executor tests to error during setup. These were order-dependent lifecycle errors, not executor algorithm failures.
- `run_phase09_batch.dry_run()` also wrote `phase09_executor_validation_audit.json` in production; it now accepts an explicit audit path and tests use a temporary directory.
- `verify_phase09_freeze.py` incorrectly treated the existence of a `phase_10*` directory as proof that Phase 10 had executed. It now checks the two frozen execution flags and retains the Phase 10 path list as diagnostic evidence.
- The remaining Phase 09 failure is a real strict integrity failure: four manifest-listed lifecycle files no longer match the exact frozen hashes. Prediction, checkpoint, canonical OOF, statistical, and upstream protected hashes still match.

## Repairs implemented

- Phase 04B finalization accepts an injected phase directory; the integration test copies Phase 04B to `pytest`'s temporary directory, executes finalization there, and verifies the production tree hash map is unchanged.
- Phase 06 preflight separates the read-only source/project root from an isolated evidence output directory. Tests build a temporary skeleton and preserve all checksum, fold-isolation, contract, evidence, and OOF assertions.
- Phase 09 initialization and contract tests read frozen evidence without invoking production writers.
- Phase 09 executor dry-run writes only when an explicit audit path is supplied; tests supply a temporary path.
- Two Phase 04B Elastic Net recovery scripts had the same missing quote in a dictionary lookup. Syntax was repaired without executing either script.
- No model training, model selection, prediction regeneration, OOF regeneration, statistical recomputation, Notebook rewriting, UI development, or Phase 10 synthesis was performed.

## Phase 04B isolation result

- Isolation audit: `audits/pre_submission_repair/phase04b_test_isolation_audit.json`
- Temporary-directory execution: PASS.
- Production files modified by the test: **0**.
- Backup/restore strategy used by the test: **NO**.

## Phase 06 manifest recovery decision

- The current final manifest was backed up before recovery work.
- Manifest entries: 114; exact current matches: 108; mismatches: 6.
- Mismatches are initialization/interface JSON records, not Primary data, folds, predictions, canonical OOF, statistics, notebooks, or frozen model selections.
- No exact original copies matching the six recorded hashes were found in the scoped audit, snapshot, temporary, config, or audit evidence locations.
- Rebuilding the manifest would bless unverified overwritten records, so the production manifest was not written.
- Status: `BLOCKED_UNVERIFIED_MANIFEST_RECOVERY`.

## Layered verification

- Syntax compilation (`python -m compileall -q experiments`): PASS.
- Unit layer (Phase 05 core and Phase 06 variants): 26 passed.
- Lifecycle/integration layer (Phase 04B isolation, Phase 06 preflight, Phase 09 initialization/contract/executor): 20 passed.
- Notebook persistence: 10 notebooks parse, 0 error outputs, 219 code cells with persisted outputs.
- Full suite: 125 passed, 1 failed, 0 errors.
- Freeze/manifest layer: FAIL due to unresolved exact lifecycle artifact hashes.

## Frozen scientific artifact immutability

- Primary checksum unchanged: YES.
- Frozen folds checksum unchanged: YES.
- Prediction files changed: 0 of 1,406.
- Canonical OOF files changed: 0 of 10.
- Statistical artifacts changed: 0 of 5.
- Canonical notebooks changed: 0 of 10.
- Frozen model configurations changed: 0 of 7.
- Final manifests changed: 0 of 12.
- Captured freeze/lifecycle records changed from the pre-repair baseline: 2. This prevents a PASS declaration even though scientific outputs are unchanged.

## Git readiness

- Branch: `codex/tables`.
- `git diff --check`: PASS.
- Worktree entries: 27 (3 tracked deletions and 24 untracked paths).
- No staging, commit, push, clean, reset, or deletion was performed.
- Git readiness: NO.

## Evidence index

- `audits/pre_submission_repair/pre_repair_state.json`
- `audits/pre_submission_repair/pre_repair_test_output.log`
- `audits/pre_submission_repair/pre_repair_freeze_hashes.json`
- `audits/pre_submission_repair/post_repair_state.json`
- `audits/pre_submission_repair/post_repair_test_output.log`
- `audits/pre_submission_repair/frozen_artifact_immutability_audit.json`
- `audits/pre_submission_repair/phase04b_test_isolation_audit.json`
- `audits/pre_submission_repair/phase06_manifest_recovery_audit.json`
- `audits/pre_submission_repair/phase09_lifecycle_isolation_audit.json`
- `audits/pre_submission_repair/notebook_persistence_audit.json`
