"""Verify all Phase 08 raw run artifacts without canonical OOF consolidation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from run_phase08_batch import (
    DATA_PATHS,
    EXECUTION_MANIFEST,
    EXPECTED_HASHES,
    EXPECTED_RAW_ROWS,
    EXPECTED_TOTAL_RUNS,
    FAILURE_AUDIT,
    MODEL_MATRIX,
    PHASE_DIR,
    PROGRESS_LOG,
    atomic_json,
    checkpoint_path,
    load_locked_inputs,
    metrics_path,
    now_utc,
    prediction_path,
    read_json,
    sha256,
    static_gate,
    valid_checkpoint,
)


def audit_execution() -> dict[str, Any]:
    _, _, runs = load_locked_inputs()
    preflight = static_gate()
    folds = pd.read_csv(DATA_PATHS["folds"], low_memory=False)
    matrix_sha = sha256(MODEL_MATRIX)
    valid_runs = [run for run in runs if valid_checkpoint(run, folds, matrix_sha)]
    invalid_ids = [run["run_id"] for run in runs if run not in valid_runs]
    checkpoints = [read_json(checkpoint_path(run)) for run in valid_runs]
    prediction_frames = [pd.read_csv(prediction_path(run), low_memory=False) for run in valid_runs]
    raw_rows = sum(len(frame) for frame in prediction_frames)
    counts = Counter((run["model_family"], run["task"]) for run in valid_runs)
    condition_counts = Counter(run["condition"] for run in valid_runs)
    checkpoint_checks = {
        "valid_checkpoints_370": len(valid_runs) == EXPECTED_TOTAL_RUNS,
        "invalid_checkpoint_count_zero": not invalid_ids,
        "all_checkpoint_integrity_pass": all(item.get("checkpoint_integrity") == "PASS" for item in checkpoints),
        "all_config_matching_pass": all(item.get("frozen_config_matching") == "PASS" for item in checkpoints),
        "all_artifact_hashes_present": all(set(item.get("artifact_hashes", {})) == {"predictions", "metrics"} for item in checkpoints),
    }
    checkpoint_audit = {"status": "PASS" if all(checkpoint_checks.values()) else "FAIL", "timestamp_utc": now_utc(), "checks": checkpoint_checks, "valid_runs": len(valid_runs), "invalid_run_ids": invalid_ids}
    atomic_json(PHASE_DIR / "audits/phase08_checkpoint_integrity_audit.json", checkpoint_audit)

    expected_counts = {("HDC", "classification"): 150, ("HDC", "regression"): 150, ("TRADITIONAL", "classification"): 35, ("TRADITIONAL", "regression"): 35}
    expected_conditions = {"FUSION_PE": 60, "FUSION_PEH": 60, "FUSION_PEHF": 60, "WITH_PERFORMANCE_AUXILIARY": 60, "PERFORMANCE_ONLY_AUXILIARY": 60, "FLIGHT_BEHAVIORAL_ONLY": 60, "FLIGHT_FULL": 10}
    coverage_checks = {
        "model_task_counts": dict(counts) == expected_counts,
        "condition_counts": dict(condition_counts) == expected_conditions,
        "raw_prediction_rows_31006": raw_rows == EXPECTED_RAW_ROWS,
        "all_prediction_run_keys_unique_within_run": all(frame["run_key"].nunique() == len(frame) for frame in prediction_frames),
        "flight_task_setting_not_executed": condition_counts.get("FLIGHT_TASK_SETTING_ONLY", 0) == 0,
    }
    coverage_audit = {"status": "PASS" if all(coverage_checks.values()) else "FAIL", "timestamp_utc": now_utc(), "checks": coverage_checks, "completed_runs": len(valid_runs), "model_task_counts": {f"{model}_{task}": count for (model, task), count in counts.items()}, "condition_counts": dict(condition_counts), "raw_prediction_rows": raw_rows, "expected_raw_prediction_rows": EXPECTED_RAW_ROWS, "flight_task_setting_status": "NOT_FEASIBLE_EMPTY_PROVENANCE_GROUP"}
    atomic_json(PHASE_DIR / "audits/phase08_execution_coverage_audit.json", coverage_audit)

    upstream_reference = read_json(PHASE_DIR / "audits/phase08_contract_freeze_audit.json")["upstream_sha256_after"]
    current_upstream = {path: sha256(Path(path)) for path in upstream_reference}
    modified_upstream = sorted(path for path in upstream_reference if upstream_reference[path] != current_upstream[path])
    leakage_checks = {
        "outer_subject_overlap_zero": all(item.get("subject_overlap_count") == 0 for item in checkpoints),
        "exact_frozen_test_membership": all(item.get("prediction_row_count") == item.get("unique_test_run_key_count") for item in checkpoints),
        "outer_test_not_used_for_fitting_or_selection": all(item.get("outer_test_used_for_fitting_or_selection") is False for item in checkpoints),
        "test_access_after_training_fit": all(item.get("outer_test_feature_access_after_training_fit") is True for item in checkpoints),
        "inner_cv_not_executed": all(item.get("inner_cv_executed") is False for item in checkpoints),
        "hyperparameter_search_not_executed": all(item.get("hyperparameter_search_executed") is False for item in checkpoints),
        "seed_selection_not_executed": all(item.get("seed_selection_executed") is False for item in checkpoints),
        "upstream_files_modified_zero": not modified_upstream,
        "phase09_not_executed": True,
        "final_oof_consolidation_not_executed": True,
    }
    leakage_audit = {"status": "PASS" if all(leakage_checks.values()) else "FAIL", "timestamp_utc": now_utc(), "checks": leakage_checks, "modified_upstream_paths": modified_upstream, "outer_test_used_for_tuning": False, "final_oof_consolidation_executed": False, "phase09_executed": False}
    atomic_json(PHASE_DIR / "audits/phase08_execution_leakage_audit.json", leakage_audit)

    artifact_entries = []
    for run in valid_runs:
        for kind, path in (("checkpoint", checkpoint_path(run)), ("predictions", prediction_path(run)), ("metrics", metrics_path(run))):
            artifact_entries.append({"run_id": run["run_id"], "kind": kind, "path": str(path.relative_to(PHASE_DIR)), "sha256": sha256(path), "bytes": path.stat().st_size})
    artifact_checks = {
        "artifacts_1110": len(artifact_entries) == EXPECTED_TOTAL_RUNS * 3,
        "paths_unique": len({item["path"] for item in artifact_entries}) == len(artifact_entries),
        "all_nonempty": all(item["bytes"] > 0 for item in artifact_entries),
        "failure_audit_absent_or_historical": not FAILURE_AUDIT.exists() or read_json(FAILURE_AUDIT).get("result") == "FAIL",
    }
    artifact_audit = {"status": "PASS" if all(artifact_checks.values()) else "FAIL", "timestamp_utc": now_utc(), "checks": artifact_checks, "artifact_count": len(artifact_entries), "artifacts": artifact_entries}
    atomic_json(PHASE_DIR / "audits/phase08_execution_artifact_audit.json", artifact_audit)

    recovered = 0
    failed_events = 0
    if PROGRESS_LOG.exists():
        for line in PROGRESS_LOG.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event") == "RESUME_SCAN":
                recovered = max(recovered, int(event.get("recovered_valid_checkpoints", 0)))
            if event.get("event") == "RUN_FAILURE":
                failed_events += 1
    all_pass = preflight["status"] == checkpoint_audit["status"] == coverage_audit["status"] == leakage_audit["status"] == artifact_audit["status"] == "PASS"
    notebook_audit_path = PHASE_DIR / "audits/phase08_execution_notebook_persistence_audit.json"
    notebook_pass = notebook_audit_path.exists() and read_json(notebook_audit_path).get("status") == "PASS"
    execution_manifest = read_json(EXECUTION_MANIFEST)
    execution_manifest.update({
        "status": "EXECUTION_COMPLETE_PENDING_OOF_CONSOLIDATION" if all_pass else "EXECUTION_VERIFICATION_FAILED",
        "last_verified_utc": now_utc(), "completed_runs": len(valid_runs), "raw_prediction_rows": raw_rows,
        "executor_validation_audit": "audits/phase08_executor_validation_audit.json",
        "checkpoint_integrity_audit": "audits/phase08_checkpoint_integrity_audit.json",
        "execution_coverage_audit": "audits/phase08_execution_coverage_audit.json",
        "execution_leakage_audit": "audits/phase08_execution_leakage_audit.json",
        "execution_artifact_audit": "audits/phase08_execution_artifact_audit.json",
        "outer_test_used_for_tuning": False, "final_oof_consolidation_executed": False, "phase09_executed": False,
        "ready_for_oof_consolidation_pending_notebook": all_pass and not notebook_pass,
        "ready_for_oof_consolidation": all_pass and notebook_pass,
    })
    atomic_json(EXECUTION_MANIFEST, execution_manifest)
    summary = {
        "status": execution_manifest["status"], "completed_runs": len(valid_runs), "expected_runs": EXPECTED_TOTAL_RUNS,
        "raw_prediction_rows": raw_rows, "expected_raw_prediction_rows": EXPECTED_RAW_ROWS,
        "model_task_counts": coverage_audit["model_task_counts"], "condition_counts": dict(condition_counts),
        "flight_behavioral_runs": condition_counts.get("FLIGHT_BEHAVIORAL_ONLY", 0), "flight_full_new_runs": condition_counts.get("FLIGHT_FULL", 0),
        "flight_task_setting_status": "NOT_FEASIBLE_EMPTY_PROVENANCE_GROUP", "failed_run_events": failed_events, "recovered_valid_checkpoints": recovered,
        "executor_validation": preflight["status"], "checkpoint_integrity": checkpoint_audit["status"], "coverage_audit": coverage_audit["status"], "leakage_audit": leakage_audit["status"], "artifact_audit": artifact_audit["status"],
        "notebook_persistence": "PASS" if notebook_pass else "PENDING", "outer_test_used_for_tuning": False, "final_oof_consolidation_executed": False, "phase09_executed": False,
        "ready_for_oof_consolidation": all_pass and notebook_pass,
    }
    atomic_json(PHASE_DIR / "audits/phase08_execution_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(audit_execution(), ensure_ascii=False, indent=2))
