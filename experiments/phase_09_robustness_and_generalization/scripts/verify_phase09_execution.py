"""Verify Phase 09 raw execution artifacts without OOF consolidation or statistics."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_phase09_batch import (
    EXECUTION_MANIFEST,
    EXPECTED_FOLDS,
    EXPECTED_PRIMARY,
    FOLDS,
    PHASE09,
    PRIMARY,
    atomic_json,
    canonical_json,
    loadable_model,
    output_paths,
    read_json,
    reusable_run,
    sha256,
)


REQUIRED_RUN_AUDIT_CHECKS = [
    "frozen_config_matching", "expected_feature_count", "removed_modality_feature_intersection_0",
    "expected_test_rows", "unique_test_run_key", "exact_split_membership", "train_test_subject_overlap_0",
    "test_subject_not_used_for_config_selection", "no_test_fitted_preprocessing", "finite_predictions",
    "classification_prediction_domain", "bounded_regression_range", "checkpoint_loadable",
    "prediction_artifact_nonempty", "performance_feature_count_0",
]


def current_snapshot(reference: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        path: {"bytes": Path(path).stat().st_size, "sha256": sha256(Path(path))}
        for path in reference
    }


def verify_execution() -> dict[str, Any]:
    manifest = read_json(EXECUTION_MANIFEST)
    records = manifest["training_runs"]
    validation = read_json(PHASE09 / "audits" / "phase09_executor_validation_audit.json")
    fold_table = pd.read_csv(FOLDS)
    all_predictions: list[pd.DataFrame] = []
    artifact_records: list[dict[str, Any]] = []
    invalid_runs: list[str] = []
    checkpoint_failures: list[str] = []
    feature_failures: list[str] = []
    leakage_failures: list[str] = []
    schema_failures: list[str] = []
    completed_by_protocol: Counter[str] = Counter()
    completed_by_model: Counter[str] = Counter()
    completed_conditions: set[str] = set()
    completed_subjects: set[str] = set()

    for record in records:
        paths = output_paths(record)
        if not reusable_run(record, paths):
            invalid_runs.append(record["run_identifier"])
            continue
        checkpoint = read_json(paths["checkpoint"])
        audit = read_json(paths["audit"])
        prediction = pd.read_csv(paths["prediction"])
        prediction["_source_prediction_path"] = str(paths["prediction"].relative_to(PHASE09)).replace("\\", "/")
        all_predictions.append(prediction)
        if not loadable_model(paths["model"], record["model_key"]):
            checkpoint_failures.append(record["run_identifier"])
        if not all(audit["checks"].get(name) is True for name in REQUIRED_RUN_AUDIT_CHECKS):
            if not audit["checks"].get("expected_feature_count") or not audit["checks"].get("removed_modality_feature_intersection_0") or not audit["checks"].get("performance_feature_count_0"):
                feature_failures.append(record["run_identifier"])
            if not audit["checks"].get("train_test_subject_overlap_0") or not audit["checks"].get("test_subject_not_used_for_config_selection") or not audit["checks"].get("no_test_fitted_preprocessing"):
                leakage_failures.append(record["run_identifier"])
            invalid_runs.append(record["run_identifier"])
        expected_schema = {
            "run_id", "protocol", "condition", "model_family", "seed", "outer_fold", "loso_subject",
            "run_key", "subject_id", "y_true", "config_hash", "feature_manifest_hash",
        }
        expected_schema |= ({"y_pred", "class_score_0", "class_score_1", "class_score_2", "class_score_3"} if record["task"] == "classification" else {"y_pred_raw", "y_pred_bounded"})
        if not expected_schema.issubset(prediction.columns):
            schema_failures.append(record["run_identifier"])
        completed_by_protocol[record["protocol"]] += 1
        completed_by_model[record["model_key"]] += 1
        if record["protocol"] == "RETRAIN_WITHOUT_MODALITY":
            completed_conditions.add(record["condition"])
        else:
            completed_subjects.add(str(record["loso_subject"]))
        for name in ["checkpoint", "model", "audit", "prediction", "metrics"]:
            path = paths[name]
            artifact_records.append({
                "run_identifier": record["run_identifier"], "role": name,
                "path": str(path.relative_to(PHASE09)).replace("\\", "/"),
                "bytes": path.stat().st_size, "sha256": sha256(path),
            })

    combined = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    duplicate_run_key_pairs = int(combined.duplicated(["run_id", "run_key"]).sum()) if len(combined) else -1
    protocol_rows = combined.groupby("protocol").size().to_dict() if len(combined) else {}
    model_rows = {}
    if len(combined):
        run_model = {record["run_identifier"]: record["model_key"] for record in records}
        model_rows = Counter(run_model[run_id] for run_id in combined.run_id.astype(str))
    row_counts = {
        "total": len(combined),
        "missing_modality": int(protocol_rows.get("RETRAIN_WITHOUT_MODALITY", 0)),
        "loso": int(protocol_rows.get("LEAVE_ONE_SUBJECT_OUT", 0)),
        "hdc_classification": int(model_rows.get("hdc_classification", 0)),
        "hdc_regression": int(model_rows.get("hdc_regression", 0)),
        "traditional_classification": int(model_rows.get("traditional_classification", 0)),
        "traditional_regression": int(model_rows.get("traditional_regression", 0)),
    }
    expected_rows = {
        "total": 30168, "missing_modality": 25140, "loso": 5028,
        "hdc_classification": 12570, "hdc_regression": 12570,
        "traditional_classification": 2514, "traditional_regression": 2514,
    }
    bounded_pass = bool(
        len(combined)
        and combined.loc[combined.run_id.map({record["run_identifier"]: record["task"] for record in records}) == "regression", "y_pred_bounded"].dropna().between(1.0, 4.0).all()
    )
    finite_columns = [column for column in ["y_pred", "class_score_0", "class_score_1", "class_score_2", "class_score_3", "y_pred_raw", "y_pred_bounded"] if column in combined]
    finite_pass = bool(len(combined) and all(np.isfinite(combined[column].dropna().to_numpy(dtype=float)).all() for column in finite_columns))

    upstream_reference = validation["upstream_snapshot"]
    upstream_modified = [path for path, value in upstream_reference.items() if current_snapshot(upstream_reference)[path] != value]
    contract_reference = validation["frozen_contract_snapshot"]
    contract_modified = [path for path, value in contract_reference.items() if current_snapshot(contract_reference)[path] != value]
    common_checks = {
        "primary_checksum": sha256(PRIMARY) == EXPECTED_PRIMARY,
        "fold_checksum": sha256(FOLDS) == EXPECTED_FOLDS,
        "executor_validation": validation["status"] == "PASS",
        "runs_720": len(records) == 720 and sum(completed_by_protocol.values()) == 720,
        "raw_rows_30168": row_counts == expected_rows,
        "duplicate_run_id_run_key_0": duplicate_run_key_pairs == 0,
        "invalid_runs_0": not invalid_runs,
        "schema_failures_0": not schema_failures,
        "finite_predictions": finite_pass,
        "bounded_predictions": bounded_pass,
        "upstream_files_modified_0": not upstream_modified,
        "frozen_contract_files_modified_0": not contract_modified,
    }
    checkpoint_status = "PASS" if not checkpoint_failures and not invalid_runs and len(artifact_records) == 3600 else "FAIL"
    feature_status = "PASS" if not feature_failures and all(read_json(output_paths(record)["audit"])["checks"]["performance_feature_count_0"] for record in records if output_paths(record)["audit"].exists()) and len(completed_conditions) == 5 else "FAIL"
    leakage_status = "PASS" if not leakage_failures and all(read_json(output_paths(record)["audit"])["checks"]["train_test_subject_overlap_0"] for record in records if output_paths(record)["audit"].exists()) and len(completed_subjects) == 35 else "FAIL"
    coverage_status = "PASS" if all(common_checks.values()) and completed_by_protocol == {"RETRAIN_WITHOUT_MODALITY": 300, "LEAVE_ONE_SUBJECT_OUT": 420} and completed_by_model == {"hdc_classification": 300, "hdc_regression": 300, "traditional_classification": 60, "traditional_regression": 60} else "FAIL"

    atomic_json(PHASE09 / "audits" / "phase09_checkpoint_integrity_audit.json", {
        "phase": "09", "audit": "checkpoint_integrity", "status": checkpoint_status,
        "completed_checkpoints": sum(completed_by_protocol.values()), "expected_checkpoints": 720,
        "load_or_hash_failures": checkpoint_failures, "invalid_runs": sorted(set(invalid_runs)),
    })
    atomic_json(PHASE09 / "audits" / "phase09_feature_exclusion_audit.json", {
        "phase": "09", "audit": "feature_exclusion", "status": feature_status,
        "missing_modality_conditions_completed": len(completed_conditions), "expected_conditions": 5,
        "feature_failures": sorted(set(feature_failures)), "performance_features_included": False,
    })
    atomic_json(PHASE09 / "audits" / "phase09_execution_leakage_audit.json", {
        "phase": "09", "audit": "execution_leakage", "status": leakage_status,
        "loso_subjects_completed": len(completed_subjects), "expected_loso_subjects": 35,
        "leakage_failures": sorted(set(leakage_failures)), "outer_test_used_for_parameter_selection": False,
        "new_inner_cv_executed": False, "test_fitted_preprocessing": False,
    })
    atomic_json(PHASE09 / "audits" / "phase09_execution_coverage_audit.json", {
        "phase": "09", "audit": "execution_coverage", "status": coverage_status,
        "checks": common_checks, "completed_runs": sum(completed_by_protocol.values()),
        "run_counts_by_protocol": dict(completed_by_protocol), "run_counts_by_model": dict(completed_by_model),
        "raw_prediction_rows": row_counts, "expected_raw_prediction_rows": expected_rows,
        "duplicate_run_id_run_key_pairs": duplicate_run_key_pairs,
        "missing_modality_conditions_completed": sorted(completed_conditions), "loso_subjects_completed": sorted(completed_subjects),
        "canonical_oof_consolidation_executed": False, "seed_aggregation_executed": False,
    })
    artifact_status = "PASS" if len(artifact_records) == 3600 and not invalid_runs and not upstream_modified and not contract_modified else "FAIL"
    atomic_json(PHASE09 / "audits" / "phase09_execution_artifact_audit.json", {
        "phase": "09", "audit": "execution_artifacts", "status": artifact_status,
        "artifact_count": len(artifact_records), "expected_artifact_count": 3600,
        "artifacts": artifact_records, "missing_or_invalid_runs": sorted(set(invalid_runs)),
        "upstream_files_modified": upstream_modified, "frozen_contract_files_modified": contract_modified,
        "canonical_oof_artifacts_created": False, "formal_statistics_executed": False, "phase10_executed": False,
    })

    all_audits_pass = all(status == "PASS" for status in [checkpoint_status, feature_status, leakage_status, coverage_status, artifact_status])
    manifest["status"] = "EXECUTION_COMPLETE_PENDING_NOTEBOOK_PERSISTENCE" if all_audits_pass else "EXECUTION_VERIFICATION_FAILED"
    manifest["completed_training_runs"] = sum(completed_by_protocol.values())
    manifest["raw_prediction_rows"] = len(combined)
    manifest["execution_verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(EXECUTION_MANIFEST, manifest)
    result = {
        "status": manifest["status"], "completed_runs": manifest["completed_training_runs"],
        "raw_prediction_rows": len(combined), "checkpoint_integrity": checkpoint_status,
        "coverage": coverage_status, "leakage": leakage_status, "feature_exclusion": feature_status,
        "artifacts": artifact_status, "ready_pending_notebook_persistence": all_audits_pass,
    }
    if not all_audits_pass:
        raise RuntimeError(result)
    return result


def finalize_execution_status() -> dict[str, Any]:
    audit_names = [
        "phase09_executor_validation_audit.json", "phase09_checkpoint_integrity_audit.json",
        "phase09_execution_coverage_audit.json", "phase09_execution_leakage_audit.json",
        "phase09_feature_exclusion_audit.json", "phase09_execution_artifact_audit.json",
        "phase09_execution_notebook_persistence_audit.json",
    ]
    statuses = {name: read_json(PHASE09 / "audits" / name)["status"] for name in audit_names}
    coverage = read_json(PHASE09 / "audits" / "phase09_execution_coverage_audit.json")
    manifest = read_json(EXECUTION_MANIFEST)
    pass_status = (
        all(value == "PASS" for value in statuses.values())
        and coverage["completed_runs"] == 720
        and coverage["raw_prediction_rows"]["total"] == 30168
        and coverage["duplicate_run_id_run_key_pairs"] == 0
    )
    manifest["status"] = "EXECUTION_COMPLETE_PENDING_OOF_CONSOLIDATION" if pass_status else "EXECUTION_FINALIZATION_FAILED"
    manifest["ready_for_oof_consolidation"] = pass_status
    manifest["canonical_oof_consolidation_executed"] = False
    manifest["formal_statistical_analysis_executed"] = False
    manifest["phase10_executed"] = False
    manifest["finalized_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(EXECUTION_MANIFEST, manifest)
    return {"status": manifest["status"], "ready_for_oof_consolidation": pass_status, "audit_statuses": statuses}


if __name__ == "__main__":
    print(json.dumps(verify_execution(), ensure_ascii=False, indent=2))
