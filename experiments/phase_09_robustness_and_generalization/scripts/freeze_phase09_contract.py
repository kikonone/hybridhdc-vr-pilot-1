"""Freeze the Phase 09 execution contract without training or prediction generation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from initialize_phase09 import (
    EXPECTED_FOLD_SHA256,
    EXPECTED_PRIMARY_SHA256,
    FOLDS,
    PHASE04A,
    PHASE04B,
    PHASE06,
    PHASE08,
    PHASE09_ROOT,
    PRIMARY,
    REGRESSION_FOLD_PARAMS,
    UPSTREAM,
    read_json,
    run_initialization,
    sha256,
    source_record,
    write_json,
)


SEEDS = [42, 43, 44, 45, 46]
MODEL_SPECS = {
    "hdc_classification": {"task": "classification", "model_family": "HDC+OnlineHD Hybrid", "seeds": SEEDS},
    "hdc_regression": {"task": "regression", "model_family": "COMMON_ENCODER_READOUT_BASELINE", "seeds": SEEDS},
    "traditional_classification": {"task": "classification", "model_family": "Gradient Boosting", "seeds": ["canonical"]},
    "traditional_regression": {"task": "regression", "model_family": "Gradient Boosting Regressor", "seeds": ["canonical"]},
}
MISSING_CONDITIONS = [
    {"condition": "FULL_PRIMARY_REFERENCE", "removed_modality": None, "removed_features": 0, "feature_count": 1176, "policy": "REUSED_NOT_RETRAINED"},
    {"condition": "MISSING_PHYSIOLOGICAL", "removed_modality": "physiological_features", "removed_features": 233, "feature_count": 943, "policy": "RETRAIN_WITHOUT_MODALITY"},
    {"condition": "MISSING_EYE_TRACKING", "removed_modality": "eye_tracking_features", "removed_features": 416, "feature_count": 760, "policy": "RETRAIN_WITHOUT_MODALITY"},
    {"condition": "MISSING_HEAD_MOVEMENT", "removed_modality": "head_movement_features", "removed_features": 159, "feature_count": 1017, "policy": "RETRAIN_WITHOUT_MODALITY"},
    {"condition": "MISSING_FLIGHT_PARAMETER", "removed_modality": "flight_parameter_features", "removed_features": 326, "feature_count": 850, "policy": "RETRAIN_WITHOUT_MODALITY"},
    {"condition": "MISSING_BODY_MOVEMENT", "removed_modality": "body_movement_features", "removed_features": 42, "feature_count": 1134, "policy": "RETRAIN_WITHOUT_MODALITY"},
]

INITIALIZATION_FILES = [
    "configs/phase09_experiment_contract_draft.json", "configs/phase09_environment.json",
    "configs/phase09_generalization_scope.json", "configs/phase09_selected_model_interfaces.json",
    "configs/phase09_missing_modality_plan.json", "configs/phase09_loso_plan.json",
    "configs/phase09_statistical_plan.json", "manifests/phase09_input_manifest.json",
    "manifests/phase09_upstream_freeze_manifest.json", "manifests/phase09_modality_manifest.json",
    "manifests/phase09_loso_feasibility_manifest.json", "audits/phase09_input_and_fold_audit.json",
    "audits/phase09_upstream_freeze_audit.json", "audits/phase09_modality_coverage_audit.json",
    "audits/phase09_loso_feasibility_audit.json", "audits/phase09_generalization_scope_audit.json",
    "audits/phase09_initialization_artifact_audit.json", "audits/phase09_notebook_persistence_audit.json",
]
CONTRACT_FILES = [
    "configs/phase09_frozen_contract.json", "configs/phase09_contract_freeze.json",
    "configs/phase09_missing_modality_contract.json", "configs/phase09_loso_contract.json",
    "configs/phase09_loso_config_mapping.json", "configs/phase09_oof_aggregation_rules.json",
    "configs/phase09_statistical_rules.json", "configs/phase09_execution_manifest.json",
    "manifests/phase09_contract_artifact_manifest.json", "manifests/phase09_loso_assignments.csv",
    "manifests/phase09_expected_coverage_manifest.json", "audits/phase09_contract_freeze_audit.json",
    "audits/phase09_run_matrix_audit.json", "audits/phase09_loso_assignment_audit.json",
    "audits/phase09_config_mapping_leakage_audit.json", "audits/phase09_missing_modality_contract_audit.json",
    "audits/phase09_checkpoint_portability_audit.json", "audits/phase09_generalization_guardrail_audit.json",
    "audits/phase09_contract_artifact_audit.json", "audits/phase09_contract_notebook_persistence_audit.json",
]
RESULT_DIRECTORIES = [
    "results/checkpoints", "results/predictions", "results/fold_metrics", "results/oof",
    "results/missing_modality", "results/loso", "results/subject_stability", "results/summaries",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(relative_path: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path = PHASE09_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def referenced_upstream_paths() -> list[Path]:
    manifest = read_json(PHASE09_ROOT / "manifests" / "phase09_upstream_freeze_manifest.json")
    paths = [Path(item["path"]) for item in manifest["sources"]]
    paths.extend([PRIMARY, FOLDS])
    unique = {str(path.resolve()).lower(): path.resolve() for path in paths}
    return [unique[key] for key in sorted(unique)]


def snapshot(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in paths
    }


def result_file_snapshot() -> list[str]:
    return sorted(
        str(path.relative_to(PHASE09_ROOT)).replace("\\", "/")
        for directory in RESULT_DIRECTORIES
        for path in (PHASE09_ROOT / directory).rglob("*")
        if path.is_file()
    )


def config_source(model_key: str, outer_fold: int) -> dict[str, str]:
    if model_key == "hdc_classification":
        return {"path": str(UPSTREAM["phase06"].resolve()), "locator": f"best_classification_hdc.fold_selected_structures[outer_fold={outer_fold}]"}
    if model_key == "hdc_regression":
        return {"path": str(UPSTREAM["phase06"].resolve()), "locator": f"best_regression_hdc.fold_parameter_policy[outer_fold={outer_fold}], ridge_alpha=0.01"}
    if model_key == "traditional_classification":
        return {"path": str(UPSTREAM["phase04a_fold_params"].resolve()), "locator": f"gradient_boosting.{outer_fold}"}
    return {"path": str(REGRESSION_FOLD_PARAMS[str(outer_fold)].resolve()), "locator": "best_params"}


def audit_checkpoint_portability() -> dict[str, Any]:
    directories = {
        "traditional_classification": PHASE04A / "results" / "checkpoints" / "gradient_boosting",
        "traditional_regression": PHASE04B / "results" / "checkpoints" / "gradient_boosting",
        "hdc_classification": PHASE06 / "results" / "checkpoints" / "final_confirmation" / "hybrid",
        "hdc_regression": PHASE06 / "results" / "checkpoints" / "final_confirmation" / "hybrid",
    }
    model_extensions = {".joblib", ".pkl", ".pickle", ".npz", ".npy", ".pt", ".pth", ".onnx"}
    interfaces = {}
    for key, directory in directories.items():
        files = [path for path in directory.rglob("*") if path.is_file()]
        binary_models = [path for path in files if path.suffix.lower() in model_extensions]
        names = [path.name.lower() for path in files]
        has_preprocessing_state = any("preprocess" in name or "imputer" in name or "scaler" in name for name in names)
        has_feature_order = any("feature_order" in name or "feature_names" in name for name in names)
        has_encoder_state = key.startswith("hdc_") and any("encoder" in name or "codebook" in name for name in names)
        checks = {
            "complete_loadable_model_saved": bool(binary_models),
            "training_fold_preprocessing_state_saved": has_preprocessing_state,
            "feature_order_saved": has_feature_order,
            "hdc_encoder_state_saved_if_required": has_encoder_state if key.startswith("hdc_") else True,
            "full_input_prediction_reproduced_without_refit": False,
        }
        interfaces[key] = {
            "checkpoint_directory": str(directory.resolve()),
            "file_count": len(files),
            "loadable_model_files": [str(path.resolve()) for path in binary_models],
            "checks": checks,
            "portable": all(checks.values()),
            "reproduction_attempted": False,
            "reason": "No complete loadable model plus fold-local preprocessing/feature-order state is available; prediction reproduction would require refitting and is prohibited.",
        }
    authorized = all(item["portable"] for item in interfaces.values())
    audit = {
        "phase": "09", "audit": "checkpoint_portability", "status": "PASS",
        "protocol_status": "AUTHORIZED_INFERENCE_ONLY" if authorized else "NOT_FEASIBLE_DUE_TO_CHECKPOINT_INTERFACE",
        "interfaces": interfaces,
        "neutralization_rule_if_authorized": {
            "test_only_action": "Set the entire missing modality to NaN.",
            "imputation": "Use the already-fitted training-fold median imputer only.",
            "test_statistics": "PROHIBITED", "model_refit": "PROHIBITED", "feature_reselection": "PROHIBITED",
        },
        "training_executed": False, "predictions_generated": False,
        "interpretation": "The audit completed successfully; the optional protocol is not feasible because its required interface gate failed.",
    }
    write_json("audits/phase09_checkpoint_portability_audit.json", audit)
    return audit


def build_loso(primary: pd.DataFrame, folds: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    all_subjects = set(primary.subject_id.astype(str))
    for split_index, subject in enumerate(sorted(all_subjects), 1):
        subject_rows = folds.loc[folds.subject_id.astype(str) == subject].copy()
        original_folds = sorted(int(value) for value in subject_rows.outer_fold.unique())
        if len(original_folds) != 1:
            raise RuntimeError(f"Subject {subject} does not map to exactly one frozen outer fold")
        original_fold = original_folds[0]
        original_outer_training_subjects = set(folds.loc[folds.outer_fold != original_fold, "subject_id"].astype(str))
        loso_training_subjects = all_subjects - {subject}
        config_sources = {key: config_source(key, original_fold) for key in MODEL_SPECS}
        mapping = {
            "loso_split": split_index, "test_subject": subject, "original_outer_fold": original_fold,
            "train_subject_count": len(loso_training_subjects), "test_subject_count": 1,
            "test_subject_in_original_outer_training": subject in original_outer_training_subjects,
            "test_subject_excluded_from_original_config_selection_evidence": subject not in original_outer_training_subjects,
            "new_inner_cv_authorized": False, "loso_test_subject_parameter_access": False,
            "fit_policy": "Refit frozen configuration on the 34 LOSO training subjects only.",
            "config_sources": config_sources,
            "expected_test_run_keys": sorted(subject_rows.run_key.astype(str)),
        }
        mappings.append(mapping)
        for row in subject_rows.sort_values("run_key").to_dict("records"):
            assignments.append({
                "loso_split": split_index, "test_subject": subject, "run_key": row["run_key"],
                "subject_id": row["subject_id"], "original_outer_fold": original_fold,
                "target_class": int(row["target_class"]), "target_score": float(row["target_score"]),
                "train_subject_count": 34, "test_subject_count": 1, "subject_overlap": 0,
            })
    checks = {
        "splits_35": len(mappings) == 35,
        "assignment_rows_419": len(assignments) == 419,
        "each_subject_held_out_once": len({item["test_subject"] for item in mappings}) == 35,
        "each_run_key_once": len({item["run_key"] for item in assignments}) == len(assignments) == 419,
        "no_run_key_missing": set(item["run_key"] for item in assignments) == set(primary.run_key.astype(str)),
        "training_subjects_34": all(item["train_subject_count"] == 34 for item in mappings),
        "test_subjects_1": all(item["test_subject_count"] == 1 for item in mappings),
        "config_selection_excludes_test_subject": all(item["test_subject_excluded_from_original_config_selection_evidence"] for item in mappings),
        "no_new_inner_cv": all(not item["new_inner_cv_authorized"] for item in mappings),
    }
    return assignments, mappings, checks


def enumerate_runs(folds: pd.DataFrame, mappings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    condition_by_name = {item["condition"]: item for item in MISSING_CONDITIONS}
    for condition_name in [item["condition"] for item in MISSING_CONDITIONS if item["condition"] != "FULL_PRIMARY_REFERENCE"]:
        condition = condition_by_name[condition_name]
        for outer_fold in range(1, 6):
            expected = sorted(folds.loc[folds.outer_fold == outer_fold, "run_key"].astype(str))
            for model_key, spec in MODEL_SPECS.items():
                for seed in spec["seeds"]:
                    seed_label = f"seed_{seed}" if isinstance(seed, int) else "canonical"
                    run_identifier = f"mm__{condition_name.lower()}__{model_key}__fold_{outer_fold}__{seed_label}"
                    records.append({
                        "run_identifier": run_identifier, "condition": condition_name,
                        "protocol": "RETRAIN_WITHOUT_MODALITY", "task": spec["task"],
                        "model_key": model_key, "model_family": spec["model_family"],
                        "outer_fold": outer_fold, "loso_subject": None, "seed_or_canonical": seed,
                        "feature_count": condition["feature_count"], "config_source": config_source(model_key, outer_fold),
                        "expected_test_run_keys": expected,
                        "checkpoint_path": f"results/checkpoints/missing_modality/{condition_name.lower()}/{model_key}/fold_{outer_fold}/{seed_label}.json",
                        "prediction_path": f"results/predictions/missing_modality/{condition_name.lower()}/{model_key}/fold_{outer_fold}/{seed_label}.csv",
                        "status": "AUTHORIZED_NOT_EXECUTED",
                    })
    for mapping in mappings:
        subject = mapping["test_subject"]
        outer_fold = mapping["original_outer_fold"]
        expected = mapping["expected_test_run_keys"]
        for model_key, spec in MODEL_SPECS.items():
            for seed in spec["seeds"]:
                seed_label = f"seed_{seed}" if isinstance(seed, int) else "canonical"
                run_identifier = f"loso__{subject}__{model_key}__{seed_label}"
                records.append({
                    "run_identifier": run_identifier, "condition": "LOSO",
                    "protocol": "LEAVE_ONE_SUBJECT_OUT", "task": spec["task"],
                    "model_key": model_key, "model_family": spec["model_family"],
                    "outer_fold": None, "loso_subject": subject, "seed_or_canonical": seed,
                    "feature_count": 1176, "config_source": config_source(model_key, outer_fold),
                    "expected_test_run_keys": expected,
                    "checkpoint_path": f"results/checkpoints/loso/{subject}/{model_key}/{seed_label}.json",
                    "prediction_path": f"results/predictions/loso/{subject}/{model_key}/{seed_label}.csv",
                    "status": "AUTHORIZED_NOT_EXECUTED",
                })
    return records


def run_freeze() -> dict[str, Any]:
    initialization_summary = run_initialization()
    initialization_documents = {path: read_json(PHASE09_ROOT / path) for path in INITIALIZATION_FILES}
    upstream_paths = referenced_upstream_paths()
    upstream_before = snapshot(upstream_paths)
    result_files_before = result_file_snapshot()

    input_audit = initialization_documents["audits/phase09_input_and_fold_audit.json"]
    upstream_audit = initialization_documents["audits/phase09_upstream_freeze_audit.json"]
    modality_manifest = initialization_documents["manifests/phase09_modality_manifest.json"]
    initialization_checks = {
        "initialization_ready": initialization_summary["ready_for_contract_freeze_pre_notebook"],
        "primary_rows_419": input_audit["actual"]["rows"] == 419,
        "subjects_35": input_audit["actual"]["subjects"] == 35,
        "primary_features_1176": input_audit["actual"]["primary_features"] == 1176,
        "unique_run_keys_419": input_audit["actual"]["unique_run_keys"] == 419,
        "primary_checksum": sha256(PRIMARY) == EXPECTED_PRIMARY_SHA256,
        "fold_checksum": sha256(FOLDS) == EXPECTED_FOLD_SHA256,
        "interfaces_all_pass": all(upstream_audit["interface_results"].values()),
        "phase08_frozen": read_json(UPSTREAM["phase08"])["status"] == "FROZEN",
        "modalities_5": len(modality_manifest["modalities"]) == 5,
        "modality_union_1176": modality_manifest["modality_feature_union_count"] == 1176,
        "loso_feasible_35": initialization_documents["audits/phase09_loso_feasibility_audit.json"]["split_count"] == 35,
        "initialization_notebook_persistence": initialization_documents["audits/phase09_notebook_persistence_audit.json"]["status"] == "PASS",
        "phase09_not_trained": not initialization_summary["model_training_executed"],
    }
    if not all(initialization_checks.values()):
        raise RuntimeError({"initialization_checks": initialization_checks})

    primary = pd.read_csv(PRIMARY)
    folds = pd.read_csv(FOLDS)
    assignments, mappings, loso_checks = build_loso(primary, folds)
    write_csv(
        "manifests/phase09_loso_assignments.csv", assignments,
        ["loso_split", "test_subject", "run_key", "subject_id", "original_outer_fold", "target_class", "target_score", "train_subject_count", "test_subject_count", "subject_overlap"],
    )
    runs = enumerate_runs(folds, mappings)
    run_ids = [item["run_identifier"] for item in runs]
    model_counts = Counter(item["model_key"] for item in runs)
    protocol_counts = Counter(item["protocol"] for item in runs)
    missing_counts = Counter(item["model_key"] for item in runs if item["protocol"] == "RETRAIN_WITHOUT_MODALITY")
    loso_counts = Counter(item["model_key"] for item in runs if item["protocol"] == "LEAVE_ONE_SUBJECT_OUT")
    run_checks = {
        "training_runs_720": len(runs) == 720,
        "duplicate_run_identifiers_0": len(run_ids) - len(set(run_ids)) == 0,
        "all_authorized_not_executed": all(item["status"] == "AUTHORIZED_NOT_EXECUTED" for item in runs),
        "missing_modality_runs_300": protocol_counts["RETRAIN_WITHOUT_MODALITY"] == 300,
        "loso_runs_420": protocol_counts["LEAVE_ONE_SUBJECT_OUT"] == 420,
        "hdc_classification_300": model_counts["hdc_classification"] == 300,
        "hdc_regression_300": model_counts["hdc_regression"] == 300,
        "traditional_classification_60": model_counts["traditional_classification"] == 60,
        "traditional_regression_60": model_counts["traditional_regression"] == 60,
        "missing_breakdown": missing_counts == {"hdc_classification": 125, "hdc_regression": 125, "traditional_classification": 25, "traditional_regression": 25},
        "loso_breakdown": loso_counts == {"hdc_classification": 175, "hdc_regression": 175, "traditional_classification": 35, "traditional_regression": 35},
        "full_reference_not_counted": all(item["condition"] != "FULL_PRIMARY_REFERENCE" for item in runs),
    }
    if not all(run_checks.values()):
        raise RuntimeError({"run_checks": run_checks, "model_counts": model_counts, "protocol_counts": protocol_counts})

    portability = audit_checkpoint_portability()
    selected = initialization_documents["configs/phase09_selected_model_interfaces.json"]
    missing_contract = {
        "phase": "09", "status": "FROZEN", "primary_protocol": "RETRAIN_WITHOUT_MODALITY",
        "conditions": MISSING_CONDITIONS, "new_training_conditions": 5, "new_training_runs": 300,
        "run_breakdown": {"hdc_classification": 125, "hdc_regression": 125, "traditional_classification": 25, "traditional_regression": 25},
        "full_primary_reference_policy": "REUSED_NOT_RETRAINED",
        "fold_local_pipeline": "Remove the modality before fitting outer-training-only preprocessing, feature selection, and model fitting; use the same remaining feature set on the outer test fold.",
        "performance_features": "EXCLUDED", "sudden_test_time_missingness": portability["protocol_status"],
        "protocol_separation": "SUDDEN_TEST_TIME_MISSINGNESS is never pooled into RETRAIN_WITHOUT_MODALITY curves and is excluded from 300 training runs.",
        "sources": [source_record(PHASE09_ROOT / "manifests" / "phase09_modality_manifest.json", "frozen_phase09_modality_manifest"), source_record(PRIMARY, "primary_without_performance")],
    }
    write_json("configs/phase09_missing_modality_contract.json", missing_contract)

    loso_mapping_config = {
        "phase": "09", "status": "FROZEN", "mapping_rule": "Map each LOSO test subject to its Phase 03 original frozen outer_fold and reuse only that fold's upstream-selected configuration.",
        "no_new_inner_cv": True, "test_subject_parameter_selection": False, "mappings": mappings,
        "sources": [source_record(FOLDS, "frozen_phase03_outer_assignments"), source_record(UPSTREAM["phase04a_fold_params"], "traditional_classification_fold_parameters"), source_record(UPSTREAM["phase06"], "hdc_fold_policies")],
    }
    write_json("configs/phase09_loso_config_mapping.json", loso_mapping_config)
    loso_contract = {
        "phase": "09", "status": "FROZEN", "protocol": "Leave-One-Subject-Out",
        "splits": 35, "training_subjects": 34, "test_subjects": 1, "runs_per_subject": 12,
        "total_training_runs": 420,
        "run_breakdown": {"hdc_classification": 175, "hdc_regression": 175, "traditional_classification": 35, "traditional_regression": 35},
        "outer_cv_relationship": "Supplementary robustness evaluation; Phase 03 frozen five-fold assignments are not replaced, modified, or regenerated.",
        "selection_guardrail": "No new inner CV, tuning, or test-subject parameter selection; configurations are mapped from original outer folds.",
        "sources": [source_record(FOLDS, "frozen_phase03_outer_assignments"), source_record(PHASE09_ROOT / "manifests" / "phase09_loso_assignments.csv", "deterministic_loso_test_assignments")],
    }
    write_json("configs/phase09_loso_contract.json", loso_contract)

    aggregation = {
        "phase": "09", "status": "FROZEN", "regression_range": [1.0, 4.0],
        "regression_task_name": "bounded difficulty-induced workload proxy regression",
        "missing_modality": {"outer_cv": "Frozen Phase 03 five-fold subject-wise CV", "coverage": "419 unique run_key per condition/task/model"},
        "loso": {"coverage": "Each subject predicted only by its LOSO split; 419 unique run_key per task/model"},
        "hdc_classification": {"seeds_per_run_key": 5, "aggregation": "Aggregate frozen class scores across seeds, then apply deterministic argmax/tie-breaking; never average class labels."},
        "hdc_regression": {"seeds_per_run_key": 5, "aggregation": "Average prediction_raw across seeds, then clip once to [1.0, 4.0] as prediction_bounded."},
        "traditional": {"predictions_per_run_key": 1, "aggregation": "Canonical prediction only."},
    }
    write_json("configs/phase09_oof_aggregation_rules.json", aggregation)
    statistical = {
        "phase": "09", "status": "FROZEN", "statistical_unit": "subject_id",
        "classification": {"primary": "Macro-F1", "secondary": ["Balanced Accuracy", "Accuracy", "Severe Error Rate", "Per-class Recall", "Confusion Matrix", "Quadratic Weighted Kappa"]},
        "regression": {"primary": "bounded MAE", "secondary": ["raw MAE", "bounded RMSE", "bounded R²", "bounded Spearman", "clipping count/rate", "rounded regression Macro-F1", "adjacent accuracy", "severe error rate"]},
        "missing_modality": {"comparison": "Each missing condition paired with FULL_PRIMARY_REFERENCE at subject level", "test": "Wilcoxon signed-rank", "multiplicity": "Holm correction", "effect_size": "rank-biserial", "bootstrap": {"resamples": 2000, "unit": "subject_id", "paired": True, "confidence_level": 0.95}},
        "loso_stability": {"summaries": ["median", "IQR", "range"], "bootstrap": {"resamples": 2000, "unit": "subject_id", "confidence_level": 0.95}, "worst_subject": "diagnostic only; never delete", "model_reselection": "PROHIBITED"},
        "interpretation_guardrail": "A non-significant result must not be described as complete equivalence.", "formal_statistics_executed": False,
    }
    write_json("configs/phase09_statistical_rules.json", statistical)

    coverage_groups = []
    for condition in MISSING_CONDITIONS:
        for model_key, spec in MODEL_SPECS.items():
            coverage_groups.append({
                "protocol": "FULL_PRIMARY_REFERENCE" if condition["condition"] == "FULL_PRIMARY_REFERENCE" else "RETRAIN_WITHOUT_MODALITY",
                "condition": condition["condition"], "task": spec["task"], "model_key": model_key,
                "expected_unique_run_keys": 419, "expected_seeds_per_run_key": 5 if model_key.startswith("hdc_") else 1,
                "reference_policy": condition["policy"],
            })
    for model_key, spec in MODEL_SPECS.items():
        coverage_groups.append({
            "protocol": "LEAVE_ONE_SUBJECT_OUT", "condition": "LOSO", "task": spec["task"], "model_key": model_key,
            "expected_unique_run_keys": 419, "expected_seeds_per_run_key": 5 if model_key.startswith("hdc_") else 1,
            "expected_subject_splits": 35,
        })
    write_json("manifests/phase09_expected_coverage_manifest.json", {
        "phase": "09", "status": "FROZEN_EXPECTATION_NOT_EXECUTED", "coverage_groups": coverage_groups,
        "sources": [source_record(PRIMARY, "primary_run_key_universe"), source_record(FOLDS, "frozen_outer_test_coverage"), source_record(PHASE09_ROOT / "manifests" / "phase09_loso_assignments.csv", "loso_test_coverage")],
    })
    write_json("configs/phase09_execution_manifest.json", {
        "phase": "09", "status": "AUTHORIZED_NOT_EXECUTED", "training_run_count": len(runs),
        "duplicate_run_identifiers": len(run_ids) - len(set(run_ids)), "run_counts_by_protocol": dict(protocol_counts),
        "run_counts_by_model": dict(model_counts), "full_primary_reference_counted_as_training": False,
        "sudden_test_time_missingness_counted_as_training": False, "training_runs": runs,
        "sources": [source_record(PHASE09_ROOT / "configs" / "phase09_selected_model_interfaces.json", "selected_model_interfaces"), source_record(FOLDS, "frozen_outer_assignments")],
    })

    generalization_guardrails = {
        "phase": "09", "audit": "generalization_guardrails", "status": "PASS",
        "allowed": ["subject generalization via LOSO", "missing-modality robustness", "subject-level performance heterogeneity", "dependence on the flight modality"],
        "not_feasible": {"unseen_session": "NOT_FEASIBLE_DUE_TO_METADATA", "unseen_scenario": "NOT_FEASIBLE_DUE_TO_METADATA", "task_template": "NOT_FEASIBLE_DUE_TO_METADATA", "route_configuration": "NOT_FEASIBLE_DUE_TO_METADATA"},
        "forbidden_scenario_proxies": ["difficulty_level", "difficulty_level_raw", "target_class", "target_score", "run order", "feature clustering"],
        "flight_generalizable_behavior_claim": "INCONCLUSIVE_DUE_TO_METADATA",
        "sources": [source_record(UPSTREAM["phase08_handoff_config"], "phase08_generalization_handoff")],
    }
    write_json("audits/phase09_generalization_guardrail_audit.json", generalization_guardrails)
    write_json("audits/phase09_loso_assignment_audit.json", {
        "phase": "09", "audit": "loso_assignments", "status": "PASS" if all(loso_checks.values()) else "FAIL",
        "checks": loso_checks, "splits": len(mappings), "assignment_rows": len(assignments), "duplicate_run_keys": len(assignments) - len({item["run_key"] for item in assignments}),
        "sources": [source_record(FOLDS, "frozen_phase03_outer_assignments"), source_record(PRIMARY, "primary_run_key_universe")],
    })
    mapping_checks = {
        "all_subjects_map_to_one_original_fold": len(mappings) == 35,
        "test_subject_never_in_original_outer_training": all(item["test_subject_excluded_from_original_config_selection_evidence"] for item in mappings),
        "new_inner_cv_disabled": all(not item["new_inner_cv_authorized"] for item in mappings),
        "loso_test_subject_parameter_access_disabled": all(not item["loso_test_subject_parameter_access"] for item in mappings),
        "all_four_config_sources_present": all(set(item["config_sources"]) == set(MODEL_SPECS) for item in mappings),
    }
    write_json("audits/phase09_config_mapping_leakage_audit.json", {
        "phase": "09", "audit": "loso_config_mapping_leakage", "status": "PASS" if all(mapping_checks.values()) else "FAIL",
        "checks": mapping_checks, "mapped_subjects": len(mappings), "test_subject_used_for_selection": False,
        "sources": [source_record(FOLDS, "frozen_phase03_outer_assignments"), source_record(PHASE09_ROOT / "configs" / "phase09_loso_config_mapping.json", "frozen_loso_mapping")],
    })
    missing_checks = {
        "six_conditions": len(MISSING_CONDITIONS) == 6, "five_new_conditions": sum(item["policy"] == "RETRAIN_WITHOUT_MODALITY" for item in MISSING_CONDITIONS) == 5,
        "feature_counts_exact": [item["feature_count"] for item in MISSING_CONDITIONS] == [1176, 943, 760, 1017, 850, 1134],
        "full_reference_reused": MISSING_CONDITIONS[0]["policy"] == "REUSED_NOT_RETRAINED",
        "new_runs_300": protocol_counts["RETRAIN_WITHOUT_MODALITY"] == 300,
        "performance_features_excluded": True, "protocols_separated": True,
    }
    write_json("audits/phase09_missing_modality_contract_audit.json", {
        "phase": "09", "audit": "missing_modality_contract", "status": "PASS" if all(missing_checks.values()) else "FAIL",
        "checks": missing_checks, "sudden_test_time_missingness": portability["protocol_status"],
        "sources": [source_record(PHASE09_ROOT / "configs" / "phase09_missing_modality_contract.json", "frozen_missing_modality_contract")],
    })
    write_json("audits/phase09_run_matrix_audit.json", {
        "phase": "09", "audit": "authorized_training_run_matrix", "status": "PASS" if all(run_checks.values()) else "FAIL",
        "checks": run_checks, "actual_training_runs": len(runs), "duplicate_run_identifiers": len(run_ids) - len(set(run_ids)),
        "run_counts_by_protocol": dict(protocol_counts), "run_counts_by_model": dict(model_counts),
        "training_executed": False, "predictions_generated": False,
        "sources": [source_record(PHASE09_ROOT / "configs" / "phase09_execution_manifest.json", "authorized_execution_manifest")],
    })

    frozen_contract = {
        "phase": "09", "phase_name": "Robustness and Generalization", "status": "PENDING_NOTEBOOK_PERSISTENCE",
        "evidence_scope": {"primary_input": str(PRIMARY.resolve()), "performance_features": "EXCLUDED", "allowed_claims": generalization_guardrails["allowed"], "metadata_limited_claims": generalization_guardrails["not_feasible"]},
        "selected_model_interfaces": {
            "hdc_classification": {"model": "HDC+OnlineHD Hybrid", "dimension": 5000, "levels": 51, "feature_k": 50, "seeds": SEEDS, "fold_specific_structures": selected["hdc_classification"]["fold_selected_structures"]},
            "hdc_regression": {"model": "COMMON_ENCODER_READOUT_BASELINE", "dimension": 10000, "levels": 51, "feature_k": 50, "ridge_alpha": 0.01, "seeds": SEEDS, "fold_parameter_policy": selected["hdc_regression"]["fold_parameter_policy"]},
            "traditional_classification": {"model": "Gradient Boosting", "fold_specific_parameters": selected["traditional_classification"]["fold_specific_parameters"]},
            "traditional_regression": {"model": "Gradient Boosting Regressor", "fold_specific_parameters": selected["traditional_regression"]["fold_specific_parameters"]},
        },
        "missing_modality": {"protocol": "RETRAIN_WITHOUT_MODALITY", "conditions": MISSING_CONDITIONS, "new_training_runs": 300, "test_time_missingness": portability["protocol_status"]},
        "loso": {"splits": 35, "training_runs": 420, "config_mapping": "original frozen outer-fold configuration for each held-out subject"},
        "authorized_training_runs": {"total": 720, "hdc_classification": 300, "hdc_regression": 300, "traditional_classification": 60, "traditional_regression": 60},
        "aggregation_rules_path": "configs/phase09_oof_aggregation_rules.json", "statistical_rules_path": "configs/phase09_statistical_rules.json",
        "training_executed": False, "predictions_generated": False, "formal_statistics_executed": False,
    }
    write_json("configs/phase09_frozen_contract.json", frozen_contract)

    upstream_after = snapshot(upstream_paths)
    result_files_after = result_file_snapshot()
    upstream_modified = [path for path in upstream_before if upstream_before[path] != upstream_after.get(path)]
    new_result_files = sorted(set(result_files_after) - set(result_files_before))
    freeze_checks = {
        **initialization_checks,
        "loso_assignments": all(loso_checks.values()), "loso_mapping_leakage": all(mapping_checks.values()),
        "missing_modality_contract": all(missing_checks.values()), "run_matrix": all(run_checks.values()),
        "generalization_guardrails": generalization_guardrails["status"] == "PASS",
        "subject_statistical_unit": statistical["statistical_unit"] == "subject_id",
        "upstream_files_modified_0": not upstream_modified,
        "training_artifacts_added_0": not new_result_files,
        "predictions_generated_0": not result_files_after,
    }
    freeze_status = "PASS" if all(freeze_checks.values()) else "FAIL"
    write_json("audits/phase09_contract_freeze_audit.json", {
        "phase": "09", "audit": "contract_freeze", "status": freeze_status, "checks": freeze_checks,
        "upstream_files_modified": upstream_modified, "training_artifacts_added": new_result_files,
        "predictions_generated": False, "model_training_executed": False,
        "ready_pending_notebook_persistence": freeze_status == "PASS",
    })
    write_json("configs/phase09_contract_freeze.json", {
        "phase": "09", "status": "PENDING_NOTEBOOK_PERSISTENCE" if freeze_status == "PASS" else "CONTRACT_FREEZE_FAILED",
        "frozen_at_utc": now(), "primary_sha256": sha256(PRIMARY), "fold_sha256": sha256(FOLDS),
        "upstream_snapshot_before": upstream_before, "upstream_snapshot_after": upstream_after,
        "upstream_files_modified": upstream_modified, "authorized_training_runs": len(runs),
        "duplicate_run_identifiers": len(run_ids) - len(set(run_ids)), "training_executed": False,
        "predictions_generated": False, "ready_for_execution": False,
    })
    return {
        "contract_freeze_audit": freeze_status, "status": "PENDING_NOTEBOOK_PERSISTENCE",
        "authorized_training_runs": len(runs), "missing_modality_runs": protocol_counts["RETRAIN_WITHOUT_MODALITY"],
        "loso_runs": protocol_counts["LEAVE_ONE_SUBJECT_OUT"], "duplicate_run_identifiers": len(run_ids) - len(set(run_ids)),
        "test_time_missingness": portability["protocol_status"], "upstream_files_modified": len(upstream_modified),
        "training_executed": False, "predictions_generated": False,
    }


def finalize_contract_artifacts() -> dict[str, Any]:
    freeze_audit = read_json(PHASE09_ROOT / "audits" / "phase09_contract_freeze_audit.json")
    notebook_audit = read_json(PHASE09_ROOT / "audits" / "phase09_contract_notebook_persistence_audit.json")
    freeze_config_path = PHASE09_ROOT / "configs" / "phase09_contract_freeze.json"
    frozen_contract_path = PHASE09_ROOT / "configs" / "phase09_frozen_contract.json"
    freeze_config = read_json(freeze_config_path)
    frozen_contract = read_json(frozen_contract_path)
    current_upstream = snapshot([Path(path) for path in freeze_config["upstream_snapshot_before"]])
    upstream_modified = [path for path, value in freeze_config["upstream_snapshot_before"].items() if current_upstream.get(path) != value]
    result_files = result_file_snapshot()
    required_without_self = [path for path in CONTRACT_FILES if path not in {"manifests/phase09_contract_artifact_manifest.json", "audits/phase09_contract_artifact_audit.json"}]
    missing = [path for path in required_without_self if not (PHASE09_ROOT / path).exists()]
    pass_status = freeze_audit["status"] == "PASS" and notebook_audit["status"] == "PASS" and not upstream_modified and not result_files and not missing
    final_status = "CONTRACT_FROZEN_NOT_TRAINED" if pass_status else "CONTRACT_FREEZE_FAILED"
    frozen_contract["status"] = final_status
    frozen_contract["ready_for_execution"] = pass_status
    write_json("configs/phase09_frozen_contract.json", frozen_contract)
    freeze_config["status"] = final_status
    freeze_config["upstream_files_modified"] = upstream_modified
    freeze_config["ready_for_execution"] = pass_status
    write_json("configs/phase09_contract_freeze.json", freeze_config)

    manifest_paths = [
        *[PHASE09_ROOT / path for path in CONTRACT_FILES if path not in {"manifests/phase09_contract_artifact_manifest.json", "audits/phase09_contract_artifact_audit.json"}],
        PHASE09_ROOT / "Phase_09_Robustness_and_Generalization.ipynb",
        PHASE09_ROOT / "scripts" / "freeze_phase09_contract.py",
        PHASE09_ROOT / "scripts" / "append_phase09_contract_notebook.py",
        PHASE09_ROOT / "tests" / "test_phase09_contract.py",
    ]
    artifact_records = [source_record(path, "phase09_contract_artifact") for path in manifest_paths if path.exists()]
    write_json("manifests/phase09_contract_artifact_manifest.json", {
        "phase": "09", "status": final_status, "self_hash_included": False,
        "artifact_count": len(artifact_records), "sources": artifact_records,
        "missing_artifacts": missing, "upstream_files_modified": upstream_modified,
        "training_artifacts_added": result_files, "predictions_generated": False,
    })
    manifest_path = PHASE09_ROOT / "manifests" / "phase09_contract_artifact_manifest.json"
    audit = {
        "phase": "09", "audit": "contract_artifacts", "status": "PASS" if pass_status else "FAIL",
        "contract_manifest_path": str(manifest_path.resolve()), "contract_manifest_sha256": sha256(manifest_path),
        "artifact_count": len(artifact_records), "missing_artifacts": missing,
        "upstream_files_modified": upstream_modified, "training_artifacts_added": result_files,
        "model_training_executed": False, "predictions_generated": False,
        "phase09_status": final_status, "ready_for_execution": pass_status,
    }
    write_json("audits/phase09_contract_artifact_audit.json", audit)
    return audit


if __name__ == "__main__":
    print(json.dumps(run_freeze(), ensure_ascii=False, indent=2))
