"""Initialize and audit Phase 09 without training models or generating predictions."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PHASE09_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PHASE09_ROOT.parent
PROJECT_ROOT = EXPERIMENTS_ROOT.parent

PHASE03 = EXPERIMENTS_ROOT / "phase_03_multimodal_dataset_labeling"
PHASE04A = EXPERIMENTS_ROOT / "phase_04a_traditional_classification_baselines"
PHASE04B = EXPERIMENTS_ROOT / "phase_04b_traditional_regression_baselines"
PHASE06 = EXPERIMENTS_ROOT / "phase_06_hdc_variant_screening"
PHASE07 = EXPERIMENTS_ROOT / "phase_07_unimodal_contribution"
PHASE08 = EXPERIMENTS_ROOT / "phase_08_fusion_and_shortcut_analysis"

PRIMARY = PHASE03 / "data" / "primary_without_performance.csv"
FOLDS = PHASE03 / "data" / "fold_assignments.csv"
PRIMARY_FEATURES = PHASE03 / "manifests" / "primary_feature_manifest.json"
FEATURE_GROUPS = PHASE03 / "manifests" / "feature_group_manifest.json"
MODALITY_SOURCE = PHASE07 / "manifests" / "phase07_modality_feature_manifest.json"

EXPECTED_PRIMARY_SHA256 = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLD_SHA256 = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
EXPECTED_UPSTREAM_FREEZE_SHA256 = {
    "phase04a": "34ea8100d9406f9701750a441aa6537323c28bcdb194cb3fd3645c4f7de4a2e1",
    "phase04b": "e2c88b1139a50aab6d47b6477c7bceff74f8443095f9d039ea9af84b715ee790",
    "phase06": "cd94acdf0bcb463b5f48a7c84110a23059899b6b5bf156131694827d7d7bde66",
    "phase07": "8569b48a8210f0ca1316d5a140d292edb892e2a556ea13ee299e9b97699af492",
    "phase08": "f36211f1b793d1868573973f8ea46a7bc89aee03759079e268df75d2a4894595",
}

UPSTREAM = {
    "phase04a": PHASE04A / "configs" / "phase04a_freeze.json",
    "phase04a_best_classifier": PHASE04A / "configs" / "best_classifier.json",
    "phase04a_final_configuration": PHASE04A / "configs" / "phase04a_final_configuration.json",
    "phase04a_fold_params": PHASE04A / "configs" / "classification_best_params_by_fold.json",
    "phase04b": PHASE04B / "configs" / "phase04b_freeze.json",
    "phase04b_gradient_boosting_configuration": PHASE04B / "configs" / "gradient_boosting_configuration.json",
    "phase06": PHASE06 / "configs" / "phase06_freeze.json",
    "phase06_best_classification": PHASE06 / "configs" / "phase06_best_classification_hdc.json",
    "phase06_best_regression": PHASE06 / "configs" / "phase06_best_regression_hdc.json",
    "phase07": PHASE07 / "configs" / "phase07_freeze.json",
    "phase07_modality_manifest": MODALITY_SOURCE,
    "phase08": PHASE08 / "configs" / "phase08_freeze.json",
    "phase08_final_manifest": PHASE08 / "manifests" / "phase08_final_manifest.json",
    "phase08_handoff_config": PHASE08 / "configs" / "phase09_generalization_handoff.json",
    "phase08_handoff_manifest": PHASE08 / "manifests" / "phase08_to_phase09_generalization_handoff.json",
}

REGRESSION_FOLD_PARAMS = {
    str(fold): PHASE04B / "results" / "checkpoints" / "gradient_boosting" / f"gradient_boosting_fold_{fold}_best_params.json"
    for fold in range(1, 6)
}

REQUIRED_DIRECTORIES = [
    "data", "manifests", "audits", "configs", "scripts", "tests", "figures", "logs", "reports",
    "results/checkpoints", "results/predictions", "results/fold_metrics", "results/oof",
    "results/missing_modality", "results/loso", "results/subject_stability", "results/summaries",
]

REQUIRED_ARTIFACTS = [
    "Phase_09_Robustness_and_Generalization.ipynb", "README.md", "notes.md",
    "configs/phase09_experiment_contract_draft.json", "configs/phase09_environment.json",
    "configs/phase09_generalization_scope.json", "configs/phase09_selected_model_interfaces.json",
    "configs/phase09_missing_modality_plan.json", "configs/phase09_loso_plan.json",
    "configs/phase09_statistical_plan.json", "manifests/phase09_input_manifest.json",
    "manifests/phase09_upstream_freeze_manifest.json", "manifests/phase09_modality_manifest.json",
    "manifests/phase09_loso_feasibility_manifest.json", "audits/phase09_input_and_fold_audit.json",
    "audits/phase09_upstream_freeze_audit.json", "audits/phase09_modality_coverage_audit.json",
    "audits/phase09_loso_feasibility_audit.json", "audits/phase09_generalization_scope_audit.json",
    "audits/phase09_initialization_artifact_audit.json", "audits/phase09_notebook_persistence_audit.json",
    "scripts/initialize_phase09.py", "scripts/build_phase09_notebook.py", "tests/test_phase09_initialization.py",
]

NON_FEATURE_COLUMNS = {
    "subject_id", "session_id", "run_id", "difficulty_level_raw", "difficulty_level",
    "run_key", "target_class", "target_score", "outer_fold",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(relative_path: str, value: Any) -> None:
    path = PHASE09_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def source_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "read_only_policy": True,
    }


def freeze_status(value: Any) -> bool:
    return str(value).upper() in {"FROZEN", "YES"}


def run_initialization() -> dict[str, Any]:
    for directory in REQUIRED_DIRECTORIES:
        (PHASE09_ROOT / directory).mkdir(parents=True, exist_ok=True)

    required_inputs = [PRIMARY, FOLDS, PRIMARY_FEATURES, FEATURE_GROUPS, *UPSTREAM.values(), *REGRESSION_FOLD_PARAMS.values()]
    missing_inputs = [str(path) for path in required_inputs if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing required frozen inputs: {missing_inputs}")

    upstream_hash_before = {str(path): sha256(path) for path in required_inputs}
    primary = pd.read_csv(PRIMARY)
    folds = pd.read_csv(FOLDS)
    primary_manifest = read_json(PRIMARY_FEATURES)
    modality_source = read_json(MODALITY_SOURCE)

    feature_names = list(primary_manifest["features"])
    actual_csv_features = [name for name in primary.columns.astype(str) if name not in NON_FEATURE_COLUMNS]
    primary_sha = sha256(PRIMARY)
    fold_sha = sha256(FOLDS)
    merged = primary[["run_key", "subject_id", "target_class", "target_score", "outer_fold"]].merge(
        folds[["run_key", "subject_id", "target_class", "target_score", "outer_fold"]],
        on="run_key", how="outer", suffixes=("_data", "_folds"), indicator=True,
    )
    fold_alignment = bool(
        (merged["_merge"] == "both").all()
        and (merged["subject_id_data"] == merged["subject_id_folds"]).all()
        and (merged["target_class_data"] == merged["target_class_folds"]).all()
        and (merged["target_score_data"] == merged["target_score_folds"]).all()
        and (merged["outer_fold_data"] == merged["outer_fold_folds"]).all()
    )
    outer_folds = sorted(int(value) for value in folds["outer_fold"].unique())
    fold_details = []
    for fold in outer_folds:
        test_subjects = set(folds.loc[folds.outer_fold == fold, "subject_id"].astype(str))
        train_subjects = set(folds.loc[folds.outer_fold != fold, "subject_id"].astype(str))
        fold_details.append({
            "outer_fold": fold,
            "train_subjects": len(train_subjects),
            "test_subjects": len(test_subjects),
            "subject_overlap": len(train_subjects & test_subjects),
            "train_rows": int((folds.outer_fold != fold).sum()),
            "test_rows": int((folds.outer_fold == fold).sum()),
        })
    outer_isolation = all(item["subject_overlap"] == 0 for item in fold_details)

    input_checks = {
        "rows": len(primary) == 419,
        "subjects": primary.subject_id.nunique() == 35,
        "primary_features": len(feature_names) == 1176,
        "unique_run_keys": primary.run_key.nunique() == len(primary) == 419,
        "target_class_values": sorted(int(v) for v in primary.target_class.unique()) == [0, 1, 2, 3],
        "target_score_values": sorted(float(v) for v in primary.target_score.unique()) == [1.0, 2.0, 3.0, 4.0],
        "targets_complete": not primary[["target_class", "target_score"]].isna().any().any(),
        "manifest_features_unique": len(feature_names) == len(set(feature_names)),
        "manifest_equals_csv_feature_universe": set(feature_names) == set(actual_csv_features),
        "fold_rows_and_run_keys": len(folds) == 419 and folds.run_key.nunique() == 419,
        "fold_alignment": fold_alignment,
        "frozen_outer_folds": outer_folds == [1, 2, 3, 4, 5],
        "outer_subject_isolation": outer_isolation,
        "primary_checksum": primary_sha == EXPECTED_PRIMARY_SHA256,
        "fold_checksum": fold_sha == EXPECTED_FOLD_SHA256,
    }
    input_status = "PASS" if all(input_checks.values()) else "FAIL"

    modality_aliases = {"body_movement": "body_movement_features"}
    normalized_modalities: list[dict[str, Any]] = []
    feature_owners: dict[str, list[str]] = {}
    for item in modality_source["modalities"]:
        normalized_name = modality_aliases.get(item["name"], item["name"])
        features = list(item["features"])
        normalized_modalities.append({
            "name": normalized_name,
            "source_name": item["name"],
            "feature_count": len(features),
            "declared_feature_count": item["feature_count"],
            "features": features,
            "missing_condition": f"missing_{normalized_name.removesuffix('_features')}",
            "remaining_feature_count": len(feature_names) - len(features),
        })
        for feature in features:
            feature_owners.setdefault(feature, []).append(normalized_name)
    modality_union = set(feature_owners)
    overlaps = {feature: owners for feature, owners in feature_owners.items() if len(owners) > 1}
    absent_features = sorted(set(feature_names) - modality_union)
    extra_features = sorted(modality_union - set(feature_names))
    modality_checks = {
        "five_modalities": len(normalized_modalities) == 5,
        "declared_counts_match": all(item["feature_count"] == item["declared_feature_count"] for item in normalized_modalities),
        "features_unique_within_modalities": all(len(item["features"]) == len(set(item["features"])) for item in normalized_modalities),
        "modalities_mutually_exclusive": not overlaps,
        "union_count_1176": len(modality_union) == 1176,
        "union_equals_primary_manifest": modality_union == set(feature_names),
        "all_missing_conditions_nonempty": all(item["remaining_feature_count"] > 0 for item in normalized_modalities),
        "traditional_pipeline_feature_capacity": all(item["remaining_feature_count"] >= 200 for item in normalized_modalities),
        "hdc_feature_k_capacity": all(item["remaining_feature_count"] >= 50 for item in normalized_modalities),
    }
    modality_status = "PASS" if all(modality_checks.values()) else "FAIL"

    loso_splits = []
    for subject in sorted(primary.subject_id.astype(str).unique()):
        test = primary.loc[primary.subject_id.astype(str) == subject]
        train = primary.loc[primary.subject_id.astype(str) != subject]
        train_subjects = set(train.subject_id.astype(str))
        test_subjects = set(test.subject_id.astype(str))
        loso_splits.append({
            "fold_id": f"loso_{subject}",
            "test_subject": subject,
            "train_subject_count": len(train_subjects),
            "test_subject_count": len(test_subjects),
            "train_rows": len(train),
            "test_rows": len(test),
            "subject_overlap": len(train_subjects & test_subjects),
            "train_target_class_values": sorted(int(v) for v in train.target_class.unique()),
            "test_target_class_values": sorted(int(v) for v in test.target_class.unique()),
            "train_four_class_coverage": train.target_class.nunique() == 4,
            "test_four_class_coverage": test.target_class.nunique() == 4,
            "train_run_keys_unique": train.run_key.nunique() == len(train),
            "test_run_keys_unique": test.run_key.nunique() == len(test),
            "run_key_overlap": len(set(train.run_key.astype(str)) & set(test.run_key.astype(str))),
            "nonempty_train_and_test": bool(len(train) and len(test)),
        })
    loso_checks = {
        "deterministic_subject_order": [item["test_subject"] for item in loso_splits] == sorted(primary.subject_id.astype(str).unique()),
        "split_count_35": len(loso_splits) == 35,
        "one_test_subject_each": all(item["test_subject_count"] == 1 for item in loso_splits),
        "thirty_four_training_subjects_each": all(item["train_subject_count"] == 34 for item in loso_splits),
        "subject_isolation": all(item["subject_overlap"] == 0 for item in loso_splits),
        "run_key_isolation": all(item["run_key_overlap"] == 0 for item in loso_splits),
        "nonempty_splits": all(item["nonempty_train_and_test"] for item in loso_splits),
        "unique_run_keys": all(item["train_run_keys_unique"] and item["test_run_keys_unique"] for item in loso_splits),
        "training_four_class_coverage": all(item["train_four_class_coverage"] for item in loso_splits),
        "test_four_class_coverage": all(item["test_four_class_coverage"] for item in loso_splits),
    }
    loso_status = "PASS" if all(loso_checks.values()) else "FAIL"

    p04a = read_json(UPSTREAM["phase04a"])
    p04a_best = read_json(UPSTREAM["phase04a_best_classifier"])
    p04a_params = read_json(UPSTREAM["phase04a_fold_params"])
    p04b = read_json(UPSTREAM["phase04b"])
    p06 = read_json(UPSTREAM["phase06"])
    p06_classification = read_json(UPSTREAM["phase06_best_classification"])
    p06_regression = read_json(UPSTREAM["phase06_best_regression"])
    p07 = read_json(UPSTREAM["phase07"])
    p08 = read_json(UPSTREAM["phase08"])
    p08_handoff = read_json(UPSTREAM["phase08_handoff_config"])
    regression_params = {fold: read_json(path) for fold, path in REGRESSION_FOLD_PARAMS.items()}

    actual_freeze_hashes = {key: sha256(UPSTREAM[key]) for key in EXPECTED_UPSTREAM_FREEZE_SHA256}
    interface_checks = {
        "phase04a": {
            "status_frozen": p04a.get("phase04a_frozen") == "YES",
            "freeze_hash": actual_freeze_hashes["phase04a"] == EXPECTED_UPSTREAM_FREEZE_SHA256["phase04a"],
            "selected_model": p04a.get("best_traditional_classifier") == p04a_best.get("model") == "Gradient Boosting",
            "five_fold_parameter_interfaces": set(p04a_params.get("gradient_boosting", {})) == {"1", "2", "3", "4", "5"},
            "fold_checksum": p04a.get("frozen_phase03_sha256") == EXPECTED_FOLD_SHA256,
        },
        "phase04b": {
            "status_frozen": freeze_status(p04b.get("status")),
            "freeze_hash": actual_freeze_hashes["phase04b"] == EXPECTED_UPSTREAM_FREEZE_SHA256["phase04b"],
            "selected_model": p04b.get("best_model") == "Gradient Boosting Regressor",
            "configuration_hash": next(item for item in p04b["configuration_files"] if item["path"].endswith("gradient_boosting_configuration.json"))["sha256"] == sha256(UPSTREAM["phase04b_gradient_boosting_configuration"]),
            "five_fold_parameter_interfaces": set(regression_params) == {"1", "2", "3", "4", "5"} and all(value["frozen_fold_sha256"] == EXPECTED_FOLD_SHA256 for value in regression_params.values()),
            "fold_checksum": p04b["frozen_fold"]["sha256"] == EXPECTED_FOLD_SHA256,
        },
        "phase06": {
            "status_frozen": freeze_status(p06.get("status")),
            "freeze_hash": actual_freeze_hashes["phase06"] == EXPECTED_UPSTREAM_FREEZE_SHA256["phase06"],
            "classification_interface_exact": p06.get("best_classification_hdc") == p06_classification,
            "regression_interface_exact": p06.get("best_regression_hdc") == p06_regression,
            "classification_model": p06_classification.get("selected_variant_name") == "HDC+OnlineHD Hybrid" and p06_classification.get("selected_fixed_dimension") == 5000,
            "regression_model": p06_regression.get("selected_regression_head") == "COMMON_ENCODER_READOUT_BASELINE" and p06_regression.get("selected_fixed_dimension") == 10000,
            "final_manifest_hash": p06.get("final_manifest_sha256") == sha256(PHASE06 / "manifests" / "phase06_final_artifact_manifest.json"),
        },
        "phase07": {
            "status_frozen": freeze_status(p07.get("status")),
            "freeze_hash": actual_freeze_hashes["phase07"] == EXPECTED_UPSTREAM_FREEZE_SHA256["phase07"],
            "five_modalities": len(modality_source.get("modalities", [])) == 5,
            "final_manifest_hash": p07.get("final_manifest_sha256") == sha256(PHASE07 / "manifests" / "phase07_final_artifact_manifest.json"),
            "all_final_audits_pass": p07.get("all_final_audits_pass") is True,
        },
        "phase08": {
            "status_frozen": freeze_status(p08.get("status")),
            "freeze_hash": actual_freeze_hashes["phase08"] == EXPECTED_UPSTREAM_FREEZE_SHA256["phase08"],
            "final_manifest_hash": p08["final_manifest"]["sha256"] == sha256(UPSTREAM["phase08_final_manifest"]),
            "upstream_interface": p08.get("upstream_freeze_integrity") == "PASS",
            "phase09_handoff": p08.get("phase09_handoff_saved") is True and p08_handoff.get("phase09_executed") is False,
        },
    }
    interface_results = {phase: all(checks.values()) for phase, checks in interface_checks.items()}
    upstream_status = "PASS" if all(interface_results.values()) else "FAIL"

    scope_expected = {
        "unseen_session": "NOT_FEASIBLE_DUE_TO_METADATA",
        "unseen_scenario": "NOT_FEASIBLE_DUE_TO_METADATA",
        "task_template": "NOT_FEASIBLE_DUE_TO_METADATA",
        "route_or_configuration": "NOT_FEASIBLE_DUE_TO_METADATA",
    }
    scope_actual = {name: p08_handoff["holdouts"][name]["feasibility"] for name in scope_expected}
    forbidden_metadata = ["scenario_id", "task_template_id", "route_id", "configuration_id"]
    scope_checks = {
        "phase08_handoff_limits_match": scope_actual == scope_expected,
        "session_equals_subject_partition": bool(primary.session_id.nunique() == primary.subject_id.nunique() == 35 and primary.groupby("session_id").subject_id.nunique().max() == 1 and primary.groupby("subject_id").session_id.nunique().max() == 1),
        "required_grouping_metadata_absent": not any(column in primary.columns for column in forbidden_metadata),
        "difficulty_not_used_as_scenario": bool("difficulty_level_raw" in primary.columns and primary.groupby("difficulty_level_raw").target_class.nunique().max() == 1),
    }
    scope_status = "PASS" if all(scope_checks.values()) else "FAIL"

    common_sources = [source_record(PRIMARY, "primary_without_performance"), source_record(FOLDS, "frozen_outer_fold_assignments")]
    write_json("configs/phase09_environment.json", {
        "phase": "09", "status": "INITIALIZED", "generated_at_utc": utc_now(),
        "python_version": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "working_directory": os.getcwd(),
        "pandas_version": pd.__version__, "modeling_dependencies_invoked": False,
    })
    write_json("configs/phase09_generalization_scope.json", {
        "phase": "09", "status": "PENDING_CONTRACT_FREEZE",
        "subject_generalization": "FEASIBLE_VIA_LOSO",
        "missing_modality_robustness": "FEASIBLE_PENDING_CONTRACT",
        "unseen_session_generalization": "NOT_FEASIBLE_DUE_TO_METADATA",
        "unseen_scenario_generalization": "NOT_FEASIBLE_DUE_TO_METADATA",
        "task_template_generalization": "NOT_FEASIBLE_DUE_TO_METADATA",
        "route_configuration_generalization": "NOT_FEASIBLE_DUE_TO_METADATA",
        "flight_generalizable_behavior_claim": "INCONCLUSIVE_DUE_TO_METADATA",
        "allowed_claim": "Phase 09 can evaluate subject generalization and missing-modality robustness.",
        "prohibited_claim": "Existing metadata cannot establish cross-scenario, cross-route, or cross-task-template generalization of flight-parameter advantages.",
        "forbidden_proxies": ["difficulty_level_raw as scenario", "target_class or target_score as scenario", "run-order inference", "feature-cluster-derived real scenario"],
        "sources": [source_record(UPSTREAM["phase08_handoff_config"], "phase08_generalization_handoff"), source_record(UPSTREAM["phase08_handoff_manifest"], "phase08_to_phase09_handoff_manifest")],
    })
    model_interfaces = {
        "phase": "09", "status": "PARSED_READ_ONLY_PENDING_CONTRACT_FREEZE",
        "traditional_classification": {
            "model": "Gradient Boosting", "source_phase": "04A", "fold_specific_parameters": p04a_params["gradient_boosting"],
            "interface_pass": interface_results["phase04a"],
        },
        "traditional_regression": {
            "model": "Gradient Boosting Regressor", "source_phase": "04B",
            "fold_specific_parameters": {fold: value["best_params"] for fold, value in regression_params.items()},
            "interface_pass": interface_results["phase04b"], "task_interpretation": "bounded difficulty-induced workload proxy regression",
        },
        "hdc_classification": {**p06_classification, "interface_pass": interface_results["phase06"]},
        "hdc_regression": {**p06_regression, "interface_pass": interface_results["phase06"]},
        "policy": "No model is instantiated, trained, tuned, or reselected during initialization.",
        "sources": [source_record(path, role) for role, path in UPSTREAM.items() if role.startswith(("phase04a", "phase04b", "phase06"))],
    }
    write_json("configs/phase09_selected_model_interfaces.json", model_interfaces)
    write_json("configs/phase09_missing_modality_plan.json", {
        "phase": "09", "status": "PENDING_CONTRACT_FREEZE", "reference": "Full Primary",
        "conditions": [{"condition": "full_primary", "feature_count": 1176, "removed_modality": None}] + [
            {"condition": item["missing_condition"], "removed_modality": item["name"], "removed_feature_count": item["feature_count"], "feature_count": item["remaining_feature_count"]}
            for item in normalized_modalities
        ],
        "protocols_requiring_separate_contracts": {
            "retrain_without_modality": "Remove the frozen modality before fold-local preprocessing, fit a new model using the upstream-selected model family and frozen configuration policy, then test on the same feature subset.",
            "sudden_test_time_missingness": "Train the full-input reference model and define a contract-frozen neutralization/masking rule only at test time; do not label this protocol as retraining without a modality.",
        },
        "hdc_feasibility": "Feature identities are deterministically keyed by feature name and the encoder iterates over the supplied selected features; omission and neutralization are computationally possible, but the exact missingness protocol is not authorized until Contract Freeze.",
        "traditional_feasibility": "Frozen pipelines accept reduced feature matrices and all missing-modality conditions retain at least 200 features; test-time neutralization must use training-fold preprocessing state and be specified at Contract Freeze.",
        "training_executed": False, "predictions_generated": False,
        "sources": [source_record(MODALITY_SOURCE, "phase07_frozen_modality_membership"), source_record(PRIMARY, "primary_without_performance")],
    })
    write_json("configs/phase09_loso_plan.json", {
        "phase": "09", "status": "PENDING_CONTRACT_FREEZE", "protocol": "Leave-One-Subject-Out",
        "split_count": 35, "training_subjects_per_split": 34, "test_subjects_per_split": 1,
        "relationship_to_outer_cv": "New supplementary robustness protocol; does not replace or modify the frozen Phase 03 five-fold outer CV.",
        "selection_policy": "LOSO outputs may not tune hyperparameters or reselect models; all model interfaces come from upstream freezes.",
        "conditional_fit_inventory": {
            "traditional_models": 70,
            "hdc_models_across_five_frozen_evaluation_seeds": 350,
            "total_if_all_four_selected_interfaces_are_later_authorized": 420,
            "timing_and_memory": "NOT_MEASURED; execute in checkpointed batches only after Contract Freeze.",
        },
        "predictions_generated": False, "sources": common_sources,
    })
    write_json("configs/phase09_statistical_plan.json", {
        "phase": "09", "status": "PLAN_ONLY_PENDING_CONTRACT_FREEZE", "statistical_unit": "subject_id",
        "subject_level_metrics": ["classification Macro-F1", "Balanced Accuracy", "Severe Error Rate", "bounded MAE", "bounded RMSE"],
        "planned_analyses": ["between-subject performance distribution", "worst-subject diagnosis", "subject-ranking stability", "subject-level bootstrap confidence intervals"],
        "formal_analysis_executed": False, "regression_wording": "bounded difficulty-induced workload proxy regression",
        "sources": common_sources,
    })
    write_json("configs/phase09_experiment_contract_draft.json", {
        "phase": "09", "phase_name": "Robustness and Generalization", "status": "PENDING_CONTRACT_FREEZE",
        "primary_input_only": str(PRIMARY.resolve()), "performance_features_in_primary_analysis": False,
        "workstreams": ["missing-modality robustness", "selected-model LOSO", "subject-level stability"],
        "frozen_outer_cv_preserved": True, "upstream_models_reselected": False, "phase08_retuning": False,
        "training_authorized": False, "prediction_generation_authorized": False, "formal_statistics_authorized": False,
        "next_action": "Freeze the Phase 09 experiment contract in a separate step before modeling.",
        "sources": common_sources + [source_record(PROJECT_ROOT / "最新完整实验计划_分类回归双任务.md", "authoritative_experiment_plan")],
    })

    write_json("manifests/phase09_input_manifest.json", {
        "phase": "09", "status": input_status,
        "sources": common_sources + [source_record(PRIMARY_FEATURES, "phase03_primary_feature_manifest"), source_record(FEATURE_GROUPS, "phase03_feature_group_manifest")],
        "rows": len(primary), "subjects": int(primary.subject_id.nunique()), "primary_features": len(feature_names), "unique_run_keys": int(primary.run_key.nunique()),
    })
    upstream_records = [source_record(path, role) for role, path in UPSTREAM.items()] + [source_record(path, f"phase04b_gradient_boosting_fold_{fold}_params") for fold, path in REGRESSION_FOLD_PARAMS.items()]
    write_json("manifests/phase09_upstream_freeze_manifest.json", {
        "phase": "09", "status": upstream_status, "sources": upstream_records,
        "freeze_interface_results": interface_results, "upstream_mutation_allowed": False,
    })
    write_json("manifests/phase09_modality_manifest.json", {
        "phase": "09", "status": modality_status,
        "sources": [source_record(MODALITY_SOURCE, "phase07_frozen_modality_feature_manifest"), source_record(PRIMARY_FEATURES, "phase03_primary_feature_manifest"), source_record(PRIMARY, "primary_without_performance")],
        "primary_feature_count": len(feature_names), "modality_feature_union_count": len(modality_union), "modalities": normalized_modalities,
        "overlapping_features": overlaps, "primary_features_absent_from_modalities": absent_features, "nonprimary_features_in_modalities": extra_features,
    })
    write_json("manifests/phase09_loso_feasibility_manifest.json", {
        "phase": "09", "status": loso_status, "sources": common_sources,
        "split_generation": "lexicographically sorted subject_id; each subject held out exactly once", "splits": loso_splits,
    })

    write_json("audits/phase09_input_and_fold_audit.json", {
        "phase": "09", "audit": "input_and_fold", "status": input_status,
        "actual": {"rows": len(primary), "subjects": int(primary.subject_id.nunique()), "primary_features": len(feature_names), "unique_run_keys": int(primary.run_key.nunique()), "target_class_values": sorted(int(v) for v in primary.target_class.unique()), "target_score_values": sorted(float(v) for v in primary.target_score.unique()), "outer_folds": len(outer_folds), "primary_sha256": primary_sha, "fold_sha256": fold_sha},
        "expected": {"rows": 419, "subjects": 35, "primary_features": 1176, "unique_run_keys": 419, "target_class_values": [0, 1, 2, 3], "target_score_values": [1.0, 2.0, 3.0, 4.0], "outer_folds": 5, "primary_sha256": EXPECTED_PRIMARY_SHA256, "fold_sha256": EXPECTED_FOLD_SHA256},
        "checks": input_checks, "fold_details": fold_details, "sources": common_sources,
    })
    write_json("audits/phase09_upstream_freeze_audit.json", {
        "phase": "09", "audit": "upstream_freeze_interfaces", "status": upstream_status,
        "checks": interface_checks, "interface_results": interface_results,
        "actual_freeze_sha256": actual_freeze_hashes, "expected_freeze_sha256": EXPECTED_UPSTREAM_FREEZE_SHA256,
        "sources": upstream_records, "model_instantiation_or_training": False,
    })
    write_json("audits/phase09_modality_coverage_audit.json", {
        "phase": "09", "audit": "modality_coverage_and_missingness_feasibility", "status": modality_status,
        "checks": modality_checks, "modality_counts": {item["name"]: item["feature_count"] for item in normalized_modalities},
        "remaining_feature_counts": {item["missing_condition"]: item["remaining_feature_count"] for item in normalized_modalities},
        "union_count": len(modality_union), "overlap_count": len(overlaps), "missing_from_union_count": len(absent_features), "extra_in_union_count": len(extra_features),
        "protocol_separation_required": True, "training_or_prediction_executed": False,
        "sources": [source_record(MODALITY_SOURCE, "phase07_frozen_modality_feature_manifest"), source_record(PRIMARY_FEATURES, "phase03_primary_feature_manifest")],
    })
    write_json("audits/phase09_loso_feasibility_audit.json", {
        "phase": "09", "audit": "loso_feasibility", "status": loso_status, "checks": loso_checks,
        "split_count": len(loso_splits), "subject_run_counts": {str(k): int(v) for k, v in primary.groupby("subject_id").size().sort_index().items()},
        "subject_target_coverage": {str(subject): sorted(int(v) for v in group.target_class.unique()) for subject, group in primary.groupby("subject_id")},
        "empty_training_sets": sum(item["train_rows"] == 0 for item in loso_splits), "empty_test_sets": sum(item["test_rows"] == 0 for item in loso_splits),
        "predictions_generated": False, "sources": common_sources,
    })
    write_json("audits/phase09_generalization_scope_audit.json", {
        "phase": "09", "audit": "generalization_scope", "status": scope_status, "checks": scope_checks,
        "handoff_holdout_feasibility": scope_actual,
        "subject_generalization": "FEASIBLE_VIA_LOSO", "missing_modality_robustness": "FEASIBLE_PENDING_CONTRACT",
        "flight_generalizable_behavior_claim": "INCONCLUSIVE_DUE_TO_METADATA",
        "sources": [source_record(UPSTREAM["phase08_handoff_config"], "phase08_generalization_handoff"), source_record(PRIMARY, "primary_metadata_audit")],
    })

    upstream_hash_after = {str(path): sha256(path) for path in required_inputs}
    upstream_unchanged = upstream_hash_before == upstream_hash_after
    core_ready = all([input_status == "PASS", upstream_status == "PASS", modality_status == "PASS", loso_status == "PASS", scope_status == "PASS", upstream_unchanged])
    summary = {
        "phase": "09", "phase_name": "Robustness and Generalization",
        "status": "PENDING_CONTRACT_FREEZE" if core_ready else "INITIALIZATION_AUDIT_FAILED",
        "input_status": input_status, "upstream_status": upstream_status, "modality_status": modality_status,
        "loso_status": loso_status, "generalization_scope_status": scope_status, "upstream_sources_unchanged": upstream_unchanged,
        "ready_for_contract_freeze_pre_notebook": core_ready, "ready_for_modeling": False,
        "model_training_executed": False, "loso_predictions_generated": False, "missing_modality_predictions_generated": False,
    }
    write_json("audits/phase09_initialization_summary.json", summary)
    return summary


def finalize_artifact_audit() -> dict[str, Any]:
    missing_directories = [path for path in REQUIRED_DIRECTORIES if not (PHASE09_ROOT / path).is_dir()]
    self_artifact = "audits/phase09_initialization_artifact_audit.json"
    missing_artifacts = [path for path in REQUIRED_ARTIFACTS if path != self_artifact and not (PHASE09_ROOT / path).is_file()]
    persistence_path = PHASE09_ROOT / "audits" / "phase09_notebook_persistence_audit.json"
    persistence_pass = persistence_path.exists() and read_json(persistence_path).get("status") == "PASS"
    summary_path = PHASE09_ROOT / "audits" / "phase09_initialization_summary.json"
    summary = read_json(summary_path)
    pass_status = not missing_directories and not missing_artifacts and persistence_pass and summary["ready_for_contract_freeze_pre_notebook"]
    audit = {
        "phase": "09", "audit": "initialization_artifacts", "status": "PASS" if pass_status else "FAIL",
        "required_directory_count": len(REQUIRED_DIRECTORIES), "required_artifact_count": len(REQUIRED_ARTIFACTS),
        "missing_directories": missing_directories, "missing_artifacts": missing_artifacts,
        "notebook_persistence": "PASS" if persistence_pass else "FAIL",
        "model_training_executed": False, "loso_predictions_generated": False, "missing_modality_predictions_generated": False,
        "phase09_status": "PENDING_CONTRACT_FREEZE" if pass_status else "INITIALIZATION_AUDIT_FAILED",
        "ready_for_contract_freeze": pass_status, "ready_for_modeling": False,
    }
    write_json("audits/phase09_initialization_artifact_audit.json", audit)
    summary.update({"status": audit["phase09_status"], "ready_for_contract_freeze": pass_status, "notebook_persistence": audit["notebook_persistence"]})
    write_json("audits/phase09_initialization_summary.json", summary)
    return audit


if __name__ == "__main__":
    print(json.dumps(run_initialization(), ensure_ascii=False, indent=2))
