from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


EXPECTED_HASHES = {
    "primary_data": "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44",
    "frozen_folds": "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f",
    "phase06_freeze": "cd94acdf0bcb463b5f48a7c84110a23059899b6b5bf156131694827d7d7bde66",
    "phase06_classification": "174a99de2d993acdea49fdebc9647b28db4648ada2bea7a33f620f4677f031a4",
    "phase06_regression": "acde51709971d57c76eefaffcf1ecd571a4d4c5c36f8d76edf39841c5e7065b8",
}

MODALITIES = [
    "physiological_features",
    "eye_tracking_features",
    "head_movement_features",
    "flight_parameter_features",
    "body_movement",
]

EXPECTED_MODALITY_COUNTS = {
    "physiological_features": 233,
    "eye_tracking_features": 416,
    "head_movement_features": 159,
    "flight_parameter_features": 326,
    "body_movement": 42,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def project_paths() -> dict[str, Path]:
    phase_dir = Path(__file__).resolve().parents[1]
    project_root = phase_dir.parents[1]
    if not (project_root / "CODEX_NOTEBOOK_RULES.md").is_file():
        raise RuntimeError(f"Project root validation failed: {project_root}")
    p3 = project_root / "experiments" / "phase_03_multimodal_dataset_labeling"
    p6 = project_root / "experiments" / "phase_06_hdc_variant_screening"
    return {
        "project_root": project_root,
        "phase_dir": phase_dir,
        "plan": project_root / "最新完整实验计划_分类回归双任务.md",
        "primary_data": p3 / "data" / "primary_without_performance.csv",
        "frozen_folds": p3 / "data" / "fold_assignments.csv",
        "feature_groups": p3 / "manifests" / "feature_group_manifest.json",
        "primary_features": p3 / "manifests" / "primary_feature_manifest.json",
        "phase06_freeze": p6 / "configs" / "phase06_freeze.json",
        "phase06_classification": p6 / "configs" / "phase06_best_classification_hdc.json",
        "phase06_regression": p6 / "configs" / "phase06_best_regression_hdc.json",
        "phase06_artifact_manifest": p6 / "manifests" / "phase06_final_artifact_manifest.json",
        "phase06_selection_audit": p6 / "audits" / "phase06_model_selection_resolution_audit.json",
        "notebook": phase_dir / "Phase_07_Unimodal_Contribution.ipynb",
    }


def distribution(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def collect_audit() -> dict[str, Any]:
    paths = project_paths()
    upstream_names = [
        "plan",
        "primary_data",
        "frozen_folds",
        "feature_groups",
        "primary_features",
        "phase06_freeze",
        "phase06_classification",
        "phase06_regression",
        "phase06_artifact_manifest",
        "phase06_selection_audit",
    ]
    existence = {name: paths[name].is_file() for name in upstream_names}
    if not all(existence.values()):
        missing = [name for name, present in existence.items() if not present]
        raise FileNotFoundError(f"Missing required upstream files: {missing}")

    hashes = {name: sha256(paths[name]) for name in upstream_names}
    data = pd.read_csv(paths["primary_data"], low_memory=False)
    folds = pd.read_csv(paths["frozen_folds"], low_memory=False)
    feature_group_manifest = load_json(paths["feature_groups"])
    primary_manifest = load_json(paths["primary_features"])
    freeze = load_json(paths["phase06_freeze"])
    classification = load_json(paths["phase06_classification"])
    regression = load_json(paths["phase06_regression"])
    selection_audit = load_json(paths["phase06_selection_audit"])

    primary_features = list(primary_manifest["features"])
    primary_set = set(primary_features)
    source_groups = feature_group_manifest["source_corrected_feature_groups"]
    modality_features = {
        modality: [feature for feature in primary_features if feature in set(source_groups[modality])]
        for modality in MODALITIES
    }

    flattened = [feature for modality in MODALITIES for feature in modality_features[modality]]
    duplicate_memberships = sorted({feature for feature in flattened if flattened.count(feature) > 1})
    union = set(flattened)
    performance_intersection = sorted(primary_set.intersection(source_groups["performance_features"]))
    feature_columns_present = all(feature in data.columns for feature in primary_features)

    merged = data[["run_key", "subject_id", "outer_fold"]].merge(
        folds[["run_key", "subject_id", "outer_fold"]],
        on="run_key",
        how="outer",
        suffixes=("_data", "_fold"),
        indicator=True,
        validate="one_to_one",
    )
    run_key_alignment = bool(
        (merged["_merge"] == "both").all()
        and (merged["subject_id_data"] == merged["subject_id_fold"]).all()
        and (merged["outer_fold_data"] == merged["outer_fold_fold"]).all()
    )

    outer_folds = sorted(int(value) for value in folds["outer_fold"].unique())
    outer_audits: list[dict[str, Any]] = []
    inner_audits: list[dict[str, Any]] = []
    for outer_fold in outer_folds:
        test_rows = folds[folds["outer_fold"] == outer_fold]
        train_rows = folds[folds["outer_fold"] != outer_fold]
        train_subjects = set(train_rows["subject_id"])
        test_subjects = set(test_rows["subject_id"])
        overlap = sorted(train_subjects.intersection(test_subjects))
        outer_audits.append(
            {
                "outer_fold": outer_fold,
                "train_rows": int(len(train_rows)),
                "test_rows": int(len(test_rows)),
                "train_subjects": int(len(train_subjects)),
                "test_subjects": int(len(test_subjects)),
                "subject_overlap": overlap,
                "subject_isolation_pass": not overlap,
            }
        )
        groups = train_rows["subject_id"].to_numpy()
        splitter = GroupKFold(n_splits=3)
        split_checks: list[bool] = []
        split_count = 0
        for inner_train, inner_valid in splitter.split(np.zeros(len(train_rows)), groups=groups):
            split_count += 1
            split_checks.append(not set(groups[inner_train]).intersection(groups[inner_valid]))
        inner_audits.append(
            {
                "outer_fold": outer_fold,
                "outer_training_subjects": int(len(set(groups))),
                "inner_splits": split_count,
                "all_inner_subject_isolation_pass": all(split_checks),
                "groupkfold_3_feasible": split_count == 3 and all(split_checks),
            }
        )

    modality_audits: list[dict[str, Any]] = []
    for modality in MODALITIES:
        features = modality_features[modality]
        fully_missing = data[features].isna().all(axis=1)
        available = ~fully_missing
        fold_details = []
        for outer_fold in outer_folds:
            test_mask = data["outer_fold"] == outer_fold
            train_mask = ~test_mask
            fold_details.append(
                {
                    "outer_fold": outer_fold,
                    "fully_missing_rows_all": int((fully_missing & test_mask).sum()),
                    "fully_missing_rows_train": int((fully_missing & train_mask).sum()),
                    "fully_missing_rows_test": int((fully_missing & test_mask).sum()),
                    "train_has_available_modality_data": bool((available & train_mask).any()),
                    "test_has_available_modality_data": bool((available & test_mask).any()),
                }
            )
        modality_audits.append(
            {
                "modality": modality,
                "feature_count": len(features),
                "fully_missing_rows": int(fully_missing.sum()),
                "subjects_with_available_data": int(data.loc[available, "subject_id"].nunique()),
                "target_class_distribution_all_rows": distribution(data["target_class"]),
                "target_class_distribution_available_rows": distribution(data.loc[available, "target_class"]),
                "target_score_distribution_all_rows": distribution(data["target_score"]),
                "target_score_distribution_available_rows": distribution(data.loc[available, "target_score"]),
                "outer_fold_details": fold_details,
            }
        )

    frozen_seeds = sorted(
        {
            int(item["seed"])
            for fold_policy in regression["fold_parameter_policy"]
            for item in json.loads(fold_policy["parameter_policy_json"])
        }
    )
    phase06_checks = {
        "freeze_checksum_pass": hashes["phase06_freeze"] == EXPECTED_HASHES["phase06_freeze"],
        "classification_checksum_pass": hashes["phase06_classification"] == EXPECTED_HASHES["phase06_classification"],
        "regression_checksum_pass": hashes["phase06_regression"] == EXPECTED_HASHES["phase06_regression"],
        "status_frozen": freeze.get("status") == "FROZEN",
        "ready_for_next_planned_phase": freeze.get("ready_for_next_planned_phase") is True,
        "classification_selection_evidence_inner_only": classification.get("selection_evidence") == "INNER_CV_ONLY",
        "regression_selection_evidence_inner_only": regression.get("selection_evidence") == "INNER_CV_ONLY",
        "outer_oof_read_for_selection": not bool(
            selection_audit.get("gates", {}).get("selector_inner_cv_and_unlabeled_efficiency_only")
        ),
        "single_seed_selected": bool(
            classification.get("single_seed_selected") or regression.get("single_seed_selected")
        ),
        "frozen_evaluation_seeds": frozen_seeds,
        "frozen_evaluation_seeds_unchanged": frozen_seeds == [42, 43, 44, 45, 46],
        "classification_interface_pass": bool(
            classification.get("selected_variant") == "hybrid"
            and classification.get("selected_variant_name") == "HDC+OnlineHD Hybrid"
            and classification.get("selected_fixed_dimension") == 5000
            and classification.get("levels") == 51
            and classification.get("feature_k") == 50
            and "fold-local inner-CV" in classification.get("structure_selection_policy", "")
        ),
        "regression_interface_pass": bool(
            regression.get("selected_variant") == "common_ridge"
            and regression.get("selected_regression_head") == "COMMON_ENCODER_READOUT_BASELINE"
            and regression.get("selected_fixed_dimension") == 10000
            and regression.get("levels") == 51
            and regression.get("feature_k") == 50
            and "fold-local inner-CV" in regression.get("parameter_policy", "")
        ),
        "model_selection_resolution_audit_pass": selection_audit.get("result") == "PASS",
    }

    data_checks = {
        "modeling_rows": int(len(data)),
        "subjects": int(data["subject_id"].nunique()),
        "unique_run_key": int(data["run_key"].nunique()),
        "primary_predictive_features": len(primary_features),
        "feature_columns_present": feature_columns_present,
        "target_class_values": sorted(int(value) for value in data["target_class"].dropna().unique()),
        "target_class_missing": int(data["target_class"].isna().sum()),
        "target_score_values": sorted(float(value) for value in data["target_score"].dropna().unique()),
        "target_score_missing": int(data["target_score"].isna().sum()),
        "fold_assignment_rows": int(len(folds)),
        "fold_assignment_unique_run_key": int(folds["run_key"].nunique()),
        "run_key_one_to_one_alignment": run_key_alignment,
        "outer_folds": outer_folds,
        "outer_subject_isolation_pass": all(item["subject_isolation_pass"] for item in outer_audits),
        "inner_groupkfold_3_feasibility_pass": all(item["groupkfold_3_feasible"] for item in inner_audits),
        "primary_data_checksum_pass": hashes["primary_data"] == EXPECTED_HASHES["primary_data"],
        "frozen_fold_checksum_pass": hashes["frozen_folds"] == EXPECTED_HASHES["frozen_folds"],
    }
    expected_data = {
        "modeling_rows": 419,
        "subjects": 35,
        "unique_run_key": 419,
        "primary_predictive_features": 1176,
        "target_class_values": [0, 1, 2, 3],
        "target_class_missing": 0,
        "target_score_values": [1.0, 2.0, 3.0, 4.0],
        "target_score_missing": 0,
        "fold_assignment_rows": 419,
        "fold_assignment_unique_run_key": 419,
        "outer_folds": [1, 2, 3, 4, 5],
    }
    data_checks["all_expected_values_pass"] = all(data_checks[key] == value for key, value in expected_data.items())

    modality_checks = {
        "feature_counts": {key: len(value) for key, value in modality_features.items()},
        "feature_union_count": len(union),
        "disjointness_pass": not duplicate_memberships,
        "duplicate_memberships": duplicate_memberships,
        "union_coverage_pass": union == primary_set,
        "missing_from_union": sorted(primary_set - union),
        "outside_primary_union": sorted(union - primary_set),
        "performance_primary_intersection_count": len(performance_intersection),
        "performance_primary_intersection": performance_intersection,
        "control_input_feature_count": len(source_groups["control_input_features"]),
        "unverified_feature_count": len(source_groups["unverified_features"]),
        "body_movement_status": primary_manifest.get("body_movement_status"),
        "body_movement_verified": primary_manifest.get("body_movement_status") == "VERIFIED_BODY_MOVEMENT:42",
    }
    modality_checks["expected_counts_pass"] = modality_checks["feature_counts"] == EXPECTED_MODALITY_COUNTS

    all_gates = [
        all(existence.values()),
        data_checks["all_expected_values_pass"],
        data_checks["feature_columns_present"],
        data_checks["run_key_one_to_one_alignment"],
        data_checks["outer_subject_isolation_pass"],
        data_checks["inner_groupkfold_3_feasibility_pass"],
        data_checks["primary_data_checksum_pass"],
        data_checks["frozen_fold_checksum_pass"],
        modality_checks["expected_counts_pass"],
        modality_checks["disjointness_pass"],
        modality_checks["union_coverage_pass"],
        modality_checks["performance_primary_intersection_count"] == 0,
        modality_checks["control_input_feature_count"] == 0,
        modality_checks["unverified_feature_count"] == 0,
        modality_checks["body_movement_verified"],
        phase06_checks["freeze_checksum_pass"],
        phase06_checks["classification_checksum_pass"],
        phase06_checks["regression_checksum_pass"],
        phase06_checks["status_frozen"],
        phase06_checks["ready_for_next_planned_phase"],
        phase06_checks["classification_selection_evidence_inner_only"],
        phase06_checks["regression_selection_evidence_inner_only"],
        not phase06_checks["outer_oof_read_for_selection"],
        not phase06_checks["single_seed_selected"],
        phase06_checks["frozen_evaluation_seeds_unchanged"],
        phase06_checks["classification_interface_pass"],
        phase06_checks["regression_interface_pass"],
        phase06_checks["model_selection_resolution_audit_pass"],
    ]
    return {
        "timestamp_utc": utc_now(),
        "paths": {key: str(value) for key, value in paths.items()},
        "upstream_existence": existence,
        "hashes": hashes,
        "data_checks": data_checks,
        "outer_fold_audits": outer_audits,
        "inner_groupkfold_audits": inner_audits,
        "modality_features": modality_features,
        "modality_checks": modality_checks,
        "modality_audits": modality_audits,
        "phase06_freeze": freeze,
        "phase06_classification": classification,
        "phase06_regression": regression,
        "phase06_checks": phase06_checks,
        "initialization_gates_without_notebook_persistence_pass": all(all_gates),
        "hdc_training_executed": False,
    }


def persist_initialization(audit: dict[str, Any]) -> None:
    phase_dir = Path(audit["paths"]["phase_dir"])
    paths = {key: Path(value) for key, value in audit["paths"].items()}
    upstream_roles = {
        "plan": "authoritative_experiment_plan",
        "primary_data": "primary_modeling_table_read_only",
        "frozen_folds": "frozen_outer_fold_assignments_read_only",
        "feature_groups": "frozen_feature_group_membership_read_only",
        "primary_features": "frozen_primary_feature_universe_read_only",
        "phase06_freeze": "phase06_freeze_interface_read_only",
        "phase06_classification": "frozen_classification_selection_read_only",
        "phase06_regression": "frozen_regression_selection_read_only",
        "phase06_artifact_manifest": "phase06_final_artifact_manifest_read_only",
        "phase06_selection_audit": "phase06_inner_only_selection_evidence_read_only",
    }
    input_manifest = {
        "phase": "07",
        "status": "PENDING_CONTRACT_FREEZE",
        "generated_at_utc": audit["timestamp_utc"],
        "source_policy": "REFERENCE_IN_PLACE_READ_ONLY_NO_COPIES",
        "inputs": [
            {
                "name": name,
                "role": role,
                "absolute_path": str(paths[name]),
                "sha256": audit["hashes"][name],
                "exists": audit["upstream_existence"][name],
            }
            for name, role in upstream_roles.items()
        ],
    }
    modality_manifest = {
        "phase": "07",
        "status": "DERIVED_READ_ONLY_MANIFEST",
        "generated_at_utc": audit["timestamp_utc"],
        "derivation": "Primary feature universe intersected with frozen Phase 03 feature-group membership; no prefix inference",
        "primary_feature_manifest_path": str(paths["primary_features"]),
        "feature_group_manifest_path": str(paths["feature_groups"]),
        "primary_feature_count": audit["data_checks"]["primary_predictive_features"],
        "modalities": [
            {
                "name": name,
                "feature_count": len(audit["modality_features"][name]),
                "status": "VERIFIED_BODY_MOVEMENT" if name == "body_movement" else "FROZEN_FEATURE_GROUP_MEMBERSHIP",
                "features": audit["modality_features"][name],
            }
            for name in MODALITIES
        ],
        "control_input": {"available": False, "feature_count": 0, "experimental_modality": False},
        "performance_features": {"included": False, "primary_intersection_count": 0},
        "unverified_features": {"included": False, "feature_count": 0},
        "checks": audit["modality_checks"],
    }
    experiment_contract = {
        "phase": "07",
        "phase_name": "Unimodal Contribution Analysis",
        "status": "PENDING_CONTRACT_FREEZE",
        "scope": "INITIALIZATION_ONLY_NO_MODELING",
        "regression_target_description": "bounded difficulty-induced workload proxy regression",
        "planned_experiments_not_executed": {
            "modalities": MODALITIES,
            "classification": {
                "interface": "frozen Phase 06 HDC+OnlineHD Hybrid",
                "primary_metric": "Macro-F1",
            },
            "regression": {
                "interface": "frozen Phase 06 Common Encoder Ridge readout",
                "primary_metric": "MAE",
            },
            "outer_validation": "reuse original five frozen outer folds",
            "fold_local_only": "all preprocessing, feature selection, and parameter handling",
            "future_outputs": [
                "classification OOF per modality",
                "regression OOF per modality",
                "modality contribution ranking",
                "per-modality error analysis",
            ],
            "multimodal_frozen_result": "reference only; no retraining or reselection",
        },
        "intentionally_unfrozen_until_contract_freeze": [
            "final handling rule for rows with an entirely missing modality",
            "formal rule when feature_k exceeds usable modality features",
            "Phase 07 seed aggregation rule",
            "statistical inference rule for unimodal-versus-multimodal deltas",
            "tie-breaking rule",
            "final modality ranking rule",
        ],
        "strictly_prohibited_in_initialization": [
            "model training",
            "quick screening",
            "hyperparameter tuning",
            "outer-test prediction",
            "OOF generation",
            "HDC hypervector generation",
            "row deletion",
            "global imputation or scaling",
            "performance or control-input experiments",
            "multimodal retraining",
        ],
    }
    environment = {
        "phase": "07",
        "captured_at_utc": audit["timestamp_utc"],
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "project_root": str(paths["project_root"]),
        "phase_directory": str(phase_dir),
        "training_executed": False,
    }
    upstream_interface = {
        "phase": "07",
        "captured_at_utc": audit["timestamp_utc"],
        "phase03": {
            "primary_data_path": str(paths["primary_data"]),
            "frozen_folds_path": str(paths["frozen_folds"]),
            "feature_group_manifest_path": str(paths["feature_groups"]),
            "primary_feature_manifest_path": str(paths["primary_features"]),
            "read_only": True,
            "data_checks": audit["data_checks"],
        },
        "phase06": {
            "freeze_path": str(paths["phase06_freeze"]),
            "classification_path": str(paths["phase06_classification"]),
            "regression_path": str(paths["phase06_regression"]),
            "read_only": True,
            "checks": audit["phase06_checks"],
            "classification_interface": audit["phase06_classification"],
            "regression_interface": audit["phase06_regression"],
        },
    }
    input_fold_audit = {
        "phase": "07",
        "audit": "input_and_fold",
        "generated_at_utc": audit["timestamp_utc"],
        "checks": audit["data_checks"],
        "outer_folds": audit["outer_fold_audits"],
        "inner_groupkfold": audit["inner_groupkfold_audits"],
        "result": "PASS" if all([
            audit["data_checks"]["all_expected_values_pass"],
            audit["data_checks"]["run_key_one_to_one_alignment"],
            audit["data_checks"]["outer_subject_isolation_pass"],
            audit["data_checks"]["inner_groupkfold_3_feasibility_pass"],
            audit["data_checks"]["primary_data_checksum_pass"],
            audit["data_checks"]["frozen_fold_checksum_pass"],
        ]) else "FAIL",
    }
    modality_audit = {
        "phase": "07",
        "audit": "modality_mapping_and_missingness",
        "generated_at_utc": audit["timestamp_utc"],
        "mapping_checks": audit["modality_checks"],
        "modality_audits": audit["modality_audits"],
        "row_policy": "AUDIT_ONLY_NO_ROWS_REMOVED_NO_GLOBAL_IMPUTATION_NO_NAN_TO_ZERO",
        "result": "PASS" if all([
            audit["modality_checks"]["expected_counts_pass"],
            audit["modality_checks"]["disjointness_pass"],
            audit["modality_checks"]["union_coverage_pass"],
            audit["modality_checks"]["performance_primary_intersection_count"] == 0,
            audit["modality_checks"]["control_input_feature_count"] == 0,
            audit["modality_checks"]["unverified_feature_count"] == 0,
            audit["modality_checks"]["body_movement_verified"],
        ]) else "FAIL",
    }
    phase06_audit = {
        "phase": "07",
        "audit": "phase06_freeze_interface",
        "generated_at_utc": audit["timestamp_utc"],
        "checks": audit["phase06_checks"],
        "classification": {
            "selected_variant": audit["phase06_classification"]["selected_variant"],
            "selected_variant_name": audit["phase06_classification"]["selected_variant_name"],
            "selected_fixed_dimension": audit["phase06_classification"]["selected_fixed_dimension"],
            "levels": audit["phase06_classification"]["levels"],
            "feature_k": audit["phase06_classification"]["feature_k"],
            "structure_selection_policy": audit["phase06_classification"]["structure_selection_policy"],
            "reselection_permitted": False,
        },
        "regression": {
            "selected_variant": audit["phase06_regression"]["selected_variant"],
            "selected_regression_head": audit["phase06_regression"]["selected_regression_head"],
            "selected_fixed_dimension": audit["phase06_regression"]["selected_fixed_dimension"],
            "levels": audit["phase06_regression"]["levels"],
            "feature_k": audit["phase06_regression"]["feature_k"],
            "parameter_policy": audit["phase06_regression"]["parameter_policy"],
            "target_description": "bounded difficulty-induced workload proxy regression",
            "reselection_permitted": False,
        },
        "result": "PASS" if all([
            audit["phase06_checks"]["freeze_checksum_pass"],
            audit["phase06_checks"]["classification_checksum_pass"],
            audit["phase06_checks"]["regression_checksum_pass"],
            audit["phase06_checks"]["status_frozen"],
            audit["phase06_checks"]["ready_for_next_planned_phase"],
            not audit["phase06_checks"]["outer_oof_read_for_selection"],
            not audit["phase06_checks"]["single_seed_selected"],
            audit["phase06_checks"]["frozen_evaluation_seeds_unchanged"],
            audit["phase06_checks"]["classification_interface_pass"],
            audit["phase06_checks"]["regression_interface_pass"],
        ]) else "FAIL",
    }

    write_json(phase_dir / "configs" / "phase07_experiment_contract.json", experiment_contract)
    write_json(phase_dir / "configs" / "phase07_environment.json", environment)
    write_json(phase_dir / "configs" / "phase07_upstream_interface.json", upstream_interface)
    write_json(phase_dir / "manifests" / "phase07_input_manifest.json", input_manifest)
    write_json(phase_dir / "manifests" / "phase07_modality_feature_manifest.json", modality_manifest)
    write_json(phase_dir / "audits" / "phase07_input_and_fold_audit.json", input_fold_audit)
    write_json(phase_dir / "audits" / "phase07_modality_mapping_audit.json", modality_audit)
    write_json(phase_dir / "audits" / "phase07_phase06_freeze_interface_audit.json", phase06_audit)

    required_artifacts = [
        "README.md",
        "task_plan.md",
        "notes.md",
        "Phase_07_Unimodal_Contribution.ipynb",
        "configs/phase07_experiment_contract.json",
        "configs/phase07_environment.json",
        "configs/phase07_upstream_interface.json",
        "manifests/phase07_input_manifest.json",
        "manifests/phase07_modality_feature_manifest.json",
        "audits/phase07_input_and_fold_audit.json",
        "audits/phase07_modality_mapping_audit.json",
        "audits/phase07_phase06_freeze_interface_audit.json",
    ]
    inventory = []
    for relative in required_artifacts:
        path = phase_dir / relative
        parseable = None
        if path.suffix == ".json" and path.is_file():
            try:
                load_json(path)
                parseable = True
            except (OSError, json.JSONDecodeError):
                parseable = False
        inventory.append(
            {
                "relative_path": relative,
                "exists": path.is_file(),
                "json_parseable": parseable,
                "sha256": sha256(path) if path.is_file() else None,
            }
        )
    artifact_pass = all(item["exists"] and item["json_parseable"] is not False for item in inventory)
    write_json(
        phase_dir / "audits" / "phase07_initialization_artifact_audit.json",
        {
            "phase": "07",
            "audit": "initialization_artifacts",
            "generated_at_utc": utc_now(),
            "inventory": inventory,
            "upstream_files_modified": False,
            "upstream_data_copied_to_phase07": False,
            "hdc_training_executed": False,
            "result": "PASS" if artifact_pass else "FAIL",
        },
    )
    write_json(
        phase_dir / "audits" / "phase07_notebook_persistence_audit.json",
        {
            "phase": "07",
            "audit": "notebook_persistence",
            "generated_at_utc": utc_now(),
            "status": "PENDING_EXECUTION",
            "notebook_path": str(paths["notebook"]),
            "hdc_training_executed": False,
            "result": "PENDING",
        },
    )


def finalize_notebook_persistence() -> dict[str, Any]:
    import nbformat

    paths = project_paths()
    phase_dir = paths["phase_dir"]
    notebook_path = paths["notebook"]
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    error_outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    all_executed = bool(code_cells) and all(cell.get("execution_count") is not None for cell in code_cells)
    all_have_outputs = bool(code_cells) and all(bool(cell.get("outputs")) for cell in code_cells)
    required_json = [
        "configs/phase07_experiment_contract.json",
        "configs/phase07_environment.json",
        "configs/phase07_upstream_interface.json",
        "manifests/phase07_input_manifest.json",
        "manifests/phase07_modality_feature_manifest.json",
        "audits/phase07_input_and_fold_audit.json",
        "audits/phase07_modality_mapping_audit.json",
        "audits/phase07_phase06_freeze_interface_audit.json",
        "audits/phase07_initialization_artifact_audit.json",
        "audits/phase07_notebook_persistence_audit.json",
    ]
    parseability = {}
    for relative in required_json:
        try:
            load_json(phase_dir / relative)
            parseability[relative] = True
        except (OSError, json.JSONDecodeError):
            parseability[relative] = False
    result_pass = all_executed and all_have_outputs and not error_outputs and all(parseability.values())
    payload = {
        "phase": "07",
        "audit": "notebook_persistence",
        "generated_at_utc": utc_now(),
        "status": "EXECUTED_AND_SAVED" if result_pass else "FAIL",
        "notebook_path": str(notebook_path),
        "notebook_sha256": sha256(notebook_path),
        "code_cell_count": len(code_cells),
        "all_code_cells_executed": all_executed,
        "all_code_cells_have_outputs": all_have_outputs,
        "error_output_count": len(error_outputs),
        "required_json_parseability": parseability,
        "hdc_training_executed": False,
        "result": "PASS" if result_pass else "FAIL",
    }
    write_json(phase_dir / "audits" / "phase07_notebook_persistence_audit.json", payload)
    artifact_audit_path = phase_dir / "audits" / "phase07_initialization_artifact_audit.json"
    artifact_audit = load_json(artifact_audit_path)
    for item in artifact_audit["inventory"]:
        artifact_path = phase_dir / item["relative_path"]
        if artifact_path.is_file():
            item["sha256"] = sha256(artifact_path)
    artifact_audit["generated_at_utc"] = utc_now()
    artifact_audit["notebook_persistence_result"] = payload["result"]
    artifact_audit["result"] = "PASS" if artifact_audit["result"] == "PASS" and result_pass else "FAIL"
    write_json(artifact_audit_path, artifact_audit)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize or finalize Phase 07 without modeling.")
    parser.add_argument("--finalize-notebook", action="store_true")
    args = parser.parse_args()
    if args.finalize_notebook:
        print(json.dumps(finalize_notebook_persistence(), indent=2))
        return
    audit = collect_audit()
    persist_initialization(audit)
    print(
        json.dumps(
            {
                "initialization_gates_without_notebook_persistence_pass": audit[
                    "initialization_gates_without_notebook_persistence_pass"
                ],
                "data_checks": audit["data_checks"],
                "modality_checks": audit["modality_checks"],
                "phase06_checks": audit["phase06_checks"],
                "hdc_training_executed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
