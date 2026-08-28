"""Initialize and audit Phase 08 without executing any model-related operation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE08_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PHASE08_ROOT.parent
PHASE03_ROOT = EXPERIMENTS_ROOT / "phase_03_multimodal_dataset_labeling"

EXPECTED_HASHES = {
    "primary": "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44",
    "with_performance": "72977a2119e30e37996fb9f0e3404988c4977fb7d2b33992f87bf54bfe5decba",
    "performance_only": "d602282ae41153886d1306494515f2e41a5e7e89a2cec5c192d44b9ca87a07a4",
    "folds": "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f",
    "with_performance_manifest": "91cc99688b0b5dca74f6ebabfe1447548cab7f86a4919725f8eff80987a492b3",
    "performance_only_manifest": "80c216dd6ece3f553d9a297dbda62b0505f0fa646c1ba4352abe9af553cb8b81",
    "phase04a": "34ea8100d9406f9701750a441aa6537323c28bcdb194cb3fd3645c4f7de4a2e1",
    "phase04b": "e2c88b1139a50aab6d47b6477c7bceff74f8443095f9d039ea9af84b715ee790",
    "phase06": "cd94acdf0bcb463b5f48a7c84110a23059899b6b5bf156131694827d7d7bde66",
    "phase07": "8569b48a8210f0ca1316d5a140d292edb892e2a556ea13ee299e9b97699af492",
}

DATA_PATHS = {
    "primary": PHASE03_ROOT / "data" / "primary_without_performance.csv",
    "with_performance": PHASE03_ROOT / "data" / "auxiliary_with_performance.csv",
    "performance_only": PHASE03_ROOT / "data" / "performance_only.csv",
    "folds": PHASE03_ROOT / "data" / "fold_assignments.csv",
}

MANIFEST_PATHS = {
    "primary": PHASE03_ROOT / "manifests" / "primary_feature_manifest.json",
    "with_performance": PHASE03_ROOT / "manifests" / "with_performance_feature_manifest.json",
    "performance_only": PHASE03_ROOT / "manifests" / "performance_only_feature_manifest.json",
    "feature_groups": PHASE03_ROOT / "manifests" / "feature_group_manifest.json",
}

UPSTREAM_PATHS = {
    "phase04a": EXPERIMENTS_ROOT / "phase_04a_traditional_classification_baselines" / "configs" / "phase04a_freeze.json",
    "phase04a_best_classifier": EXPERIMENTS_ROOT / "phase_04a_traditional_classification_baselines" / "configs" / "best_classifier.json",
    "phase04b": EXPERIMENTS_ROOT / "phase_04b_traditional_regression_baselines" / "configs" / "phase04b_freeze.json",
    "phase06": EXPERIMENTS_ROOT / "phase_06_hdc_variant_screening" / "configs" / "phase06_freeze.json",
    "phase07": EXPERIMENTS_ROOT / "phase_07_unimodal_contribution" / "configs" / "phase07_freeze.json",
}

REQUIRED_DIRECTORIES = [
    "data", "configs", "manifests", "audits", "src", "scripts", "tests",
    "figures", "logs", "reports", "results/checkpoints", "results/predictions",
    "results/fold_metrics", "results/oof", "results/fusion", "results/shortcut",
    "results/statistics", "results/efficiency", "results/summaries",
]

REQUIRED_ARTIFACTS = [
    "README.md", "task_plan.md", "notes.md",
    "configs/phase08_experiment_contract.json",
    "configs/phase08_environment.json",
    "configs/phase08_upstream_interface.json",
    "configs/phase08_dataset_conditions.json",
    "manifests/phase08_input_manifest.json",
    "manifests/phase08_fusion_feature_manifest.json",
    "manifests/phase08_performance_feature_manifest.json",
    "audits/phase08_input_and_fold_audit.json",
    "audits/phase08_dataset_alignment_audit.json",
    "audits/phase08_fusion_mapping_audit.json",
    "audits/phase08_performance_feature_risk_inventory.json",
    "audits/phase08_upstream_freeze_interface_audit.json",
    "scripts/initialize_phase08.py",
    "scripts/build_phase08_notebook.py",
    "Phase_08_Fusion_and_Shortcut_Analysis.ipynb",
]


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
    path = PHASE08_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def numeric_values(rows: list[dict[str, str]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row[field].strip() != ""]


def normalized_numeric(value: str) -> float | None:
    if value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def dataset_summary(
    headers: list[str], rows: list[dict[str, str]], feature_manifest: dict[str, Any]
) -> dict[str, Any]:
    features = feature_manifest["features"]
    return {
        "rows": len(rows),
        "subjects": len({row["subject_id"] for row in rows}),
        "unique_run_keys": len({row["run_key"] for row in rows}),
        "predictive_features": len(features),
        "manifest_features_unique": len(features) == len(set(features)),
        "manifest_features_present_in_csv": set(features).issubset(headers),
        "csv_columns": len(headers),
    }


def same_fields_by_run_key(
    left: list[dict[str, str]], right: list[dict[str, str]], fields: list[str]
) -> bool:
    left_map = {row["run_key"]: tuple(row[field] for field in fields) for row in left}
    right_map = {row["run_key"]: tuple(row[field] for field in fields) for row in right}
    return left_map == right_map


def deterministic_target_checks(
    rows: list[dict[str, str]], feature: str, target: str
) -> dict[str, Any]:
    feature_values = [normalized_numeric(row[feature]) for row in rows]
    target_values = [normalized_numeric(row[target]) for row in rows]
    complete_numeric = all(value is not None for value in feature_values + target_values)
    direct_copy = bool(
        complete_numeric
        and all(feature_value == target_value for feature_value, target_value in zip(feature_values, target_values))
    )
    mapping: dict[float, set[float]] = defaultdict(set)
    if complete_numeric:
        for target_value, feature_value in zip(target_values, feature_values):
            assert target_value is not None and feature_value is not None
            mapping[target_value].add(feature_value)
    deterministic_from_target = bool(
        complete_numeric
        and mapping
        and all(len(values) == 1 for values in mapping.values())
        and len({next(iter(values)) for values in mapping.values()}) > 1
    )
    return {
        "direct_numeric_copy": direct_copy,
        "deterministic_function_of_target": deterministic_from_target,
        "target_to_feature_unique_value_counts": {
            str(key): len(values) for key, values in sorted(mapping.items())
        } if complete_numeric else {},
    }


def run_initialization() -> dict[str, Any]:
    for directory in REQUIRED_DIRECTORIES:
        (PHASE08_ROOT / directory).mkdir(parents=True, exist_ok=True)

    missing_inputs = [
        str(path) for path in [*DATA_PATHS.values(), *MANIFEST_PATHS.values(), *UPSTREAM_PATHS.values()]
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(f"Missing frozen inputs: {missing_inputs}")

    manifests = {name: read_json(path) for name, path in MANIFEST_PATHS.items()}
    tables = {name: read_csv(path) for name, path in DATA_PATHS.items()}
    headers = {name: value[0] for name, value in tables.items()}
    rows = {name: value[1] for name, value in tables.items()}

    actual_hashes = {name: sha256(path) for name, path in DATA_PATHS.items()}
    actual_hashes.update({
        "with_performance_manifest": sha256(MANIFEST_PATHS["with_performance"]),
        "performance_only_manifest": sha256(MANIFEST_PATHS["performance_only"]),
    })
    hash_pass = {name: actual_hashes[name] == EXPECTED_HASHES[name] for name in actual_hashes}

    summaries = {
        name: dataset_summary(headers[name], rows[name], manifests[name])
        for name in ("primary", "with_performance", "performance_only")
    }
    expected_sizes = {
        "primary": (419, 35, 419, 1176),
        "with_performance": (419, 35, 419, 1235),
        "performance_only": (419, 35, 419, 59),
    }
    size_pass = {
        name: (
            summary["rows"], summary["subjects"], summary["unique_run_keys"],
            summary["predictive_features"],
        ) == expected_sizes[name]
        for name, summary in summaries.items()
    }

    primary_features = manifests["primary"]["features"]
    with_performance_features = manifests["with_performance"]["features"]
    performance_features = manifests["performance_only"]["features"]
    frozen_groups = manifests["primary"]["feature_groups"]
    source_groups = manifests["feature_groups"]["source_corrected_feature_groups"]

    primary_set = set(primary_features)
    with_performance_set = set(with_performance_features)
    performance_set = set(performance_features)
    feature_relations = {
        "with_performance_equals_primary_union_performance": with_performance_set == primary_set | performance_set,
        "primary_plus_performance_count_equals_with_performance": len(primary_set) + len(performance_set) == len(with_performance_set) == 1235,
        "primary_performance_intersection_count": len(primary_set & performance_set),
        "performance_only_equals_frozen_performance_group": performance_set == set(source_groups["performance_features"]),
    }

    identity_fields = ["subject_id", "target_class", "target_score", "outer_fold"]
    run_key_sets = {name: {row["run_key"] for row in rows[name]} for name in rows}
    three_dataset_run_key_alignment = (
        run_key_sets["primary"] == run_key_sets["with_performance"] == run_key_sets["performance_only"]
    )
    three_dataset_target_alignment = (
        same_fields_by_run_key(rows["primary"], rows["with_performance"], identity_fields)
        and same_fields_by_run_key(rows["primary"], rows["performance_only"], identity_fields)
    )
    fold_alignment = same_fields_by_run_key(
        rows["primary"], rows["folds"], ["subject_id", "target_class", "target_score", "outer_fold"]
    )
    target_class_values = sorted(set(numeric_values(rows["primary"], "target_class")))
    target_score_values = sorted(set(numeric_values(rows["primary"], "target_score")))
    targets_missing = any(
        not row["target_class"].strip() or not row["target_score"].strip()
        for row in rows["primary"]
    )
    outer_folds = sorted({int(float(row["outer_fold"])) for row in rows["folds"]})
    fold_checks = []
    for fold in outer_folds:
        test_subjects = {row["subject_id"] for row in rows["folds"] if int(float(row["outer_fold"])) == fold}
        train_subjects = {row["subject_id"] for row in rows["folds"] if int(float(row["outer_fold"])) != fold}
        fold_checks.append({
            "outer_fold": fold,
            "train_subjects": len(train_subjects),
            "test_subjects": len(test_subjects),
            "train_test_subject_overlap": sorted(train_subjects & test_subjects),
            "inner_groupkfold_3_feasible": len(train_subjects) >= 3,
        })

    identifier_and_target_fields = set(
        manifests["primary"].get("excluded_identifier_columns", [])
        + manifests["primary"].get("excluded_targets", [])
        + ["outer_fold", "difficulty_level", "difficulty_level_raw"]
    )
    control_features = set(source_groups.get("control_input_features", []))
    unverified_features = set(source_groups.get("unverified_features", []))
    fusion_specs = [
        ("physiological_plus_eye", ["physiological_features", "eye_tracking_features"], 649),
        ("physiological_plus_eye_plus_head", ["physiological_features", "eye_tracking_features", "head_movement_features"], 808),
        ("physiological_plus_eye_plus_head_plus_flight", ["physiological_features", "eye_tracking_features", "head_movement_features", "flight_parameter_features"], 1134),
        ("full_multimodal_without_performance", ["physiological_features", "eye_tracking_features", "head_movement_features", "flight_parameter_features", "body_movement"], 1176),
    ]
    fusion_combinations: dict[str, Any] = {}
    for combination_name, group_names, expected_count in fusion_specs:
        features = [feature for group_name in group_names for feature in frozen_groups[group_name]]
        feature_set = set(features)
        checks = {
            "feature_count_correct": len(features) == expected_count,
            "no_duplicate_columns": len(features) == len(feature_set),
            "membership_from_frozen_primary_manifest": feature_set.issubset(primary_set),
            "no_performance_features": not bool(feature_set & performance_set),
            "no_target_or_identifier_fields": not bool(feature_set & identifier_and_target_fields),
            "no_control_input_features": not bool(feature_set & control_features),
            "no_unverified_features": not bool(feature_set & unverified_features),
        }
        fusion_combinations[combination_name] = {
            "feature_groups": group_names,
            "feature_count": len(features),
            "expected_feature_count": expected_count,
            "features": features,
            "checks": checks,
            "pass": all(checks.values()),
        }
    fusion_mapping_pass = all(item["pass"] for item in fusion_combinations.values())

    collision_fields = {
        "target_class", "target_score", "difficulty_level", "difficulty_level_raw",
        "run_key", "subject_id", "session_id", "run_id", "outer_fold",
    }
    marker_terms = ["target", "difficulty", "level", "class", "score"]
    performance_inventory = []
    for feature in performance_features:
        missing_count = sum(not row[feature].strip() for row in rows["performance_only"])
        class_check = deterministic_target_checks(rows["performance_only"], feature, "target_class")
        score_check = deterministic_target_checks(rows["performance_only"], feature, "target_score")
        markers = [term for term in marker_terms if term in feature.casefold()]
        performance_inventory.append({
            "feature": feature,
            "source_group": "performance_features",
            "missing_count": missing_count,
            "missing_rate": missing_count / len(rows["performance_only"]),
            "reserved_field_name_collision": feature in collision_fields,
            "label_adjacent_name_markers": markers,
            "target_class_static_checks": class_check,
            "target_score_static_checks": score_check,
            "static_risk_flags": {
                "name_marker_present": bool(markers),
                "direct_target_copy_detected": class_check["direct_numeric_copy"] or score_check["direct_numeric_copy"],
                "deterministic_target_transform_detected": class_check["deterministic_function_of_target"] or score_check["deterministic_function_of_target"],
            },
        })

    upstream = {name: read_json(path) for name, path in UPSTREAM_PATHS.items()}
    upstream_hashes = {name: sha256(UPSTREAM_PATHS[name]) for name in ("phase04a", "phase04b", "phase06", "phase07")}
    upstream_hash_pass = {name: upstream_hashes[name] == EXPECTED_HASHES[name] for name in upstream_hashes}
    upstream_checks = {
        "phase04a": {
            "sha256": upstream_hashes["phase04a"],
            "checksum_pass": upstream_hash_pass["phase04a"],
            "frozen": upstream["phase04a"].get("phase04a_frozen") == "YES",
            "best_classifier": upstream["phase04a_best_classifier"].get("model"),
            "best_classifier_matches": upstream["phase04a_best_classifier"].get("model") == "Gradient Boosting"
                and upstream["phase04a"].get("best_traditional_classifier") == "Gradient Boosting",
        },
        "phase04b": {
            "sha256": upstream_hashes["phase04b"],
            "checksum_pass": upstream_hash_pass["phase04b"],
            "status": upstream["phase04b"].get("status"),
            "best_regressor": upstream["phase04b"].get("best_model"),
            "best_regressor_matches": upstream["phase04b"].get("best_model") == "Gradient Boosting Regressor",
        },
        "phase06": {
            "sha256": upstream_hashes["phase06"],
            "checksum_pass": upstream_hash_pass["phase06"],
            "status": upstream["phase06"].get("status"),
            "classification_hdc": upstream["phase06"]["best_classification_hdc"].get("selected_variant_name"),
            "classification_dimension": upstream["phase06"]["best_classification_hdc"].get("selected_fixed_dimension"),
            "regression_head": upstream["phase06"]["best_regression_hdc"].get("selected_regression_head"),
            "regression_dimension": upstream["phase06"]["best_regression_hdc"].get("selected_fixed_dimension"),
        },
        "phase07": {
            "sha256": upstream_hashes["phase07"],
            "checksum_pass": upstream_hash_pass["phase07"],
            "status": upstream["phase07"].get("status"),
            "best_classification_modality": upstream["phase07"].get("best_classification_modality"),
            "best_regression_modality": upstream["phase07"].get("best_regression_modality"),
            "all_final_audits_pass": upstream["phase07"].get("all_final_audits_pass"),
            "ready_for_next_planned_phase": upstream["phase07"].get("ready_for_next_planned_phase"),
        },
    }
    upstream_interface_pass = {
        "phase04a": all((upstream_checks["phase04a"]["checksum_pass"], upstream_checks["phase04a"]["frozen"], upstream_checks["phase04a"]["best_classifier_matches"])),
        "phase04b": all((upstream_checks["phase04b"]["checksum_pass"], upstream_checks["phase04b"]["status"] == "FROZEN", upstream_checks["phase04b"]["best_regressor_matches"])),
        "phase06": all((upstream_checks["phase06"]["checksum_pass"], upstream_checks["phase06"]["status"] == "FROZEN", upstream_checks["phase06"]["classification_hdc"] == "HDC+OnlineHD Hybrid", upstream_checks["phase06"]["classification_dimension"] == 5000, upstream_checks["phase06"]["regression_head"] == "COMMON_ENCODER_READOUT_BASELINE", upstream_checks["phase06"]["regression_dimension"] == 10000)),
        "phase07": all((upstream_checks["phase07"]["checksum_pass"], upstream_checks["phase07"]["status"] == "FROZEN", upstream_checks["phase07"]["best_classification_modality"] == "flight_parameter_features", upstream_checks["phase07"]["best_regression_modality"] == "flight_parameter_features", upstream_checks["phase07"]["all_final_audits_pass"] is True, upstream_checks["phase07"]["ready_for_next_planned_phase"] is True)),
    }

    input_fold_checks = {
        "all_expected_checksums_pass": all(hash_pass.values()),
        "all_expected_dataset_sizes_pass": all(size_pass.values()),
        "target_class_values_pass": target_class_values == [0.0, 1.0, 2.0, 3.0],
        "target_score_values_pass": target_score_values == [1.0, 2.0, 3.0, 4.0],
        "targets_have_no_missing_values": not targets_missing,
        "outer_folds_equal_5": outer_folds == [1, 2, 3, 4, 5],
        "outer_train_test_subject_overlap_zero": all(not item["train_test_subject_overlap"] for item in fold_checks),
        "inner_groupkfold_3_feasible": all(item["inner_groupkfold_3_feasible"] for item in fold_checks),
        "frozen_fold_alignment": fold_alignment,
        "folds_not_regenerated": True,
    }
    input_fold_pass = all(input_fold_checks.values())
    dataset_alignment_checks = {
        "three_dataset_run_key_alignment": three_dataset_run_key_alignment,
        "three_dataset_subject_target_fold_alignment": three_dataset_target_alignment,
        "with_performance_feature_union": feature_relations["with_performance_equals_primary_union_performance"],
        "primary_performance_disjoint": feature_relations["primary_performance_intersection_count"] == 0,
        "performance_only_matches_frozen_group": feature_relations["performance_only_equals_frozen_performance_group"],
        "manifest_roles_valid": manifests["primary"].get("intended_role") == "PRIMARY_THESIS_DATASET"
            and manifests["with_performance"].get("intended_role") == "AUXILIARY_SHORTCUT_LEARNING_COMPARISON"
            and manifests["performance_only"].get("intended_role") == "AUXILIARY_PERFORMANCE_ONLY_SHORTCUT_ANALYSIS",
    }
    dataset_alignment_pass = all(dataset_alignment_checks.values())

    environment = {
        "phase": "08",
        "captured_at_utc": utc_now(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "working_directory": os.getcwd(),
        "scope": "INITIALIZATION_ONLY_NO_MODELING",
    }
    contract = {
        "phase": "08",
        "phase_name": "Fusion and Shortcut Analysis",
        "status": "PENDING_CONTRACT_FREEZE",
        "scope": "INITIALIZATION_ONLY_NO_MODELING",
        "evidence_roles": {
            "primary_without_performance": "PRIMARY_MAIN_EVIDENCE_ONLY",
            "with_performance": "AUXILIARY_SHORTCUT_LEARNING_COMPARISON",
            "performance_only": "AUXILIARY_PERFORMANCE_ONLY_SHORTCUT_ANALYSIS",
        },
        "fusion_policy": {
            "early_fusion": "CORE",
            "late_fusion": "OPTIONAL_NOT_AUTHORIZED",
            "hdc_modality_aware_binding": "OPTIONAL_NOT_AUTHORIZED",
        },
        "regression_wording": "bounded difficulty-induced workload proxy regression",
        "candidate_model_interfaces_only": {
            "classification": ["frozen HDC+OnlineHD Hybrid", "frozen best traditional Gradient Boosting classifier"],
            "regression": ["frozen COMMON_ENCODER_READOUT_BASELINE", "frozen best traditional Gradient Boosting Regressor"],
        },
        "deferred_to_contract_freeze": [
            "fusion condition model matrix", "with-performance/performance-only model matrix",
            "fold-specific parameter reuse versus inner-CV", "HDC seeds and aggregation",
            "traditional model random seeds", "OOF canonical aggregation",
            "shortcut-risk thresholds", "statistical comparison families", "tie-breaking",
            "early-fusion ranking rules", "optional late-fusion authorization", "total model-run count",
        ],
        "training_authorized": False,
    }
    dataset_conditions = {
        "status": "PENDING_CONTRACT_FREEZE",
        "conditions": {
            "WITHOUT_PERFORMANCE_PRIMARY": {"feature_count": 1176, "role": "PRIMARY_MAIN_RESULT", "execution": "FROZEN_REFERENCE_ONLY"},
            "WITH_PERFORMANCE_AUXILIARY": {"feature_count": 1235, "composition": "primary 1176 + performance 59", "role": "AUXILIARY_UPPER_BOUND_AND_SHORTCUT_RISK"},
            "PERFORMANCE_ONLY_AUXILIARY": {"feature_count": 59, "role": "AUXILIARY_SHORTCUT_RISK_ONLY"},
        },
    }
    input_manifest = {
        "created_at_utc": utc_now(),
        "read_only": True,
        "inputs": {
            name: {"path": str(path), "sha256": actual_hashes.get(name, sha256(path)), "checksum_pass": hash_pass.get(name)}
            for name, path in {**DATA_PATHS, **MANIFEST_PATHS}.items()
        },
        "data_copied_into_phase08": False,
    }
    fusion_manifest = {
        "source": str(MANIFEST_PATHS["primary"]),
        "membership_rule": "EXPLICIT_FROZEN_MANIFEST_LISTS_ONLY_NO_PREFIX_INFERENCE",
        "core_fusion": "EARLY_FUSION",
        "combinations": fusion_combinations,
        "best_single_modality_reference": {"group": "flight_parameter_features", "feature_count": 326, "source": "Phase 07 frozen results", "retraining_authorized": False},
        "full_primary_reference": {"feature_count": 1176, "includes_body_movement": True, "source": "Phase 06 frozen multimodal OOF", "retraining_authorized": False},
        "optional": {"late_fusion": "OPTIONAL_NOT_AUTHORIZED", "hdc_modality_aware_binding": "OPTIONAL_NOT_AUTHORIZED"},
    }
    performance_manifest = {
        "source": str(MANIFEST_PATHS["performance_only"]),
        "source_group": "performance_features",
        "feature_count": len(performance_features),
        "features": performance_features,
        "interpretation_guardrail": "Static risk indicators do not by themselves prove leakage; predictive ability must not be interpreted as physiological causality.",
    }

    write_json("configs/phase08_experiment_contract.json", contract)
    write_json("configs/phase08_environment.json", environment)
    write_json("configs/phase08_upstream_interface.json", {
        "status": "FROZEN_READ_ONLY_INTERFACES",
        "interfaces": upstream_checks,
        "candidate_models_only_not_authorized_for_training": contract["candidate_model_interfaces_only"],
    })
    write_json("configs/phase08_dataset_conditions.json", dataset_conditions)
    write_json("manifests/phase08_input_manifest.json", input_manifest)
    write_json("manifests/phase08_fusion_feature_manifest.json", fusion_manifest)
    write_json("manifests/phase08_performance_feature_manifest.json", performance_manifest)
    write_json("audits/phase08_input_and_fold_audit.json", {
        "status": "PASS" if input_fold_pass else "FAIL",
        "actual_hashes": actual_hashes,
        "expected_hashes": {name: EXPECTED_HASHES[name] for name in actual_hashes},
        "checksum_pass": hash_pass,
        "dataset_summaries": summaries,
        "dataset_size_pass": size_pass,
        "target_class_values": target_class_values,
        "target_score_values": target_score_values,
        "outer_folds": outer_folds,
        "fold_checks": fold_checks,
        "checks": input_fold_checks,
    })
    write_json("audits/phase08_dataset_alignment_audit.json", {
        "status": "PASS" if dataset_alignment_pass else "FAIL",
        "checks": dataset_alignment_checks,
        "feature_set_relations": feature_relations,
        "identity_fields_compared_by_run_key": identity_fields,
    })
    write_json("audits/phase08_fusion_mapping_audit.json", {
        "status": "PASS" if fusion_mapping_pass else "FAIL",
        "membership_rule": "EXPLICIT_FROZEN_MANIFEST_LISTS_ONLY_NO_PREFIX_INFERENCE",
        "combinations": {name: {"feature_count": item["feature_count"], "checks": item["checks"], "pass": item["pass"]} for name, item in fusion_combinations.items()},
    })
    write_json("audits/phase08_performance_feature_risk_inventory.json", {
        "status": "STATIC_AUDIT_COMPLETE",
        "feature_count": len(performance_inventory),
        "features": performance_inventory,
        "summary": {
            "reserved_name_collisions": sum(item["reserved_field_name_collision"] for item in performance_inventory),
            "label_adjacent_names": sum(item["static_risk_flags"]["name_marker_present"] for item in performance_inventory),
            "direct_target_copies": sum(item["static_risk_flags"]["direct_target_copy_detected"] for item in performance_inventory),
            "deterministic_target_transforms": sum(item["static_risk_flags"]["deterministic_target_transform_detected"] for item in performance_inventory),
        },
        "interpretation": "Report-only static screening. No feature was removed or renamed; names alone do not establish leakage and predictive performance is not physiological causal evidence.",
    })
    write_json("audits/phase08_upstream_freeze_interface_audit.json", {
        "status": "PASS" if all(upstream_interface_pass.values()) else "FAIL",
        "interface_pass": upstream_interface_pass,
        "interfaces": upstream_checks,
        "upstream_artifacts_modified": False,
    })

    overall_pre_notebook_pass = all((
        input_fold_pass, dataset_alignment_pass, fusion_mapping_pass,
        all(upstream_interface_pass.values()), len(performance_inventory) == 59,
    ))
    output_summary = {
        "phase08_directory_initialized": True,
        "phase08_name": "Fusion and Shortcut Analysis",
        "summaries": summaries,
        "checksum_pass": hash_pass,
        "three_dataset_run_key_alignment": three_dataset_run_key_alignment,
        "three_dataset_target_alignment": three_dataset_target_alignment,
        "feature_relations": feature_relations,
        "fusion_counts": {name: item["feature_count"] for name, item in fusion_combinations.items()},
        "fusion_mapping_pass": fusion_mapping_pass,
        "performance_feature_risk_inventory_saved": True,
        "upstream_interface_pass": upstream_interface_pass,
        "best_traditional_classifier": upstream_checks["phase04a"]["best_classifier"],
        "best_traditional_regressor": upstream_checks["phase04b"]["best_regressor"],
        "frozen_hdc_classifier": upstream_checks["phase06"]["classification_hdc"],
        "frozen_hdc_regression_head": upstream_checks["phase06"]["regression_head"],
        "model_training_executed": False,
        "outer_test_predictions_generated": False,
        "input_fold_audit_saved": True,
        "dataset_alignment_audit_saved": True,
        "fusion_mapping_audit_saved": True,
        "upstream_interface_audit_saved": True,
        "overall_pre_notebook_pass": overall_pre_notebook_pass,
        "phase08_status": "PENDING_CONTRACT_FREEZE" if overall_pre_notebook_pass else "FAIL",
        "ready_for_contract_freeze_pre_notebook": overall_pre_notebook_pass,
        "ready_for_modeling": False,
    }
    write_json("audits/phase08_initialization_artifact_audit.json", {
        "status": "PENDING_NOTEBOOK_EXECUTION" if overall_pre_notebook_pass else "FAIL",
        "required_directories": {path: (PHASE08_ROOT / path).is_dir() for path in REQUIRED_DIRECTORIES},
        "required_artifacts": {path: (PHASE08_ROOT / path).is_file() for path in REQUIRED_ARTIFACTS},
        "phase08_data_directory_entries": sorted(path.name for path in (PHASE08_ROOT / "data").iterdir()),
        "training_executed": False,
        "predictions_generated": False,
        "overall_pre_notebook_pass": overall_pre_notebook_pass,
    })
    write_json("audits/phase08_initialization_summary.json", output_summary)
    return output_summary


if __name__ == "__main__":
    summary = run_initialization()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
