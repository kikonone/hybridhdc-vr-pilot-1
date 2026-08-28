"""Freeze the Phase 08 execution contract without training or prediction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT.parent
P2 = EXPERIMENTS / "phase_02_full_multimodal_feature_extraction"
P3 = EXPERIMENTS / "phase_03_multimodal_dataset_labeling"
P4A = EXPERIMENTS / "phase_04a_traditional_classification_baselines"
P4B = EXPERIMENTS / "phase_04b_traditional_regression_baselines"
P6 = EXPERIMENTS / "phase_06_hdc_variant_screening"
P7 = EXPERIMENTS / "phase_07_unimodal_contribution"

EXPECTED_HASHES = {
    "primary": "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44",
    "with_performance": "72977a2119e30e37996fb9f0e3404988c4977fb7d2b33992f87bf54bfe5decba",
    "performance_only": "d602282ae41153886d1306494515f2e41a5e7e89a2cec5c192d44b9ca87a07a4",
    "folds": "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f",
}
DATA_PATHS = {
    "primary": P3 / "data" / "primary_without_performance.csv",
    "with_performance": P3 / "data" / "auxiliary_with_performance.csv",
    "performance_only": P3 / "data" / "performance_only.csv",
    "folds": P3 / "data" / "fold_assignments.csv",
}
PHASE08_GATE_FILES = [
    "configs/phase08_experiment_contract.json",
    "configs/phase08_dataset_conditions.json",
    "configs/phase08_upstream_interface.json",
    "manifests/phase08_input_manifest.json",
    "manifests/phase08_fusion_feature_manifest.json",
    "manifests/phase08_performance_feature_manifest.json",
    "audits/phase08_input_and_fold_audit.json",
    "audits/phase08_dataset_alignment_audit.json",
    "audits/phase08_fusion_mapping_audit.json",
    "audits/phase08_performance_feature_risk_inventory.json",
    "audits/phase08_upstream_freeze_interface_audit.json",
    "audits/phase08_initialization_artifact_audit.json",
    "audits/phase08_notebook_persistence_audit.json",
]
CONTRACT_ARTIFACTS = [
    "configs/phase08_frozen_contract.json",
    "configs/phase08_model_matrix.json",
    "configs/phase08_execution_manifest.json",
    "configs/phase08_metric_definitions.json",
    "configs/phase08_statistical_analysis_contract.json",
    "configs/phase08_shortcut_evidence_contract.json",
    "manifests/phase08_flight_feature_provenance_manifest.json",
    "manifests/phase08_to_phase09_generalization_handoff.json",
    "audits/phase08_flight_feature_provenance_audit.json",
    "audits/phase08_scenario_metadata_feasibility_audit.json",
    "audits/phase08_contract_freeze_audit.json",
    "audits/phase08_pretraining_feasibility_audit.json",
    "scripts/freeze_phase08_contract.py",
    "tests/test_phase08_contract.py",
]
SEEDS = [42, 43, 44, 45, 46]
FOLDS = [1, 2, 3, 4, 5]
UPSTREAM_READ_PATHS = [
    *DATA_PATHS.values(),
    P3 / "manifests/primary_feature_manifest.json",
    P3 / "manifests/with_performance_feature_manifest.json",
    P3 / "manifests/performance_only_feature_manifest.json",
    P3 / "manifests/feature_group_manifest.json",
    P2 / "results/phase02_corrected_feature_provenance.csv",
    P2 / "results/phase02_corrected_feature_groups.json",
    P2 / "results/phase02_modeling_feature_manifest.json",
    P2 / "scripts/phase02_extract.py",
    P2 / "README.md",
    P4A / "configs/phase04a_freeze.json",
    P4A / "configs/best_classifier.json",
    P4A / "configs/phase04a_final_configuration.json",
    P4A / "configs/classification_best_params_by_fold.json",
    P4B / "configs/phase04b_freeze.json",
    P4B / "configs/gradient_boosting_configuration.json",
    P6 / "configs/phase06_freeze.json",
    P6 / "configs/phase06_best_classification_hdc.json",
    P6 / "configs/phase06_best_regression_hdc.json",
    P7 / "configs/phase07_freeze.json",
    P7 / "configs/phase07_frozen_unimodal_contract.json",
]
AGGREGATION_SUFFIXES = sorted([
    "sampling_rate_estimate", "missing_ratio", "num_missing", "sample_count",
    "kurtosis", "median", "range", "slope", "mean", "std", "min", "max",
    "iqr", "skew", "rms", "duration",
], key=len, reverse=True)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(relative_path: str, value: Any) -> None:
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_set_hash(features: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(features)) + "\n").encode("utf-8")).hexdigest()


def split_aggregation(feature: str) -> tuple[str, str]:
    for suffix in AGGREGATION_SUFFIXES:
        marker = "_" + suffix
        if feature.endswith(marker):
            return feature[:-len(marker)], suffix
    raise ValueError(f"Cannot parse aggregation statistic: {feature}")


def file_count(path: Path) -> int:
    return sum(item.is_file() for item in path.rglob("*")) if path.exists() else 0


def gate_checks() -> dict[str, Any]:
    missing = [relative for relative in PHASE08_GATE_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise RuntimeError(f"Contract Freeze gate missing artifacts: {missing}")
    contract = read_json(ROOT / "configs/phase08_experiment_contract.json")
    input_audit = read_json(ROOT / "audits/phase08_input_and_fold_audit.json")
    alignment = read_json(ROOT / "audits/phase08_dataset_alignment_audit.json")
    fusion = read_json(ROOT / "audits/phase08_fusion_mapping_audit.json")
    risk = read_json(ROOT / "audits/phase08_performance_feature_risk_inventory.json")
    upstream = read_json(ROOT / "audits/phase08_upstream_freeze_interface_audit.json")
    artifact = read_json(ROOT / "audits/phase08_initialization_artifact_audit.json")
    notebook = read_json(ROOT / "audits/phase08_notebook_persistence_audit.json")
    summaries = input_audit["dataset_summaries"]
    actual_hashes = {name: sha256(path) for name, path in DATA_PATHS.items()}
    checks = {
        "status_valid_for_initial_or_idempotent_freeze": contract.get("status") in {"PENDING_CONTRACT_FREEZE", "CONTRACT_FROZEN_NOT_TRAINED"},
        "ready_for_contract_freeze": artifact.get("ready_for_contract_freeze") is True
            or (artifact.get("overall_pre_notebook_pass") is True and notebook.get("status") == "PASS"),
        "ready_for_modeling_no": artifact.get("ready_for_modeling", False) is False,
        "notebook_initialization_pass": notebook.get("status") == "PASS",
        "three_rows_419": all(summaries[name]["rows"] == 419 for name in ("primary", "with_performance", "performance_only")),
        "three_subjects_35": all(summaries[name]["subjects"] == 35 for name in ("primary", "with_performance", "performance_only")),
        "three_unique_run_keys_419": all(summaries[name]["unique_run_keys"] == 419 for name in ("primary", "with_performance", "performance_only")),
        "feature_counts": [summaries[name]["predictive_features"] for name in ("primary", "with_performance", "performance_only")] == [1176, 1235, 59],
        "alignment_pass": alignment.get("status") == "PASS",
        "run_key_alignment": alignment["checks"].get("three_dataset_run_key_alignment") is True,
        "target_fold_alignment": alignment["checks"].get("three_dataset_subject_target_fold_alignment") is True,
        "primary_performance_intersection_zero": alignment["feature_set_relations"].get("primary_performance_intersection_count") == 0,
        "fusion_mapping_pass": fusion.get("status") == "PASS",
        "performance_inventory_59": risk.get("feature_count") == 59,
        "performance_direct_copy_zero": risk["summary"].get("direct_target_copies") == 0,
        "performance_deterministic_transform_zero": risk["summary"].get("deterministic_target_transforms") == 0,
        "upstream_interfaces_pass": upstream.get("status") == "PASS" and all(upstream.get("interface_pass", {}).values()),
        "phase07_frozen": upstream["interfaces"]["phase07"].get("status") == "FROZEN",
        "checksums": all(actual_hashes[name] == EXPECTED_HASHES[name] for name in EXPECTED_HASHES),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Contract Freeze gate failed: {[name for name, passed in checks.items() if not passed]}")
    return {"status": "PASS", "checks": checks, "actual_hashes": actual_hashes}


def raw_xplane_schema() -> tuple[list[str], Path, Path | None]:
    source_root = Path(r"E:\hdc-vr-pilot\vrdataset\dataPackage\task-ils")
    data_path = next(source_root.rglob("*lslxp11xpcac*_dat.csv"))
    with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
        schema = next(csv.reader(handle))
    header_candidates = list(data_path.parent.glob("*lslxp11xpcac*_hea.csv"))
    return schema, data_path, header_candidates[0] if header_candidates else None


def build_flight_provenance() -> tuple[dict[str, Any], dict[str, Any]]:
    primary_manifest = read_json(P3 / "manifests/primary_feature_manifest.json")
    flight_features = primary_manifest["feature_groups"]["flight_parameter_features"]
    provenance_path = P2 / "results/phase02_corrected_feature_provenance.csv"
    provenance_rows = {row["feature_name"]: row for row in read_csv(provenance_path)}
    schema, raw_data_path, raw_header_path = raw_xplane_schema()
    raw_fields = [field for field in schema if field != "time_dn"]
    derived_terms = {
        "xplane_airspeed": ["airspeed"],
        "xplane_altitude": ["altitude", "elevation"],
        "xplane_heading": ["heading", "yaw"],
        "xplane_ils": ["ils"],
        "xplane_latitude": ["latitude"],
        "xplane_longitude": ["longitude"],
        "xplane_attitude": ["pitch", "roll", "yaw"],
    }
    records: list[dict[str, Any]] = []
    for feature in flight_features:
        base_variable, statistic = split_aggregation(feature)
        provenance = provenance_rows.get(feature)
        if provenance is None:
            raise RuntimeError(f"Missing Phase 02 provenance for {feature}")
        source_match = re.search(r"source_column=([^|]+)$", provenance["source"])
        recorded_source_field = source_match.group(1).strip() if source_match else ""
        if base_variable in derived_terms:
            source_fields = sorted({
                field for field in raw_fields
                if any(term in field.casefold() for term in derived_terms[base_variable])
            })
            source_field: str | list[str] = source_fields
        elif base_variable == "xplane_lslxp11xpcac":
            source_field = "time_dn/acquisition_metadata"
        else:
            source_field = recorded_source_field

        if base_variable == "xplane_lslxp11xpcac":
            category = "AMBIGUOUS"
            rationale = "Acquisition duration, sample count, and sampling-rate metadata do not directly encode an aircraft behavioral response or a preset task/scenario setting."
            confidence = "HIGH"
            ambiguous_reason = "Acquisition-process metadata cannot be assigned to either permitted substantive provenance group."
        else:
            category = "BEHAVIORAL_RESPONSE"
            rationale = "Phase 02 explicitly identifies the X-Plane stream as aircraft/pilot state; the raw schema and extraction rule show this feature summarizes aircraft state, trajectory, dynamic response, trim/control result, or ILS response rather than a preset scenario/configuration field."
            confidence = "HIGH" if recorded_source_field != "derived" else "MEDIUM"
            ambiguous_reason = ""
        records.append({
            "feature_name": feature,
            "base_variable": base_variable,
            "aggregation_statistic": statistic,
            "source_stream": "lslxp11xpcac / xplane_flight_state",
            "source_field": source_field,
            "provenance_evidence": {
                "phase02_provenance_path": str(provenance_path),
                "phase02_provenance_status": provenance["provenance_status"],
                "phase02_verified_status": provenance["verified_status"],
                "phase02_source_record": provenance["source"],
                "phase02_notes": provenance["notes"],
                "raw_schema_data_path": str(raw_data_path),
                "raw_schema_header_path": str(raw_header_path) if raw_header_path else None,
                "raw_schema_fields": schema,
                "extraction_script": str(P2 / "scripts/phase02_extract.py"),
                "extraction_rule": "xplane raw-column statistics plus frozen derived combinations selected by explicit terms",
                "documentation": str(P2 / "README.md"),
                "documentation_statement": "Flight/control features include X-Plane aircraft state summaries; no explicit control-input streams were available.",
            },
            "semantic_category": category,
            "rationale": rationale,
            "confidence": confidence,
            "ambiguous_reason": ambiguous_reason,
        })
    counts = Counter(record["semantic_category"] for record in records)
    manifest = {
        "phase": "08",
        "status": "FROZEN_BEFORE_TRAINING",
        "classification_policy": "PROVENANCE_ONLY_NO_TARGET_CORRELATION_OR_MODEL_PERFORMANCE",
        "allowed_categories": ["BEHAVIORAL_RESPONSE", "TASK_SETTING_OR_SCENARIO", "AMBIGUOUS"],
        "source_flight_feature_manifest": str(P3 / "manifests/primary_feature_manifest.json"),
        "source_phase02_provenance": str(provenance_path),
        "source_stream": "lslxp11xpcac / xplane_flight_state",
        "raw_schema": schema,
        "feature_count": len(records),
        "category_counts": {name: counts.get(name, 0) for name in ("BEHAVIORAL_RESPONSE", "TASK_SETTING_OR_SCENARIO", "AMBIGUOUS")},
        "features": records,
        "training_executed": False,
    }
    write_json("manifests/phase08_flight_feature_provenance_manifest.json", manifest)
    manifest_path = ROOT / "manifests/phase08_flight_feature_provenance_manifest.json"
    manifest_hash = sha256(manifest_path)
    behavioral = [record["feature_name"] for record in records if record["semantic_category"] == "BEHAVIORAL_RESPONSE"]
    task_setting = [record["feature_name"] for record in records if record["semantic_category"] == "TASK_SETTING_OR_SCENARIO"]
    ambiguous = [record["feature_name"] for record in records if record["semantic_category"] == "AMBIGUOUS"]
    audit_checks = {
        "feature_count_326": len(records) == 326,
        "all_features_unique": len({record["feature_name"] for record in records}) == 326,
        "exactly_one_allowed_category_each": all(record["semantic_category"] in manifest["allowed_categories"] for record in records),
        "required_fields_complete": all(all(key in record and record[key] is not None for key in ("feature_name", "base_variable", "aggregation_statistic", "source_stream", "source_field", "provenance_evidence", "semantic_category", "rationale", "confidence", "ambiguous_reason")) for record in records),
        "behavioral_task_setting_disjoint": not bool(set(behavioral) & set(task_setting)),
        "ambiguous_excluded_from_sensitivity_subsets": not bool(set(ambiguous) & (set(behavioral) | set(task_setting))),
        "phase02_provenance_complete": all(record["provenance_evidence"]["phase02_provenance_status"] == "VERIFIED_OTHER" for record in records),
        "categories_not_target_informed": True,
        "manifest_frozen_before_training": True,
    }
    audit = {
        "status": "PASS" if all(audit_checks.values()) else "FAIL",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "category_counts": manifest["category_counts"],
        "behavioral_feature_set_sha256": feature_set_hash(behavioral),
        "task_setting_feature_set_sha256": feature_set_hash(task_setting),
        "ambiguous_feature_set_sha256": feature_set_hash(ambiguous),
        "checks": audit_checks,
        "caveat": "Provenance classification does not establish leakage or causal interpretation.",
    }
    write_json("audits/phase08_flight_feature_provenance_audit.json", audit)
    return manifest, audit


def condition_status(features: list[str], full_features: list[str], seen_hashes: dict[str, str]) -> dict[str, Any]:
    set_hash = feature_set_hash(features)
    if not features:
        return {"status": "NOT_FEASIBLE_EMPTY_PROVENANCE_GROUP", "feature_count": 0, "feature_set_sha256": set_hash, "alias_of": None}
    if set(features) == set(full_features):
        return {"status": "REFERENCE_ALIAS_NO_DUPLICATE_TRAINING", "feature_count": len(features), "feature_set_sha256": set_hash, "alias_of": "FLIGHT_FULL"}
    if set_hash in seen_hashes:
        return {"status": "REFERENCE_ALIAS_NO_DUPLICATE_TRAINING", "feature_count": len(features), "feature_set_sha256": set_hash, "alias_of": seen_hashes[set_hash]}
    seen_hashes[set_hash] = "UNIQUE"
    return {"status": "AUTHORIZED_UNIQUE_SENSITIVITY_CONDITION", "feature_count": len(features), "feature_set_sha256": set_hash, "alias_of": None}


def metadata_feasibility(primary_rows: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    def profile(field: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in primary_rows:
            groups[row[field]].append(row)
        subjects = [len({row["subject_id"] for row in group_rows}) for group_rows in groups.values()]
        coverage = {group: sorted({int(float(row["target_class"])) for row in group_rows}) for group, group_rows in groups.items()}
        return {
            "metadata_fields": [field],
            "unique_groups": len(groups),
            "subjects_per_group": {"minimum": min(subjects), "maximum": max(subjects)},
            "target_coverage": {"groups_with_all_four_classes": sum(values == [0, 1, 2, 3] for values in coverage.values()), "total_groups": len(groups)},
        }
    session = profile("session_id")
    session.update({
        "feasibility": "NOT_FEASIBLE_DUE_TO_METADATA",
        "leakage_risks": ["Each session belongs to exactly one subject and each subject has exactly one session; session holdout is indistinguishable from subject holdout."],
        "recommended_grouping": "Do not claim independent session generalization; retain subject-wise folds until repeated sessions per subject exist.",
    })
    difficulty = profile("difficulty_level_raw")
    missing_group = {
        "metadata_fields": [], "unique_groups": 0,
        "subjects_per_group": None, "target_coverage": None,
        "feasibility": "NOT_FEASIBLE_DUE_TO_METADATA",
        "leakage_risks": ["No explicit identifier exists; inferring one from target, run order, or feature values would create label proximity or fabricated metadata."],
        "recommended_grouping": "Collect an explicit preregistered identifier before Phase 09.",
    }
    handoff = {
        "phase": "08_to_09",
        "status": "SAVED",
        "source_rows": len(primary_rows),
        "source_subjects": len({row["subject_id"] for row in primary_rows}),
        "holdouts": {
            "unseen_session": session,
            "unseen_scenario": {**missing_group, "recommended_grouping": "Collect explicit scenario_id; do not use difficulty/target as a scenario proxy."},
            "task_template": {**missing_group, "recommended_grouping": "Collect explicit task_template_id; current data only identifies the common task-ils task."},
            "route_or_configuration": {**missing_group, "recommended_grouping": "Collect explicit route_id and configuration_id before route/configuration holdout."},
        },
        "observed_difficulty_metadata": {**difficulty, "warning": "difficulty_level_raw deterministically defines the target and is not an admissible scenario-holdout grouping for the target-prediction task."},
        "generalization_guardrails": [
            "Subject-wise LOSO only evaluates subject generalization.",
            "LOSO does not automatically establish scenario generalization.",
            "Do not fabricate a scenario identifier that is absent from the metadata.",
        ],
        "phase09_directory_created": False,
        "holdout_executed": False,
    }
    audit_checks = {
        "source_rows_419": len(primary_rows) == 419,
        "subjects_35": len({row["subject_id"] for row in primary_rows}) == 35,
        "session_groups_35": session["unique_groups"] == 35,
        "session_one_subject_each": session["subjects_per_group"] == {"minimum": 1, "maximum": 1},
        "no_scenario_identifier_fabricated": not handoff["holdouts"]["unseen_scenario"]["metadata_fields"],
        "no_task_template_identifier_fabricated": not handoff["holdouts"]["task_template"]["metadata_fields"],
        "no_route_configuration_identifier_fabricated": not handoff["holdouts"]["route_or_configuration"]["metadata_fields"],
        "loso_guardrail_present": len(handoff["generalization_guardrails"]) == 3,
    }
    audit = {"status": "PASS" if all(audit_checks.values()) else "FAIL", "checks": audit_checks, "holdout_feasibility": {name: value["feasibility"] for name, value in handoff["holdouts"].items()}}
    write_json("manifests/phase08_to_phase09_generalization_handoff.json", handoff)
    write_json("audits/phase08_scenario_metadata_feasibility_audit.json", audit)
    return handoff, audit


def load_traditional_interfaces() -> dict[str, Any]:
    classification_params = read_json(P4A / "configs/classification_best_params_by_fold.json")["gradient_boosting"]
    regression_params = {
        str(fold): read_json(P4B / f"results/checkpoints/gradient_boosting/gradient_boosting_fold_{fold}_best_params.json")["best_params"]
        for fold in FOLDS
    }
    classification = {
        "status": "FROZEN_REUSE_NO_RETUNING",
        "model": "Gradient Boosting",
        "estimator": "GradientBoostingClassifier",
        "random_state": 42,
        "fold_specific_parameters": classification_params,
        "pipeline": ["SimpleImputer(strategy=median, add_indicator=true, keep_empty_features=true)", "VarianceThreshold(0.0)", "SelectKBest(f_classif)", "GradientBoostingClassifier"],
        "effective_feature_k_rule": "min(frozen_requested_k, post_variance_feature_count)",
        "source": str(P4A / "configs/classification_best_params_by_fold.json"),
        "parameter_search_authorized": False,
    }
    regression = {
        "status": "FROZEN_REUSE_NO_RETUNING",
        "model": "Gradient Boosting Regressor",
        "estimator": "GradientBoostingRegressor",
        "random_state": 42,
        "fold_specific_parameters": regression_params,
        "pipeline": read_json(P4B / "configs/gradient_boosting_configuration.json")["pipeline"],
        "effective_feature_k_rule": "min(frozen_requested_k, post_variance_feature_count); preserve 'all' as all post-variance features",
        "source": str(P4B / "results/checkpoints/gradient_boosting"),
        "parameter_search_authorized": False,
    }
    return {"classification": classification, "regression": regression}


def build_run_records(conditions: list[str], sensitivity_conditions: list[str], traditional_flight_full_required: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    def add(condition: str, model_family: str, task: str, fold: int, seed: int | None) -> None:
        seed_token = f"seed_{seed}" if seed is not None else "canonical"
        records.append({"run_id": f"{condition}__{model_family}__{task}__fold_{fold}__{seed_token}", "condition": condition, "model_family": model_family, "task": task, "outer_fold": fold, "seed": seed, "status": "AUTHORIZED_NOT_EXECUTED"})
    for condition in conditions + sensitivity_conditions:
        for task in ("classification", "regression"):
            for fold in FOLDS:
                for seed in SEEDS:
                    add(condition, "HDC", task, fold, seed)
                add(condition, "TRADITIONAL", task, fold, None)
    if traditional_flight_full_required:
        for task in ("classification", "regression"):
            for fold in FOLDS:
                add("FLIGHT_FULL", "TRADITIONAL", task, fold, None)
    return records


def freeze_contract() -> dict[str, Any]:
    result_files_before = file_count(ROOT / "results")
    regression_parameter_paths = [P4B / f"results/checkpoints/gradient_boosting/gradient_boosting_fold_{fold}_best_params.json" for fold in FOLDS]
    schema, raw_data_path, raw_header_path = raw_xplane_schema()
    del schema
    upstream_paths = [*UPSTREAM_READ_PATHS, *regression_parameter_paths, raw_data_path]
    if raw_header_path is not None:
        upstream_paths.append(raw_header_path)
    upstream_before = {str(path): sha256(path) for path in upstream_paths}
    gate = gate_checks()
    primary_rows = read_csv(DATA_PATHS["primary"])
    provenance_manifest, provenance_audit = build_flight_provenance()
    handoff, metadata_audit = metadata_feasibility(primary_rows)
    flight_features = [record["feature_name"] for record in provenance_manifest["features"]]
    behavioral = [record["feature_name"] for record in provenance_manifest["features"] if record["semantic_category"] == "BEHAVIORAL_RESPONSE"]
    task_setting = [record["feature_name"] for record in provenance_manifest["features"] if record["semantic_category"] == "TASK_SETTING_OR_SCENARIO"]
    ambiguous = [record["feature_name"] for record in provenance_manifest["features"] if record["semantic_category"] == "AMBIGUOUS"]
    seen_hashes = {feature_set_hash(flight_features): "FLIGHT_FULL"}
    behavioral_condition = condition_status(behavioral, flight_features, seen_hashes)
    if behavioral_condition["status"] == "AUTHORIZED_UNIQUE_SENSITIVITY_CONDITION":
        seen_hashes[behavioral_condition["feature_set_sha256"]] = "FLIGHT_BEHAVIORAL_ONLY"
    task_condition = condition_status(task_setting, flight_features, seen_hashes)
    sensitivity = {
        "FLIGHT_FULL": {"status": "FROZEN_HDC_REFERENCE", "feature_count": 326, "feature_set_sha256": feature_set_hash(flight_features), "ambiguous_features_retained": len(ambiguous)},
        "FLIGHT_BEHAVIORAL_ONLY": behavioral_condition,
        "FLIGHT_TASK_SETTING_ONLY": task_condition,
    }

    traditional = load_traditional_interfaces()
    phase06_classification = read_json(P6 / "configs/phase06_best_classification_hdc.json")
    phase06_regression = read_json(P6 / "configs/phase06_best_regression_hdc.json")
    hdc = {
        "classification": {
            "status": "FROZEN_REUSE_NO_SEARCH", "model": "HDC+OnlineHD Hybrid", "dimension": 5000, "levels": 51,
            "seeds": SEEDS, "requested_feature_k": 50, "fold_specific_structures": phase06_classification["fold_selected_structures"],
            "structure_search_authorized": False, "variant_search_authorized": False,
        },
        "regression": {
            "status": "FROZEN_REUSE_NO_SEARCH", "head": "COMMON_ENCODER_READOUT_BASELINE", "dimension": 10000, "levels": 51,
            "seeds": SEEDS, "ridge_alpha": 0.01, "requested_feature_k": 50, "parameter_search_authorized": False,
        },
        "preprocessing": {
            "fit_scope": "outer-training only", "ordered_steps": ["SimpleImputer median + missing indicators", "VarianceThreshold(0.0)", "StandardScaler fold-local", "SelectKBest(f_classif)"],
            "effective_feature_k_rule": "min(50, post_variance_feature_count)", "row_deletion_authorized": False,
        },
    }
    hdc_interface_pass = (
        hdc["classification"]["dimension"] == 5000 and hdc["classification"]["levels"] == 51 and hdc["classification"]["seeds"] == SEEDS
        and len(hdc["classification"]["fold_specific_structures"]) == 5
        and hdc["regression"]["dimension"] == 10000 and hdc["regression"]["ridge_alpha"] == 0.01 and hdc["regression"]["seeds"] == SEEDS
    )
    traditional_interface_pass = (
        traditional["classification"]["random_state"] == 42 and len(traditional["classification"]["fold_specific_parameters"]) == 5
        and traditional["regression"]["random_state"] == 42 and len(traditional["regression"]["fold_specific_parameters"]) == 5
    )

    mandatory_conditions = ["FUSION_PE", "FUSION_PEH", "FUSION_PEHF", "WITH_PERFORMANCE_AUXILIARY", "PERFORMANCE_ONLY_AUXILIARY"]
    unique_sensitivity = [name for name, value in sensitivity.items() if name != "FLIGHT_FULL" and value["status"] == "AUTHORIZED_UNIQUE_SENSITIVITY_CONDITION"]
    traditional_flight_full_required = True
    run_records = build_run_records(mandatory_conditions, unique_sensitivity, traditional_flight_full_required)
    run_ids = [record["run_id"] for record in run_records]
    core_runs = 300
    traditional_flight_runs = 10 if traditional_flight_full_required else 0
    flight_sensitivity_runs = 60 * len(unique_sensitivity)
    expected_total_runs = core_runs + traditional_flight_runs + flight_sensitivity_runs
    duplicate_run_identifiers = len(run_ids) - len(set(run_ids))

    oof_rules = {
        "status": "FROZEN",
        "HDC_classification": {"seeds": SEEDS, "rule": "mean five seed class scores then argmax", "tie_break": "smaller class", "metrics": "recompute from aggregated predictions", "best_seed_selection": "PROHIBITED"},
        "HDC_regression": {"seeds": SEEDS, "rule": "mean five prediction_raw values then clip to [1.0, 4.0]", "metrics": "recompute from aggregated predictions", "best_seed_selection": "PROHIBITED"},
        "traditional": {"canonical_predictions_per_condition_task_run_key": 1, "seed_repetitions": "PROHIBITED"},
        "coverage": {"rows": 419, "unique_run_key": 419, "outer_folds": 5},
    }
    comparison_families = {
        "status": "FROZEN_SEPARATE_FAMILIES",
        "A_EARLY_FUSION": ["BEST_SINGLE_FLIGHT_REFERENCE", "FUSION_PE", "FUSION_PEH", "FUSION_PEHF", "FULL_PRIMARY_REFERENCE"],
        "B_PERFORMANCE_SHORTCUT": ["WITHOUT_PERFORMANCE_PRIMARY_REFERENCE", "WITH_PERFORMANCE_AUXILIARY", "PERFORMANCE_ONLY_AUXILIARY"],
        "C_FLIGHT_PROVENANCE_SENSITIVITY": ["FLIGHT_FULL", "FLIGHT_BEHAVIORAL_ONLY", "FLIGHT_TASK_SETTING_ONLY"],
        "single_cross_family_ranking": "PROHIBITED",
    }
    metric_definitions = {
        "status": "FROZEN", "classification_primary": "Macro-F1", "regression_primary": "bounded MAE",
        "regression_task_wording": "bounded difficulty-induced workload proxy regression", "regression_clip_range": [1.0, 4.0],
        "subject_level_recomputation_required": True,
    }
    statistics = {
        "status": "FROZEN", "statistical_unit": "subject_id", "n": 35,
        "independent_units_prohibited": ["run", "fold", "seed"], "paired_subject_analysis": True,
        "bootstrap": {"repetitions": 2000, "seed": 42, "ci": "percentile 95%", "resampling": "paired subjects"},
        "test": "Wilcoxon signed-rank", "multiplicity": "Holm within each comparison family, model, and task",
        "effect_size": "rank-biserial", "alpha": 0.05, "save_nonsignificant_results": True,
        "unregistered_cross_family_tests": "PROHIBITED",
    }
    shortcut_evidence = {
        "status": "FROZEN", "automatic_leakage_threshold": None,
        "DIRECT_TARGET_LEAKAGE": {"allowed_values": ["YES", "NO"], "yes_only_if": "direct target copy or deterministic transform is found"},
        "ADDED_PERFORMANCE_INFORMATION": {"allowed_values": ["SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"], "evidence": ["numeric difference", "95% CI", "Holm-adjusted p", "rank-biserial effect"]},
        "PERFORMANCE_ONLY_PREDICTIVE_SIGNAL": {"report": ["absolute predictive performance", "difference from dummy baseline", "uncertainty"]},
        "FLIGHT_LABEL_PROXIMITY_RISK": {"allowed_values": ["BEHAVIORAL_SIGNAL_SUPPORTED", "TASK_STRUCTURE_SIGNAL_SUPPORTED", "MIXED_OR_AMBIGUOUS", "INCONCLUSIVE_DUE_TO_PROVENANCE", "NOT_FEASIBLE_EMPTY_GROUP"], "evidence": ["provenance categories", "behavioral-only performance", "task-setting-only performance", "ambiguous feature count", "Phase 09 scenario-holdout feasibility"]},
        "high_performance_alone_equals_leakage": False,
    }
    evidence_roles = {
        "PRIMARY_MAIN_EVIDENCE": {"dataset": "primary_without_performance", "feature_count": 1176, "paper_role": "ONLY_MAIN_RESULT"},
        "AUXILIARY_UPPER_BOUND_AND_SHORTCUT_ANALYSIS": {"dataset": "auxiliary_with_performance", "feature_count": 1235},
        "AUXILIARY_PERFORMANCE_ONLY_SHORTCUT_ANALYSIS": {"dataset": "performance_only", "feature_count": 59},
    }
    fusion_conditions = {
        "FUSION_PE": {"groups": ["physiological_features", "eye_tracking_features"], "feature_count": 649, "execution": "NEW_TRAINING_AUTHORIZED_AFTER_FREEZE"},
        "FUSION_PEH": {"groups": ["physiological_features", "eye_tracking_features", "head_movement_features"], "feature_count": 808, "execution": "NEW_TRAINING_AUTHORIZED_AFTER_FREEZE"},
        "FUSION_PEHF": {"groups": ["physiological_features", "eye_tracking_features", "head_movement_features", "flight_parameter_features"], "feature_count": 1134, "execution": "NEW_TRAINING_AUTHORIZED_AFTER_FREEZE"},
        "FULL_PRIMARY_REFERENCE": {"feature_count": 1176, "execution": "READ_ONLY_PHASE06_AND_PHASE04_OOF_NO_RETRAINING"},
        "BEST_SINGLE_FLIGHT_REFERENCE": {"feature_count": 326, "execution": "READ_ONLY_PHASE07_HDC_OOF_NO_RETRAINING"},
    }
    shortcut_conditions = {
        "WITHOUT_PERFORMANCE_PRIMARY_REFERENCE": {"feature_count": 1176, "execution": "READ_ONLY_UPSTREAM_FROZEN_OOF"},
        "WITH_PERFORMANCE_AUXILIARY": {"feature_count": 1235, "execution": "NEW_TRAINING_AUTHORIZED_AFTER_FREEZE"},
        "PERFORMANCE_ONLY_AUXILIARY": {"feature_count": 59, "execution": "NEW_TRAINING_AUTHORIZED_AFTER_FREEZE"},
    }
    model_matrix = {
        "status": "CONTRACT_FROZEN_NOT_TRAINED", "HDC": hdc, "traditional": traditional,
        "mandatory_new_conditions": mandatory_conditions, "flight_sensitivity_conditions": sensitivity,
        "traditional_flight_full": {"reusable_artifact_found": False, "reason": "Phase 07 frozen flight-only artifacts are HDC; Phase 04A/04B frozen OOF are full-primary, not flight-only.", "required_runs": traditional_flight_runs},
        "run_counts": {"core_required_model_runs": core_runs, "traditional_flight_full_required_runs": traditional_flight_runs, "unique_nonempty_nonaliased_flight_sensitivity_conditions": len(unique_sensitivity), "flight_sensitivity_model_runs": flight_sensitivity_runs, "expected_total_model_runs": expected_total_runs},
        "interfaces": {"HDC_classification": "PASS" if hdc_interface_pass else "FAIL", "HDC_regression": "PASS" if hdc_interface_pass else "FAIL", "traditional_classification": "PASS" if traditional_interface_pass else "FAIL", "traditional_regression": "PASS" if traditional_interface_pass else "FAIL"},
    }
    execution_manifest = {
        "status": "AUTHORIZED_NOT_EXECUTED", "expected_total_runs": expected_total_runs,
        "duplicate_run_identifiers": duplicate_run_identifiers, "run_records": run_records,
        "outer_folds": FOLDS, "seeds": SEEDS, "training_executed": False, "outer_test_predictions_generated": False, "oof_generated": False,
    }
    frozen_contract = {
        "phase": "08", "phase_name": "Fusion and Shortcut Analysis", "status": "CONTRACT_FROZEN_NOT_TRAINED", "frozen_at_utc": now_utc(),
        "scope": "CONTRACT_FREEZE_ONLY_NO_MODELING", "evidence_roles": evidence_roles,
        "fusion_conditions": fusion_conditions, "shortcut_conditions": shortcut_conditions, "flight_sensitivity": sensitivity,
        "core_fusion": "EARLY_FUSION", "late_fusion": "OPTIONAL_NOT_AUTHORIZED", "hdc_modality_aware_binding": "OPTIONAL_NOT_AUTHORIZED",
        "validation": {"outer_folds": "frozen subject-wise folds", "outer_training_fit_only": True, "outer_test_transform_predict_only": True, "retain_rows": 419, "train_test_subject_overlap": 0, "fold_regeneration": "PROHIBITED", "global_preprocessing": "PROHIBITED", "outer_test_tuning": "PROHIBITED"},
        "oof_aggregation": oof_rules, "comparison_families": comparison_families,
        "training_executed": False, "outer_test_predictions_generated": False,
    }
    write_json("configs/phase08_frozen_contract.json", frozen_contract)
    write_json("configs/phase08_model_matrix.json", model_matrix)
    write_json("configs/phase08_execution_manifest.json", execution_manifest)
    write_json("configs/phase08_metric_definitions.json", metric_definitions)
    write_json("configs/phase08_statistical_analysis_contract.json", statistics)
    write_json("configs/phase08_shortcut_evidence_contract.json", shortcut_evidence)

    experiment_contract = read_json(ROOT / "configs/phase08_experiment_contract.json")
    experiment_contract.update({"status": "CONTRACT_FROZEN_NOT_TRAINED", "scope": "CONTRACT_FREEZE_ONLY_NO_MODELING", "frozen_contract": "configs/phase08_frozen_contract.json", "training_authorized": False, "contract_freeze_completed": True})
    write_json("configs/phase08_experiment_contract.json", experiment_contract)

    result_files_after = file_count(ROOT / "results")
    upstream_after = {str(path): sha256(path) for path in upstream_paths}
    modified_upstream_files = sorted(path for path in upstream_before if upstream_before[path] != upstream_after[path])
    freeze_checks = {
        "gate_pass": gate["status"] == "PASS", "provenance_pass": provenance_audit["status"] == "PASS",
        "metadata_feasibility_pass": metadata_audit["status"] == "PASS", "hdc_interfaces_pass": hdc_interface_pass,
        "traditional_interfaces_pass": traditional_interface_pass, "run_count_dynamic_formula_pass": expected_total_runs == 300 + traditional_flight_runs + 60 * len(unique_sensitivity),
        "run_record_count_matches": len(run_records) == expected_total_runs, "duplicate_run_identifiers_zero": duplicate_run_identifiers == 0,
        "oof_rules_frozen": oof_rules["status"] == "FROZEN", "comparison_families_frozen": comparison_families["status"] == "FROZEN_SEPARATE_FAMILIES",
        "statistics_frozen": statistics["status"] == "FROZEN", "shortcut_rules_frozen": shortcut_evidence["status"] == "FROZEN",
        "phase09_handoff_saved": (ROOT / "manifests/phase08_to_phase09_generalization_handoff.json").is_file(),
        "training_artifacts_added_zero": result_files_after - result_files_before == 0,
        "upstream_files_modified_zero": len(modified_upstream_files) == 0,
        "training_executed_no": True, "outer_test_predictions_generated_no": True,
    }
    freeze_pass = all(freeze_checks.values())
    freeze_audit = {
        "status": "PASS" if freeze_pass else "FAIL", "checks": freeze_checks, "gate": gate,
        "flight_category_counts": provenance_manifest["category_counts"], "run_counts": model_matrix["run_counts"],
        "duplicate_run_identifiers": duplicate_run_identifiers, "result_files_before": result_files_before, "result_files_after": result_files_after,
        "training_artifacts_added": result_files_after - result_files_before, "training_executed": False, "outer_test_predictions_generated": False,
        "upstream_sha256_before": upstream_before, "upstream_sha256_after": upstream_after,
        "upstream_files_modified": len(modified_upstream_files), "modified_upstream_paths": modified_upstream_files,
    }
    write_json("audits/phase08_contract_freeze_audit.json", freeze_audit)
    pretraining = {
        "status": "PASS" if freeze_pass else "FAIL", "ready_for_execution_pending_tests_and_notebook": freeze_pass,
        "input_rows": 419, "subjects": 35, "outer_folds": 5, "expected_total_runs": expected_total_runs,
        "prohibited_operations_executed": [], "training_executed": False, "outer_test_predictions_generated": False,
    }
    write_json("audits/phase08_pretraining_feasibility_audit.json", pretraining)

    artifact_entries = []
    for relative in CONTRACT_ARTIFACTS:
        path = ROOT / relative
        artifact_entries.append({"path": relative, "exists": path.is_file(), "sha256": sha256(path) if path.is_file() else None})
    artifact_manifest = {
        "phase": "08", "status": "FROZEN", "created_at_utc": now_utc(), "artifacts": artifact_entries,
        "excluded_self_reference": "manifests/phase08_contract_artifact_manifest.json",
        "notebook_audited_separately": "audits/phase08_contract_notebook_persistence_audit.json",
    }
    write_json("manifests/phase08_contract_artifact_manifest.json", artifact_manifest)
    artifact_audit = {
        "status": "PENDING_NOTEBOOK_PERSISTENCE" if freeze_pass else "FAIL",
        "all_current_contract_artifacts_present": all(item["exists"] for item in artifact_entries),
        "artifact_count": len(artifact_entries), "manifest_path": "manifests/phase08_contract_artifact_manifest.json",
        "training_artifacts_added": result_files_after - result_files_before, "upstream_files_modified": len(modified_upstream_files),
        "modified_upstream_paths": modified_upstream_files,
        "training_executed": False, "outer_test_predictions_generated": False,
        "ready_for_execution": False,
    }
    write_json("audits/phase08_contract_artifact_audit.json", artifact_audit)
    summary = {
        "phase08_contract_freeze": "PASS" if freeze_pass else "FAIL", "checksums": {name: gate["actual_hashes"][name] == EXPECTED_HASHES[name] for name in EXPECTED_HASHES},
        "fusion_conditions_frozen": len(fusion_conditions), "shortcut_conditions_frozen": len(shortcut_conditions),
        "flight_features_audited": len(provenance_manifest["features"]), "flight_category_counts": provenance_manifest["category_counts"],
        "flight_provenance_audit": provenance_audit["status"], "flight_conditions": sensitivity,
        "interfaces": model_matrix["interfaces"], "run_counts": model_matrix["run_counts"], "duplicate_run_identifiers": duplicate_run_identifiers,
        "oof_aggregation_rules": oof_rules["status"], "comparison_families": "FROZEN", "statistical_rules": statistics["status"], "shortcut_evidence_rules": shortcut_evidence["status"],
        "holdout_feasibility": metadata_audit["holdout_feasibility"], "phase09_handoff_saved": True,
        "static_unit_tests": "PENDING", "contract_artifact_audit": artifact_audit["status"], "notebook_persistence": "PENDING",
        "model_training_executed": False, "outer_test_predictions_generated": False,
        "phase08_status": "CONTRACT_FROZEN_NOT_TRAINED" if freeze_pass else "FAIL", "ready_for_phase08_execution": False,
    }
    write_json("audits/phase08_contract_freeze_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(freeze_contract(), ensure_ascii=False, indent=2))
