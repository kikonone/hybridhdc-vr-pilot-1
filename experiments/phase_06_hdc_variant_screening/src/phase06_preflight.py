"""Read-only Phase 06 initialization and upstream-interface validation."""

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
from sklearn.model_selection import GroupKFold


EXPECTED_PRIMARY_SHA256 = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLD_SHA256 = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
CANONICAL_SELECTION = "NOT_PERFORMED_OUTER_TEST_SELECTION_PROHIBITED"
REQUIRED_VARIANTS = [
    "Vanilla Prototype HDC",
    "OnlineHD-style HDC",
    "Multi-centroid HDC",
    "HDC+OnlineHD Hybrid",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    exists = resolved.is_file()
    return {
        "path": str(resolved),
        "exists": exists,
        "file_size_bytes": resolved.stat().st_size if exists else None,
        "sha256": sha256(resolved) if exists else None,
        "result": "PASS" if exists else "FAIL",
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "最新完整实验计划_分类回归双任务.md").is_file() and (candidate / "experiments").is_dir():
            return candidate
    raise FileNotFoundError("Could not validate the hdc-vr-pilot project root")


def audit_primary_and_folds(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    phase03 = root / "experiments" / "phase_03_multimodal_dataset_labeling"
    primary_path = phase03 / "data" / "primary_without_performance.csv"
    folds_path = phase03 / "data" / "fold_assignments.csv"
    feature_manifest_path = phase03 / "manifests" / "primary_feature_manifest.json"

    primary_record = file_record(primary_path)
    folds_record = file_record(folds_path)
    feature_record = file_record(feature_manifest_path)
    primary = pd.read_csv(primary_path, low_memory=False)
    folds = pd.read_csv(folds_path)
    feature_manifest = load_json(feature_manifest_path)
    predictive_features = feature_manifest.get("features", [])

    primary_keys = set(primary["run_key"].astype(str))
    fold_keys = set(folds["run_key"].astype(str))
    coverage_pass = (
        len(primary) == len(folds) == 419
        and primary["run_key"].nunique() == folds["run_key"].nunique() == 419
        and primary_keys == fold_keys
    )
    row_alignment = primary[["run_key", "subject_id"]].merge(
        folds[["run_key", "subject_id", "outer_fold"]],
        on="run_key",
        how="outer",
        validate="one_to_one",
        suffixes=("_primary", "_fold"),
        indicator=True,
    )
    subject_alignment_pass = bool(
        (row_alignment["_merge"] == "both").all()
        and (row_alignment["subject_id_primary"] == row_alignment["subject_id_fold"]).all()
    )

    outer_details: list[dict[str, Any]] = []
    inner_details: list[dict[str, Any]] = []
    outer_isolation_pass = True
    inner_feasibility_pass = True
    for outer_fold in sorted(int(v) for v in folds["outer_fold"].unique()):
        train = folds.loc[folds["outer_fold"] != outer_fold].reset_index(drop=True)
        test = folds.loc[folds["outer_fold"] == outer_fold].reset_index(drop=True)
        overlap = sorted(set(train["subject_id"]) & set(test["subject_id"]))
        fold_outer_pass = len(overlap) == 0
        outer_isolation_pass &= fold_outer_pass
        outer_details.append(
            {
                "outer_fold": outer_fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_subjects": int(train["subject_id"].nunique()),
                "test_subjects": int(test["subject_id"].nunique()),
                "subject_overlap": overlap,
                "result": "PASS" if fold_outer_pass else "FAIL",
            }
        )

        split_details: list[dict[str, Any]] = []
        splitter = GroupKFold(n_splits=3)
        for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(
            splitter.split(train, groups=train["subject_id"]), start=1
        ):
            inner_train_subjects = set(train.iloc[inner_train_idx]["subject_id"])
            inner_valid_subjects = set(train.iloc[inner_valid_idx]["subject_id"])
            inner_overlap = sorted(inner_train_subjects & inner_valid_subjects)
            split_pass = len(inner_overlap) == 0 and len(inner_train_idx) > 0 and len(inner_valid_idx) > 0
            inner_feasibility_pass &= split_pass
            split_details.append(
                {
                    "inner_fold": inner_fold,
                    "train_rows": len(inner_train_idx),
                    "validation_rows": len(inner_valid_idx),
                    "subject_overlap": inner_overlap,
                    "result": "PASS" if split_pass else "FAIL",
                }
            )
        inner_details.append(
            {
                "outer_fold": outer_fold,
                "group_count": int(train["subject_id"].nunique()),
                "n_splits": 3,
                "splits": split_details,
                "result": "PASS" if all(item["result"] == "PASS" for item in split_details) else "FAIL",
            }
        )

    target_class_values = sorted(int(v) for v in primary["target_class"].dropna().unique())
    target_score_values = sorted(float(v) for v in primary["target_score"].dropna().unique())
    checks = {
        "modeling_rows_419": len(primary) == 419,
        "subjects_35": primary["subject_id"].nunique() == 35,
        "primary_predictive_features_1176": len(predictive_features) == 1176,
        "feature_manifest_columns_present": set(predictive_features).issubset(primary.columns),
        "unique_run_keys_419": primary["run_key"].nunique() == 419,
        "target_class_values": target_class_values == [0, 1, 2, 3],
        "target_class_missing_0": int(primary["target_class"].isna().sum()) == 0,
        "target_score_values": target_score_values == [1.0, 2.0, 3.0, 4.0],
        "target_score_missing_0": int(primary["target_score"].isna().sum()) == 0,
        "primary_checksum": primary_record["sha256"] == EXPECTED_PRIMARY_SHA256,
        "fold_rows_419": len(folds) == 419,
        "fold_unique_run_keys_419": folds["run_key"].nunique() == 419,
        "outer_folds_5": sorted(int(v) for v in folds["outer_fold"].unique()) == [1, 2, 3, 4, 5],
        "run_coverage_one_to_one": coverage_pass,
        "subject_alignment": subject_alignment_pass,
        "outer_subject_isolation": outer_isolation_pass,
        "inner_groupkfold_3_feasible": inner_feasibility_pass,
        "frozen_fold_checksum": folds_record["sha256"] == EXPECTED_FOLD_SHA256,
    }
    result = "PASS" if all(checks.values()) else "FAIL"
    audit = {
        "phase": "06",
        "audit": "input_and_fold",
        "timestamp_utc": utc_now(),
        "result": result,
        "checks": checks,
        "actual": {
            "modeling_rows": len(primary),
            "subjects": int(primary["subject_id"].nunique()),
            "primary_predictive_features": len(predictive_features),
            "unique_run_keys": int(primary["run_key"].nunique()),
            "target_class_values": target_class_values,
            "target_class_missing": int(primary["target_class"].isna().sum()),
            "target_score_values": target_score_values,
            "target_score_missing": int(primary["target_score"].isna().sum()),
            "fold_assignment_rows": len(folds),
            "fold_unique_run_keys": int(folds["run_key"].nunique()),
            "outer_folds": int(folds["outer_fold"].nunique()),
        },
        "outer_fold_details": outer_details,
        "inner_cv_details": inner_details,
        "evidence": {
            "primary_dataset": primary_record,
            "frozen_fold_assignments": folds_record,
            "primary_feature_manifest": feature_record,
        },
        "write_protection": {
            "phase03_files_modified": False,
            "primary_or_fold_copied_to_phase06": False,
            "outer_folds_regenerated": False,
        },
    }
    facts = {
        **audit["actual"],
        "primary_sha256": primary_record["sha256"],
        "primary_checksum": "PASS" if checks["primary_checksum"] else "FAIL",
        "frozen_fold_sha256": folds_record["sha256"],
        "frozen_fold_checksum": "PASS" if checks["frozen_fold_checksum"] else "FAIL",
        "outer_subject_isolation": "PASS" if outer_isolation_pass else "FAIL",
        "inner_groupkfold_3_feasibility": "PASS" if inner_feasibility_pass else "FAIL",
    }
    return audit, facts


def validate_manifest(phase_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    entries = manifest.get("artifacts", [])
    missing = 0
    size_mismatches = 0
    hash_mismatches = 0
    for entry in entries:
        candidate = phase_dir / entry.get("relative_path", "")
        if not candidate.is_file():
            missing += 1
            continue
        if candidate.stat().st_size != entry.get("file_size_bytes"):
            size_mismatches += 1
        if sha256(candidate) != entry.get("sha256"):
            hash_mismatches += 1
    passed = bool(entries) and not (missing or size_mismatches or hash_mismatches)
    return {
        "entries_checked": len(entries),
        "missing": missing,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "result": "PASS" if passed else "FAIL",
    }


def audit_phase05(root: Path, facts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    phase05 = root / "experiments" / "phase_05_basic_dual_output_hdc"
    rels = [
        "configs/phase05_freeze.json",
        "manifests/phase05_final_artifact_manifest.json",
        "audits/phase05_final_artifact_audit.json",
        "audits/phase05_final_reproducibility_audit.json",
        "audits/phase05_upstream_freeze_integrity_audit.json",
        "audits/phase05_final_oof_leakage_audit.json",
        "audits/phase05_final_notebook_persistence_audit.json",
        "reports/phase05_final_summary.md",
        "results/summaries/vanilla_hdc_final_confirmation_execution_summary.csv",
        "results/oof/vanilla_hdc_classification_oof.csv",
        "results/oof/vanilla_hdc_similarity_regression_oof.csv",
        "results/oof/vanilla_hdc_ridge_regression_oof.csv",
    ]
    evidence = {rel: file_record(phase05 / rel) for rel in rels}
    freeze = load_json(phase05 / rels[0])
    manifest = load_json(phase05 / rels[1])
    artifact_audit = load_json(phase05 / rels[2])
    reproducibility_audit = load_json(phase05 / rels[3])
    upstream_audit = load_json(phase05 / rels[4])
    leakage_audit = load_json(phase05 / rels[5])
    notebook_audit = load_json(phase05 / rels[6])
    execution = pd.read_csv(phase05 / rels[8])
    manifest_validation = validate_manifest(phase05, manifest)
    checks = {
        "required_files_exist": all(item["exists"] for item in evidence.values()),
        "status_frozen": freeze.get("status") == "FROZEN",
        "final_confirmation_folds_5_of_5": len(execution) == 5 and execution["outer_fold"].nunique() == 5,
        "final_confirmation_runs_100_of_100": int(execution["configs_completed"].sum()) == 100,
        "dimensions": freeze.get("dimensions") == [1000, 2000, 5000, 10000],
        "seeds": freeze.get("seeds") == [42, 43, 44, 45, 46],
        "levels_51": freeze.get("levels") == 51,
        "feature_k_50": freeze.get("feature_k") == 50,
        "classification_oof_exists": evidence[rels[9]]["exists"],
        "similarity_regression_oof_exists": evidence[rels[10]]["exists"],
        "ridge_regression_oof_exists": evidence[rels[11]]["exists"],
        "leakage_audit_pass": leakage_audit.get("result") == "PASS",
        "artifact_audit_pass": artifact_audit.get("result") == "PASS",
        "reproducibility_audit_pass": reproducibility_audit.get("result") == "PASS",
        "notebook_persistence_pass": notebook_audit.get("result") == "PASS",
        "manifest_parseable_and_all_entries_valid": manifest_validation["result"] == "PASS",
        "manifest_hash_matches_freeze": evidence[rels[1]]["sha256"] == freeze.get("final_artifact_manifest_sha256"),
        "primary_checksum_matches_actual": freeze.get("primary_data_sha256") == facts["primary_sha256"],
        "fold_checksum_matches_actual": freeze.get("frozen_fold_sha256") == facts["frozen_fold_sha256"],
        "upstream_primary_checksum_pass": upstream_audit.get("primary_checksum_pass") is True,
        "upstream_fold_checksum_pass": upstream_audit.get("frozen_fold_checksum_pass") is True,
        "canonical_selection_prohibition_recorded": freeze.get("canonical_configuration_selection") == CANONICAL_SELECTION,
    }
    result = "PASS" if all(checks.values()) else "FAIL"
    audit = {
        "phase": "06",
        "audit": "phase05_freeze_interface",
        "timestamp_utc": utc_now(),
        "result": result,
        "checks": checks,
        "actual": {
            "phase05_status": freeze.get("status"),
            "final_confirmation_folds": int(execution["outer_fold"].nunique()),
            "final_confirmation_runs": int(execution["configs_completed"].sum()),
            "dimensions": freeze.get("dimensions"),
            "seeds": freeze.get("seeds"),
            "levels": freeze.get("levels"),
            "feature_k": freeze.get("feature_k"),
            "canonical_configuration_selection": freeze.get("canonical_configuration_selection"),
        },
        "manifest_validation": manifest_validation,
        "evidence": evidence,
        "vanilla_baseline_policy": {
            "source": str(phase05.resolve()),
            "access": "READ_ONLY_REUSE",
            "retraining_executed": False,
            "outer_test_observed_best_used_for_phase06_selection": False,
        },
    }
    phase05_facts = {
        "status": freeze.get("status"),
        "freeze_interface": result,
        "vanilla_results_available": "YES" if all(checks[key] for key in [
            "classification_oof_exists", "similarity_regression_oof_exists", "ridge_regression_oof_exists"
        ]) else "NO",
        "canonical_configuration_selection": freeze.get("canonical_configuration_selection"),
    }
    return audit, phase05_facts


def audit_phase04(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    phase04a = root / "experiments" / "phase_04a_traditional_classification_baselines"
    phase04b = root / "experiments" / "phase_04b_traditional_regression_baselines"
    a_paths = {
        "freeze": phase04a / "configs" / "phase04a_freeze.json",
        "summary": phase04a / "results" / "summaries" / "classification_baseline_summary.csv",
        "oof": phase04a / "results" / "oof" / "classification_oof_predictions.csv",
        "report": phase04a / "reports" / "phase04a_final_summary.md",
        "leakage_audit": phase04a / "audits" / "phase04a_leakage_audit.json",
    }
    b_paths = {
        "freeze": phase04b / "configs" / "phase04b_freeze.json",
        "manifest": phase04b / "manifests" / "phase04b_final_artifact_manifest.json",
        "summary": phase04b / "results" / "summaries" / "gradient_boosting_summary.csv",
        "oof": phase04b / "results" / "predictions" / "gradient_boosting_oof.csv",
        "report": phase04b / "reports" / "phase04b_final_summary.md",
        "artifact_audit": phase04b / "audits" / "phase04b_final_artifact_audit.json",
        "leakage_audit": phase04b / "audits" / "phase04b_final_leakage_audit.json",
        "notebook_audit": phase04b / "audits" / "phase04b_final_notebook_persistence_audit.json",
    }
    a_evidence = {key: file_record(path) for key, path in a_paths.items()}
    b_evidence = {key: file_record(path) for key, path in b_paths.items()}
    a_freeze = load_json(a_paths["freeze"])
    a_leakage = load_json(a_paths["leakage_audit"])
    b_freeze = load_json(b_paths["freeze"])
    b_artifact = load_json(b_paths["artifact_audit"])
    b_leakage = load_json(b_paths["leakage_audit"])
    b_notebook = load_json(b_paths["notebook_audit"])
    load_json(b_paths["manifest"])
    pd.read_csv(a_paths["summary"])
    pd.read_csv(a_paths["oof"])
    pd.read_csv(b_paths["summary"])
    pd.read_csv(b_paths["oof"])
    a_checks = {
        "required_files_exist": all(item["exists"] for item in a_evidence.values()),
        "frozen": a_freeze.get("phase04a_frozen") == "YES",
        "complete": a_freeze.get("phase04a_status") == "COMPLETE",
        "leakage_audit_pass": a_leakage.get("status") == "PASS",
    }
    b_checks = {
        "required_files_exist": all(item["exists"] for item in b_evidence.values()),
        "frozen": b_freeze.get("status") == "FROZEN",
        "artifact_audit_pass": b_artifact.get("overall_pass") is True,
        "leakage_audit_pass": b_leakage.get("overall_pass") is True,
        "notebook_persistence_pass": b_notebook.get("overall_pass") is True,
    }
    a_result = "PASS" if all(a_checks.values()) else "FAIL"
    b_result = "PASS" if all(b_checks.values()) else "FAIL"
    audit = {
        "phase": "06",
        "audit": "phase04_baseline_interface",
        "timestamp_utc": utc_now(),
        "result": "PASS" if a_result == b_result == "PASS" else "FAIL",
        "phase04a": {"result": a_result, "checks": a_checks, "evidence": a_evidence},
        "phase04b": {"result": b_result, "checks": b_checks, "evidence": b_evidence},
        "policy": "READ_ONLY_COMPARISON_INTERFACE_NO_RECOMPUTATION",
    }
    return audit, {"phase04a": a_result, "phase04b": b_result}


def build_contract(root: Path, phase_dir: Path) -> dict[str, Any]:
    sources = {
        "full_experiment_plan": file_record(root / "最新完整实验计划_分类回归双任务.md"),
        "notebook_rules": file_record(root / "CODEX_NOTEBOOK_RULES.md"),
        "phase05_freeze": file_record(root / "experiments" / "phase_05_basic_dual_output_hdc" / "configs" / "phase05_freeze.json"),
    }
    return {
        "phase": "06",
        "phase_name": "HDC Variant Screening",
        "status": "PENDING_CONTRACT_FREEZE",
        "result": "PASS" if all(item["exists"] for item in sources.values()) else "FAIL",
        "required_variants": REQUIRED_VARIANTS,
        "classification_primary_metric": "Macro-F1",
        "regression_primary_metric": "MAE",
        "outer_cv": "frozen Phase 03 five-fold subject-wise split",
        "inner_cv": "GroupKFold(n_splits=3, groups=subject_id)",
        "primary_dataset": "primary_without_performance.csv",
        "vanilla_baseline_source": "frozen Phase 05",
        "canonical_phase05_configuration_selection": "not performed",
        "canonical_phase05_configuration_selection_record": CANONICAL_SELECTION,
        "model_training_executed": False,
        "contract_freeze_required_before_modeling": True,
        "interpretation": "bounded difficulty-induced workload proxy regression",
        "project_claim_scope": "workload-proxy classification and regression",
        "deferred_to_contract_freeze": [
            "OnlineHD update formula and learning-rate grid",
            "low-confidence definition",
            "prototype update count",
            "Multi-centroid center counts and initialization",
            "empty-center handling",
            "Hybrid initialization and update order",
            "quick-screen space",
            "final-confirmation space",
            "variant elimination rule",
            "Pareto selection rule",
            "canonical configuration selection rule",
        ],
        "source_evidence": sources,
        "phase_directory": str(phase_dir.resolve()),
    }


def build_environment(root: Path) -> dict[str, Any]:
    evidence = {
        "python_executable": file_record(Path(sys.executable)),
        "requirements": file_record(root / "requirements.txt"),
    }
    return {
        "phase": "06",
        "artifact": "environment",
        "timestamp_utc": utc_now(),
        "result": "PASS" if all(item["exists"] for item in evidence.values()) else "FAIL",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "working_directory": os.getcwd(),
        "package_versions": {
            "pandas": pd.__version__,
        },
        "evidence": evidence,
    }


def build_upstream_interface(root: Path) -> dict[str, Any]:
    paths = {
        "phase03_primary": root / "experiments" / "phase_03_multimodal_dataset_labeling" / "data" / "primary_without_performance.csv",
        "phase03_folds": root / "experiments" / "phase_03_multimodal_dataset_labeling" / "data" / "fold_assignments.csv",
        "phase04a_freeze": root / "experiments" / "phase_04a_traditional_classification_baselines" / "configs" / "phase04a_freeze.json",
        "phase04b_freeze": root / "experiments" / "phase_04b_traditional_regression_baselines" / "configs" / "phase04b_freeze.json",
        "phase05_freeze": root / "experiments" / "phase_05_basic_dual_output_hdc" / "configs" / "phase05_freeze.json",
    }
    evidence = {key: file_record(path) for key, path in paths.items()}
    return {
        "phase": "06",
        "artifact": "upstream_interface",
        "timestamp_utc": utc_now(),
        "result": "PASS" if all(item["exists"] for item in evidence.values()) else "FAIL",
        "access_mode": "READ_ONLY_REFERENCE",
        "copied_inputs": [],
        "phase03": {"role": "frozen data and outer folds"},
        "phase04a": {"role": "frozen classification baseline comparison"},
        "phase04b": {"role": "frozen regression baseline comparison"},
        "phase05": {"role": "frozen Vanilla Prototype HDC baseline"},
        "evidence": evidence,
    }


def make_input_manifest(audits: list[dict[str, Any]]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for audit in audits:
        for key, value in audit.get("evidence", {}).items():
            records[f"{audit['audit']}::{key}"] = value
        for phase_key in ("phase04a", "phase04b"):
            for key, value in audit.get(phase_key, {}).get("evidence", {}).items():
                records[f"{audit['audit']}::{phase_key}::{key}"] = value
    unique = {record["path"]: record for record in records.values()}
    return {
        "phase": "06",
        "manifest": "input_manifest",
        "timestamp_utc": utc_now(),
        "result": "PASS" if unique and all(item["result"] == "PASS" for item in unique.values()) else "FAIL",
        "access_mode": "READ_ONLY",
        "input_count": len(unique),
        "inputs": list(unique.values()),
        "primary_or_fold_files_copied": False,
    }


def build_initialization_audit(phase_dir: Path, include_notebook: bool) -> dict[str, Any]:
    required = [
        "README.md",
        "configs/phase06_experiment_contract.json",
        "configs/phase06_environment.json",
        "configs/phase06_upstream_interface.json",
        "manifests/phase06_input_manifest.json",
        "audits/phase06_input_and_fold_audit.json",
        "audits/phase06_phase05_freeze_interface_audit.json",
        "audits/phase06_phase04_baseline_interface_audit.json",
        "src/phase06_preflight.py",
        "scripts/initialize_phase06.py",
        "scripts/build_phase06_notebook.py",
        "tests/test_phase06_preflight.py",
    ]
    if include_notebook:
        required.extend([
            "Phase_06_HDC_Variant_Screening.ipynb",
            "audits/phase06_notebook_persistence_audit.json",
        ])
    evidence = {rel: file_record(phase_dir / rel) for rel in required}
    expected_dirs = [
        "data", "manifests", "audits", "configs", "src", "scripts", "tests", "figures", "logs", "reports",
        "results/checkpoints", "results/predictions", "results/fold_metrics", "results/oof", "results/summaries", "results/efficiency",
    ]
    directory_checks = {rel: (phase_dir / rel).is_dir() for rel in expected_dirs}
    result = "PASS" if all(item["exists"] for item in evidence.values()) and all(directory_checks.values()) else "FAIL"
    return {
        "phase": "06",
        "audit": "initialization_artifact",
        "timestamp_utc": utc_now(),
        "result": result,
        "finalized_after_notebook_execution": include_notebook,
        "checks": {
            "required_files_exist_and_hashed": all(item["exists"] for item in evidence.values()),
            "required_directories_exist": all(directory_checks.values()),
            "model_training_executed": False,
            "phase03_phase04_phase05_files_modified": False,
            "primary_or_fold_files_copied": False,
        },
        "directory_checks": directory_checks,
        "evidence": evidence,
        "self_hash_exclusion": "This audit cannot contain its own SHA-256 without self-reference.",
    }


def run_preflight(start: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    """Validate frozen upstream inputs and write initialization evidence.

    ``start`` identifies the real project/Phase 06 source tree used for read-only
    validation.  ``output_dir`` may point at an isolated test tree; production is
    used only when callers deliberately omit it.
    """
    phase_dir = (start or Path(__file__).resolve().parents[1]).resolve()
    evidence_dir = (output_dir or phase_dir).resolve()
    root = find_project_root(phase_dir)
    input_audit, facts = audit_primary_and_folds(root)
    phase05_audit, phase05_facts = audit_phase05(root, facts)
    phase04_audit, phase04_facts = audit_phase04(root)
    contract = build_contract(root, phase_dir)
    environment = build_environment(root)
    upstream = build_upstream_interface(root)
    manifest = make_input_manifest([input_audit, phase05_audit, phase04_audit])

    outputs = {
        evidence_dir / "configs" / "phase06_experiment_contract.json": contract,
        evidence_dir / "configs" / "phase06_environment.json": environment,
        evidence_dir / "configs" / "phase06_upstream_interface.json": upstream,
        evidence_dir / "manifests" / "phase06_input_manifest.json": manifest,
        evidence_dir / "audits" / "phase06_input_and_fold_audit.json": input_audit,
        evidence_dir / "audits" / "phase06_phase05_freeze_interface_audit.json": phase05_audit,
        evidence_dir / "audits" / "phase06_phase04_baseline_interface_audit.json": phase04_audit,
    }
    for path, payload in outputs.items():
        write_json(path, payload)
    initialization_audit = build_initialization_audit(evidence_dir, include_notebook=False)
    write_json(evidence_dir / "audits" / "phase06_initialization_artifact_audit.json", initialization_audit)

    all_gates_pass = all(
        item == "PASS"
        for item in [input_audit["result"], phase05_audit["result"], phase04_audit["result"], contract["result"], environment["result"], upstream["result"], manifest["result"], initialization_audit["result"]]
    )
    return {
        "phase06_directory_initialized": "YES" if initialization_audit["result"] == "PASS" else "NO",
        "phase06_name": "HDC Variant Screening",
        **facts,
        "phase05": phase05_facts,
        "phase04": phase04_facts,
        "required_hdc_variants": REQUIRED_VARIANTS,
        "input_manifest_saved": "YES" if manifest["result"] == "PASS" else "NO",
        "initialization_audit": initialization_audit["result"],
        "model_training_executed": "NO",
        "phase06_status": "PENDING_CONTRACT_FREEZE" if all_gates_pass else "FAIL",
        "ready_for_contract_freeze_before_notebook_gate": all_gates_pass,
    }


def finalize_after_notebook(phase_dir: Path) -> dict[str, Any]:
    notebook_path = phase_dir / "Phase_06_HDC_Variant_Screening.ipynb"
    notebook_record = file_record(notebook_path)
    notebook_data = load_json(notebook_path)
    code_cells = [cell for cell in notebook_data.get("cells", []) if cell.get("cell_type") == "code"]
    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]
    text = "\n".join(
        "".join(output.get("text", [])) if isinstance(output.get("text", []), list) else str(output.get("text", ""))
        for output in outputs
    )
    checks = {
        "notebook_exists_and_hashed": notebook_record["result"] == "PASS",
        "code_cells_executed": bool(code_cells) and all(cell.get("execution_count") is not None for cell in code_cells),
        "outputs_persisted": bool(outputs),
        "no_error_outputs": not any(output.get("output_type") == "error" for output in outputs),
        "training_no_marker": "HDC VARIANT TRAINING EXECUTED: NO" in text,
        "pending_contract_freeze_marker": "PHASE 06 STATUS: PENDING_CONTRACT_FREEZE" in text,
        "all_required_audit_markers": all(marker in text for marker in [
            "INPUT AND FOLD AUDIT: PASS",
            "PHASE 05 FREEZE INTERFACE AUDIT: PASS",
            "PHASE 04 BASELINE INTERFACE AUDIT: PASS",
            "INITIALIZATION ARTIFACT AUDIT: PASS",
        ]),
    }
    notebook_audit = {
        "phase": "06",
        "audit": "notebook_persistence",
        "timestamp_utc": utc_now(),
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "evidence": {"notebook": notebook_record},
        "model_training_executed": False,
    }
    write_json(phase_dir / "audits" / "phase06_notebook_persistence_audit.json", notebook_audit)
    initialization_audit = build_initialization_audit(phase_dir, include_notebook=True)
    initialization_audit["checks"]["notebook_persistence"] = notebook_audit["result"] == "PASS"
    if notebook_audit["result"] != "PASS":
        initialization_audit["result"] = "FAIL"
    write_json(phase_dir / "audits" / "phase06_initialization_artifact_audit.json", initialization_audit)
    return {"notebook_persistence": notebook_audit["result"], "initialization_audit": initialization_audit["result"]}
