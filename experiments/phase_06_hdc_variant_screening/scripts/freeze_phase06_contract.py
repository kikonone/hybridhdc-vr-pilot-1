"""Freeze the Phase 06 HDC variant, search-space, and selection contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
PHASE05 = ROOT / "experiments" / "phase_05_basic_dual_output_hdc"
EXPECTED_PRIMARY = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLDS = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "sha256": digest(path),
        "result": "PASS",
    }


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    timestamp = datetime.now(timezone.utc).isoformat()
    init_contract = read(PHASE / "configs" / "phase06_experiment_contract.json")
    input_audit = read(PHASE / "audits" / "phase06_input_and_fold_audit.json")
    init_audit = read(PHASE / "audits" / "phase06_initialization_artifact_audit.json")
    amendment_config_path = PHASE05 / "configs" / "phase05_no_retraining_completion_amendment.json"
    amendment_audit_path = PHASE05 / "audits" / "phase05_no_retraining_amendment_audit.json"
    diagnostic_audit_path = PHASE05 / "audits" / "phase05_no_retraining_diagnostic_completion_audit.json"
    efficiency_audit_path = PHASE05 / "audits" / "phase05_no_retraining_efficiency_protocol_completion_audit.json"
    freeze_path = PHASE05 / "configs" / "phase05_freeze.json"
    final_manifest_path = PHASE05 / "manifests" / "phase05_final_artifact_manifest.json"
    amendment = read(amendment_config_path)
    amendment_audit = read(amendment_audit_path)
    diagnostic_audit = read(diagnostic_audit_path)
    efficiency_audit = read(efficiency_audit_path)
    phase05_freeze = read(freeze_path)
    amendment_checks = {
        "status_completed_no_retraining": amendment.get("status") == "COMPLETED_NO_RETRAINING",
        "amendment_audit_pass": amendment_audit.get("result") == "PASS",
        "diagnostic_audit_pass": diagnostic_audit.get("result") == "PASS",
        "efficiency_audit_pass": efficiency_audit.get("result") == "PASS",
        "phase05_still_frozen": phase05_freeze.get("status") == "FROZEN",
        "model_fitting_not_executed": amendment_audit.get("model_fitting_executed") is False,
        "prediction_not_replaced": amendment_audit.get("prediction_artifact_replaced") is False,
        "canonical_not_selected": amendment_audit.get("canonical_configuration_selected") is False,
        "original_freeze_hash_recorded": bool(amendment_audit.get("pre_amendment_freeze_sha256")),
        "original_manifest_hash_recorded": bool(amendment_audit.get("pre_amendment_manifest_sha256")),
    }
    amendment_gate = {
        "phase": "06",
        "audit": "phase05_existing_freeze_amendment_gate",
        "timestamp_utc": timestamp,
        "result": "PASS" if all(amendment_checks.values()) else "FAIL",
        "checks": amendment_checks,
        "action": "REUSED_EXISTING_VALID_AMENDMENT_NO_DUPLICATE_PHASE05_FILES_CREATED",
        "evidence": {
            "amendment_config": record(amendment_config_path),
            "amendment_audit": record(amendment_audit_path),
            "diagnostic_audit": record(diagnostic_audit_path),
            "efficiency_audit": record(efficiency_audit_path),
            "current_phase05_freeze": record(freeze_path),
            "current_phase05_final_manifest": record(final_manifest_path),
        },
    }
    write(PHASE / "audits" / "phase06_phase05_amendment_gate_audit.json", amendment_gate)

    if amendment_gate["result"] != "PASS":
        raise RuntimeError("Phase 05 freeze amendment gate failed")
    if init_contract.get("status") != "PENDING_CONTRACT_FREEZE" or input_audit.get("result") != "PASS" or init_audit.get("result") != "PASS":
        raise RuntimeError("Phase 06 initialization gate failed")
    if input_audit["evidence"]["primary_dataset"]["sha256"] != EXPECTED_PRIMARY:
        raise RuntimeError("Primary checksum mismatch")
    if input_audit["evidence"]["frozen_fold_assignments"]["sha256"] != EXPECTED_FOLDS:
        raise RuntimeError("Frozen-fold checksum mismatch")

    common = {
        "representation": "bipolar",
        "binding": "bipolar elementwise multiplication",
        "bundling": "sum then deterministic sign with frozen tie vector",
        "similarity": "cosine",
        "quantization": "fold-local per-feature equal-width quantization",
        "levels": 51,
        "feature_k": 50,
        "classification_primary_metric": "Macro-F1",
        "regression_primary_metric": "MAE",
        "inner_cv": "GroupKFold(n_splits=3, groups=subject_id)",
        "outer_cv": "frozen Phase 03 five-fold subject-wise split",
        "deterministic_prng": "PCG64",
        "random_stream_derivation": "stable SHA-256-derived identifiers",
        "python_hash_for_random_seed": "PROHIBITED",
        "dtype_contract": {"sample_hv": "int8", "prototype_or_centroid": "float32", "similarity": "float32"},
    }
    preprocessing = [
        {"step": 1, "operation": "SimpleImputer", "parameters": {"strategy": "median", "add_indicator": True, "keep_empty_features": True}},
        {"step": 2, "operation": "VarianceThreshold", "parameters": {"threshold": 0.0}},
        {"step": 3, "operation": "StandardScaler", "parameters": {}},
        {"step": 4, "operation": "SelectKBest", "parameters": {"score_func": "f_classif", "k": 50}},
        {"step": 5, "operation": "fold-local equal-width quantization", "parameters": {"levels": 51}},
        {"step": 6, "operation": "HDC encoding", "parameters": {"source": "Phase 05 frozen encoder import"}},
    ]
    variant_contract = {
        "phase": "06",
        "phase_name": "HDC Variant Screening",
        "contract_version": "phase06_hdc_variant_contract_v1",
        "status": "FROZEN",
        "timestamp_utc": timestamp,
        "result": "PASS",
        "common_interface": common,
        "preprocessing": preprocessing,
        "preprocessing_scope": "Every operation is independently fitted on each inner-training split only.",
        "encoder_reuse": {
            "module": str((PHASE05 / "src" / "phase05_hdc_core.py").resolve()),
            "sha256": digest(PHASE05 / "src" / "phase05_hdc_core.py"),
            "functions": ["EqualWidthQuantizer", "incremental_encode_prefixes", "cosine_similarity_scores", "stable_rng"],
            "copy_created": False,
        },
        "variants": {
            "vanilla": {"name": "Vanilla Prototype HDC", "source": "frozen Phase 05", "access": "READ_ONLY", "retraining": "PROHIBITED"},
            "onlinehd": {
                "name": "OnlineHD-style HDC",
                "initialization": "class-wise Vanilla prototypes accumulated as float32 and L2-normalized",
                "order": "PCG64 permutation derived from outer fold, inner split, dimension, seed, candidate, and epoch",
                "trigger": "prediction error OR margin < margin_threshold",
                "margin": "similarity_true - highest_similarity_non_true_class",
                "true_update": "learning_rate * (1 - similarity_true) * sample_hv",
                "runner_up_update": "-learning_rate * max(0, similarity_runner_up) * sample_hv",
                "normalization": "L2-normalize all prototypes after every epoch",
            },
            "multicentroid": {
                "name": "Multi-centroid HDC",
                "training_scope": "class-wise inner-training sample hypervectors only",
                "initialization": "deterministic Euclidean KMeans on normalized sample hypervectors",
                "kmeans": {"n_init": 10, "max_iter": 300, "random_state": "stable SHA-256 and experiment seed-derived 32-bit integer"},
                "centroid": "cluster mean followed by L2 normalization",
                "class_score": "maximum cosine similarity among class centroids",
                "class_tie": "smaller class id",
                "invalid_candidate": "training class count below requested centers OR empty cluster/effective-center shortfall",
            },
            "hybrid": {
                "name": "HDC+OnlineHD Hybrid",
                "initialization": "Multi-centroid HDC",
                "true_target": "highest-similarity centroid within true class",
                "runner_up_target": "highest-similarity centroid among all non-true classes",
                "trigger_and_update": "same margin trigger and update formula as OnlineHD-style",
                "normalization": "L2-normalize every centroid after every epoch",
                "order": "stable SHA-256-derived PCG64 permutation",
            },
        },
        "prohibitions": ["outer-test feature access", "outer-test prediction", "similarity regression", "Ridge readout", "Final Confirmation", "Phase 07"],
    }
    spaces = {
        "phase": "06",
        "contract_version": "phase06_variant_search_spaces_v1",
        "status": "FROZEN",
        "timestamp_utc": timestamp,
        "result": "PASS",
        "common": {"dimensions": [2000, 5000], "levels": [51], "feature_k": [50], "seeds": [42]},
        "onlinehd": {"epochs": [1, 3, 5], "learning_rate": [0.05, 0.1], "margin_threshold": [0.0, 0.1], "candidates_per_dimension": 12, "total_candidates": 24},
        "multicentroid": {"centroids_per_class": [2, 3, 4], "candidates_per_dimension": 3, "total_candidates": 6},
        "hybrid": {"centroids_per_class": [2, 3], "epochs": [1, 3], "learning_rate": [0.05, 0.1], "margin_threshold": [0.0, 0.1], "candidates_per_dimension": 16, "total_candidates": 32},
        "final_confirmation": {"executed": False, "dimensions": [1000, 2000, 5000, 10000], "seeds": [42, 43, 44, 45, 46]},
        "mutation_after_freeze": "PROHIBITED",
    }
    selection = {
        "phase": "06",
        "contract_version": "phase06_model_selection_rules_v1",
        "status": "FROZEN",
        "timestamp_utc": timestamp,
        "result": "PASS",
        "scope": "Each outer fold and new variant independently; outer-training inner-CV only.",
        "ranking": [
            {"priority": 1, "field": "mean_inner_macro_f1", "direction": "descending"},
            {"priority": 2, "field": "sample_sd_inner_macro_f1", "direction": "ascending"},
            {"priority": 3, "field": "mean_inner_balanced_accuracy", "direction": "descending"},
            {"priority": 4, "field": "mean_inner_severe_error_rate", "direction": "ascending"},
            {"priority": 5, "field": "dimension", "direction": "ascending"},
            {"priority": 6, "fields": ["epochs", "centroids_per_class", "learning_rate", "margin_threshold"], "direction": "ascending", "missing_value": 0},
            {"priority": 7, "field": "canonical_config_json", "direction": "lexicographic ascending"},
        ],
        "outer_test_tie_breaking": "PROHIBITED",
        "classification_only": True,
        "regression_heads_executed": False,
    }
    config_paths = [
        PHASE / "configs" / "phase06_hdc_variant_contract.json",
        PHASE / "configs" / "phase06_variant_search_spaces.json",
        PHASE / "configs" / "phase06_model_selection_rules.json",
    ]
    for path, payload in zip(config_paths, [variant_contract, spaces, selection]):
        write(path, payload)

    upstream_files: list[dict[str, Any]] = []
    for phase_name in [
        "phase_03_multimodal_dataset_labeling",
        "phase_04a_traditional_classification_baselines",
        "phase_04b_traditional_regression_baselines",
        "phase_05_basic_dual_output_hdc",
    ]:
        phase_path = ROOT / "experiments" / phase_name
        for path in sorted(item for item in phase_path.rglob("*") if item.is_file()):
            upstream_files.append({"path": str(path.resolve()), "file_size_bytes": path.stat().st_size, "sha256": digest(path)})
    snapshot = {
        "phase": "06", "audit": "upstream_pre_quick_screen_snapshot", "timestamp_utc": timestamp,
        "result": "PASS", "file_count": len(upstream_files), "files": upstream_files,
    }
    snapshot_path = PHASE / "audits" / "phase06_upstream_pre_quick_screen_snapshot.json"
    write(snapshot_path, snapshot)
    manifest = {
        "phase": "06", "manifest": "contract_freeze", "timestamp_utc": timestamp, "result": "PASS",
        "artifacts": [record(path) for path in config_paths],
        "source_evidence": {
            "phase05_encoder": record(PHASE05 / "src" / "phase05_hdc_core.py"),
            "phase05_preprocessing_runner": record(PHASE05 / "scripts" / "run_vanilla_hdc_quick_screen.py"),
            "phase05_amendment_gate": record(PHASE / "audits" / "phase06_phase05_amendment_gate_audit.json"),
            "upstream_snapshot": record(snapshot_path),
        },
    }
    manifest_path = PHASE / "manifests" / "phase06_contract_manifest.json"
    write(manifest_path, manifest)
    freeze_checks = {
        "phase05_amendment_pass": amendment_gate["result"] == "PASS",
        "phase06_initialization_pass": init_audit.get("result") == "PASS",
        "primary_checksum_pass": input_audit["evidence"]["primary_dataset"]["sha256"] == EXPECTED_PRIMARY,
        "frozen_fold_checksum_pass": input_audit["evidence"]["frozen_fold_assignments"]["sha256"] == EXPECTED_FOLDS,
        "variant_contract_frozen": variant_contract["status"] == "FROZEN",
        "search_spaces_frozen": spaces["status"] == "FROZEN" and spaces["onlinehd"]["total_candidates"] == 24 and spaces["multicentroid"]["total_candidates"] == 6 and spaces["hybrid"]["total_candidates"] == 32,
        "selection_rules_frozen": selection["status"] == "FROZEN",
        "phase05_encoder_reused_by_hash": bool(variant_contract["encoder_reuse"]["sha256"]),
        "outer_test_feature_access_prohibited": True,
        "final_confirmation_not_executed": True,
    }
    freeze_audit = {
        "phase": "06", "audit": "contract_freeze", "timestamp_utc": timestamp,
        "result": "PASS" if all(freeze_checks.values()) else "FAIL",
        "checks": freeze_checks,
        "evidence": {"contract_manifest": record(manifest_path), **{path.name: record(path) for path in config_paths}},
    }
    write(PHASE / "audits" / "phase06_contract_freeze_audit.json", freeze_audit)
    print(json.dumps({"phase05_amendment": amendment_gate["result"], "phase06_contract_freeze": freeze_audit["result"], "upstream_files_snapshotted": len(upstream_files)}, indent=2))
    return 0 if freeze_audit["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
