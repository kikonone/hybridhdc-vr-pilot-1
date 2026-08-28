"""Canonical Phase 09 OOF consolidation from frozen raw predictions only."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PHASE09 = Path(__file__).resolve().parents[1]
EXPERIMENTS = PHASE09.parent
PHASE03 = EXPERIMENTS / "phase_03_multimodal_dataset_labeling"
PHASE04A = EXPERIMENTS / "phase_04a_traditional_classification_baselines"
PHASE04B = EXPERIMENTS / "phase_04b_traditional_regression_baselines"
PHASE05 = EXPERIMENTS / "phase_05_basic_dual_output_hdc"
PHASE06 = EXPERIMENTS / "phase_06_hdc_variant_screening"
MANIFEST_PATH = PHASE09 / "configs" / "phase09_execution_manifest.json"
SEEDS = {42, 43, 44, 45, 46}
MODEL_KEYS = [
    "hdc_classification", "hdc_regression",
    "traditional_classification", "traditional_regression",
]
MISSING_CONDITIONS = [
    "MISSING_PHYSIOLOGICAL", "MISSING_EYE_TRACKING", "MISSING_HEAD_MOVEMENT",
    "MISSING_FLIGHT_PARAMETER", "MISSING_BODY_MOVEMENT",
]
EXPECTED_PRIMARY = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLDS = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"

sys.path.insert(0, str(PHASE09 / "scripts"))
from run_phase09_batch import (  # noqa: E402
    FROZEN_CONTRACT_PATHS,
    atomic_csv,
    atomic_json,
    output_paths,
    read_json,
    reusable_run,
    sha256,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    """Convert NumPy scalars used by validation dictionaries to JSON scalars."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def metadata() -> pd.DataFrame:
    frame = pd.read_csv(PHASE03 / "data" / "fold_assignments.csv")
    required = ["run_key", "subject_id", "outer_fold", "target_class", "target_score", "difficulty_level"]
    if len(frame) != 419 or frame.run_key.nunique() != 419 or not set(required).issubset(frame.columns):
        raise RuntimeError("Frozen Phase 03 assignment coverage is invalid")
    return frame[required].copy()


def execution_preflight() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    required_audits = [
        "phase09_checkpoint_integrity_audit.json", "phase09_execution_coverage_audit.json",
        "phase09_execution_leakage_audit.json", "phase09_feature_exclusion_audit.json",
        "phase09_execution_artifact_audit.json", "phase09_config_mapping_leakage_audit.json",
    ]
    audit_statuses = {
        name: read_json(PHASE09 / "audits" / name)["status"] for name in required_audits
    }
    records = manifest["training_runs"]
    invalid = [record["run_identifier"] for record in records if not reusable_run(record, output_paths(record))]
    contract_statuses = {
        relative: (PHASE09 / relative).exists() for relative in FROZEN_CONTRACT_PATHS
    }
    checks = {
        "status_ready": manifest["status"] in {
            "EXECUTION_COMPLETE_PENDING_OOF_CONSOLIDATION", "OOF_CONSOLIDATED_PENDING_ANALYSIS",
            "ANALYSIS_COMPLETE_PENDING_FINAL_VERIFICATION", "ANALYSIS_COMPLETE_PENDING_NOTEBOOK",
            "ANALYSIS_COMPLETE_PENDING_FREEZE",
        },
        "runs_720": manifest.get("completed_training_runs") == 720 and len(records) == 720,
        "raw_rows_30168": manifest.get("raw_prediction_rows") == 30168,
        "protocol_counts": manifest.get("run_counts_by_protocol") == {
            "RETRAIN_WITHOUT_MODALITY": 300, "LEAVE_ONE_SUBJECT_OUT": 420,
        },
        "model_counts": manifest.get("run_counts_by_model") == {
            "hdc_classification": 300, "hdc_regression": 300,
            "traditional_classification": 60, "traditional_regression": 60,
        },
        "execution_audits_pass": all(value == "PASS" for value in audit_statuses.values()),
        "all_checkpoints_and_predictions_hash_valid": not invalid,
        "all_contract_files_exist": all(contract_statuses.values()),
        "primary_checksum": sha256(PHASE03 / "data" / "primary_without_performance.csv") == EXPECTED_PRIMARY,
        "fold_checksum": sha256(PHASE03 / "data" / "fold_assignments.csv") == EXPECTED_FOLDS,
        "performance_features_included_no": not read_json(PHASE09 / "audits" / "phase09_feature_exclusion_audit.json")["performance_features_included"],
        "full_reference_retrained_no": not manifest.get("full_primary_reference_counted_as_training", True),
        "test_time_missingness_no": not manifest.get("sudden_test_time_missingness_counted_as_training", True),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "invalid_runs": invalid[:10], "audits": audit_statuses})
    return {
        "checks": checks, "records": records, "audit_statuses": audit_statuses,
        "invalid_runs": invalid, "contract_files": contract_statuses,
    }


def load_raw(records: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for record in records:
        path = output_paths(record)["prediction"]
        frame = pd.read_csv(path)
        frame["model_key"] = record["model_key"]
        frame["source_prediction_path"] = str(path.resolve())
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if len(combined) != 30168:
        raise RuntimeError(f"Raw row count {len(combined)} != 30168")
    return combined


def _constant(group: pd.DataFrame, column: str) -> Any:
    values = group[column].drop_duplicates()
    if len(values) != 1:
        raise RuntimeError(f"Non-constant {column} for {group.run_key.iloc[0]}")
    return values.iloc[0]


def aggregate_group(group: pd.DataFrame, model_key: str) -> dict[str, Any]:
    base = {
        "run_key": _constant(group, "run_key"),
        "subject_id": _constant(group, "subject_id"),
        "source_raw_rows": len(group),
    }
    if model_key.startswith("hdc_"):
        seed_values = set(pd.to_numeric(group.seed, errors="raise").astype(int))
        if len(group) != 5 or seed_values != SEEDS:
            raise RuntimeError(f"Incomplete five-seed group {base['run_key']}: {seed_values}")
        base["aggregation"] = "FIVE_SEED_CLASS_SCORE_MEAN_ARGMAX" if model_key.endswith("classification") else "FIVE_SEED_RAW_MEAN_THEN_CLIP"
        base["seed_count"] = 5
        base["seeds"] = "42,43,44,45,46"
    else:
        if len(group) != 1:
            raise RuntimeError(f"Traditional canonical group is not unique: {base['run_key']}")
        base["aggregation"] = "CANONICAL_FROZEN_PREDICTION"
        base["seed_count"] = 1
        base["seeds"] = "canonical"

    if model_key.endswith("classification"):
        base["y_true"] = int(_constant(group, "y_true"))
        scores = [float(group[f"class_score_{label}"].mean()) for label in range(4)]
        base.update({f"class_score_{label}": scores[label] for label in range(4)})
        base["y_pred"] = int(np.argmax(np.asarray(scores, dtype=float)))
    else:
        base["y_true"] = float(_constant(group, "y_true"))
        raw = float(group["y_pred_raw"].mean())
        base["y_pred_raw"] = raw
        base["y_pred_bounded"] = float(np.clip(raw, 1.0, 4.0))
        base["clipped"] = bool(raw < 1.0 or raw > 4.0)
    return base


def consolidate_protocol(raw: pd.DataFrame, protocol: str, meta: pd.DataFrame) -> pd.DataFrame:
    subset = raw[raw.protocol == protocol].copy()
    rows: list[dict[str, Any]] = []
    for (condition, model_key, run_key), group in subset.groupby(
        ["condition", "model_key", "run_key"], sort=True, dropna=False
    ):
        row = aggregate_group(group, model_key)
        row.update({
            "protocol": protocol,
            "condition": condition,
            "model_key": model_key,
            "task": "classification" if model_key.endswith("classification") else "regression",
            "source_phase": "09",
            "reference_policy": "RETRAIN_WITHOUT_MODALITY" if protocol == "RETRAIN_WITHOUT_MODALITY" else "LEAVE_ONE_SUBJECT_OUT",
        })
        rows.append(row)
    result = pd.DataFrame(rows).merge(
        meta, on=["run_key", "subject_id"], validate="many_to_one", suffixes=("", "_frozen")
    )
    result["outer_fold"] = result["outer_fold"].astype(int)
    if result.task.eq("classification").any():
        mask = result.task.eq("classification")
        if not np.array_equal(result.loc[mask, "y_true"].astype(int), result.loc[mask, "target_class"].astype(int)):
            raise RuntimeError("Classification targets do not align with frozen assignments")
    if result.task.eq("regression").any():
        mask = result.task.eq("regression")
        if not np.allclose(result.loc[mask, "y_true"].astype(float), result.loc[mask, "target_score"].astype(float)):
            raise RuntimeError("Regression targets do not align with frozen assignments")
    result["canonical_key"] = result.apply(
        lambda row: f"{row.protocol}|{row.condition}|{row.model_key}|{row.run_key}", axis=1
    )
    return result.sort_values(["condition", "model_key", "outer_fold", "run_key"], kind="mergesort").reset_index(drop=True)


def aggregate_reference(frame: pd.DataFrame, model_key: str, meta: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    if model_key == "hdc_classification":
        renamed = renamed[(renamed.variant == "hybrid") & (renamed.dimension == 5000)].copy()
        renamed = renamed.rename(columns={"true_class": "y_true", "predicted_class": "y_pred"})
        rows = []
        for _, group in renamed.groupby("run_key", sort=True):
            proxy = group.rename(columns={"true_class": "y_true"}) if "y_true" not in group else group
            row = aggregate_group(proxy, model_key)
            rows.append(row)
        result = pd.DataFrame(rows)
    elif model_key == "hdc_regression":
        renamed = renamed[renamed.dimension == 10000].rename(columns={
            "target_score": "y_true", "ridge_prediction_raw": "y_pred_raw",
            "ridge_prediction_bounded": "y_pred_bounded",
        })
        rows = [aggregate_group(group, model_key) for _, group in renamed.groupby("run_key", sort=True)]
        result = pd.DataFrame(rows)
    elif model_key == "traditional_classification":
        result = renamed.rename(columns={
            "true_class": "y_true", "predicted_class": "y_pred",
            "probability_class_0": "class_score_0", "probability_class_1": "class_score_1",
            "probability_class_2": "class_score_2", "probability_class_3": "class_score_3",
        })[["run_key", "subject_id", "y_true", "y_pred", "class_score_0", "class_score_1", "class_score_2", "class_score_3"]].copy()
        result["aggregation"] = "CANONICAL_FROZEN_PREDICTION"
        result["seed_count"] = 1
        result["seeds"] = "canonical"
        result["source_raw_rows"] = 1
    else:
        result = renamed.rename(columns={
            "target_score": "y_true", "prediction_raw": "y_pred_raw",
            "prediction_bounded": "y_pred_bounded",
        })[["run_key", "subject_id", "y_true", "y_pred_raw", "y_pred_bounded"]].copy()
        result["aggregation"] = "CANONICAL_FROZEN_PREDICTION"
        result["seed_count"] = 1
        result["seeds"] = "canonical"
        result["source_raw_rows"] = 1
        result["clipped"] = ~np.isclose(result.y_pred_raw, result.y_pred_bounded)

    result = result.merge(meta, on=["run_key", "subject_id"], validate="one_to_one", suffixes=("", "_frozen"))
    result["protocol"] = "FULL_PRIMARY_REFERENCE"
    result["condition"] = "FULL_PRIMARY_REFERENCE"
    result["model_key"] = model_key
    result["task"] = "classification" if model_key.endswith("classification") else "regression"
    result["reference_policy"] = "REUSED_FROZEN_REFERENCE"
    result["canonical_key"] = result.apply(lambda row: f"FULL_PRIMARY_REFERENCE|{model_key}|{row.run_key}", axis=1)
    if len(result) != 419 or result.run_key.nunique() != 419:
        raise RuntimeError(f"Reference coverage failure for {model_key}")
    return result.sort_values(["outer_fold", "run_key"], kind="mergesort").reset_index(drop=True)


def reference_sources() -> dict[str, dict[str, Any]]:
    sources = {
        "hdc_classification": {
            "path": PHASE06 / "results" / "oof" / "phase06_hybrid_final_oof.csv",
            "expected_sha256": "ff619baf4be600279482c9e1f4f4139000fc05c1dfaf41555d644674b45d875a",
            "source_phase": "06",
            "freeze_evidence": str((PHASE06 / "manifests" / "phase06_final_artifact_manifest.json").resolve()),
        },
        "hdc_regression": {
            "path": PHASE05 / "results" / "oof" / "vanilla_hdc_ridge_regression_oof.csv",
            "expected_sha256": "a449d8f43a0935f0a3fcf8cf901894e426a83e552807dcef9551bc983ba22758",
            "source_phase": "06_SELECTION_USING_FROZEN_PHASE05_COMMON_ENCODER_INTERFACE",
            "freeze_evidence": str((PHASE06 / "configs" / "phase06_freeze.json").resolve()),
        },
        "traditional_classification": {
            "path": PHASE04A / "results" / "predictions" / "gradient_boosting_oof.csv",
            "expected_sha256": "2900b8dda8a2a51982e1f731c00ff1aff7d028ea43eae41c9b9e4294d3c00c19",
            "source_phase": "04A",
            "freeze_evidence": str((PHASE04A / "configs" / "phase04a_freeze.json").resolve()),
        },
        "traditional_regression": {
            "path": PHASE04B / "results" / "predictions" / "gradient_boosting_oof.csv",
            "expected_sha256": "e05c23a526a00eaaa49d0fd07e1eae68ff2c53e94a05418d47474367aa0ae3ed",
            "source_phase": "04B",
            "freeze_evidence": str((PHASE04B / "configs" / "phase04b_freeze.json").resolve()),
        },
    }
    return sources


def build_references(meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    records = []
    for model_key, source in reference_sources().items():
        path = source["path"]
        actual = sha256(path)
        if actual != source["expected_sha256"]:
            raise RuntimeError(f"Frozen reference hash mismatch: {model_key}")
        canonical = aggregate_reference(pd.read_csv(path), model_key, meta)
        canonical["source_phase"] = source["source_phase"]
        canonical["source_path"] = str(path.resolve())
        canonical["source_sha256"] = actual
        frames.append(canonical)
        records.append({
            "model_key": model_key, "task": canonical.task.iloc[0], "rows": len(canonical),
            "unique_run_keys": canonical.run_key.nunique(), "source_phase": source["source_phase"],
            "source_path": str(path.resolve()), "expected_sha256": source["expected_sha256"],
            "actual_sha256": actual, "hash_pass": actual == source["expected_sha256"],
            "freeze_evidence": source["freeze_evidence"], "reference_policy": "REUSED_FROZEN_REFERENCE",
        })
    combined = pd.concat(frames, ignore_index=True, sort=False)
    alignment = combined.groupby("model_key").run_key.apply(lambda values: set(values) == set(meta.run_key)).to_dict()
    audit = {
        "phase": "09", "audit": "full_primary_reference_integrity", "status": "PASS",
        "audited_at_utc": utc_now(), "reference_rows": len(combined),
        "expected_reference_rows": 1676, "records": records, "run_key_alignment": alignment,
        "subject_outer_fold_alignment": bool(all(
            combined.merge(meta[["run_key", "subject_id", "outer_fold"]], on="run_key", suffixes=("", "_expected"))
            .eval("subject_id == subject_id_expected and outer_fold == outer_fold_expected")
        )),
        "full_primary_retrained": False,
    }
    if len(combined) != 1676 or not all(alignment.values()) or not audit["subject_outer_fold_alignment"]:
        audit["status"] = "FAIL"
        raise RuntimeError(audit)
    return combined, audit


def validate_canonical(missing: pd.DataFrame, loso: pd.DataFrame) -> dict[str, Any]:
    combined = pd.concat([missing, loso], ignore_index=True, sort=False)
    group_counts = Counter(zip(combined.protocol, combined.condition, combined.model_key))
    expected_groups = 20 + 4
    finite_columns = [column for column in [
        "y_true", "y_pred", "class_score_0", "class_score_1", "class_score_2",
        "class_score_3", "y_pred_raw", "y_pred_bounded",
    ] if column in combined]
    finite = all(np.isfinite(pd.to_numeric(combined[column].dropna(), errors="coerce")).all() for column in finite_columns)
    checks = {
        "missing_rows_8380": len(missing) == 8380,
        "loso_rows_1676": len(loso) == 1676,
        "canonical_rows_10056": len(combined) == 10056,
        "groups_24": len(group_counts) == expected_groups and all(value == 419 for value in group_counts.values()),
        "unique_canonical_keys": combined.canonical_key.nunique() == len(combined),
        "run_key_coverage": all(group.run_key.nunique() == 419 for _, group in combined.groupby(["protocol", "condition", "model_key"])),
        "five_seed_coverage": all(group.seed_count.eq(5).all() for _, group in combined[combined.model_key.str.startswith("hdc_")].groupby(["protocol", "condition", "model_key"])),
        "finite_values": finite,
        "classification_labels": set(pd.to_numeric(combined.loc[combined.task == "classification", "y_pred"]).astype(int)).issubset({0, 1, 2, 3}),
        "regression_bounded": combined.loc[combined.task == "regression", "y_pred_bounded"].between(1.0, 4.0).all(),
        "loso_subjects_35": loso.subject_id.nunique() == 35,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "group_counts": {"|".join(key): value for key, value in group_counts.items()}}


def dry_run() -> dict[str, Any]:
    preflight = execution_preflight()
    meta = metadata()
    raw = load_raw(preflight["records"])
    missing = consolidate_protocol(raw, "RETRAIN_WITHOUT_MODALITY", meta)
    loso = consolidate_protocol(raw, "LEAVE_ONE_SUBJECT_OUT", meta)
    references, reference_audit = build_references(meta)
    validation = validate_canonical(missing, loso)
    result = {
        "status": "PASS" if validation["status"] == "PASS" and reference_audit["status"] == "PASS" else "FAIL",
        "raw_runs_verified": 720, "raw_prediction_rows_verified": len(raw),
        "missing_modality_canonical_rows": len(missing), "loso_canonical_rows": len(loso),
        "canonical_oof_rows": len(missing) + len(loso), "reference_rows": len(references),
        "five_seed_coverage": validation["checks"]["five_seed_coverage"],
        "canonical_validation": validation, "reference_integrity": reference_audit["status"],
        "writes_performed": False, "model_retraining_executed": False,
        "raw_predictions_regenerated": False,
    }
    if result["status"] != "PASS":
        raise RuntimeError(result)
    return result


def consolidate_and_save() -> dict[str, Any]:
    preflight = execution_preflight()
    meta = metadata()
    raw = load_raw(preflight["records"])
    missing = consolidate_protocol(raw, "RETRAIN_WITHOUT_MODALITY", meta)
    loso = consolidate_protocol(raw, "LEAVE_ONE_SUBJECT_OUT", meta)
    references, reference_audit = build_references(meta)
    validation = validate_canonical(missing, loso)
    if validation["status"] != "PASS":
        raise RuntimeError(validation)

    out = PHASE09 / "results" / "oof"
    atomic_csv(out / "phase09_missing_modality_canonical_classification_oof.csv", missing[missing.task == "classification"])
    atomic_csv(out / "phase09_missing_modality_canonical_regression_oof.csv", missing[missing.task == "regression"])
    atomic_csv(out / "phase09_loso_canonical_classification_oof.csv", loso[loso.task == "classification"])
    atomic_csv(out / "phase09_loso_canonical_regression_oof.csv", loso[loso.task == "regression"])
    index_columns = [
        "canonical_key", "protocol", "condition", "model_key", "task", "run_key",
        "subject_id", "outer_fold", "aggregation", "seed_count", "seeds", "source_phase",
        "reference_policy",
    ]
    index = pd.concat([missing, loso], ignore_index=True, sort=False)
    atomic_csv(out / "phase09_canonical_oof_index.csv", index[index_columns])
    atomic_csv(out / "phase09_full_primary_reference_index.csv", references)
    atomic_json(PHASE09 / "audits" / "phase09_full_primary_reference_integrity_audit.json", reference_audit)
    coverage_audit = {
        "phase": "09", "audit": "oof_coverage", "status": "PASS",
        "audited_at_utc": utc_now(), "raw_runs_verified": 720,
        "raw_prediction_rows_verified": len(raw), "missing_modality_rows": len(missing),
        "loso_rows": len(loso), "canonical_rows": len(index),
        "five_seed_coverage": "PASS", "checks": validation["checks"],
    }
    alignment_audit = {
        "phase": "09", "audit": "oof_alignment", "status": "PASS",
        "audited_at_utc": utc_now(), "canonical_run_key_coverage": "PASS",
        "subject_outer_fold_alignment": "PASS", "full_primary_reference_alignment": "PASS",
        "duplicate_canonical_keys": int(index.canonical_key.duplicated().sum()),
    }
    leakage_audit = {
        "phase": "09", "audit": "oof_leakage", "status": "PASS",
        "audited_at_utc": utc_now(), "source_execution_leakage_audit": "PASS",
        "source_loso_mapping_leakage_audit": "PASS", "aggregation_only": True,
        "model_retraining_executed": False, "raw_predictions_regenerated": False,
        "hyperparameter_tuning_executed": False, "outer_test_used_for_selection": False,
    }
    atomic_json(PHASE09 / "audits" / "phase09_oof_coverage_audit.json", coverage_audit)
    atomic_json(PHASE09 / "audits" / "phase09_oof_alignment_audit.json", alignment_audit)
    atomic_json(PHASE09 / "audits" / "phase09_oof_leakage_audit.json", leakage_audit)
    manifest = read_json(MANIFEST_PATH)
    manifest.update({
        "status": "OOF_CONSOLIDATED_PENDING_ANALYSIS", "canonical_oof_rows": len(index),
        "missing_modality_canonical_rows": len(missing), "loso_canonical_rows": len(loso),
        "full_primary_reference_rows": len(references), "canonical_oof_consolidated_at_utc": utc_now(),
        "canonical_oof_consolidation_executed": True, "formal_statistical_analysis_executed": False,
        "model_retraining_during_analysis": False, "raw_predictions_regenerated_during_analysis": False,
    })
    atomic_json(MANIFEST_PATH, manifest)
    return {
        "status": manifest["status"], "raw_runs_verified": 720,
        "raw_prediction_rows_verified": len(raw), "missing_modality_canonical_rows": len(missing),
        "loso_canonical_rows": len(loso), "canonical_oof_rows": len(index),
        "full_primary_reference_rows": len(references), "five_seed_coverage": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = dry_run() if args.dry_run else consolidate_and_save()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
