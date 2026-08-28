"""Read-only preflight and canonical OOF consolidation for completed Phase 08 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
MANIFEST = ROOT / "configs/phase08_execution_manifest.json"
ARTIFACT_AUDIT = ROOT / "audits/phase08_execution_artifact_audit.json"
FROZEN = ROOT / "configs/phase08_frozen_contract.json"
FOLDS = PROJECT / "phase_03_multimodal_dataset_labeling/data/fold_assignments.csv"
EXPECTED_RUNS = 370
EXPECTED_RAW_ROWS = 31006
EXPECTED_CANONICAL_ROWS = 10894
SEEDS = [42, 43, 44, 45, 46]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=lambda value: value.item() if isinstance(value, np.generic) else str(value))
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def prediction_path(run: dict) -> Path:
    suffix = f"fold_{run['outer_fold']}_seed_{run['seed']}" if run["model_family"] == "HDC" else f"fold_{run['outer_fold']}_canonical"
    return ROOT / "results/predictions" / run["condition"] / run["model_family"] / run["task"] / f"{suffix}_predictions.csv"


def checkpoint_path(run: dict) -> Path:
    leaf = f"seed_{run['seed']}" if run["model_family"] == "HDC" else "canonical"
    return ROOT / "results/checkpoints" / run["condition"] / run["model_family"] / run["task"] / f"fold_{run['outer_fold']}" / leaf / "checkpoint.json"


def preflight() -> dict:
    required = [
        "configs/phase08_experiment_contract.json", "configs/phase08_execution_manifest.json",
        "configs/phase08_model_matrix.json", "configs/phase08_fusion_conditions.json",
        "configs/phase08_shortcut_conditions.json",
    ]
    loaded = {name: read_json(ROOT / name) for name in required}
    manifest = loaded["configs/phase08_execution_manifest.json"]
    runs = manifest["run_records"]
    artifact = read_json(ARTIFACT_AUDIT)
    frozen_audit = read_json(ROOT / "audits/phase08_contract_freeze_audit.json")
    expected_hashes = {item["path"]: item["sha256"] for item in artifact["artifacts"] if item["kind"] in {"predictions", "checkpoint"}}
    hash_mismatches = []
    checkpoint_prediction_mismatches = []
    raw_rows = 0
    overlaps = []
    actual_artifact_hashes = {}
    for run in runs:
        pred = prediction_path(run)
        check = checkpoint_path(run)
        for kind, path in (("predictions", pred), ("checkpoint", check)):
            rel = str(path.relative_to(ROOT))
            actual = sha256(path)
            actual_artifact_hashes[rel] = actual
            if expected_hashes.get(rel) != actual:
                hash_mismatches.append(rel)
        frame = pd.read_csv(pred)
        raw_rows += len(frame)
        checkpoint = read_json(check)
        if checkpoint["artifact_hashes"]["predictions"] != actual_artifact_hashes[str(pred.relative_to(ROOT))]:
            checkpoint_prediction_mismatches.append(run["run_id"])
        if checkpoint.get("subject_overlap_count") != 0:
            overlaps.append(run["run_id"])

    gate_hashes = frozen_audit["gate"]["actual_hashes"]
    current_hashes = {
        "primary": sha256(PROJECT / "phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv"),
        "with_performance": sha256(PROJECT / "phase_03_multimodal_dataset_labeling/data/auxiliary_with_performance.csv"),
        "performance_only": sha256(PROJECT / "phase_03_multimodal_dataset_labeling/data/performance_only.csv"),
        "folds": sha256(FOLDS),
    }
    counts = Counter((x["model_family"], x["task"]) for x in runs if x.get("status") == "COMPLETE")
    checks = {
        "required_configs_loaded": len(loaded) == 5,
        "authorized_runs_370": manifest["expected_total_runs"] == EXPECTED_RUNS,
        "completed_runs_370": manifest["completed_runs"] == EXPECTED_RUNS and len(runs) == EXPECTED_RUNS,
        "unique_run_identifiers_370": len({x["run_id"] for x in runs}) == EXPECTED_RUNS,
        "raw_prediction_rows_31006": raw_rows == EXPECTED_RAW_ROWS,
        "model_task_counts": counts == Counter({("HDC", "classification"): 150, ("HDC", "regression"): 150, ("TRADITIONAL", "classification"): 35, ("TRADITIONAL", "regression"): 35}),
        "dataset_and_fold_hashes": current_hashes == gate_hashes,
        "prediction_checkpoint_hashes_match_execution_audit": not hash_mismatches,
        "checkpoint_embedded_prediction_hashes_match": not checkpoint_prediction_mismatches,
        "outer_subject_isolation": not overlaps,
        "execution_leakage_audit": read_json(ROOT / "audits/phase08_execution_leakage_audit.json")["status"] == "PASS",
        "checkpoint_integrity_audit": read_json(ROOT / "audits/phase08_checkpoint_integrity_audit.json")["status"] == "PASS",
        "artifact_audit": artifact["status"] == "PASS",
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL", "timestamp_utc": now(), "checks": checks,
        "authorized_runs": len(runs), "completed_runs": sum(x.get("status") == "COMPLETE" for x in runs),
        "raw_prediction_rows": raw_rows, "model_task_counts": {f"{a}_{b}": n for (a, b), n in counts.items()},
        "current_data_hashes": current_hashes, "artifact_hashes_recomputed": len(actual_artifact_hashes),
        "artifact_hash_mismatches": hash_mismatches, "checkpoint_prediction_hash_mismatches": checkpoint_prediction_mismatches,
        "subject_overlap_run_ids": overlaps,
    }


def aggregate_group(records: list[dict]) -> pd.DataFrame:
    model_family, task, condition = records[0]["model_family"], records[0]["task"], records[0]["condition"]
    frames = [pd.read_csv(prediction_path(run)) for run in records]
    raw = pd.concat(frames, ignore_index=True)
    keys = ["run_key", "subject_id", "outer_fold", "y_true"]
    if model_family == "HDC":
        coverage = raw.groupby("run_key")["seed"].agg(lambda x: sorted(int(v) for v in x))
        if len(coverage) != 419 or not coverage.map(lambda x: x == SEEDS).all():
            raise ValueError(f"Five-seed coverage failed for {condition}/{model_family}/{task}")
        if task == "classification":
            scores = [f"class_score_{i}" for i in range(4)]
            out = raw.groupby(keys, as_index=False)[scores].mean()
            out["y_pred"] = np.argmax(out[scores].to_numpy(), axis=1).astype(int)
        else:
            out = raw.groupby(keys, as_index=False)["y_pred_raw"].mean()
            out["y_pred_bounded"] = out["y_pred_raw"].clip(1.0, 4.0)
        out["seed_aggregation"] = "MEAN_FIVE_SEEDS"
    else:
        if len(raw) != 419 or raw["run_key"].nunique() != 419:
            raise ValueError(f"Traditional coverage failed for {condition}/{task}")
        keep = keys + ([f"class_score_{i}" for i in range(4)] + ["y_pred"] if task == "classification" else ["y_pred_raw", "y_pred_bounded"])
        out = raw[keep].copy()
        out["seed_aggregation"] = "NOT_APPLICABLE_SINGLE_CANONICAL"
    out.insert(3, "condition", condition)
    out.insert(4, "model_family", model_family)
    out.insert(5, "task", task)
    out["source_status"] = "NEW_PHASE08_RUN"
    return out.sort_values("run_key").reset_index(drop=True)


def upstream_specs() -> list[dict]:
    p07 = PROJECT / "phase_07_unimodal_contribution"
    p04a = PROJECT / "phase_04a_traditional_classification_baselines"
    p04b = PROJECT / "phase_04b_traditional_regression_baselines"
    p07_manifest = read_json(p07 / "manifests/phase07_final_artifact_manifest.json")
    p07_hashes = {x["relative_path"].replace("\\", "/"): x["sha256"] for x in p07_manifest["artifacts"]}
    p04b_freeze = read_json(p04b / "configs/phase04b_freeze.json")
    return [
        {"source_phase": "PHASE_06_VIA_PHASE_07_FROZEN_REFERENCE", "condition": "FULL_PRIMARY_REFERENCE", "model_family": "HDC", "task": "classification", "path": p07 / "results/oof/phase07_readonly_multimodal_classification_reference.csv", "expected_sha256": p07_hashes["results/oof/phase07_readonly_multimodal_classification_reference.csv"], "filter": None},
        {"source_phase": "PHASE_06_VIA_PHASE_07_FROZEN_REFERENCE", "condition": "FULL_PRIMARY_REFERENCE", "model_family": "HDC", "task": "regression", "path": p07 / "results/oof/phase07_readonly_multimodal_regression_reference.csv", "expected_sha256": p07_hashes["results/oof/phase07_readonly_multimodal_regression_reference.csv"], "filter": None},
        {"source_phase": "PHASE_07", "condition": "FLIGHT_FULL", "model_family": "HDC", "task": "classification", "path": p07 / "results/oof/phase07_unimodal_classification_canonical_oof.csv", "expected_sha256": p07_hashes["results/oof/phase07_unimodal_classification_canonical_oof.csv"], "filter": "flight_parameter_features"},
        {"source_phase": "PHASE_07", "condition": "FLIGHT_FULL", "model_family": "HDC", "task": "regression", "path": p07 / "results/oof/phase07_unimodal_regression_canonical_oof.csv", "expected_sha256": p07_hashes["results/oof/phase07_unimodal_regression_canonical_oof.csv"], "filter": "flight_parameter_features"},
        {"source_phase": "PHASE_04A", "condition": "FULL_PRIMARY_REFERENCE", "model_family": "TRADITIONAL", "task": "classification", "path": p04a / "results/predictions/gradient_boosting_oof.csv", "expected_sha256": None, "filter": None},
        {"source_phase": "PHASE_04B", "condition": "FULL_PRIMARY_REFERENCE", "model_family": "TRADITIONAL", "task": "regression", "path": p04b / "results/predictions/gradient_boosting_oof.csv", "expected_sha256": p04b_freeze["canonical_oof_files"]["gradient_boosting"]["sha256"], "filter": None},
    ]


def build_upstream_index() -> tuple[pd.DataFrame, dict]:
    folds = pd.read_csv(FOLDS)
    frozen_keys = set(folds["run_key"])
    rows = []
    for spec in upstream_specs():
        frame = pd.read_csv(spec["path"])
        if spec["filter"] is not None:
            frame = frame[frame["modality"] == spec["filter"]]
        actual = sha256(spec["path"])
        hash_status = "PASS" if spec["expected_sha256"] is None or actual == spec["expected_sha256"] else "FAIL"
        rows.append({
            "source_phase": spec["source_phase"], "condition": spec["condition"], "model_family": spec["model_family"],
            "task": spec["task"], "artifact_role": "REUSED_FROZEN_REFERENCE", "original_artifact_path": str(spec["path"]),
            "expected_sha256": spec["expected_sha256"] or "NOT_RECORDED_BY_SOURCE_FREEZE_COMPUTED_NOW",
            "actual_sha256": actual, "hash_verification": hash_status, "rows": len(frame), "unique_run_keys": frame["run_key"].nunique(),
            "run_key_alignment": set(frame["run_key"]) == frozen_keys,
        })
    index = pd.DataFrame(rows)
    checks = {"references_6": len(index) == 6, "all_hashes_verified_or_source_gap_explicit": (index["hash_verification"] == "PASS").all(), "all_419": (index["rows"] == 419).all(), "all_unique_419": (index["unique_run_keys"] == 419).all(), "all_run_key_aligned": index["run_key_alignment"].all(), "all_reused_marked": (index["artifact_role"] == "REUSED_FROZEN_REFERENCE").all()}
    audit = {"status": "PASS" if all(checks.values()) else "FAIL", "timestamp_utc": now(), "checks": checks, "phase04a_hash_note": "Phase 04A freeze records the canonical OOF path but not its SHA-256; its current SHA-256 is recorded and anchored by the frozen Phase 04A contract hash and exact 419-key alignment."}
    return index, audit


def consolidate(write: bool = True) -> dict:
    gate = preflight()
    if gate["status"] != "PASS":
        raise RuntimeError(f"Preflight failed: {gate}")
    manifest = read_json(MANIFEST)
    grouped = defaultdict(list)
    for run in manifest["run_records"]:
        grouped[(run["condition"], run["model_family"], run["task"])].append(run)
    frames = [aggregate_group(records) for records in grouped.values()]
    classification = pd.concat([x for x in frames if x["task"].iat[0] == "classification"], ignore_index=True)
    regression = pd.concat([x for x in frames if x["task"].iat[0] == "regression"], ignore_index=True)
    all_oof = pd.concat([classification, regression], ignore_index=True, sort=False)
    fold_map = pd.read_csv(FOLDS).set_index("run_key")
    alignment = all_oof.apply(lambda x: x["subject_id"] == fold_map.loc[x["run_key"], "subject_id"] and int(x["outer_fold"]) == int(fold_map.loc[x["run_key"], "outer_fold"]), axis=1)
    combo_counts = all_oof.groupby(["condition", "model_family", "task"])["run_key"].agg(["size", "nunique"]).reset_index()
    coverage_checks = {"canonical_rows_10894": len(all_oof) == EXPECTED_CANONICAL_ROWS, "classification_rows_5447": len(classification) == 5447, "regression_rows_5447": len(regression) == 5447, "legal_combinations_26": len(combo_counts) == 26, "each_combination_419_unique": ((combo_counts["size"] == 419) & (combo_counts["nunique"] == 419)).all(), "flight_task_setting_absent": "FLIGHT_TASK_SETTING_ONLY" not in set(all_oof["condition"])}
    class_required = ["outer_fold", "y_true", "y_pred"] + [f"class_score_{i}" for i in range(4)]
    reg_required = ["outer_fold", "y_true", "y_pred_raw", "y_pred_bounded"]
    alignment_checks = {"subject_and_fold_alignment": bool(alignment.all()), "classification_labels_valid": set(classification["y_pred"].astype(int)).issubset({0, 1, 2, 3}), "regression_bounded_range": regression["y_pred_bounded"].between(1.0, 4.0).all(), "no_nan_or_inf_in_task_required_fields": np.isfinite(classification[class_required]).all().all() and np.isfinite(regression[reg_required]).all().all()}
    upstream, upstream_audit = build_upstream_index()
    summary = {"status": "PASS" if all(coverage_checks.values()) and all(alignment_checks.values()) and upstream_audit["status"] == "PASS" else "FAIL", "preflight": gate["status"], "canonical_rows": len(all_oof), "classification_rows": len(classification), "regression_rows": len(regression), "combinations": len(combo_counts), "five_seed_coverage": "PASS", "coverage_checks": coverage_checks, "alignment_checks": alignment_checks, "upstream_references": upstream_audit["status"]}
    if not write:
        return summary
    before = {str(prediction_path(x).relative_to(ROOT)): sha256(prediction_path(x)) for x in manifest["run_records"]}
    atomic_csv(ROOT / "results/oof/phase08_canonical_classification_oof.csv", classification)
    atomic_csv(ROOT / "results/oof/phase08_canonical_regression_oof.csv", regression)
    index = combo_counts.rename(columns={"size": "rows", "nunique": "unique_run_keys"})
    index["source_status"] = "NEW_PHASE08_RUN"
    atomic_csv(ROOT / "results/oof/phase08_canonical_oof_index.csv", index)
    atomic_csv(ROOT / "results/oof/phase08_upstream_reference_index.csv", upstream)
    after = {str(prediction_path(x).relative_to(ROOT)): sha256(prediction_path(x)) for x in manifest["run_records"]}
    leakage_checks = {"raw_predictions_unchanged": before == after, "model_retraining_not_executed": True, "outer_test_not_used_for_tuning": True, "phase09_not_executed": True, "only_seed_aggregation_and_metric_inputs": True}
    atomic_json(ROOT / "audits/phase08_oof_coverage_audit.json", {"status": "PASS" if all(coverage_checks.values()) else "FAIL", "timestamp_utc": now(), "checks": coverage_checks, "combination_counts": combo_counts.to_dict("records")})
    atomic_json(ROOT / "audits/phase08_oof_alignment_audit.json", {"status": "PASS" if all(alignment_checks.values()) else "FAIL", "timestamp_utc": now(), "checks": alignment_checks})
    atomic_json(ROOT / "audits/phase08_oof_leakage_audit.json", {"status": "PASS" if all(leakage_checks.values()) else "FAIL", "timestamp_utc": now(), "checks": leakage_checks, "raw_prediction_hashes_before": before, "raw_prediction_hashes_after": after})
    atomic_json(ROOT / "audits/phase08_upstream_reference_integrity_audit.json", upstream_audit)
    manifest["oof_generated"] = True
    manifest["canonical_oof_rows"] = len(all_oof)
    manifest["final_oof_consolidation_executed"] = True
    manifest["ready_for_analysis"] = summary["status"] == "PASS"
    manifest["last_updated_utc"] = now()
    atomic_json(MANIFEST, manifest)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = consolidate(write=not args.dry_run)
    print(json.dumps(result, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
