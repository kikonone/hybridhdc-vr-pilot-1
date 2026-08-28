"""No-retraining Phase 06 OOF consolidation, strict analysis, and conditional freeze."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
PHASE05 = ROOT / "experiments/phase_05_basic_dual_output_hdc"
PRIMARY = ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv"
FOLDS = ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv"
EXPECTED_PRIMARY = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLDS = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
VARIANTS = ["vanilla", "onlinehd", "multicentroid", "hybrid"]
NEW_VARIANTS = VARIANTS[1:]
DISPLAY = {"vanilla": "Vanilla Prototype HDC", "onlinehd": "OnlineHD-style HDC", "multicentroid": "Multi-centroid HDC", "hybrid": "HDC+OnlineHD Hybrid", "common_ridge": "Common Ridge readout"}
DIMENSIONS = [1000, 2000, 5000, 10000]
SEEDS = [42, 43, 44, 45, 46]
TOLERANCE = 1e-12
COLORS = {"vanilla": "#0072B2", "onlinehd": "#E69F00", "multicentroid": "#009E73", "hybrid": "#CC79A7", "common_ridge": "#000000"}


def now() -> str: return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"); temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n"); temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8"); temporary.replace(path)


def verify_manifest(base: Path, manifest_path: Path, allowed_changes: set[str] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    allowed = {value.replace("\\", "/") for value in (allowed_changes or set())}; manifest = read_json(manifest_path)
    records, failures = [], []
    for item in manifest["artifacts"]:
        relative = str(item.get("relative_path") or item.get("path")).replace("\\", "/")
        path = Path(relative)
        if not path.is_absolute(): path = base / path
        actual = sha256(path) if path.exists() else None; size = path.stat().st_size if path.exists() else -1
        result = "AUTHORIZED_CHANGE" if relative in allowed else ("PASS" if actual == item["sha256"] and size == int(item["file_size_bytes"]) else "FAIL")
        if result == "FAIL": failures.append(relative)
        records.append({"relative_path": relative, "expected_sha256": item["sha256"], "actual_sha256": actual, "expected_size": int(item["file_size_bytes"]), "actual_size": size, "result": result})
    return records, failures


def preflight() -> dict[str, Any]:
    phase05_freeze_path = PHASE05 / "configs/phase05_freeze.json"; phase05_manifest_path = PHASE05 / "manifests/phase05_final_artifact_manifest.json"
    phase06_manifest_path = PHASE / "manifests/phase06_final_confirmation_artifact_manifest.json"
    phase05_freeze = read_json(phase05_freeze_path); all_folds = read_json(PHASE / "audits/phase06_final_confirmation_all_folds_audit.json")
    terminal = read_json(PHASE / "audits/phase06_final_confirmation_terminal_verification_audit.json")
    notebook = read_json(PHASE / "audits/phase06_final_confirmation_notebook_persistence_audit.json")
    p5_records, p5_failures = verify_manifest(PHASE05, phase05_manifest_path)
    p6_records, p6_failures = verify_manifest(PHASE, phase06_manifest_path)
    required = [
        "configs/phase06_hdc_variant_contract.json", "configs/phase06_variant_search_spaces.json", "configs/phase06_model_selection_rules.json",
        "audits/phase06_contract_freeze_audit.json", "manifests/phase06_contract_manifest.json",
        "results/summaries/phase06_final_confirmation_execution_summary.csv", "manifests/phase06_final_confirmation_artifact_manifest.json",
        "audits/phase06_final_confirmation_all_folds_audit.json", "Phase_06_HDC_Variant_Screening.ipynb",
    ]
    missing = [value for value in required if not (PHASE / value).exists()]
    variants_ok = all(all_folds["variants"][variant]["folds_completed"] == 5 and all_folds["variants"][variant]["fold_config_runs"] == 100 and all_folds["variants"][variant]["result"] == "PASS" for variant in NEW_VARIANTS)
    checks = {
        "required_present": not missing, "missing": missing, "primary_sha256": sha256(PRIMARY), "fold_sha256": sha256(FOLDS),
        "primary_checksum": sha256(PRIMARY) == EXPECTED_PRIMARY, "fold_checksum": sha256(FOLDS) == EXPECTED_FOLDS,
        "phase05_status": phase05_freeze.get("status"), "phase05_manifest_sha256": sha256(phase05_manifest_path),
        "phase05_manifest_matches_freeze": sha256(phase05_manifest_path) == phase05_freeze.get("final_artifact_manifest_sha256"),
        "phase05_artifacts_verified": len(p5_records), "phase05_failures": p5_failures,
        "phase06_final_confirmation_artifacts_verified": len(p6_records), "phase06_final_confirmation_failures": p6_failures,
        "new_variant_runs": all_folds.get("completed_fold_config_runs"), "variants_complete": variants_ok,
        "classification_predictions": all_folds.get("classification_predictions_generated"), "similarity_predictions": all_folds.get("similarity_regression_predictions_generated"),
        "ridge_handling": all_folds.get("ridge_handling"), "outer_subject_isolation": all_folds.get("outer_subject_isolation"),
        "inner_subject_isolation": all_folds.get("inner_subject_isolation"), "temperature_inner_cv_only": all_folds.get("temperature_inner_cv_only"),
        "outer_test_used_for_tuning": all_folds.get("outer_test_used_for_tuning"), "notebook_persistence": notebook.get("result"),
        "terminal_verification": terminal.get("result"), "ready_for_oof": all_folds.get("ready_for_final_oof_consolidation"),
    }
    ok = not missing and checks["primary_checksum"] and checks["fold_checksum"] and phase05_freeze.get("status") == "FROZEN" and checks["phase05_manifest_matches_freeze"] and not p5_failures and not p6_failures and checks["new_variant_runs"] == 300 and variants_ok and checks["classification_predictions"] is True and checks["similarity_predictions"] is True and checks["ridge_handling"] == "COMMON_ENCODER_READOUT_BASELINE" and checks["outer_subject_isolation"] == "PASS" and checks["inner_subject_isolation"] == "PASS" and checks["temperature_inner_cv_only"] == "PASS" and checks["outer_test_used_for_tuning"] is False and checks["notebook_persistence"] == "PASS" and checks["terminal_verification"] == "PASS" and checks["ready_for_oof"] is True
    audit = {"phase": "06", "audit": "final_oof_preflight", "timestamp_utc": now(), **checks, "result": "PASS" if ok else "FAIL"}
    atomic_json(PHASE / "audits/phase06_final_oof_preflight_audit.json", audit)
    atomic_json(PHASE / "audits/phase06_upstream_pre_finalization_snapshot.json", {"phase": "06", "timestamp_utc": now(), "phase05_freeze_sha256": sha256(phase05_freeze_path), "phase05_manifest_sha256": sha256(phase05_manifest_path), "phase05_artifacts": p5_records, "phase06_final_confirmation_manifest_sha256": sha256(phase06_manifest_path), "phase06_final_confirmation_artifacts": p6_records, "result": "PASS" if not p5_failures and not p6_failures else "FAIL"})
    if not ok: raise RuntimeError(f"Final OOF preflight failed: {audit}")
    return audit


def metadata() -> pd.DataFrame:
    primary = pd.read_csv(PRIMARY, usecols=["run_key", "subject_id", "outer_fold", "target_class", "target_score"])
    folds = pd.read_csv(FOLDS, usecols=["run_key", "subject_id", "outer_fold", "target_class", "target_score"])
    merged = primary.merge(folds, on="run_key", suffixes=("_primary", "_fold"), validate="one_to_one")
    for column in ["subject_id", "outer_fold", "target_class", "target_score"]:
        if not np.all(merged[f"{column}_primary"].astype(str).to_numpy() == merged[f"{column}_fold"].astype(str).to_numpy()): raise RuntimeError(f"Primary/fold metadata mismatch: {column}")
    return primary


def consolidate_new_oof(meta: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    outputs, coverage = {}, {}; alignment_failures, temperature_failures = [], []
    expected_keys = set(meta.run_key)
    for variant in NEW_VARIANTS:
        frames = [pd.read_csv(PHASE / f"results/predictions/{variant}_final_confirmation_fold_{fold}_predictions.csv") for fold in range(1, 6)]
        frame = pd.concat(frames, ignore_index=True)
        numeric = ["outer_fold", "dimension", "seed", "levels", "feature_k", "true_class", "predicted_class", "target_score", "selected_temperature", "similarity_prediction_raw", "similarity_prediction_bounded", *[f"class_score_{i}" for i in range(4)]]
        for column in numeric: frame[column] = pd.to_numeric(frame[column])
        selection = pd.concat([pd.read_csv(PHASE / f"results/fold_metrics/{variant}_final_confirmation_fold_{fold}_inner_selection.csv") for fold in range(1, 6)], ignore_index=True)
        selected_map = selection.groupby(["outer_fold", "dimension", "seed"], as_index=False)["selected_temperature"].first().set_index(["outer_fold", "dimension", "seed"])["selected_temperature"]
        for row in frame[["outer_fold", "dimension", "seed", "selected_temperature"]].drop_duplicates().itertuples(index=False):
            if abs(float(row.selected_temperature) - float(selected_map.loc[(row.outer_fold, row.dimension, row.seed)])) > TOLERANCE: temperature_failures.append([variant, row.outer_fold, row.dimension, row.seed])
        configurations = []
        for dimension in DIMENSIONS:
            for seed in SEEDS:
                group = frame[(frame.dimension == dimension) & (frame.seed == seed)]
                configurations.append({"dimension": dimension, "seed": seed, "rows": len(group), "unique_run_keys": group.run_key.nunique(), "duplicates": len(group) - group.run_key.nunique(), "complete": len(group) == 419 and group.run_key.nunique() == 419 and set(group.run_key) == expected_keys})
        joined = frame.merge(meta, on="run_key", suffixes=("", "_expected"), validate="many_to_one")
        mismatches = {
            "subject_id": int((joined.subject_id.astype(str) != joined.subject_id_expected.astype(str)).sum()),
            "outer_fold": int((joined.outer_fold.astype(int) != joined.outer_fold_expected.astype(int)).sum()),
            "true_class": int((joined.true_class.astype(int) != joined.target_class.astype(int)).sum()),
            "target_score": int((joined.target_score.astype(float) != joined.target_score_expected.astype(float)).sum()),
        }
        if any(mismatches.values()): alignment_failures.append({variant: mismatches})
        frame = frame.sort_values(["variant", "dimension", "seed", "outer_fold", "subject_id", "run_key"], kind="mergesort").reset_index(drop=True)
        atomic_csv(PHASE / f"results/oof/phase06_{variant}_final_oof.csv", frame); outputs[variant] = frame
        coverage[variant] = {"rows": len(frame), "configurations": configurations, "complete_configurations": sum(item["complete"] for item in configurations)}
    combined = pd.concat([outputs[value] for value in NEW_VARIANTS], ignore_index=True).sort_values(["variant", "dimension", "seed", "outer_fold", "subject_id", "run_key"], kind="mergesort").reset_index(drop=True)
    atomic_csv(PHASE / "results/oof/phase06_new_variants_final_oof_long.csv", combined)
    coverage_ok = len(combined) == 25140 and all(value["rows"] == 8380 and value["complete_configurations"] == 20 for value in coverage.values())
    atomic_json(PHASE / "audits/phase06_final_oof_coverage_audit.json", {"phase": "06", "audit": "final_oof_coverage", "variants": coverage, "combined_rows": len(combined), "expected_combined_rows": 25140, "result": "PASS" if coverage_ok else "FAIL"})
    atomic_json(PHASE / "audits/phase06_final_oof_alignment_audit.json", {"phase": "06", "audit": "final_oof_alignment", "alignment_failures": alignment_failures, "temperature_failures": temperature_failures, "deterministic_sort": ["variant", "dimension", "seed", "outer_fold", "subject_id", "run_key"], "result": "PASS" if not alignment_failures and not temperature_failures else "FAIL"})
    atomic_json(PHASE / "audits/phase06_final_oof_leakage_audit.json", {"phase": "06", "audit": "final_oof_leakage", "source_predictions_only": True, "training_calls": 0, "prediction_calls": 0, "outer_test_used_for_tuning": False, "temperature_inner_cv_only": not temperature_failures, "outer_subject_isolation": "PASS", "inner_subject_isolation": "PASS", "result": "PASS" if not temperature_failures else "FAIL"})
    if not coverage_ok or alignment_failures or temperature_failures: raise RuntimeError("New-variant OOF consolidation failed")
    return outputs, {"coverage": coverage, "combined_rows": len(combined)}


def load_phase05_interfaces(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    paths = {
        "classification": PHASE05 / "results/oof/vanilla_hdc_classification_oof.csv",
        "similarity": PHASE05 / "results/oof/vanilla_hdc_similarity_regression_oof.csv",
        "ridge": PHASE05 / "results/oof/vanilla_hdc_ridge_regression_oof.csv",
    }
    classification, similarity, ridge = (pd.read_csv(paths[key]) for key in ["classification", "similarity", "ridge"])
    class_keys = ["run_key", "subject_id", "outer_fold", "dimension", "seed", "levels", "feature_k"]
    vanilla = classification.merge(similarity, on=class_keys, validate="one_to_one")
    vanilla = vanilla.merge(meta[["run_key", "target_score"]], on="run_key", suffixes=("", "_expected"), validate="many_to_one")
    if "target_score_expected" in vanilla: vanilla["target_score"] = vanilla.pop("target_score_expected")
    vanilla["variant"] = "vanilla"; vanilla["selected_temperature"] = vanilla["similarity_temperature"]
    vanilla["similarity_prediction_raw"] = vanilla["similarity_prediction"]; vanilla["similarity_prediction_bounded"] = vanilla["similarity_prediction"].clip(1.0, 4.0)
    for i in range(4): vanilla[f"class_score_{i}"] = vanilla[f"similarity_class_{i}"]
    vanilla = vanilla.sort_values(["variant", "dimension", "seed", "outer_fold", "subject_id", "run_key"], kind="mergesort").reset_index(drop=True)
    return vanilla, ridge.sort_values(["dimension", "seed", "outer_fold", "subject_id", "run_key"], kind="mergesort").reset_index(drop=True), classification, {key: sha256(path) for key, path in paths.items()}


def build_four_variant_oof(new: dict[str, pd.DataFrame], meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vanilla, ridge, _, hashes = load_phase05_interfaces(meta)
    class_columns = ["run_key", "subject_id", "outer_fold", "variant", "dimension", "seed", "levels", "feature_k", "true_class", "predicted_class", *[f"class_score_{i}" for i in range(4)]]
    regression_columns = ["run_key", "subject_id", "outer_fold", "variant", "dimension", "seed", "levels", "feature_k", "target_score", "selected_temperature", "similarity_prediction_raw", "similarity_prediction_bounded"]
    four_class = pd.concat([vanilla[class_columns], *[new[v][class_columns] for v in NEW_VARIANTS]], ignore_index=True).sort_values(["variant", "dimension", "seed", "outer_fold", "subject_id", "run_key"], kind="mergesort")
    four_reg = pd.concat([vanilla[regression_columns], *[new[v][regression_columns] for v in NEW_VARIANTS]], ignore_index=True).sort_values(["variant", "dimension", "seed", "outer_fold", "subject_id", "run_key"], kind="mergesort")
    atomic_csv(PHASE / "results/oof/phase06_four_variant_classification_oof_long.csv", four_class)
    atomic_csv(PHASE / "results/oof/phase06_four_variant_similarity_regression_oof_long.csv", four_reg)
    alignment = []
    reference = set(vanilla.run_key)
    for variant in VARIANTS:
        for dimension in DIMENSIONS:
            for seed in SEEDS:
                group = four_class[(four_class.variant == variant) & (four_class.dimension == dimension) & (four_class.seed == seed)]
                alignment.append(len(group) == 419 and group.run_key.nunique() == 419 and set(group.run_key) == reference)
    audit = read_json(PHASE / "audits/phase06_final_oof_alignment_audit.json")
    audit["four_variant_configurations_checked"] = len(alignment); audit["four_variant_all_aligned"] = all(alignment); audit["phase05_interface_hashes"] = hashes
    audit["ridge_status"] = "COMMON_ENCODER_READOUT_BASELINE"; audit["ridge_copies_created"] = 0; audit["result"] = "PASS" if all(alignment) and len(four_class) == 33520 and len(four_reg) == 33520 else "FAIL"
    atomic_json(PHASE / "audits/phase06_final_oof_alignment_audit.json", audit)
    if audit["result"] != "PASS": raise RuntimeError("Four-variant OOF alignment failed")
    return four_class.reset_index(drop=True), four_reg.reset_index(drop=True), ridge


def classification_row(group: pd.DataFrame) -> tuple[dict[str, Any], list[list[int]]]:
    truth = group.true_class.to_numpy(int); pred = group.predicted_class.to_numpy(int); matrix = confusion_matrix(truth, pred, labels=[0, 1, 2, 3])
    return {"oof_rows": len(group), "accuracy": float(accuracy_score(truth, pred)), "balanced_accuracy": float(balanced_accuracy_score(truth, pred)), "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)), "weighted_f1": float(f1_score(truth, pred, average="weighted", zero_division=0)), "severe_error_rate": float(np.mean(np.abs(truth - pred) >= 2))}, matrix.astype(int).tolist()


def regression_row(group: pd.DataFrame, raw_column: str, bounded_column: str) -> dict[str, Any]:
    truth = group.target_score.to_numpy(float); raw = group[raw_column].to_numpy(float); bounded = group[bounded_column].to_numpy(float)
    rho = spearmanr(truth, bounded).statistic
    return {"oof_rows": len(group), "mae_raw": float(mean_absolute_error(truth, raw)), "mae_bounded": float(mean_absolute_error(truth, bounded)), "rmse_bounded": float(mean_squared_error(truth, bounded) ** 0.5), "r2_bounded": float(r2_score(truth, bounded)), "spearman_bounded": float(rho)}


def max_metric_diff(actual: pd.DataFrame, saved: pd.DataFrame, keys: list[str], metrics: list[str]) -> float:
    merged = actual.merge(saved, on=keys, suffixes=("_actual", "_saved"), validate="one_to_one")
    if len(merged) != len(actual) or len(merged) != len(saved): return float("inf")
    return max(abs(pd.to_numeric(merged[f"{metric}_actual"]) - pd.to_numeric(merged[f"{metric}_saved"])).max() for metric in metrics)


def recalculate_metrics(four_class: pd.DataFrame, four_reg: pd.DataFrame, ridge: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_rows, reg_rows, ridge_rows, matrices = [], [], [], {}
    for (variant, dimension, seed), group in four_class.groupby(["variant", "dimension", "seed"], sort=True):
        metrics, matrix = classification_row(group); class_rows.append({"variant": variant, "dimension": dimension, "seed": seed, **metrics}); matrices[f"{variant}|dimension={dimension}|seed={seed}"] = matrix
    for (variant, dimension, seed), group in four_reg.groupby(["variant", "dimension", "seed"], sort=True): reg_rows.append({"variant": variant, "regression_head": "similarity", "dimension": dimension, "seed": seed, **regression_row(group, "similarity_prediction_raw", "similarity_prediction_bounded")})
    for (dimension, seed), group in ridge.groupby(["dimension", "seed"], sort=True): ridge_rows.append({"variant": "common_ridge", "regression_head": "COMMON_ENCODER_READOUT_BASELINE", "dimension": dimension, "seed": seed, **regression_row(group, "ridge_prediction_raw", "ridge_prediction_bounded")})
    classification = pd.DataFrame(class_rows); regression = pd.DataFrame(reg_rows); ridge_metrics = pd.DataFrame(ridge_rows)
    atomic_csv(PHASE / "results/summaries/phase06_classification_metrics_by_config.csv", classification)
    atomic_csv(PHASE / "results/summaries/phase06_similarity_regression_metrics_by_config.csv", regression)
    atomic_csv(PHASE / "results/summaries/phase06_common_ridge_metrics_by_config.csv", ridge_metrics)
    atomic_json(PHASE / "results/summaries/phase06_confusion_matrices_by_config.json", {"labels": [0, 1, 2, 3], "matrices": matrices})
    class_metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "severe_error_rate"]
    reg_metrics = ["mae_raw", "mae_bounded", "rmse_bounded", "r2_bounded", "spearman_bounded"]
    diffs = {
        "phase05_classification": max_metric_diff(classification[classification.variant == "vanilla"].drop(columns="variant"), pd.read_csv(PHASE05 / "results/summaries/vanilla_hdc_classification_oof_metrics_by_config.csv"), ["dimension", "seed"], class_metrics),
        "phase05_similarity": max_metric_diff(regression[regression.variant == "vanilla"].drop(columns=["variant", "regression_head"]), pd.read_csv(PHASE05 / "results/summaries/vanilla_hdc_similarity_regression_oof_metrics_by_config.csv"), ["dimension", "seed"], reg_metrics),
        "phase05_ridge": max_metric_diff(ridge_metrics.drop(columns=["variant", "regression_head"]), pd.read_csv(PHASE05 / "results/summaries/vanilla_hdc_ridge_regression_oof_metrics_by_config.csv"), ["dimension", "seed"], reg_metrics),
    }
    fold_diffs = []
    for variant in NEW_VARIANTS:
        source = pd.read_csv(PHASE / f"results/oof/phase06_{variant}_final_oof.csv")
        for fold in range(1, 6):
            saved_c = pd.read_csv(PHASE / f"results/fold_metrics/{variant}_final_confirmation_fold_{fold}_classification_metrics.csv")
            saved_r = pd.read_csv(PHASE / f"results/fold_metrics/{variant}_final_confirmation_fold_{fold}_similarity_regression_metrics.csv")
            calc_c, calc_r = [], []
            for (dimension, seed), group in source[source.outer_fold == fold].groupby(["dimension", "seed"]):
                row, _ = classification_row(group); calc_c.append({"dimension": dimension, "seed": seed, **row})
                calc_r.append({"dimension": dimension, "seed": seed, **regression_row(group, "similarity_prediction_raw", "similarity_prediction_bounded")})
            fold_diffs.append({"variant": variant, "outer_fold": fold, "classification_max_abs_diff": max_metric_diff(pd.DataFrame(calc_c), saved_c, ["dimension", "seed"], class_metrics), "regression_max_abs_diff": max_metric_diff(pd.DataFrame(calc_r), saved_r, ["dimension", "seed"], reg_metrics)})
    maximum = max([*diffs.values(), *[max(row["classification_max_abs_diff"], row["regression_max_abs_diff"]) for row in fold_diffs]])
    audit = {"phase": "06", "audit": "metric_recalculation", "tolerance": TOLERANCE, "phase05_differences": diffs, "phase06_fold_differences": fold_diffs, "maximum_absolute_difference": maximum, "copied_fold_metrics": False, "result": "PASS" if maximum <= TOLERANCE else "FAIL"}
    atomic_json(PHASE / "audits/phase06_metric_recalculation_audit.json", audit)
    if audit["result"] != "PASS": raise RuntimeError(f"Metric recalculation mismatch {maximum}")
    return classification, regression, ridge_metrics


def seed_aggregate(frame: pd.DataFrame, group_columns: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple): keys = (keys,)
        row = dict(zip(group_columns, keys)); row["seeds"] = ",".join(str(value) for value in sorted(group.seed.unique()))
        for metric in metrics:
            values = group[metric].astype(float)
            row.update({f"{metric}_count": len(values), f"{metric}_mean": values.mean(), f"{metric}_sd_sample": values.std(ddof=1), f"{metric}_median": values.median(), f"{metric}_min": values.min(), f"{metric}_max": values.max()})
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_stability(classification: pd.DataFrame, regression: pd.DataFrame, ridge: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_metrics = ["macro_f1", "balanced_accuracy", "accuracy", "severe_error_rate"]
    reg_metrics = ["mae_bounded", "rmse_bounded", "r2_bounded", "spearman_bounded"]
    class_agg = seed_aggregate(classification, ["variant", "dimension"], class_metrics)
    reg_agg = seed_aggregate(regression, ["variant", "regression_head", "dimension"], reg_metrics)
    ridge_agg = seed_aggregate(ridge, ["variant", "regression_head", "dimension"], reg_metrics)
    atomic_csv(PHASE / "results/summaries/phase06_classification_seed_aggregate.csv", class_agg)
    atomic_csv(PHASE / "results/summaries/phase06_similarity_regression_seed_aggregate.csv", reg_agg)
    atomic_csv(PHASE / "results/summaries/phase06_common_ridge_seed_aggregate.csv", ridge_agg)
    atomic_json(PHASE / "results/summaries/phase06_dimension_and_seed_stability.json", {"phase": "06", "seed_count": 5, "seeds": SEEDS, "dimensions": DIMENSIONS, "confidence_intervals_generated": False, "confidence_interval_reason": "No final subject-level CI method was preregistered in phase06_model_selection_rules_v1.", "classification": class_agg.to_dict(orient="records"), "similarity_regression": reg_agg.to_dict(orient="records"), "common_ridge": ridge_agg.to_dict(orient="records"), "statistical_units": {"inference": "subject", "prediction": "run", "partition": "outer_fold_not_independent", "repeat": "seed_not_independent_subject"}})
    return class_agg, reg_agg, ridge_agg


def efficiency() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    vanilla = pd.read_csv(PHASE05 / "results/summaries/vanilla_hdc_efficiency_by_config.csv")
    protocol = pd.read_csv(PHASE05 / "results/summaries/vanilla_hdc_inference_efficiency_protocol_by_config.csv")
    vanilla = vanilla.merge(protocol, on=["dimension", "seed", "folds"], validate="one_to_one")
    for row in vanilla.itertuples(index=False):
        rows.append({"variant": "vanilla", "dimension": row.dimension, "seed": row.seed, "fit_time_seconds": row.fit_time_seconds, "prediction_time_seconds": row.prediction_time_seconds, "total_runtime_seconds": row.measured_total_runtime_seconds, "peak_memory_bytes": row.maximum_inference_python_peak_allocated_bytes_across_folds, "peak_memory_scope": "maximum inference Python allocation across folds", "model_size_bytes": row.model_artifact_size_bytes_across_folds, "encoding_throughput_rows_per_second": row.encoding_throughput_rows_per_second, "source": "Phase 05 frozen measured efficiency"})
    for variant in NEW_VARIANTS:
        combined = pd.concat([pd.read_csv(PHASE / f"results/efficiency/{variant}_final_confirmation_fold_{fold}_efficiency.csv") for fold in range(1, 6)], ignore_index=True)
        for (dimension, seed), group in combined.groupby(["dimension", "seed"], sort=True):
            model_time = group.model_training_and_inference_seconds.astype(float).sum(); prep = group.preprocessing_seconds.astype(float).sum(); encoding = group.encoding_seconds.astype(float).sum()
            rows.append({"variant": variant, "dimension": dimension, "seed": seed, "fit_time_seconds": "NOT_AVAILABLE", "prediction_time_seconds": "NOT_AVAILABLE", "total_runtime_seconds": model_time + prep + encoding, "peak_memory_bytes": "NOT_AVAILABLE", "peak_memory_scope": "NOT_AVAILABLE", "model_size_bytes": group.model_bytes.astype(float).max(), "encoding_throughput_rows_per_second": "NOT_AVAILABLE", "source": "Phase 06 recorded preprocessing + encoding + combined model training/inference times"})
    frame = pd.DataFrame(rows); atomic_csv(PHASE / "results/summaries/phase06_efficiency_by_config.csv", frame)
    numeric = ["total_runtime_seconds", "model_size_bytes"]
    aggregates = []
    for (variant, dimension), group in frame.groupby(["variant", "dimension"], sort=True):
        row: dict[str, Any] = {"variant": variant, "dimension": dimension, "seed_count": len(group), "seeds": ",".join(str(v) for v in sorted(group.seed))}
        for metric in numeric:
            values = pd.to_numeric(group[metric]); row.update({f"{metric}_mean": values.mean(), f"{metric}_sd_sample": values.std(ddof=1), f"{metric}_median": values.median(), f"{metric}_min": values.min(), f"{metric}_max": values.max()})
        row["peak_memory_complete"] = bool((group.peak_memory_bytes != "NOT_AVAILABLE").all()); row["encoding_throughput_complete"] = bool((group.encoding_throughput_rows_per_second != "NOT_AVAILABLE").all())
        aggregates.append(row)
    agg = pd.DataFrame(aggregates); atomic_csv(PHASE / "results/summaries/phase06_efficiency_seed_aggregate.csv", agg)
    return frame, agg


def pareto_mask(performance: np.ndarray, runtime: np.ndarray, maximize: bool) -> np.ndarray:
    keep = np.ones(len(performance), dtype=bool)
    for i in range(len(performance)):
        better_perf = performance >= performance[i] if maximize else performance <= performance[i]
        no_slower = runtime <= runtime[i]
        strict = (performance > performance[i] if maximize else performance < performance[i]) | (runtime < runtime[i])
        if np.any(better_perf & no_slower & strict): keep[i] = False
    return keep


def pareto_analysis(classification: pd.DataFrame, regression: pd.DataFrame, ridge: pd.DataFrame, efficiency_frame: pd.DataFrame) -> pd.DataFrame:
    base = efficiency_frame[["variant", "dimension", "seed", "total_runtime_seconds", "peak_memory_bytes", "model_size_bytes"]]
    class_rows = classification.merge(base, on=["variant", "dimension", "seed"], validate="one_to_one"); class_rows["task"] = "classification"; class_rows["head"] = "classification"; class_rows["performance_metric"] = "macro_f1"; class_rows["performance_value"] = class_rows.macro_f1; class_rows["performance_direction"] = "maximize"
    similarity = regression.merge(base, on=["variant", "dimension", "seed"], validate="one_to_one"); similarity["task"] = "regression"; similarity["head"] = "similarity"; similarity["performance_metric"] = "mae_bounded"; similarity["performance_value"] = similarity.mae_bounded; similarity["performance_direction"] = "minimize"
    ridge_base = base[base.variant == "vanilla"].copy(); ridge_base["variant"] = "common_ridge"
    ridge_rows = ridge.merge(ridge_base, on=["variant", "dimension", "seed"], validate="one_to_one"); ridge_rows["task"] = "regression"; ridge_rows["head"] = "COMMON_ENCODER_READOUT_BASELINE"; ridge_rows["performance_metric"] = "mae_bounded"; ridge_rows["performance_value"] = ridge_rows.mae_bounded; ridge_rows["performance_direction"] = "minimize"
    columns = ["task", "variant", "head", "dimension", "seed", "performance_metric", "performance_value", "performance_direction", "total_runtime_seconds", "peak_memory_bytes", "model_size_bytes"]
    result = pd.concat([class_rows[columns], similarity[columns], ridge_rows[columns]], ignore_index=True)
    result["pareto_dimensions"] = "performance_time_2d"; result["memory_pareto_available"] = False; result["is_pareto"] = False
    for task, indices in result.groupby("task").groups.items():
        subset = result.loc[indices]; result.loc[indices, "is_pareto"] = pareto_mask(subset.performance_value.to_numpy(float), subset.total_runtime_seconds.to_numpy(float), maximize=task == "classification")
    atomic_csv(PHASE / "results/summaries/phase06_performance_efficiency_pareto.csv", result)
    return result


def model_selection(class_agg: pd.DataFrame, reg_agg: pd.DataFrame, ridge_agg: pd.DataFrame, efficiency_agg: pd.DataFrame, pareto: pd.DataFrame) -> dict[str, Any]:
    rules_path = PHASE / "configs/phase06_model_selection_rules.json"; rules = read_json(rules_path)
    missing = [
        "final four-variant outer-OOF classification candidate-family ranking scope",
        "final classification tie-breaking across variant × dimension families using five-seed aggregates",
        "final regression candidate ranking including similarity heads and COMMON_ENCODER_READOUT_BASELINE",
        "final regression tie-breaking across five-seed aggregates",
        "rule for incorporating efficiency/Pareto status into final selection",
    ]
    contradictions = ["classification_only=true", "regression_heads_executed=false", "scope is inner-CV Quick Screen per outer fold/new variant"]
    candidates = []
    for row in class_agg.itertuples(index=False):
        eff = efficiency_agg[(efficiency_agg.variant == row.variant) & (efficiency_agg.dimension == row.dimension)].iloc[0]
        candidates.append({"task": "classification", "variant": row.variant, "dimension": row.dimension, "primary_metric": "macro_f1", "seed_mean": row.macro_f1_mean, "seed_sd_sample": row.macro_f1_sd_sample, "runtime_seconds_mean": eff.total_runtime_seconds_mean, "pareto_seed_config_count": int(pareto[(pareto.task == "classification") & (pareto.variant == row.variant) & (pareto.dimension == row.dimension)].is_pareto.sum()), "selection_status": "NOT_RANKED", "exclusion_reason": "FINAL_SELECTION_RULE_NOT_PREREGISTERED"})
    heads = pd.concat([reg_agg, ridge_agg], ignore_index=True)
    for row in heads.itertuples(index=False):
        eff_variant = "vanilla" if row.variant == "common_ridge" else row.variant
        eff = efficiency_agg[(efficiency_agg.variant == eff_variant) & (efficiency_agg.dimension == row.dimension)].iloc[0]
        candidates.append({"task": "regression", "variant": row.variant, "dimension": row.dimension, "primary_metric": "mae_bounded", "seed_mean": row.mae_bounded_mean, "seed_sd_sample": row.mae_bounded_sd_sample, "runtime_seconds_mean": eff.total_runtime_seconds_mean, "pareto_seed_config_count": int(pareto[(pareto.task == "regression") & (pareto.variant == row.variant) & (pareto.dimension == row.dimension)].is_pareto.sum()), "selection_status": "NOT_RANKED", "exclusion_reason": "FINAL_SELECTION_RULE_NOT_PREREGISTERED"})
    trace = pd.DataFrame(candidates).sort_values(["task", "variant", "dimension"], kind="mergesort"); atomic_csv(PHASE / "results/summaries/phase06_model_selection_trace.csv", trace)
    audit = {"phase": "06", "audit": "model_selection", "timestamp_utc": now(), "status": "MODEL_SELECTION_BLOCKED", "selection_rule": str(rules_path), "selection_rule_sha256": sha256(rules_path), "rule_contract_version": rules.get("contract_version"), "rule_scope": rules.get("scope"), "rule_classification_only": rules.get("classification_only"), "rule_regression_heads_executed": rules.get("regression_heads_executed"), "missing_fields": missing, "contradictions_with_requested_final_selection": contradictions, "candidate_count": len(candidates), "best_classification_hdc_saved": False, "best_regression_hdc_saved": False, "outer_test_scores_used_to_invent_rules": False, "result": "FAIL"}
    atomic_json(PHASE / "audits/phase06_model_selection_audit.json", audit)
    return audit


def comparison_tables(class_agg: pd.DataFrame, reg_agg: pd.DataFrame, ridge_agg: pd.DataFrame, efficiency_agg: pd.DataFrame, pareto: pd.DataFrame) -> None:
    classification = class_agg.merge(efficiency_agg[["variant", "dimension", "total_runtime_seconds_mean", "peak_memory_complete"]], on=["variant", "dimension"], validate="one_to_one")
    classification["macro_f1_mean_sd"] = classification.apply(lambda r: f"{r.macro_f1_mean:.6f} ± {r.macro_f1_sd_sample:.6f} SD (5 seeds)", axis=1); classification["pareto_seed_config_count"] = classification.apply(lambda r: int(pareto[(pareto.task == "classification") & (pareto.variant == r.variant) & (pareto.dimension == r.dimension)].is_pareto.sum()), axis=1); classification["final_selection_status"] = "MODEL_SELECTION_BLOCKED"
    atomic_csv(PHASE / "results/summaries/phase06_four_hdc_classification_comparison.csv", classification)
    similarity = reg_agg.merge(efficiency_agg[["variant", "dimension", "total_runtime_seconds_mean", "peak_memory_complete"]], on=["variant", "dimension"], validate="one_to_one"); similarity["mae_mean_sd"] = similarity.apply(lambda r: f"{r.mae_bounded_mean:.6f} ± {r.mae_bounded_sd_sample:.6f} SD (5 seeds)", axis=1); similarity["final_selection_status"] = "MODEL_SELECTION_BLOCKED"
    atomic_csv(PHASE / "results/summaries/phase06_four_hdc_similarity_regression_comparison.csv", similarity)
    ridge_copy = ridge_agg.copy(); ridge_copy["runtime_reference_variant"] = "vanilla"; heads = pd.concat([similarity, ridge_copy], ignore_index=True, sort=False); heads["regression_estimand"] = "bounded difficulty-induced workload proxy regression"; heads["common_ridge_status"] = heads.variant.map(lambda value: "COMMON_ENCODER_READOUT_BASELINE" if value == "common_ridge" else "NOT_APPLICABLE")
    atomic_csv(PHASE / "results/summaries/phase06_hdc_regression_head_comparison.csv", heads)
    final = classification[["variant", "dimension", "seeds", "macro_f1_mean", "macro_f1_sd_sample", "total_runtime_seconds_mean", "peak_memory_complete", "pareto_seed_config_count", "final_selection_status"]].merge(similarity[["variant", "dimension", "mae_bounded_mean", "mae_bounded_sd_sample"]], on=["variant", "dimension"], validate="one_to_one")
    final["common_ridge_reference"] = "See phase06_hdc_regression_head_comparison.csv; not duplicated per variant"; final["regression_estimand"] = "bounded difficulty-induced workload proxy regression"
    atomic_csv(PHASE / "results/summaries/phase06_final_hdc_variant_comparison.csv", final)


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); plt.tight_layout(); plt.savefig(path, dpi=600, bbox_inches="tight"); plt.close()


def figures(classification: pd.DataFrame, regression: pd.DataFrame, ridge: pd.DataFrame, class_agg: pd.DataFrame, reg_agg: pd.DataFrame, ridge_agg: pd.DataFrame, pareto: pd.DataFrame) -> list[dict[str, Any]]:
    catalog = []; figdir = PHASE / "figures"
    path = figdir / "phase06_classification_macro_f1_by_variant_dimension.png"; plt.figure(figsize=(8, 5))
    for variant in VARIANTS:
        rows = class_agg[class_agg.variant == variant].sort_values("dimension"); plt.errorbar(rows.dimension, rows.macro_f1_mean, yerr=rows.macro_f1_sd_sample, marker="o", capsize=3, label=DISPLAY[variant], color=COLORS[variant])
    plt.xscale("log"); plt.xticks(DIMENSIONS, [str(v) for v in DIMENSIONS]); plt.ylim(0, 1); plt.xlabel("Hypervector dimension"); plt.ylabel("OOF Macro-F1 (mean ± SD across 5 seeds)"); plt.legend(fontsize=8); plt.grid(alpha=.25); save_figure(path)
    catalog.append({"filename": path.name, "purpose": "Compare classification performance and seed variability across variants and dimensions.", "data_source": "phase06_classification_seed_aggregate.csv", "reader_notice": "Points are five-seed means; bars are sample SD, not confidence intervals.", "decision_change": "Describes performance/stability trade-offs but cannot select a final family without a preregistered final rule.", "caveat": "Seeds are repeated runs, not independent subjects."})
    path = figdir / "phase06_classification_seed_stability.png"; fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, variant in zip(axes.ravel(), VARIANTS):
        rows = classification[classification.variant == variant]
        for seed in SEEDS:
            values = rows[rows.seed == seed].sort_values("dimension"); ax.plot(values.dimension, values.macro_f1, marker="o", alpha=.8, label=f"seed {seed}")
        ax.set_title(DISPLAY[variant]); ax.set_xscale("log"); ax.set_xticks(DIMENSIONS, [str(v) for v in DIMENSIONS]); ax.set_ylim(0, 1); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7); fig.supxlabel("Hypervector dimension"); fig.supylabel("OOF Macro-F1 per seed"); save_figure(path)
    catalog.append({"filename": path.name, "purpose": "Expose seed-specific classification trajectories rather than only means.", "data_source": "phase06_classification_metrics_by_config.csv", "reader_notice": "Each line is one registered seed across dimensions.", "decision_change": "Reveals whether dimension trends are consistent across seeds.", "caveat": "No seed is selected individually."})
    path = figdir / "phase06_similarity_regression_mae_by_variant_dimension.png"; plt.figure(figsize=(8, 5))
    for variant in VARIANTS:
        rows = reg_agg[reg_agg.variant == variant].sort_values("dimension"); plt.errorbar(rows.dimension, rows.mae_bounded_mean, yerr=rows.mae_bounded_sd_sample, marker="o", capsize=3, label=DISPLAY[variant], color=COLORS[variant])
    plt.xscale("log"); plt.xticks(DIMENSIONS, [str(v) for v in DIMENSIONS]); plt.ylim(bottom=0); plt.xlabel("Hypervector dimension"); plt.ylabel("Bounded MAE (mean ± SD across 5 seeds)"); plt.legend(fontsize=8); plt.grid(alpha=.25); save_figure(path)
    catalog.append({"filename": path.name, "purpose": "Compare similarity-regression error and seed variability.", "data_source": "phase06_similarity_regression_seed_aggregate.csv", "reader_notice": "Lower MAE is better; bars are sample SD.", "decision_change": "Shows the descriptive regression trade-off by representation size.", "caveat": "The estimand is a bounded difficulty-induced workload proxy."})
    path = figdir / "phase06_regression_head_comparison.png"; plt.figure(figsize=(9, 5))
    heads = pd.concat([reg_agg, ridge_agg], ignore_index=True)
    for variant in [*VARIANTS, "common_ridge"]:
        rows = heads[heads.variant == variant].sort_values("dimension"); plt.errorbar(rows.dimension, rows.mae_bounded_mean, yerr=rows.mae_bounded_sd_sample, marker="o", capsize=3, label=DISPLAY[variant], color=COLORS[variant])
    plt.xscale("log"); plt.xticks(DIMENSIONS, [str(v) for v in DIMENSIONS]); plt.ylim(bottom=0); plt.xlabel("Hypervector dimension"); plt.ylabel("Bounded MAE (mean ± SD across 5 seeds)"); plt.legend(fontsize=8); plt.grid(alpha=.25); save_figure(path)
    catalog.append({"filename": path.name, "purpose": "Compare four similarity heads with the single common Ridge baseline.", "data_source": "phase06_similarity_regression_seed_aggregate.csv and phase06_common_ridge_seed_aggregate.csv", "reader_notice": "Common Ridge is plotted once, not copied to four variants.", "decision_change": "Shows whether a regularized sample-HV readout changes the descriptive error range.", "caveat": "No final regression head can be selected because the frozen final ranking rule is absent."})
    path = figdir / "phase06_performance_time_pareto.png"; fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, task in zip(axes, ["classification", "regression"]):
        rows = pareto[pareto.task == task]
        for variant in rows.variant.unique():
            group = rows[rows.variant == variant]; dominated = group[~group.is_pareto]; frontier = group[group.is_pareto]
            ax.scatter(dominated.total_runtime_seconds, dominated.performance_value, s=18, alpha=.4, label=DISPLAY.get(variant, variant), color=COLORS.get(variant, "gray"))
            ax.scatter(frontier.total_runtime_seconds, frontier.performance_value, s=55, alpha=.95, color=COLORS.get(variant, "gray"), edgecolors="black", linewidths=.4)
        ax.set_xscale("log"); ax.set_xlabel("Recorded total runtime (seconds, log scale)"); ax.set_ylabel("Macro-F1 (higher better)" if task == "classification" else "Bounded MAE (lower better)"); ax.set_title(task.capitalize() + " performance–time Pareto"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); save_figure(path)
    catalog.append({"filename": path.name, "purpose": "Identify nondominated performance–time configurations.", "data_source": "phase06_performance_efficiency_pareto.csv", "reader_notice": "Larger markers are on the two-dimensional Pareto front.", "decision_change": "Separates efficient trade-offs from dominated configurations.", "caveat": "Runtime protocols differ in granularity; peak memory is incomplete, so this is not a three-objective Pareto analysis."})
    return catalog


def analysis_reports(class_agg: pd.DataFrame, reg_agg: pd.DataFrame, ridge_agg: pd.DataFrame, pareto: pd.DataFrame, catalog: list[dict[str, Any]], selection: dict[str, Any]) -> None:
    top_class = class_agg.sort_values(["macro_f1_mean", "macro_f1_sd_sample"], ascending=[False, True]).iloc[0]
    all_reg = pd.concat([reg_agg, ridge_agg], ignore_index=True); top_reg = all_reg.sort_values(["mae_bounded_mean", "mae_bounded_sd_sample"]).iloc[0]
    report = f"""# Phase 06 Strict Analysis Report

## Comparison question

Across four P1 HDC families and four registered dimensions, how do OOF classification performance, bounded difficulty-induced workload proxy regression, five-seed stability, and recorded efficiency compare, and can the frozen rules select final working models?

## Comparability verification

All four classification and similarity-regression variants contain 20 complete dimension × seed configurations, each aligned on the same 419 run keys, 35 subjects, and five frozen outer folds. Vanilla predictions are read directly from the frozen Phase 05 interface. The common Ridge readout is referenced once and is not duplicated by variant.

## Descriptive observations

- The highest observed five-seed mean classification Macro-F1 is `{top_class.macro_f1_mean:.6f}` for {DISPLAY[top_class.variant]} at dimension {int(top_class.dimension)} (sample SD `{top_class.macro_f1_sd_sample:.6f}`). This is a descriptive maximum, not a frozen final selection.
- The lowest observed five-seed mean bounded MAE is `{top_reg.mae_bounded_mean:.6f}` for {DISPLAY[top_reg.variant]} at dimension {int(top_reg.dimension)} (sample SD `{top_reg.mae_bounded_sd_sample:.6f}`). This is a descriptive minimum, not a frozen final selection.
- The valid Pareto analysis is performance–time two-dimensional because peak memory is unavailable for all three new variants. No three-objective memory claim is made.

## Model-selection outcome

`MODEL_SELECTION_BLOCKED`. The frozen file governs inner-CV Quick Screen selection only, is classification-only, and explicitly says regression heads were not executed. It does not preregister final outer-OOF classification or regression ranking/tie-breaking. Adding such rules after seeing OOF outcomes would violate the frozen protocol. Consequently, neither best-classification nor best-regression configuration is announced and Phase 06 is not frozen.

## Statistical boundary

Subject is the inferential unit; run is the prediction unit; fold is a partition; seed is a repeated algorithmic run. No subject-level paired bootstrap, permutation test, effect-size definition, multiplicity procedure, or final-selection CI was preregistered. The analysis therefore reports descriptive five-seed summaries only and does not use “significantly better.”

## Selection-induced optimism

Any future selection using these complete outer-OOF results would make the selected OOF score selection-conditioned, not an independent confirmation estimate. A later preregistered LOSO or robustness phase is required for independent confirmation wording.

## Claim candidates

- Claim: Four P1 HDC families were compared on exactly aligned OOF prediction sets.
  - Source evidence: final OOF coverage/alignment audits.
  - Allowed wording: “All compared configurations covered the same 419 runs.”
  - Forbidden stronger wording: “One method significantly outperformed the others.”
  - Uncertainty: No preregistered subject-level inference was run.
  - Next check: Freeze a final selection rule before ranking models.
  - Decision: keep.
- Claim: The common Ridge readout is a single encoder-level baseline.
  - Source evidence: frozen Phase 05 OOF and Phase 06 Ridge contract.
  - Allowed wording: “Common Ridge was referenced once across the comparison.”
  - Forbidden stronger wording: “Each prototype variant has a separate Ridge head.”
  - Uncertainty: Runtime attribution uses the frozen Vanilla interface.
  - Next check: none for Phase 06.
  - Decision: keep.
"""
    stats = """# Phase 06 Statistics Appendix

## Units and sample structure

- Inferential unit: subject (35).
- Prediction unit: run (419 per configuration).
- Outer folds: 5 partition units; not treated as independent samples.
- Seeds: 5 registered repeats; not treated as independent subjects.
- Dimensions: 1000, 2000, 5000, 10000.

## Descriptive statistics

For every variant × dimension, the bundle reports count, mean, sample SD, median, minimum, and maximum across the five registered seeds for classification and regression metrics. No seed predictions are averaged into a new ensemble, no seed is deleted, and no single seed is selected.

## Inferential analysis

The frozen `phase06_model_selection_rules_v1` does not preregister a subject-level paired bootstrap, permutation test, effect-size estimator, confidence-interval method, or multiple-comparison correction for the final four-model comparison. No inferential test, effect size, CI, or multiplicity-adjusted p-value was added post hoc. Inferential analysis is deferred until a future statistical protocol is frozen.

## Metric audit

Metrics were independently recomputed from OOF prediction rows. Phase 05 OOF metrics and Phase 06 saved fold metrics were cross-checked at maximum absolute tolerance 1e-12.

## Limitations

- Seed SD describes algorithmic variability, not subject-level uncertainty.
- OOF predictions support aligned descriptive comparison but a future selection on these scores induces optimism.
- Peak memory is incomplete across variants, limiting Pareto analysis to performance and time.
- Runtime instrumentation differs in field granularity between Phase 05 and the new variants; all missing fields remain `NOT_AVAILABLE`.
"""
    figure_lines = ["# Phase 06 Figure Catalog", ""]
    for item in catalog:
        figure_lines.extend([f"## {item['filename']}", "", f"- Purpose: {item['purpose']}", f"- Data source: {item['data_source']}", f"- Reader should notice: {item['reader_notice']}", f"- What this changes: {item['decision_change']}", f"- Caveat: {item['caveat']}", ""])
    figure_lines.extend(["## Figures intentionally not created", "", "- `phase06_performance_memory_pareto.png`: peak memory is incomplete across variants; creating it would imply a false three-objective comparison.", "- Best-classification confusion matrix and best-regression prediction/residual figures: final selection is blocked by the incomplete frozen rule, so no best model may be declared.", ""])
    atomic_text(PHASE / "reports/analysis-output/analysis-report.md", report); atomic_text(PHASE / "reports/analysis-output/stats-appendix.md", stats); atomic_text(PHASE / "reports/analysis-output/figure-catalog.md", "\n".join(figure_lines))
    summary = f"""# Phase 06 Final Summary

OOF consolidation, four-family alignment, independent metric recalculation, seed/dimension stability summaries, actual-record efficiency tables, two-dimensional Pareto analysis, scientific figures, and the strict analysis bundle are complete.

Phase 06 remains **NOT_FROZEN** because final model selection is `MODEL_SELECTION_BLOCKED`. The frozen selection rules cover inner-CV Quick Screen classification only and do not define final four-family classification or regression selection. No best HDC configuration was created, and Phase 07 must not begin.

Key audits: final OOF coverage PASS; alignment PASS; leakage PASS; metric recalculation PASS; model selection FAIL (preregistered-rule blocker).
"""
    atomic_text(PHASE / "reports/phase06_final_summary.md", summary)


def append_readme() -> None:
    path = PHASE / "README.md"; text = path.read_text(encoding="utf-8")
    marker = "## Final OOF consolidation status"
    if marker not in text:
        text += f"\n\n{marker}\n\n- OOF consolidation, independent metric recalculation, stability analysis, Pareto analysis, figures, reports, and final manifest: complete.\n- Model selection: `MODEL_SELECTION_BLOCKED` because the frozen rules do not define final four-family classification/regression ranking.\n- Phase 06 status: `NOT_FROZEN`; Phase 07 is not authorized.\n- Key artifacts: `results/oof/`, `results/summaries/phase06_*`, `reports/analysis-output/`, `reports/phase06_final_summary.md`, and `manifests/phase06_final_artifact_manifest.json`.\n"
        atomic_text(path, text)


def persist_notebook(selection: dict[str, Any]) -> dict[str, Any]:
    path = PHASE / "Phase_06_HDC_Variant_Screening.ipynb"; notebook = read_json(path); cells = notebook["cells"]; prior_count = len(cells)
    marker = "phase06_final_oof_consolidation_executed_v1"
    if any(marker in "".join(cell.get("source", [])) for cell in cells): raise RuntimeError("Final OOF notebook section already exists")
    markdown = """## Final OOF consolidation and conditional freeze

The existing Phase 05/06 predictions were consolidated without retraining or reprediction. Four classification and similarity-regression OOF libraries contain 20 dimension × seed configurations per variant, each covering 419 aligned runs. Metrics were independently recalculated; seed/dimension stability and performance–time Pareto analyses were generated from actual records.

Statistical boundary: subject is the inferential unit; run is the prediction unit; fold is a partition; seed is a repeated run. No final subject-level inferential method was preregistered, so this section reports descriptive statistics only.

Final selection status is `MODEL_SELECTION_BLOCKED`: the frozen rules govern inner-CV Quick Screen classification only and do not define final four-family classification or regression ranking. No best models were saved, no freeze file was created, and Phase 07 is not authorized.
"""
    code = """# phase06_final_oof_consolidation_executed_v1
from pathlib import Path
import json
import pandas as pd
phase = Path.cwd()
coverage = json.loads((phase / 'audits/phase06_final_oof_coverage_audit.json').read_text(encoding='utf-8'))
alignment = json.loads((phase / 'audits/phase06_final_oof_alignment_audit.json').read_text(encoding='utf-8'))
metrics = json.loads((phase / 'audits/phase06_metric_recalculation_audit.json').read_text(encoding='utf-8'))
selection = json.loads((phase / 'audits/phase06_model_selection_audit.json').read_text(encoding='utf-8'))
summary = {
  'new_variant_oof_rows': coverage['combined_rows'],
  'new_variant_complete_configs': {k: v['complete_configurations'] for k, v in coverage['variants'].items()},
  'four_variant_alignment': alignment['result'],
  'metric_recalculation': metrics['result'],
  'statistical_unit': 'subject (run prediction; fold partition; seed repeat)',
  'pareto_scope': 'performance_time_2d; memory incomplete',
  'model_selection': selection['status'],
  'best_classification_hdc': 'NOT_SELECTED',
  'best_regression_hdc': 'NOT_SELECTED',
  'primary_checksum': 'PASS',
  'frozen_fold_checksum': 'PASS',
  'phase06_status': 'NOT_FROZEN',
  'ready_for_phase07': False,
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
"""
    output = io.StringIO(); previous = Path.cwd()
    try:
        os.chdir(PHASE)
        with contextlib.redirect_stdout(output): exec(compile(code, str(path) + "#final-oof", "exec"), {"__name__": "__phase06_final_oof_cell__"})
    finally: os.chdir(previous)
    count = max([cell.get("execution_count") or 0 for cell in cells if cell.get("cell_type") == "code"] + [0]) + 1
    cells.extend([{"cell_type": "markdown", "metadata": {"phase06_stage": "final_oof"}, "source": markdown.splitlines(keepends=True)}, {"cell_type": "code", "execution_count": count, "metadata": {"phase06_stage": "final_oof", "executed": True}, "outputs": [{"name": "stdout", "output_type": "stream", "text": output.getvalue().splitlines(keepends=True)}], "source": code.splitlines(keepends=True)}])
    temporary = path.with_suffix(".ipynb.tmp"); temporary.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"); temporary.replace(path)
    persisted = read_json(path); section = persisted["cells"][prior_count:]; section_text = json.dumps(section, ensure_ascii=False)
    required = [marker, "four_variant_alignment", "metric_recalculation", "statistical_unit", "model_selection", "phase06_status", "ready_for_phase07"]
    audit = {"phase": "06", "audit": "final_notebook_persistence", "timestamp_utc": now(), "prior_cell_count": prior_count, "final_cell_count": len(persisted["cells"]), "append_only": len(persisted["cells"]) == prior_count + 2, "executed_code_cells": 1, "execution_count": count, "required_content": {value: value in section_text for value in required}, "initialization_contract_quick_screen_final_confirmation_cells_retained": prior_count == 20, "result": "PASS" if len(persisted["cells"]) == prior_count + 2 and output.getvalue().strip() and all(value in section_text for value in required) else "FAIL"}
    atomic_json(PHASE / "audits/phase06_final_notebook_persistence_audit.json", audit)
    if audit["result"] != "PASS": raise RuntimeError("Final notebook persistence failed")
    return audit


def update_work_files(status: str) -> None:
    plan = (PHASE / "final_consolidation_task_plan.md").read_text(encoding="utf-8").replace("- [ ] Phase 1:", "- [x] Phase 1:").replace("- [ ] Phase 2:", "- [x] Phase 2:").replace("- [ ] Phase 3:", "- [x] Phase 3:").replace("- [ ] Phase 4:", "- [x] Phase 4:").replace("- [ ] Phase 5:", "- [x] Phase 5:").replace("- [ ] Phase 6:", "- [x] Phase 6:").replace("- [ ] Phase 7:", "- [x] Phase 7:").replace("- [ ] Phase 8:", "- [x] Phase 8:").replace("- [ ] Phase 9:", "- [x] Phase 9:")
    plan = plan.replace("**Currently in Phase 1** — running the complete freeze preflight.", f"**Completed with {status}** — all permitted consolidation and analysis work finished; stopped before Phase 07.")
    atomic_text(PHASE / "final_consolidation_task_plan.md", plan)
    notes = (PHASE / "final_consolidation_notes.md").read_text(encoding="utf-8").replace("- Pending preflight.", "- Preflight, OOF coverage/alignment, metric recalculation, stability, efficiency, Pareto, reports, figures, and Notebook persistence passed.\n- Final model selection is blocked because the frozen rules do not specify final outer-OOF classification/regression ranking or tie-breaking.\n- No best-model configs or phase06_freeze.json were created; status is NOT_FROZEN.")
    atomic_text(PHASE / "final_consolidation_notes.md", notes)


def upstream_integrity() -> dict[str, Any]:
    snapshot = read_json(PHASE / "audits/phase06_upstream_pre_finalization_snapshot.json"); failures = []
    for item in snapshot["phase05_artifacts"]:
        path = PHASE05 / item["relative_path"]
        if not path.exists() or sha256(path) != item["actual_sha256"] or path.stat().st_size != item["actual_size"]: failures.append("phase05:" + item["relative_path"])
    authorized = {"Phase_06_HDC_Variant_Screening.ipynb"}
    for item in snapshot["phase06_final_confirmation_artifacts"]:
        if item["relative_path"].replace("\\", "/") in authorized: continue
        path = PHASE / item["relative_path"]
        if not path.exists() or sha256(path) != item["actual_sha256"] or path.stat().st_size != item["actual_size"]: failures.append("phase06_final_confirmation:" + item["relative_path"])
    audit = {"phase": "06", "audit": "upstream_freeze_integrity", "timestamp_utc": now(), "phase05_artifacts_checked": len(snapshot["phase05_artifacts"]), "phase06_final_confirmation_artifacts_checked": len(snapshot["phase06_final_confirmation_artifacts"]) - 1, "authorized_phase06_notebook_append": True, "failures": failures, "result": "PASS" if not failures else "FAIL"}
    atomic_json(PHASE / "audits/phase06_upstream_freeze_integrity_audit.json", audit)
    if failures: raise RuntimeError(f"Upstream integrity failed: {failures}")
    return audit


def reproducibility_audit(selection: dict[str, Any]) -> dict[str, Any]:
    source = (PHASE / "scripts/finalize_phase06.py").read_text(encoding="utf-8")
    call_tokens = ["train_" + "onlinehd(", "train_" + "multicentroid(", "train_" + "hybrid(", "incremental_" + "encode_prefixes("]
    forbidden = [token for token in call_tokens if token in source]
    audit = {"phase": "06", "audit": "final_reproducibility", "timestamp_utc": now(), "executor_sha256": sha256(PHASE / "scripts/finalize_phase06.py"), "training_or_prediction_calls": forbidden, "source_predictions_only": not forbidden, "deterministic_oof_sort": ["variant", "dimension", "seed", "outer_fold", "subject_id", "run_key"], "metric_tolerance": TOLERANCE, "model_selection_status": selection["status"], "frozen_rule_not_modified": True, "result": "PASS" if not forbidden else "FAIL"}
    atomic_json(PHASE / "audits/phase06_final_reproducibility_audit.json", audit)
    if audit["result"] != "PASS": raise RuntimeError("Reproducibility audit failed")
    return audit


def category(path: Path) -> str:
    relative = str(path.relative_to(PHASE)).replace("\\", "/")
    if relative.startswith("results/"): return "result"
    if relative.startswith("figures/"): return "figure"
    if relative.startswith("reports/"): return "report"
    if relative.startswith("audits/"): return "audit"
    if relative.startswith("configs/"): return "config"
    if relative.startswith("scripts/"): return "reproducibility_code"
    if relative.endswith(".ipynb"): return "notebook"
    return "documentation"


def final_manifest() -> Path:
    excluded = {PHASE / "manifests/phase06_final_artifact_manifest.json", PHASE / "configs/phase06_freeze.json", PHASE / "audits/phase06_final_artifact_audit.json"}
    paths: set[Path] = set()
    patterns = ["configs/phase06_*.json", "audits/phase06_*.json", "results/oof/phase06_*", "results/summaries/phase06_*", "figures/phase06_*", "reports/phase06_*", "reports/analysis-output/*", "scripts/*phase06*.py", "scripts/finalize_phase06.py"]
    for pattern in patterns: paths.update(path for path in PHASE.glob(pattern) if path.is_file())
    paths.update([PHASE / "README.md", PHASE / "Phase_06_HDC_Variant_Screening.ipynb", PHASE / "task_plan.md", PHASE / "notes.md", PHASE / "final_confirmation_task_plan.md", PHASE / "final_confirmation_notes.md", PHASE / "final_consolidation_task_plan.md", PHASE / "final_consolidation_notes.md"])
    artifacts = [{"relative_path": str(path.relative_to(PHASE)), "category": category(path), "file_size_bytes": path.stat().st_size, "sha256": sha256(path), "completion_status": "EXISTS_AND_HASHED"} for path in sorted(paths - excluded, key=str) if path.exists()]
    manifest_path = PHASE / "manifests/phase06_final_artifact_manifest.json"
    atomic_json(manifest_path, {"phase": "06", "manifest": "final_artifacts", "timestamp_utc": now(), "artifact_count": len(artifacts), "artifacts": artifacts, "self_hash_excluded": True, "freeze_file_excluded": True, "final_artifact_audit_excluded_to_avoid_circular_hash": True, "phase_status": "NOT_FROZEN", "model_selection_status": "MODEL_SELECTION_BLOCKED"})
    return manifest_path


def artifact_audit(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path); mismatches = []
    for item in manifest["artifacts"]:
        path = PHASE / item["relative_path"]
        if not path.exists() or path.stat().st_size != item["file_size_bytes"] or sha256(path) != item["sha256"]: mismatches.append(item["relative_path"])
    audit = {"phase": "06", "audit": "final_artifact", "timestamp_utc": now(), "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "artifacts_verified": len(manifest["artifacts"]), "mismatches": mismatches, "result": "PASS" if not mismatches else "FAIL"}
    atomic_json(PHASE / "audits/phase06_final_artifact_audit.json", audit)
    if mismatches: raise RuntimeError(f"Final artifact audit failed: {mismatches}")
    return audit


def main() -> int:
    preflight(); meta = metadata(); new, _ = consolidate_new_oof(meta); four_class, four_reg, ridge = build_four_variant_oof(new, meta)
    classification, regression, ridge_metrics = recalculate_metrics(four_class, four_reg, ridge)
    class_agg, reg_agg, ridge_agg = aggregate_stability(classification, regression, ridge_metrics)
    eff, eff_agg = efficiency(); pareto = pareto_analysis(classification, regression, ridge_metrics, eff)
    selection = model_selection(class_agg, reg_agg, ridge_agg, eff_agg, pareto); comparison_tables(class_agg, reg_agg, ridge_agg, eff_agg, pareto)
    catalog = figures(classification, regression, ridge_metrics, class_agg, reg_agg, ridge_agg, pareto)
    analysis_reports(class_agg, reg_agg, ridge_agg, pareto, catalog, selection); append_readme(); persist_notebook(selection)
    update_work_files(selection["status"]); upstream_integrity(); reproducibility_audit(selection); manifest = final_manifest(); artifact_audit(manifest)
    # Selection is blocked by contract, so freeze creation is prohibited.
    if (PHASE / "configs/phase06_freeze.json").exists(): raise RuntimeError("Unexpected phase06_freeze.json exists despite blocked selection")
    print("PHASE 06 FINALIZATION COMPLETE WITH MODEL_SELECTION_BLOCKED; STATUS NOT_FROZEN")
    return 0


if __name__ == "__main__": raise SystemExit(main())
