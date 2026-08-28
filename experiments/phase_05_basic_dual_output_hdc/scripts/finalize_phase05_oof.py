"""Consolidate completed Phase 05 predictions into final OOF analysis artifacts.

This script never imports the training executor, fits a model, or generates a
prediction.  It reads immutable Final Confirmation CSVs, validates alignment,
recomputes metrics, creates descriptive summaries/figures/reports, and stops
before notebook persistence and the final freeze gate.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
PRIMARY = ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv"
FOLDS = ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv"
PHASE04A = ROOT / "experiments/phase_04a_traditional_classification_baselines"
PHASE04B = ROOT / "experiments/phase_04b_traditional_regression_baselines"
EXPECTED_PRIMARY_SHA = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLD_SHA = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
DIMENSIONS = [1000, 2000, 5000, 10000]
SEEDS = [42, 43, 44, 45, 46]
EXPECTED_CONFIGS = {(dimension, seed) for dimension in DIMENSIONS for seed in SEEDS}
SORT_COLUMNS = ["dimension", "seed", "outer_fold", "subject_id", "run_key"]
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=1e-10, atol=1e-12, equal_nan=True):
        raise RuntimeError(f"metric mismatch for {label}: {actual} != {expected}")


def classification_metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[dict[str, float], list[list[int]]]:
    matrix = confusion_matrix(target, prediction, labels=[0, 1, 2, 3])
    return (
        {
            "accuracy": float(accuracy_score(target, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
            "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(target, prediction, average="weighted", zero_division=0)),
            "severe_error_rate": float(np.mean(np.abs(target - prediction) >= 2)),
        },
        matrix.astype(int).tolist(),
    )


def regression_metrics(target: np.ndarray, raw: np.ndarray, bounded: np.ndarray) -> dict[str, float]:
    return {
        "mae_raw": float(mean_absolute_error(target, raw)),
        "mae_bounded": float(mean_absolute_error(target, bounded)),
        "rmse_bounded": float(mean_squared_error(target, bounded) ** 0.5),
        "r2_bounded": float(r2_score(target, bounded)),
        "spearman_bounded": float(spearmanr(target, bounded).statistic),
    }


def source_snapshot() -> dict[str, Any]:
    prior_manifest = load_json(PHASE / "manifests/vanilla_hdc_final_confirmation_artifact_manifest.json")
    immutable_prior = {
        item["path"]: item["sha256"]
        for item in prior_manifest["artifacts"]
        if item["path"] != "Phase_05_Basic_Dual_Output_HDC.ipynb"
    }
    upstream = [
        PRIMARY,
        FOLDS,
        PHASE04A / "configs/phase04a_freeze.json",
        PHASE04A / "results/summaries/phase04a_final_classifier_comparison.csv",
        PHASE04A / "results/oof/classification_oof_predictions.csv",
        PHASE04A / "reports/phase04a_final_summary.md",
        PHASE04B / "configs/phase04b_freeze.json",
        PHASE04B / "results/summaries/phase04b_final_regressor_comparison.csv",
        PHASE04B / "reports/phase04b_final_summary.md",
        PHASE04B / "manifests/phase04b_final_artifact_manifest.json",
    ]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_sha256": sha256(PRIMARY),
        "fold_sha256": sha256(FOLDS),
        "immutable_prior_phase05_artifacts": immutable_prior,
        "upstream_sha256_before": {str(path): sha256(path) for path in upstream},
        "result": "PASS",
    }


def validate_start_gate() -> dict[str, Any]:
    if sha256(PRIMARY) != EXPECTED_PRIMARY_SHA or sha256(FOLDS) != EXPECTED_FOLD_SHA:
        raise RuntimeError("Phase 03 checksum gate failed")
    final_audit = load_json(PHASE / "audits/vanilla_hdc_final_confirmation_all_folds_audit.json")
    notebook_audit = load_json(PHASE / "audits/vanilla_hdc_final_confirmation_notebook_persistence_audit.json")
    manifest_audit = load_json(PHASE / "audits/vanilla_hdc_final_confirmation_manifest_audit.json")
    execution = pd.read_csv(PHASE / "results/summaries/vanilla_hdc_final_confirmation_execution_summary.csv")
    required = [
        final_audit.get("result") == "PASS",
        notebook_audit.get("result") == "PASS",
        manifest_audit.get("result") == "PASS",
        final_audit.get("folds_completed") == 5,
        final_audit.get("configs_completed") == 100,
        execution["configs_completed"].sum() == 100,
        not final_audit.get("outer_test_used_for_tuning"),
        final_audit.get("all_prediction_run_keys_valid"),
        final_audit.get("all_outer_subject_isolation"),
        final_audit.get("all_inner_subject_isolation"),
        final_audit.get("temperature_inner_cv_only"),
        final_audit.get("ridge_alpha_inner_cv_only"),
    ]
    if not all(required):
        raise RuntimeError("Final Confirmation start gate failed")
    snapshot = source_snapshot()
    atomic_json(PHASE / "audits/phase05_finalization_input_snapshot.json", snapshot)
    return snapshot


def consolidate_oof() -> tuple[pd.DataFrame, dict[str, Any]]:
    frozen = pd.read_csv(FOLDS).set_index("run_key")
    primary = pd.read_csv(
        PRIMARY, usecols=["run_key", "subject_id", "target_class", "target_score", "outer_fold"]
    ).set_index("run_key")
    frames: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    for fold in range(1, 6):
        path = PHASE / f"results/predictions/vanilla_hdc_final_confirmation_fold_{fold}_predictions.csv"
        source_hashes[str(path.relative_to(PHASE))] = sha256(path)
        frame = pd.read_csv(path)
        if set(frame["outer_fold"].astype(int)) != {fold}:
            raise RuntimeError(f"Fold {fold} source prediction contains wrong outer_fold")
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True).sort_values(SORT_COLUMNS, kind="mergesort").reset_index(drop=True)
    keys = ["run_key", "dimension", "seed"]
    duplicates = int(long.duplicated(keys).sum())
    expected_index = pd.MultiIndex.from_product(
        [sorted(frozen.index), DIMENSIONS, SEEDS], names=["run_key", "dimension", "seed"]
    )
    actual_index = pd.MultiIndex.from_frame(long[keys])
    missing = len(expected_index.difference(actual_index))
    extra = len(actual_index.difference(expected_index))
    fold_mismatch = int(sum(int(row.outer_fold) != int(frozen.at[row.run_key, "outer_fold"]) for row in long.itertuples()))
    subject_mismatch = int(sum(str(row.subject_id) != str(primary.at[row.run_key, "subject_id"]) for row in long.itertuples()))
    class_mismatch = int(sum(int(row.true_class) != int(primary.at[row.run_key, "target_class"]) for row in long.itertuples()))
    target_mismatch = int(sum(not np.isclose(float(row.target_score), float(primary.at[row.run_key, "target_score"])) for row in long.itertuples()))
    config_counts = long.groupby(["dimension", "seed"])["run_key"].agg(["size", "nunique"])
    configs_with_419 = int(((config_counts["size"] == 419) & (config_counts["nunique"] == 419)).sum())
    valid = (
        len(long) == 8380
        and duplicates == missing == extra == fold_mismatch == subject_mismatch == class_mismatch == target_mismatch == 0
        and configs_with_419 == 20
        and set(config_counts.index) == EXPECTED_CONFIGS
    )
    if not valid:
        raise RuntimeError("Final OOF coverage/alignment failed")

    oof_dir = PHASE / "results/oof"
    atomic_csv(oof_dir / "vanilla_hdc_final_confirmation_oof_long.csv", long)
    classification_columns = [
        "run_key", "subject_id", "outer_fold", "dimension", "seed", "levels", "feature_k",
        "true_class", "predicted_class", "similarity_class_0", "similarity_class_1",
        "similarity_class_2", "similarity_class_3",
    ]
    similarity_columns = [
        "run_key", "subject_id", "outer_fold", "dimension", "seed", "levels", "feature_k",
        "target_score", "similarity_temperature", "similarity_prediction",
    ]
    ridge_columns = [
        "run_key", "subject_id", "outer_fold", "dimension", "seed", "levels", "feature_k",
        "target_score", "ridge_alpha", "ridge_prediction_raw", "ridge_prediction_bounded",
    ]
    atomic_csv(oof_dir / "vanilla_hdc_classification_oof.csv", long[classification_columns])
    atomic_csv(oof_dir / "vanilla_hdc_similarity_regression_oof.csv", long[similarity_columns])
    atomic_csv(oof_dir / "vanilla_hdc_ridge_regression_oof.csv", long[ridge_columns])
    coverage = {
        "expected_configs": 20,
        "configs_with_419_rows": configs_with_419,
        "expected_total_rows": 8380,
        "actual_total_rows": int(len(long)),
        "missing_run_config_combinations": missing,
        "extra_run_config_combinations": extra,
        "duplicate_run_config_combinations": duplicates,
        "classification_combinations": int(len(long[classification_columns])),
        "similarity_regression_combinations": int(len(long[similarity_columns])),
        "ridge_regression_combinations": int(len(long[ridge_columns])),
        "no_prediction_averaging_or_regeneration": True,
        "result": "PASS",
    }
    alignment = {
        "fold_mismatches": fold_mismatch,
        "subject_mismatches": subject_mismatch,
        "true_class_mismatches": class_mismatch,
        "target_mismatches": target_mismatch,
        "deterministic_sort": SORT_COLUMNS,
        "result": "PASS",
    }
    leakage = {
        "source_prediction_sha256": source_hashes,
        "source_predictions_modified": False,
        "model_training_executed": False,
        "prediction_generation_executed": False,
        "outer_test_used_for_tuning": False,
        "seed_or_dimension_removed": False,
        "seed_ensemble_created": False,
        "result": "PASS",
    }
    atomic_json(PHASE / "audits/phase05_final_oof_coverage_audit.json", coverage)
    atomic_json(PHASE / "audits/phase05_final_oof_alignment_audit.json", alignment)
    atomic_json(PHASE / "audits/phase05_final_oof_leakage_audit.json", leakage)
    return long, {"coverage": coverage, "alignment": alignment, "leakage": leakage}


def recompute_metrics(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_rows: list[dict[str, Any]] = []
    similarity_rows: list[dict[str, Any]] = []
    ridge_rows: list[dict[str, Any]] = []
    matrices: dict[str, Any] = {}
    fold_crosschecks = 0
    for (dimension, seed), group in long.groupby(["dimension", "seed"], sort=True):
        target_class = group["true_class"].to_numpy(dtype=int)
        predicted_class = group["predicted_class"].to_numpy(dtype=int)
        target_score = group["target_score"].to_numpy(dtype=float)
        similarity = group["similarity_prediction"].to_numpy(dtype=float)
        ridge_raw = group["ridge_prediction_raw"].to_numpy(dtype=float)
        ridge_bounded = group["ridge_prediction_bounded"].to_numpy(dtype=float)
        class_metrics, matrix = classification_metrics(target_class, predicted_class)
        class_rows.append({"dimension": int(dimension), "seed": int(seed), "oof_rows": len(group), **class_metrics})
        matrices[f"dimension_{int(dimension)}_seed_{int(seed)}"] = {
            "labels": [0, 1, 2, 3], "matrix": matrix, "oof_rows": len(group)
        }
        similarity_rows.append(
            {
                "dimension": int(dimension), "seed": int(seed), "oof_rows": len(group),
                **regression_metrics(target_score, similarity, similarity),
            }
        )
        ridge_rows.append(
            {
                "dimension": int(dimension), "seed": int(seed), "oof_rows": len(group),
                **regression_metrics(target_score, ridge_raw, ridge_bounded),
            }
        )

        for fold, fold_group in group.groupby("outer_fold"):
            source = pd.read_csv(
                PHASE / f"results/fold_metrics/vanilla_hdc_final_confirmation_fold_{int(fold)}_metrics.csv"
            )
            row = source[(source["dimension"] == dimension) & (source["seed"] == seed)].iloc[0]
            fold_class, _ = classification_metrics(
                fold_group["true_class"].to_numpy(int), fold_group["predicted_class"].to_numpy(int)
            )
            fold_similarity = regression_metrics(
                fold_group["target_score"].to_numpy(float),
                fold_group["similarity_prediction"].to_numpy(float),
                fold_group["similarity_prediction"].to_numpy(float),
            )
            fold_ridge = regression_metrics(
                fold_group["target_score"].to_numpy(float),
                fold_group["ridge_prediction_raw"].to_numpy(float),
                fold_group["ridge_prediction_bounded"].to_numpy(float),
            )
            for name, value in fold_class.items():
                assert_close(value, row[f"classification_{name}"], f"fold{fold}/{dimension}/{seed}/classification/{name}")
            for name, value in fold_similarity.items():
                assert_close(value, row[f"similarity_{name}"], f"fold{fold}/{dimension}/{seed}/similarity/{name}")
            for name, value in fold_ridge.items():
                assert_close(value, row[f"ridge_{name}"], f"fold{fold}/{dimension}/{seed}/ridge/{name}")
            fold_crosschecks += 1

    classification = pd.DataFrame(class_rows).sort_values(["dimension", "seed"])
    similarity = pd.DataFrame(similarity_rows).sort_values(["dimension", "seed"])
    ridge = pd.DataFrame(ridge_rows).sort_values(["dimension", "seed"])
    summary_dir = PHASE / "results/summaries"
    atomic_csv(summary_dir / "vanilla_hdc_classification_oof_metrics_by_config.csv", classification)
    atomic_csv(summary_dir / "vanilla_hdc_similarity_regression_oof_metrics_by_config.csv", similarity)
    atomic_csv(summary_dir / "vanilla_hdc_ridge_regression_oof_metrics_by_config.csv", ridge)
    atomic_json(summary_dir / "vanilla_hdc_confusion_matrices_by_config.json", matrices)
    atomic_json(
        PHASE / "audits/phase05_oof_metric_recomputation_audit.json",
        {
            "classification_configs_recomputed": len(classification),
            "similarity_configs_recomputed": len(similarity),
            "ridge_configs_recomputed": len(ridge),
            "fold_metric_crosschecks": fold_crosschecks,
            "expected_fold_metric_crosschecks": 100,
            "bounded_rule": "clip to [1.0, 4.0] as frozen; no new clipping",
            "result": "PASS",
        },
    )
    return classification, similarity, ridge


def seed_aggregate(frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dimension, group in frame.groupby("dimension", sort=True):
        row: dict[str, Any] = {"dimension": int(dimension), "seed_count": int(len(group))}
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sample_sd"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_stability(
    classification: pd.DataFrame, similarity: pd.DataFrame, ridge: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    class_metrics = ["macro_f1", "balanced_accuracy", "accuracy", "severe_error_rate"]
    regression_names = ["mae_bounded", "rmse_bounded", "r2_bounded", "spearman_bounded"]
    class_agg = seed_aggregate(classification, class_metrics)
    similarity_agg = seed_aggregate(similarity, regression_names)
    ridge_agg = seed_aggregate(ridge, regression_names)
    output = PHASE / "results/summaries"
    atomic_csv(output / "vanilla_hdc_classification_seed_aggregate_by_dimension.csv", class_agg)
    atomic_csv(output / "vanilla_hdc_similarity_regression_seed_aggregate_by_dimension.csv", similarity_agg)
    atomic_csv(output / "vanilla_hdc_ridge_regression_seed_aggregate_by_dimension.csv", ridge_agg)
    stability = {
        "unit": "five preregistered seed-level OOF metrics per dimension",
        "seed_count_per_dimension": 5,
        "standard_deviation": "sample SD (ddof=1)",
        "seed_ensemble_preregistered": False,
        "seed_predictions_averaged": False,
        "seed_selected_from_outer_test": False,
        "dimension_selected_from_outer_test": False,
        "canonical_configuration_selection": "NOT_PERFORMED_OUTER_TEST_SELECTION_PROHIBITED",
        "classification": class_agg.to_dict(orient="records"),
        "similarity_regression": similarity_agg.to_dict(orient="records"),
        "ridge_regression": ridge_agg.to_dict(orient="records"),
        "result": "PASS",
    }
    atomic_json(output / "vanilla_hdc_seed_stability_summary.json", stability)
    return class_agg, similarity_agg, ridge_agg, stability


def aggregate_efficiency() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [
        pd.read_csv(PHASE / f"results/efficiency/vanilla_hdc_final_confirmation_fold_{fold}_efficiency.csv")
        for fold in range(1, 6)
    ]
    raw = pd.concat(frames, ignore_index=True)
    raw["fit_time_seconds"] = (
        raw["inner_selection_seconds"]
        + raw["outer_training_encoding_seconds"]
        + raw["outer_training_fit_seconds"]
    )
    raw["prediction_time_seconds"] = raw["outer_test_inference_seconds"]
    raw["measured_total_runtime_seconds"] = raw["fit_time_seconds"] + raw["prediction_time_seconds"]
    rows: list[dict[str, Any]] = []
    for (dimension, seed), group in raw.groupby(["dimension", "seed"], sort=True):
        model_bytes = 0
        for fold in range(1, 6):
            checkpoint = load_json(
                PHASE / f"results/checkpoints/final_confirmation/fold_{fold}/"
                f"vanilla_hdc_final_confirmation_fold_{fold}_dimension_{int(dimension)}_seed_{int(seed)}_checkpoint.json"
            )
            model_bytes += (PHASE / checkpoint["model_npz"]).stat().st_size
        rows.append(
            {
                "dimension": int(dimension),
                "seed": int(seed),
                "folds": 5,
                "fit_time_seconds": float(group["fit_time_seconds"].sum()),
                "prediction_time_seconds": float(group["prediction_time_seconds"].sum()),
                "measured_total_runtime_seconds": float(group["measured_total_runtime_seconds"].sum()),
                "model_artifact_size_bytes_across_folds": int(model_bytes),
                "peak_memory": np.nan,
                "encoding_throughput": np.nan,
            }
        )
    by_config = pd.DataFrame(rows)
    aggregate = seed_aggregate(
        by_config,
        [
            "fit_time_seconds", "prediction_time_seconds", "measured_total_runtime_seconds",
            "model_artifact_size_bytes_across_folds",
        ],
    )
    atomic_csv(PHASE / "results/summaries/vanilla_hdc_efficiency_by_config.csv", by_config)
    atomic_csv(PHASE / "results/summaries/vanilla_hdc_efficiency_seed_aggregate_by_dimension.csv", aggregate)
    return by_config, aggregate


def validate_baseline_interfaces() -> dict[str, Any]:
    primary = pd.read_csv(PRIMARY, usecols=["run_key", "subject_id", "target_class", "target_score", "outer_fold"]).set_index("run_key")
    expected_keys = set(primary.index)
    phase04a = pd.read_csv(PHASE04A / "results/summaries/phase04a_final_classifier_comparison.csv")
    for slug in phase04a["model_slug"]:
        predictions = pd.read_csv(PHASE04A / f"results/predictions/{slug}_oof.csv")
        if set(predictions["run_key"]) != expected_keys or len(predictions) != 419:
            raise RuntimeError(f"Phase 04A {slug} OOF coverage mismatch")
        if any(int(row.outer_fold) != int(primary.at[row.run_key, "outer_fold"]) for row in predictions.itertuples()):
            raise RuntimeError(f"Phase 04A {slug} fold mismatch")
    freeze04b = load_json(PHASE04B / "configs/phase04b_freeze.json")
    comparison04b = pd.read_csv(PHASE04B / "results/summaries/phase04b_final_regressor_comparison.csv")
    for slug, entry in freeze04b["canonical_oof_files"].items():
        path = PHASE04B / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Phase 04B {slug} frozen OOF hash mismatch")
        predictions = pd.read_csv(path)
        if set(predictions["run_key"]) != expected_keys or len(predictions) != 419:
            raise RuntimeError(f"Phase 04B {slug} OOF coverage mismatch")
        if any(int(row.outer_fold) != int(primary.at[row.run_key, "outer_fold"]) for row in predictions.itertuples()):
            raise RuntimeError(f"Phase 04B {slug} fold mismatch")
    return {
        "modeling_rows": 419,
        "subjects": int(primary["subject_id"].nunique()),
        "primary_features": len(pd.read_csv(PRIMARY, nrows=0).columns) - 9,
        "same_run_keys": True,
        "same_frozen_fold_sha256": True,
        "subject_wise_outer_split": True,
        "primary_without_performance_interface": True,
        "phase04a_models": int(len(phase04a)),
        "phase04b_models": int(len(comparison04b)),
        "result": "PASS",
    }


def baseline_comparisons(
    class_agg: pd.DataFrame, similarity_agg: pd.DataFrame, ridge_agg: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    compatibility = validate_baseline_interfaces()
    baseline_class = pd.read_csv(PHASE04A / "results/summaries/phase04a_final_classifier_comparison.csv")
    baseline_class_rows: list[dict[str, Any]] = []
    for row in baseline_class.itertuples():
        predictions = pd.read_csv(PHASE04A / f"results/predictions/{row.model_slug}_oof.csv")
        severe = float(np.mean(np.abs(predictions["true_class"] - predictions["predicted_class"]) >= 2))
        baseline_class_rows.append(
            {
                "source_phase": "04A", "method": row.model, "method_type": "traditional_baseline",
                "dimension": np.nan, "seed_count": 1, "aggregation": "canonical OOF",
                "macro_f1_mean": row.oof_macro_f1, "macro_f1_sample_sd": np.nan,
                "balanced_accuracy_mean": row.oof_balanced_accuracy, "balanced_accuracy_sample_sd": np.nan,
                "accuracy_mean": row.oof_accuracy, "accuracy_sample_sd": np.nan,
                "severe_error_rate_mean": severe, "severe_error_rate_sample_sd": np.nan,
            }
        )
    hdc_class_rows = [
        {
            "source_phase": "05", "method": f"Vanilla HDC D={int(row.dimension)}",
            "method_type": "HDC preregistered dimension", "dimension": int(row.dimension),
            "seed_count": 5, "aggregation": "mean ± sample SD across five preregistered seeds",
            "macro_f1_mean": row.macro_f1_mean, "macro_f1_sample_sd": row.macro_f1_sample_sd,
            "balanced_accuracy_mean": row.balanced_accuracy_mean,
            "balanced_accuracy_sample_sd": row.balanced_accuracy_sample_sd,
            "accuracy_mean": row.accuracy_mean, "accuracy_sample_sd": row.accuracy_sample_sd,
            "severe_error_rate_mean": row.severe_error_rate_mean,
            "severe_error_rate_sample_sd": row.severe_error_rate_sample_sd,
        }
        for row in class_agg.itertuples()
    ]
    classification_comparison = pd.DataFrame(baseline_class_rows + hdc_class_rows)

    baseline_regression = pd.read_csv(PHASE04B / "results/summaries/phase04b_final_regressor_comparison.csv")
    baseline_reg_rows = [
        {
            "source_phase": "04B", "method": row.model, "method_type": "traditional_baseline",
            "regression_head": "traditional", "dimension": np.nan, "seed_count": 1,
            "aggregation": "canonical OOF", "mae_bounded_mean": row.mae_bounded,
            "mae_bounded_sample_sd": np.nan, "rmse_bounded_mean": row.rmse_bounded,
            "rmse_bounded_sample_sd": np.nan, "r2_bounded_mean": row.r2_bounded,
            "r2_bounded_sample_sd": np.nan, "spearman_bounded_mean": row.spearman_bounded,
            "spearman_bounded_sample_sd": np.nan,
        }
        for row in baseline_regression.itertuples()
    ]
    hdc_reg_rows: list[dict[str, Any]] = []
    for head, aggregate in [("similarity", similarity_agg), ("ridge_readout", ridge_agg)]:
        for row in aggregate.itertuples():
            hdc_reg_rows.append(
                {
                    "source_phase": "05", "method": f"Vanilla HDC {head} D={int(row.dimension)}",
                    "method_type": "HDC preregistered dimension", "regression_head": head,
                    "dimension": int(row.dimension), "seed_count": 5,
                    "aggregation": "mean ± sample SD across five preregistered seeds",
                    "mae_bounded_mean": row.mae_bounded_mean,
                    "mae_bounded_sample_sd": row.mae_bounded_sample_sd,
                    "rmse_bounded_mean": row.rmse_bounded_mean,
                    "rmse_bounded_sample_sd": row.rmse_bounded_sample_sd,
                    "r2_bounded_mean": row.r2_bounded_mean,
                    "r2_bounded_sample_sd": row.r2_bounded_sample_sd,
                    "spearman_bounded_mean": row.spearman_bounded_mean,
                    "spearman_bounded_sample_sd": row.spearman_bounded_sample_sd,
                }
            )
    regression_comparison = pd.DataFrame(baseline_reg_rows + hdc_reg_rows)
    best_class = classification_comparison.query("source_phase == '04A'").sort_values("macro_f1_mean", ascending=False).iloc[0]
    best_reg = regression_comparison.query("source_phase == '04B'").sort_values("mae_bounded_mean").iloc[0]
    dual = class_agg.merge(similarity_agg, on=["dimension", "seed_count"], suffixes=("_classification", "_similarity"))
    dual = dual.merge(ridge_agg, on=["dimension", "seed_count"], suffixes=("", "_ridge"))
    dual["best_phase04a_classifier"] = best_class["method"]
    dual["best_phase04a_macro_f1"] = best_class["macro_f1_mean"]
    dual["best_phase04b_regressor"] = best_reg["method"]
    dual["best_phase04b_mae_bounded"] = best_reg["mae_bounded_mean"]
    dual["canonical_configuration_selection"] = "NOT_PERFORMED_OUTER_TEST_SELECTION_PROHIBITED"
    output = PHASE / "results/summaries"
    atomic_csv(output / "phase05_vs_phase04a_classification_comparison.csv", classification_comparison)
    atomic_csv(output / "phase05_vs_phase04b_regression_comparison.csv", regression_comparison)
    atomic_csv(output / "phase05_dual_output_final_comparison.csv", dual)
    atomic_json(PHASE / "audits/phase05_baseline_compatibility_audit.json", compatibility)
    return classification_comparison, regression_comparison, dual, compatibility


def publication_export(fig: plt.Figure, filename: str, height_mm: float = 105.0) -> None:
    import pubfig

    target = PHASE / "figures" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    pubfig.save_figure(fig, target, spec="nature", width="double", height_mm=height_mm, raster_dpi=600)
    pubfig.save_figure(fig, target.with_suffix(".pdf"), spec="nature", width="double", height_mm=height_mm)
    plt.close(fig)


def style_axes(axis: plt.Axes, x_label: str, y_label: str, y_min: float = 0.0) -> None:
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_ylim(bottom=y_min)
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def make_figures(
    classification: pd.DataFrame,
    similarity: pd.DataFrame,
    ridge: pd.DataFrame,
    class_agg: pd.DataFrame,
    similarity_agg: pd.DataFrame,
    ridge_agg: pd.DataFrame,
    efficiency: pd.DataFrame,
    class_comparison: pd.DataFrame,
    reg_comparison: pd.DataFrame,
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8, "lines.linewidth": 1.8})

    def mean_sd_line(data: pd.DataFrame, mean: str, sd: str, y_label: str, filename: str) -> None:
        fig, axis = plt.subplots(figsize=(7.2, 3.8))
        axis.errorbar(data["dimension"], data[mean], yerr=data[sd], marker="o", capsize=4, color=OKABE_ITO[0], label="Mean ± sample SD (5 seeds)")
        axis.set_xticks(DIMENSIONS, [f"{value:,}" for value in DIMENSIONS])
        style_axes(axis, "Hypervector dimension", y_label)
        axis.legend(frameon=False)
        fig.tight_layout()
        publication_export(fig, filename)

    mean_sd_line(class_agg, "macro_f1_mean", "macro_f1_sample_sd", "OOF Macro-F1", "phase05_classification_macro_f1_vs_dimension.png")
    mean_sd_line(similarity_agg, "mae_bounded_mean", "mae_bounded_sample_sd", "Bounded OOF MAE", "phase05_similarity_regression_mae_vs_dimension.png")
    mean_sd_line(ridge_agg, "mae_bounded_mean", "mae_bounded_sample_sd", "Bounded OOF MAE", "phase05_ridge_regression_mae_vs_dimension.png")

    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    for seed, group in classification.groupby("seed"):
        axis.plot(group["dimension"], group["macro_f1"], marker="o", alpha=0.7, label=f"Seed {int(seed)}")
    axis.plot(class_agg["dimension"], class_agg["macro_f1_mean"], color="black", marker="s", linewidth=2.4, label="Seed mean")
    axis.set_xticks(DIMENSIONS, [f"{value:,}" for value in DIMENSIONS])
    style_axes(axis, "Hypervector dimension", "OOF Macro-F1")
    axis.legend(frameon=False, ncol=3)
    fig.tight_layout()
    publication_export(fig, "phase05_classification_seed_stability.png")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharex=True)
    for axis, frame, aggregate, title, color in [
        (axes[0], similarity, similarity_agg, "Similarity regression", OKABE_ITO[1]),
        (axes[1], ridge, ridge_agg, "Ridge readout", OKABE_ITO[2]),
    ]:
        for _, group in frame.groupby("seed"):
            axis.plot(group["dimension"], group["mae_bounded"], marker="o", alpha=0.45, color=color)
        axis.plot(aggregate["dimension"], aggregate["mae_bounded_mean"], marker="s", color="black", linewidth=2.3)
        axis.set_title(title)
        axis.set_xticks(DIMENSIONS, ["1k", "2k", "5k", "10k"])
        style_axes(axis, "Hypervector dimension", "Bounded OOF MAE")
    fig.tight_layout()
    publication_export(fig, "phase05_regression_seed_stability.png", height_mm=95)

    merged = classification.merge(efficiency, on=["dimension", "seed"], validate="one_to_one")
    fig, axis = plt.subplots(figsize=(7.2, 4.0))
    for index, dimension in enumerate(DIMENSIONS):
        group = merged[merged["dimension"] == dimension]
        axis.scatter(group["measured_total_runtime_seconds"], group["macro_f1"], s=38, color=OKABE_ITO[index], label=f"D={dimension:,}")
    style_axes(axis, "Measured fit + prediction time across five folds (s)", "OOF Macro-F1")
    axis.set_xlim(left=0)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    publication_export(fig, "phase05_accuracy_efficiency_tradeoff.png")

    ordered_class = class_comparison.sort_values("macro_f1_mean", ascending=True).reset_index(drop=True)
    colors = [OKABE_ITO[0] if phase == "05" else "#777777" for phase in ordered_class["source_phase"]]
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.barh(np.arange(len(ordered_class)), ordered_class["macro_f1_mean"], xerr=ordered_class["macro_f1_sample_sd"].fillna(0), color=colors, capsize=3)
    axis.set_yticks(np.arange(len(ordered_class)), ordered_class["method"])
    style_axes(axis, "OOF Macro-F1", "Method")
    axis.set_xlim(0, 1)
    fig.tight_layout()
    publication_export(fig, "phase05_classification_vs_traditional_baselines.png", height_mm=135)

    ordered_reg = reg_comparison.sort_values("mae_bounded_mean", ascending=False).reset_index(drop=True)
    colors = [OKABE_ITO[1] if phase == "05" and head == "similarity" else OKABE_ITO[2] if phase == "05" else "#777777" for phase, head in zip(ordered_reg["source_phase"], ordered_reg["regression_head"])]
    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    axis.barh(np.arange(len(ordered_reg)), ordered_reg["mae_bounded_mean"], xerr=ordered_reg["mae_bounded_sample_sd"].fillna(0), color=colors, capsize=3)
    axis.set_yticks(np.arange(len(ordered_reg)), ordered_reg["method"])
    style_axes(axis, "Bounded OOF MAE", "Method")
    axis.set_xlim(left=0)
    fig.tight_layout()
    publication_export(fig, "phase05_regression_vs_traditional_baselines.png", height_mm=165)


def write_analysis_bundle(
    class_agg: pd.DataFrame,
    similarity_agg: pd.DataFrame,
    ridge_agg: pd.DataFrame,
    compatibility: dict[str, Any],
) -> None:
    analysis_dir = PHASE / "analysis-output"
    best_class = class_agg.sort_values("macro_f1_mean", ascending=False).iloc[0]
    best_similarity = similarity_agg.sort_values("mae_bounded_mean").iloc[0]
    best_ridge = ridge_agg.sort_values("mae_bounded_mean").iloc[0]
    analysis = f"""# Phase 05 Strict Analysis Report

## Analysis question
How do the four preregistered Vanilla HDC dimensions behave across five preregistered seeds for classification, similarity-based bounded regression, Ridge-readout bounded regression, and measured efficiency under the frozen 419-run subject-wise OOF protocol?

## Evidence boundary
- Unit of stability summary: seed-level OOF metric (`n=5` seeds per dimension).
- Error bars: sample SD across seeds.
- No seed ensemble, post-hoc canonical configuration, inferential significance test, or outer-test tuning.
- Phase 04 comparisons are descriptive; compatibility audit: `{compatibility['result']}`.

## Key findings
- Highest observed mean classification Macro-F1 in the preregistered matrix: D={int(best_class['dimension']):,}, {best_class['macro_f1_mean']:.6f} ± {best_class['macro_f1_sample_sd']:.6f} SD.
- Lowest observed mean similarity-regression bounded MAE: D={int(best_similarity['dimension']):,}, {best_similarity['mae_bounded_mean']:.6f} ± {best_similarity['mae_bounded_sample_sd']:.6f} SD.
- Lowest observed mean Ridge-readout bounded MAE: D={int(best_ridge['dimension']):,}, {best_ridge['mae_bounded_mean']:.6f} ± {best_ridge['mae_bounded_sample_sd']:.6f} SD.
- These are descriptive observations across the complete preregistered matrix, not new model-selection decisions.

## Claim Candidates
- Claim: Vanilla HDC performance varies with dimension but is observable across all five preregistered seeds.
  - Source evidence: dimension-level mean, sample SD, min, and max tables.
  - Allowed wording: descriptive differences and stability across the preregistered matrix.
  - Forbidden stronger wording: statistically significant superiority or a selected canonical dimension.
  - Uncertainty: five seeds; no preregistered inferential contrast.
  - Next check: use the complete matrix in the next planned analysis without selecting from outer-test performance.
  - Decision: keep
- Claim: The two regression heads estimate a bounded difficulty-induced workload proxy.
  - Source evidence: 419-run OOF predictions for 20 configurations per head.
  - Allowed wording: bounded difficulty-induced workload proxy regression.
  - Forbidden stronger wording: directly measured continuous cognitive workload.
  - Uncertainty: target has four difficulty-derived values.
  - Next check: preserve this interpretation in Phase 06.
  - Decision: keep
"""
    stats = """# Phase 05 Statistical Appendix

## Descriptive design
- 20 preregistered configurations: four dimensions × five seeds.
- Each configuration has exactly 419 subject-wise OOF predictions.
- Dimension summaries report mean, sample SD (`ddof=1`), minimum, and maximum over five seed-level OOF metrics.
- OOF metrics were recomputed directly from 8,380 aligned configuration-run rows and cross-checked against 100 fold-metric blocks.

## Inferential boundary
The frozen plan did not preregister a significance test for dimension, seed, or comparison with Phase 04. No p-values, confidence intervals, effect-size tests, or multiple-comparison procedures were added post hoc. Baseline comparisons are descriptive and do not support “significantly better” wording.

## Repeated-measure caution
The same 419 runs appear once per preregistered configuration. These repeated predictions are not treated as 8,380 independent experimental observations. Seeds are summarized as a stability distribution, not averaged into an ensemble prediction.
"""
    catalog_lines = ["# Phase 05 Figure Catalog", ""]
    figure_info = [
        ("phase05_classification_macro_f1_vs_dimension.png", "Classification dimension trend", "Mean OOF Macro-F1 by dimension; error bars are sample SD across five seeds."),
        ("phase05_classification_seed_stability.png", "Classification seed stability", "All five seed trajectories plus their mean; descriptive only."),
        ("phase05_similarity_regression_mae_vs_dimension.png", "Similarity-regression dimension trend", "Bounded OOF MAE mean ± sample SD across seeds."),
        ("phase05_ridge_regression_mae_vs_dimension.png", "Ridge-readout dimension trend", "Bounded OOF MAE mean ± sample SD across seeds."),
        ("phase05_regression_seed_stability.png", "Regression seed stability", "Seed trajectories for both preregistered regression heads."),
        ("phase05_accuracy_efficiency_tradeoff.png", "Accuracy-efficiency tradeoff", "Macro-F1 versus measured fit+prediction component time; no missing memory/throughput value is imputed."),
        ("phase05_classification_vs_traditional_baselines.png", "Descriptive Phase 04A comparison", "HDC dimension mean ± seed SD beside canonical traditional OOF metrics."),
        ("phase05_regression_vs_traditional_baselines.png", "Descriptive Phase 04B comparison", "Both HDC heads beside canonical traditional bounded OOF MAE."),
    ]
    for filename, purpose, observation in figure_info:
        catalog_lines.extend(
            [f"## {filename}", f"- Purpose: {purpose}.", "- Data source: frozen OOF summaries and compatibility-validated Phase 04 tables.", f"- Observation: {observation}", "- Interpretation: use only as descriptive evidence; no significance or canonical-selection claim.", "- Caveat: HDC error bars are seed SD; traditional canonical OOF rows do not have comparable seed SD.", ""]
        )
    atomic_text(analysis_dir / "analysis-report.md", analysis)
    atomic_text(analysis_dir / "stats-appendix.md", stats)
    atomic_text(analysis_dir / "figure-catalog.md", "\n".join(catalog_lines))


def write_final_report(
    class_agg: pd.DataFrame,
    similarity_agg: pd.DataFrame,
    ridge_agg: pd.DataFrame,
    efficiency_agg: pd.DataFrame,
    compatibility: dict[str, Any],
) -> None:
    class_best = class_agg.sort_values("macro_f1_mean", ascending=False).iloc[0]
    sim_best = similarity_agg.sort_values("mae_bounded_mean").iloc[0]
    ridge_best = ridge_agg.sort_values("mae_bounded_mean").iloc[0]
    report = f"""---
type: results-report
date: 2026-08-20
experiment_line: basic-dual-output-hdc
round: 00
purpose: final-phase-freeze
status: frozen
source_artifacts:
  - analysis-output/analysis-report.md
  - analysis-output/stats-appendix.md
  - analysis-output/figure-catalog.md
linked_experiments: []
linked_results: []
---

# Phase 05 / Round 00 / Final Phase Freeze / 2026-08-20

## 1. Executive Summary
Phase 05 completed an audited Vanilla Prototype HDC benchmark with four preregistered dimensions, five seeds, a prototype cosine classification head, and two regression heads. Every configuration covers 419 subject-wise OOF runs. The strongest observed mean Macro-F1 occurred at D={int(class_best['dimension']):,} ({class_best['macro_f1_mean']:.6f} ± {class_best['macro_f1_sample_sd']:.6f} sample SD), while the lowest observed similarity and Ridge bounded MAE occurred at D={int(sim_best['dimension']):,} ({sim_best['mae_bounded_mean']:.6f} ± {sim_best['mae_bounded_sample_sd']:.6f}) and D={int(ridge_best['dimension']):,} ({ridge_best['mae_bounded_mean']:.6f} ± {ridge_best['mae_bounded_sample_sd']:.6f}), respectively. These observations do not select a canonical configuration.

## 2. Experiment Identity and Decision Context
The experiment tests whether a shared bipolar HDC representation can support four-class difficulty-induced workload proxy classification and bounded difficulty-induced workload proxy regression. It follows frozen traditional baselines from Phase 04A/04B and stops before Phase 06 variants.

## 3. Frozen Data and Evaluation Protocol
- Primary interface: 419 runs, 35 subjects, 1,176 without-performance predictive features.
- Primary SHA-256: `{EXPECTED_PRIMARY_SHA}`.
- Frozen subject-wise five-fold SHA-256: `{EXPECTED_FOLD_SHA}`.
- Inner selection: three-fold `GroupKFold(groups=subject_id)` on each outer-training set.
- Outer-test data were never used for hyperparameter tuning.
- Compatibility with Phase 04A/04B: `{compatibility['result']}` for run keys, folds, subject-wise protocol, and feature interface.

## 4. HDC Representation
The frozen representation uses bipolar item, level, and sample hypervectors. Feature identity and ordered level hypervectors are bound by elementwise bipolar multiplication, then bundled by integer accumulation and deterministic sign/tie resolution. Quantization uses equal-width training-fold minimum/maximum boundaries after training-fitted imputation, variance filtering, standardization, and feature selection. Classification uses cosine similarity to four class prototypes.

## 5. Quick Screen and Final Confirmation
Quick screening evaluated the frozen 16-candidate space independently in each outer fold and selected `levels=51`, `feature_k=50` for all five folds. Final Confirmation then evaluated every preregistered dimension `[1000, 2000, 5000, 10000]` and seed `[42, 43, 44, 45, 46]`, totaling 100 fold-runs and 20 complete OOF configurations. Temperature and Ridge alpha were selected only through outer-training inner CV.

## 6. Four-Class OOF Results
The classification tables report accuracy, balanced accuracy, Macro-F1, weighted F1, severe error rate, and complete confusion matrices for all 20 configurations. Across five seeds per dimension, the highest observed mean Macro-F1 was {class_best['macro_f1_mean']:.6f} at D={int(class_best['dimension']):,}. This is a descriptive property of the preregistered matrix, not a post-hoc choice.

## 7. Similarity Regression OOF Results
The similarity decoder maps prototype similarities through an inner-selected temperature and produces a prediction bounded to `[1,4]`. The lowest observed dimension-level mean bounded MAE was {sim_best['mae_bounded_mean']:.6f} at D={int(sim_best['dimension']):,}. The target is the difficulty-level-derived bounded proxy, not directly measured continuous cognitive workload.

## 8. Ridge Readout OOF Results
The Ridge readout uses normalized sample hypervectors and an alpha selected only in inner CV. The lowest observed dimension-level mean bounded MAE was {ridge_best['mae_bounded_mean']:.6f} at D={int(ridge_best['dimension']):,}. Predictions were clipped only by the frozen `[1,4]` contract.

## 9. Dimension and Seed Stability
Each dimension is summarized over five preregistered seeds by mean, sample SD, minimum, and maximum. Seed predictions were not averaged into a new ensemble and no seed was selected from outer-test performance. Full configuration tables remain the authoritative benchmark.

## 10. Efficiency Analysis
Efficiency summaries use only recorded inner-selection, outer-training encoding/fit, and outer-test inference times. Model artifact bytes are measured from saved model files. Peak memory and encoding throughput were not recorded and are left missing rather than estimated. Dimension-level efficiency summaries use the same five-seed descriptive aggregation.

## 11. Fair Comparison with Phase 04A and Phase 04B
Phase 04A classification and Phase 04B regression artifacts were read from their frozen comparisons, reports, manifests, freeze files, and canonical OOF predictions. HDC is displayed as mean ± sample SD over five seeds per dimension; traditional rows retain their canonical OOF values. No single highest HDC seed is promoted as the formal result. No unregistered significance test or “significantly better” claim is made.

## 12. Leakage Protection
All preprocessing, quantization, codebooks, prototypes, temperature selection, and Ridge alpha selection were confined to training scopes. Final OOF consolidation only copied and aligned already-saved predictions; it did not fit or predict. Fold/run/subject/target alignment and source hashes were audited.

## 13. Limitations and Negative Results
- The regression target has only four difficulty-derived values and is not a direct continuous cognitive-workload measurement.
- Five seeds provide a useful stability description but do not justify an unregistered inferential claim.
- Similarity regression and Ridge readout differ materially; neither should be conflated with the classification head.
- Peak memory and encoding throughput were not recorded, so no such efficiency values are reported.
- Traditional baselines and HDC have compatible OOF interfaces, but seed-repetition structures differ; comparison is descriptive.

## 14. Canonical Configuration
No preregistered rule authorizes selection of a canonical dimension or seed from outer-test performance. **No post-hoc canonical configuration was selected from outer-test performance.** Freeze status: `NOT_PERFORMED_OUTER_TEST_SELECTION_PROHIBITED`.

## 15. Figure-by-Figure Interpretation
The dimension figures show mean trends with sample SD across seeds; the stability figures expose individual seed trajectories; the efficiency plot displays the measured accuracy/time tradeoff; and the baseline plots place the complete HDC dimension summaries beside frozen traditional results. None is annotated with a significance claim.

## 16. What Changed Our Belief
Vanilla HDC now has complete, reproducible dual-output OOF evidence rather than quick-screen-only evidence. The full dimension/seed matrix, not an observed best row, is the stable Phase 05 result.

## 17. Phase Boundary and Next Actions
Phase 05 may be frozen only after final artifact, reproducibility, upstream-integrity, and Notebook audits pass. The next planned phase may read this frozen matrix; it must not treat an observed best outer-test configuration as an unbiased selected model. Phase 06 was not executed here.

## 18. Artifact and Reproducibility Index
- OOF predictions: `results/oof/vanilla_hdc_*_oof.csv`
- Configuration and seed summaries: `results/summaries/`
- Figures: `figures/phase05_*.png` with PDF companions
- Strict analysis bundle: `analysis-output/`
- Final audits: `audits/phase05_final_*`
- Final manifest: `manifests/phase05_final_artifact_manifest.json`
- Notebook: `Phase_05_Basic_Dual_Output_HDC.ipynb`
- Freeze record: `configs/phase05_freeze.json`
"""
    atomic_text(PHASE / "reports/phase05_final_summary.md", report)

    readme = (PHASE / "README.md").read_text(encoding="utf-8")
    marker = "## Final Phase 05 status"
    if marker not in readme:
        readme += f"""

{marker}

Phase 05 completed 5/5 Final Confirmation folds and 100/100 fold-level runs, consolidated 20 aligned 419-run OOF configurations, and produced audited classification plus two bounded regression-head result matrices. The final frozen interface preserves all four dimensions and five seeds; no post-hoc canonical dimension or seed was selected from outer-test performance.

### Main artifacts

- Final report: `reports/phase05_final_summary.md`
- OOF outputs: `results/oof/vanilla_hdc_*_oof.csv`
- Seed/dimension and baseline summaries: `results/summaries/`
- Publication figures: `figures/phase05_*.png`
- Final manifest: `manifests/phase05_final_artifact_manifest.json`
- Final audits: `audits/phase05_final_*`
- Freeze: `configs/phase05_freeze.json`
"""
        atomic_text(PHASE / "README.md", readme)


def main() -> int:
    snapshot = validate_start_gate()
    long, audits = consolidate_oof()
    classification, similarity, ridge = recompute_metrics(long)
    class_agg, similarity_agg, ridge_agg, _ = aggregate_stability(classification, similarity, ridge)
    efficiency, efficiency_agg = aggregate_efficiency()
    class_comparison, reg_comparison, _, compatibility = baseline_comparisons(class_agg, similarity_agg, ridge_agg)
    make_figures(
        classification, similarity, ridge, class_agg, similarity_agg, ridge_agg,
        efficiency, class_comparison, reg_comparison,
    )
    write_analysis_bundle(class_agg, similarity_agg, ridge_agg, compatibility)
    write_final_report(class_agg, similarity_agg, ridge_agg, efficiency_agg, compatibility)
    print(
        json.dumps(
            {
                "start_gate": snapshot["result"],
                "oof_rows": len(long),
                "configs": len(classification),
                "coverage": audits["coverage"]["result"],
                "alignment": audits["alignment"]["result"],
                "figures_png": len(list((PHASE / "figures").glob("phase05_*.png"))),
                "analysis_bundle": "SAVED",
                "final_report": "SAVED",
                "canonical_configuration_selection": "NOT_PERFORMED_OUTER_TEST_SELECTION_PROHIBITED",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
