"""Resumable Phase 07 frozen unimodal batch executor.

This module runs only the frozen Hybrid classification model and frozen Common
Encoder Ridge regression head. It contains no model-search or inner-CV path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import nbformat
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


PHASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE_DIR.parents[1]
PHASE05_SRC = PROJECT_ROOT / "experiments" / "phase_05_basic_dual_output_hdc" / "src"
PHASE06_SRC = PROJECT_ROOT / "experiments" / "phase_06_hdc_variant_screening" / "src"
sys.path.insert(0, str(PHASE05_SRC))
sys.path.insert(0, str(PHASE06_SRC))

from phase05_hdc_core import EqualWidthQuantizer, incremental_encode_prefixes  # noqa: E402
from phase06_hybrid import predict_hybrid, train_hybrid  # noqa: E402


PRIMARY = PROJECT_ROOT / "experiments" / "phase_03_multimodal_dataset_labeling" / "data" / "primary_without_performance.csv"
FOLDS = PROJECT_ROOT / "experiments" / "phase_03_multimodal_dataset_labeling" / "data" / "fold_assignments.csv"
CONTRACT_PATH = PHASE_DIR / "configs" / "phase07_frozen_unimodal_contract.json"
EXECUTION_MANIFEST = PHASE_DIR / "configs" / "phase07_execution_manifest.json"
MODALITY_MANIFEST = PHASE_DIR / "manifests" / "phase07_modality_feature_manifest.json"
EXPERIMENT_CONTRACT = PHASE_DIR / "configs" / "phase07_experiment_contract.json"
PROGRESS_PATH = PHASE_DIR / "logs" / "phase07_unimodal_progress.jsonl"
STDOUT_PATH = PHASE_DIR / "logs" / "phase07_unimodal_batch_stdout.log"
STDERR_PATH = PHASE_DIR / "logs" / "phase07_unimodal_batch_stderr.log"

EXPECTED_PRIMARY = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLDS = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
TASKS = ["classification", "regression"]


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def append_progress(payload: dict[str, Any]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def payload_digest(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "checkpoint_payload_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_frozen_inputs() -> tuple[dict[str, Any], dict[str, list[str]]]:
    contract = read_json(CONTRACT_PATH)
    manifest = read_json(MODALITY_MANIFEST)
    modalities = {item["name"]: list(item["features"]) for item in manifest["modalities"]}
    if contract["status"] != "CONTRACT_FROZEN_NOT_TRAINED":
        raise RuntimeError("Phase 07 contract is not CONTRACT_FROZEN_NOT_TRAINED")
    if sha256(PRIMARY) != EXPECTED_PRIMARY or sha256(FOLDS) != EXPECTED_FOLDS:
        raise RuntimeError("Frozen Phase 03 checksum mismatch")
    if list(modalities) != [item["name"] for item in contract["modalities"]]:
        raise RuntimeError("Modality order or membership source mismatch")
    if {name: len(features) for name, features in modalities.items()} != {item["name"]: item["feature_count"] for item in contract["modalities"]}:
        raise RuntimeError("Modality feature counts mismatch")
    return contract, modalities


def enumerate_runs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    identifiers: list[dict[str, Any]] = []
    for modality in [item["name"] for item in contract["modalities"]]:
        for outer_fold in range(1, 6):
            for seed in contract["randomness"]["seeds"]:
                for task in TASKS:
                    identifiers.append(
                        {
                            "run_id": f"{modality}|{task}|fold={outer_fold}|seed={seed}",
                            "modality": modality,
                            "task": task,
                            "outer_fold": outer_fold,
                            "seed": int(seed),
                        }
                    )
    return identifiers


def checkpoint_path(run: dict[str, Any]) -> Path:
    return PHASE_DIR / "results" / "checkpoints" / run["modality"] / run["task"] / f"fold_{run['outer_fold']}" / f"seed_{run['seed']}" / "checkpoint.json"


def prediction_path(run: dict[str, Any]) -> Path:
    return PHASE_DIR / "results" / "predictions" / run["modality"] / run["task"] / f"fold_{run['outer_fold']}_seed_{run['seed']}_predictions.csv"


def metrics_path(run: dict[str, Any]) -> Path:
    return PHASE_DIR / "results" / "fold_metrics" / run["modality"] / run["task"] / f"fold_{run['outer_fold']}_seed_{run['seed']}_metrics.json"


def efficiency_path(run: dict[str, Any]) -> Path:
    return PHASE_DIR / "results" / "efficiency" / run["modality"] / run["task"] / f"fold_{run['outer_fold']}_seed_{run['seed']}_efficiency.json"


def frozen_structure(contract: dict[str, Any], outer_fold: int) -> dict[str, Any]:
    matches = [item for item in contract["classification"]["fold_structures"] if int(item["outer_fold"]) == outer_fold]
    if len(matches) != 1:
        raise RuntimeError(f"Missing frozen Hybrid structure for outer fold {outer_fold}")
    return {key: value for key, value in matches[0].items() if key != "outer_fold"}


def valid_checkpoint(run: dict[str, Any], contract: dict[str, Any], expected_test_rows: int | None = None) -> bool:
    path = checkpoint_path(run)
    try:
        payload = read_json(path)
        if payload.get("status") != "COMPLETE" or payload.get("result") != "PASS":
            return False
        if payload.get("checkpoint_payload_sha256") != payload_digest(payload):
            return False
        for key in ["modality", "task", "outer_fold", "seed"]:
            if payload.get(key) != run[key]:
                return False
        if payload.get("primary_sha256") != EXPECTED_PRIMARY or payload.get("frozen_fold_sha256") != EXPECTED_FOLDS:
            return False
        if payload.get("leakage_audit_result") != "PASS" or payload.get("checkpoint_integrity") != "PASS":
            return False
        if expected_test_rows is not None and payload.get("prediction_row_count") != expected_test_rows:
            return False
        artifacts = payload.get("artifact_hashes", {})
        for label, artifact in [("predictions", prediction_path(run)), ("metrics", metrics_path(run)), ("efficiency", efficiency_path(run))]:
            if not artifact.is_file() or artifacts.get(label) != sha256(artifact):
                return False
        frame = pd.read_csv(prediction_path(run))
        if len(frame) != payload["prediction_row_count"] or frame["run_key"].nunique() != payload["unique_test_run_key_count"]:
            return False
        prediction_column = "predicted_class" if run["task"] == "classification" else "prediction_bounded"
        if frame[prediction_column].isna().any() or not np.isfinite(frame[prediction_column].to_numpy(dtype=float)).all():
            return False
        if run["task"] == "classification" and not set(frame[prediction_column].astype(int)).issubset({0, 1, 2, 3}):
            return False
        if run["task"] == "regression" and not frame[prediction_column].between(1.0, 4.0).all():
            return False
        return True
    except Exception:
        return False


def fit_preprocessing(
    train_values: np.ndarray,
    test_values: np.ndarray,
    train_labels: np.ndarray,
    feature_names: list[str],
    requested_k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    train_imputed = imputer.fit_transform(train_values)
    test_imputed = imputer.transform(test_values)
    imputed_names = np.asarray(imputer.get_feature_names_out(feature_names), dtype=object)
    missing_indicator_names = [str(name) for name in imputed_names if str(name).startswith("missingindicator_")]
    variance = VarianceThreshold(threshold=0.0)
    train_variable = variance.fit_transform(train_imputed)
    test_variable = variance.transform(test_imputed)
    variable_names = imputed_names[variance.get_support()]
    post_variance_count = int(train_variable.shape[1])
    effective_k = min(int(requested_k), post_variance_count)
    if effective_k <= 0:
        raise RuntimeError("No usable features remain after fold-local variance filtering")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_variable)
    test_scaled = scaler.transform(test_variable)
    selector = SelectKBest(score_func=f_classif, k=effective_k)
    with np.errstate(divide="ignore", invalid="ignore"):
        train_selected = selector.fit_transform(train_scaled, train_labels)
    test_selected = selector.transform(test_scaled)
    selected_names = [str(name) for name in variable_names[selector.get_support()]]
    if not np.isfinite(train_selected).all() or not np.isfinite(test_selected).all():
        raise RuntimeError("Non-finite values remain after fold-local preprocessing")
    arrays: Iterable[np.ndarray] = [
        np.asarray(imputer.statistics_),
        np.asarray(variance.variances_),
        np.asarray(scaler.mean_),
        np.asarray(scaler.scale_),
        np.asarray(selector.scores_),
    ]
    state = {
        "requested_feature_k": int(requested_k),
        "post_variance_feature_count": post_variance_count,
        "effective_feature_k": effective_k,
        "selected_feature_names": selected_names,
        "missing_indicator_feature_names": missing_indicator_names,
        "imputer_input_feature_count": len(feature_names),
        "imputer_output_feature_count": int(train_imputed.shape[1]),
        "preprocessing_state_bytes": int(sum(array.nbytes for array in arrays)),
        "fit_scope": "outer-training only",
    }
    timing = {"preprocessing_seconds": time.perf_counter() - start}
    return train_selected, test_selected, state, timing


def encode_values(
    train_values: np.ndarray,
    test_values: np.ndarray,
    selected_names: list[str],
    levels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    quantizer = EqualWidthQuantizer(levels).fit(train_values)
    quantized = np.vstack([quantizer.transform(train_values), quantizer.transform(test_values)])
    effective_k = len(selected_names)
    encoded = incremental_encode_prefixes(
        quantized,
        selected_names,
        levels,
        seed,
        [effective_k],
        10_000,
    )
    values = encoded.samples_by_k[str(effective_k)]
    return values[: len(train_values)], values[len(train_values) :], {
        "encoding_seconds": time.perf_counter() - start,
        "quantizer_state_sha256": quantizer.state_digest(),
        "codebook_hashes": dict(encoded.codebook_hashes),
        "work_dimension": 10_000,
        "classification_dimension_prefix": 5_000,
        "regression_dimension": 10_000,
    }


def classification_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    recalls = recall_score(truth, prediction, labels=[0, 1, 2, 3], average=None, zero_division=0)
    matrix = confusion_matrix(truth, prediction, labels=[0, 1, 2, 3])
    return {
        "macro_f1": float(f1_score(truth, prediction, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "severe_error_rate": float(np.mean(np.abs(prediction - truth) >= 2)),
        "per_class_recall": {str(index): float(value) for index, value in enumerate(recalls)},
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def regression_metrics(truth: np.ndarray, raw: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    bounded = np.clip(raw, 1.0, 4.0)
    correlation = spearmanr(truth, bounded).statistic
    per_level = {
        str(float(level)): float(mean_absolute_error(truth[truth == level], bounded[truth == level]))
        for level in sorted(np.unique(truth))
    }
    clipped = raw != bounded
    metrics = {
        "raw_mae": float(mean_absolute_error(truth, raw)),
        "bounded_mae": float(mean_absolute_error(truth, bounded)),
        "bounded_rmse": float(mean_squared_error(truth, bounded) ** 0.5),
        "bounded_r2": float(r2_score(truth, bounded)),
        "bounded_spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        "per_target_level_mae": per_level,
        "clipping_count": int(clipped.sum()),
        "clipping_rate": float(clipped.mean()),
    }
    return metrics, bounded


def write_run_artifacts(
    run: dict[str, Any],
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    efficiency: dict[str, Any],
    checkpoint: dict[str, Any],
) -> None:
    atomic_csv(prediction_path(run), predictions)
    atomic_json(metrics_path(run), metrics)
    atomic_json(efficiency_path(run), efficiency)
    checkpoint["artifact_hashes"] = {
        "predictions": sha256(prediction_path(run)),
        "metrics": sha256(metrics_path(run)),
        "efficiency": sha256(efficiency_path(run)),
    }
    checkpoint["checkpoint_integrity"] = "PASS"
    checkpoint["status"] = "COMPLETE"
    checkpoint["result"] = "PASS"
    checkpoint["checkpoint_payload_sha256"] = payload_digest(checkpoint)
    atomic_json(checkpoint_path(run), checkpoint)
    if not valid_checkpoint(run, read_json(CONTRACT_PATH), len(predictions)):
        raise RuntimeError(f"Checkpoint failed post-write integrity validation: {run['run_id']}")


def common_checkpoint(
    run: dict[str, Any],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    preprocessing: dict[str, Any],
    encoding: dict[str, Any],
) -> dict[str, Any]:
    train_subjects = sorted(train_frame["subject_id"].astype(str).unique())
    test_subjects = sorted(test_frame["subject_id"].astype(str).unique())
    overlap = sorted(set(train_subjects).intersection(test_subjects))
    if overlap:
        raise RuntimeError(f"Subject leakage detected for {run['run_id']}")
    return {
        "phase": "07",
        "stage": "unimodal_batch_execution",
        "timestamp_utc": now_utc(),
        **run,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_subjects": len(train_subjects),
        "test_subjects": len(test_subjects),
        "subject_overlap": overlap,
        "subject_overlap_count": 0,
        "primary_sha256": EXPECTED_PRIMARY,
        "frozen_fold_sha256": EXPECTED_FOLDS,
        **preprocessing,
        "encoding": encoding,
        "prediction_row_count": int(len(test_frame)),
        "unique_test_run_key_count": int(test_frame["run_key"].nunique()),
        "outer_test_used_for_fitting_or_selection": False,
        "inner_cv_executed": False,
        "model_reselection_performed": False,
        "other_hdc_variants_executed": False,
        "leakage_audit_result": "PASS",
    }


def run_seed_pair(
    modality: str,
    outer_fold: int,
    seed: int,
    contract: dict[str, Any],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    train_processed: np.ndarray,
    test_processed: np.ndarray,
    preprocessing: dict[str, Any],
    preprocessing_timing: dict[str, Any],
    completed: set[str],
) -> None:
    pair_runs = [
        {"run_id": f"{modality}|{task}|fold={outer_fold}|seed={seed}", "modality": modality, "task": task, "outer_fold": outer_fold, "seed": seed}
        for task in TASKS
    ]
    pending = [run for run in pair_runs if run["run_id"] not in completed]
    if not pending:
        print(f"REUSE pair modality={modality} fold={outer_fold} seed={seed}")
        return
    train_hv, test_hv, encoding = encode_values(
        train_processed,
        test_processed,
        preprocessing["selected_feature_names"],
        int(contract["classification"]["levels"]),
        seed,
    )
    truth_class = train_frame["target_class"].to_numpy(dtype=np.int64)
    test_truth_class = test_frame["target_class"].to_numpy(dtype=np.int64)
    truth_score = train_frame["target_score"].to_numpy(dtype=float)
    test_truth_score = test_frame["target_score"].to_numpy(dtype=float)
    base_columns = {
        "run_key": test_frame["run_key"].astype(str).to_numpy(),
        "subject_id": test_frame["subject_id"].astype(str).to_numpy(),
        "outer_fold": np.full(len(test_frame), outer_fold),
        "modality": np.full(len(test_frame), modality),
        "seed": np.full(len(test_frame), seed),
        "modality_available": (~test_frame[preprocessing["input_feature_names"]].isna().all(axis=1)).to_numpy(),
    }
    for run in pending:
        if run["task"] == "classification":
            structure = frozen_structure(contract, outer_fold)
            training_start = time.perf_counter()
            model, model_info = train_hybrid(
                train_hv[:, :5000],
                truth_class,
                centroids_per_class=int(structure["centroids_per_class"]),
                epochs=int(structure["epochs"]),
                learning_rate=float(structure["learning_rate"]),
                margin_threshold=float(structure["margin_threshold"]),
                seed=seed,
                stream_identifier=f"phase07|modality={modality}|outer={outer_fold}|seed={seed}|hybrid",
            )
            training_seconds = time.perf_counter() - training_start
            inference_start = time.perf_counter()
            predicted, scores = predict_hybrid(test_hv[:, :5000], model)
            inference_seconds = time.perf_counter() - inference_start
            metrics = classification_metrics(test_truth_class, predicted)
            order = np.sort(scores, axis=1)
            predictions = pd.DataFrame({
                **base_columns,
                "task": "classification",
                "target_class": test_truth_class,
                "predicted_class": predicted,
                "classification_confidence": scores.max(axis=1),
                "similarity_margin": order[:, -1] - order[:, -2],
                **{f"class_score_{class_id}": scores[:, class_id] for class_id in range(4)},
            })
            efficiency = {
                **preprocessing_timing,
                "encoding_seconds": encoding["encoding_seconds"],
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                "model_bytes": int(model.nbytes),
                "preprocessing_state_bytes": preprocessing["preprocessing_state_bytes"],
            }
            checkpoint = common_checkpoint(run, train_frame, test_frame, preprocessing, encoding)
            checkpoint.update({
                "model": "HDC+OnlineHD Hybrid",
                "selected_variant": "hybrid",
                "dimension": 5000,
                "levels": 51,
                "hybrid_structure": structure,
                "classification_metrics": metrics,
                "model_info": model_info,
                "efficiency": efficiency,
            })
        else:
            training_start = time.perf_counter()
            model = Ridge(alpha=0.01, fit_intercept=True, solver="lsqr")
            model.fit(train_hv.astype(np.float32, copy=False), truth_score)
            training_seconds = time.perf_counter() - training_start
            inference_start = time.perf_counter()
            raw = model.predict(test_hv.astype(np.float32, copy=False))
            inference_seconds = time.perf_counter() - inference_start
            metrics, bounded = regression_metrics(test_truth_score, raw)
            predictions = pd.DataFrame({
                **base_columns,
                "task": "regression",
                "target_score": test_truth_score,
                "prediction_raw": raw,
                "prediction_bounded": bounded,
                "residual_bounded": test_truth_score - bounded,
            })
            model_bytes = int(np.asarray(model.coef_).nbytes + np.asarray(model.intercept_).nbytes)
            efficiency = {
                **preprocessing_timing,
                "encoding_seconds": encoding["encoding_seconds"],
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                "model_bytes": model_bytes,
                "preprocessing_state_bytes": preprocessing["preprocessing_state_bytes"],
            }
            checkpoint = common_checkpoint(run, train_frame, test_frame, preprocessing, encoding)
            checkpoint.update({
                "selected_variant": "common_ridge",
                "regression_head": "COMMON_ENCODER_READOUT_BASELINE",
                "dimension": 10000,
                "levels": 51,
                "ridge_alpha": 0.01,
                "target_description": "bounded difficulty-induced workload proxy regression",
                "regression_metrics": metrics,
                "efficiency": efficiency,
            })
        # input_feature_names is an execution-only convenience and is not duplicated into checkpoints.
        checkpoint.pop("input_feature_names", None)
        write_run_artifacts(run, predictions, metrics, efficiency, checkpoint)
        completed.add(run["run_id"])
        update_execution_manifest(completed, executor_completed=False)
        progress = {
            "timestamp_utc": now_utc(),
            "event": "CHECKPOINT_COMPLETE",
            "run_id": run["run_id"],
            "completed_runs": len(completed),
            "total_runs": 250,
            "checkpoint_sha256": sha256(checkpoint_path(run)),
        }
        append_progress(progress)
        print(f"CHECKPOINT PASS {len(completed)}/250 {run['run_id']}")


def update_execution_manifest(completed: set[str], executor_completed: bool) -> None:
    manifest = read_json(EXECUTION_MANIFEST)
    classification = sum("|classification|" in run_id for run_id in completed)
    regression = sum("|regression|" in run_id for run_id in completed)
    manifest.update({
        "status": "UNIMODAL_EXECUTION_COMPLETE_PENDING_OOF_CONSOLIDATION" if executor_completed else "UNIMODAL_EXECUTION_RUNNING",
        "last_updated_utc": now_utc(),
        "completed_runs": len(completed),
        "completed_classification_runs": classification,
        "completed_regression_runs": regression,
        "training_executed": bool(completed),
        "executor_invoked": True,
        "executor_completed": executor_completed,
        "canonical_oof_generated": False,
        "prediction_files_generated": len(completed),
        "oof_files_generated": 0,
    })
    atomic_json(EXECUTION_MANIFEST, manifest)


def valid_completed_runs(contract: dict[str, Any], folds: pd.DataFrame) -> set[str]:
    completed: set[str] = set()
    test_counts = folds.groupby("outer_fold").size().to_dict()
    for run in enumerate_runs(contract):
        if valid_checkpoint(run, contract, int(test_counts[run["outer_fold"]])):
            completed.add(run["run_id"])
    return completed


def static_gate(contract: dict[str, Any]) -> dict[str, Any]:
    runs = enumerate_runs(contract)
    run_ids = [run["run_id"] for run in runs]
    classifications = sum(run["task"] == "classification" for run in runs)
    regressions = sum(run["task"] == "regression" for run in runs)
    gate = {
        "phase": "07",
        "audit": "executor_static",
        "timestamp_utc": now_utc(),
        "primary_sha256": sha256(PRIMARY),
        "frozen_fold_sha256": sha256(FOLDS),
        "classification_runs": classifications,
        "regression_runs": regressions,
        "total_runs": len(runs),
        "duplicate_run_identifiers": len(run_ids) - len(set(run_ids)),
        "other_hdc_variants_present_in_scope": False,
        "inner_cv_or_search_present_in_execution_scope": False,
        "result": "PASS" if classifications == regressions == 125 and len(runs) == len(set(run_ids)) == 250 else "FAIL",
    }
    atomic_json(PHASE_DIR / "audits" / "phase07_executor_static_audit.json", gate)
    return gate


def execute_batch() -> None:
    contract, modalities = load_frozen_inputs()
    static = static_gate(contract)
    if static["result"] != "PASS":
        raise RuntimeError("Executor static gate failed")
    data = pd.read_csv(PRIMARY, low_memory=False)
    folds = pd.read_csv(FOLDS, low_memory=False)
    if len(data) != 419 or data["run_key"].nunique() != 419 or len(folds) != 419:
        raise RuntimeError("Unexpected frozen cohort")
    data = data.sort_values("run_key", kind="stable").reset_index(drop=True)
    completed = valid_completed_runs(contract, folds)
    update_execution_manifest(completed, executor_completed=False)
    print(f"RESUME VALID CHECKPOINTS {len(completed)}/250")
    for modality, features in modalities.items():
        for outer_fold in range(1, 6):
            train_frame = data.loc[data["outer_fold"] != outer_fold].copy()
            test_frame = data.loc[data["outer_fold"] == outer_fold].copy()
            overlap = set(train_frame["subject_id"]).intersection(test_frame["subject_id"])
            if overlap:
                raise RuntimeError(f"Outer subject leakage modality={modality} fold={outer_fold}")
            pair_ids = {
                f"{modality}|{task}|fold={outer_fold}|seed={seed}"
                for seed in contract["randomness"]["seeds"]
                for task in TASKS
            }
            if pair_ids.issubset(completed):
                print(f"REUSE modality/fold all checkpoints modality={modality} fold={outer_fold}")
                continue
            train_processed, test_processed, preprocessing, timing = fit_preprocessing(
                train_frame[features].to_numpy(dtype=float),
                test_frame[features].to_numpy(dtype=float),
                train_frame["target_class"].to_numpy(dtype=np.int64),
                features,
                int(contract["preprocessing"]["requested_feature_k"]),
            )
            preprocessing["input_feature_names"] = features
            print(
                f"PREPROCESS PASS modality={modality} fold={outer_fold} train={len(train_frame)} test={len(test_frame)} "
                f"post_variance={preprocessing['post_variance_feature_count']} effective_k={preprocessing['effective_feature_k']}"
            )
            for seed in contract["randomness"]["seeds"]:
                run_seed_pair(
                    modality,
                    outer_fold,
                    int(seed),
                    contract,
                    train_frame,
                    test_frame,
                    train_processed,
                    test_processed,
                    preprocessing,
                    timing,
                    completed,
                )
    if len(completed) != 250:
        raise RuntimeError(f"Batch ended with incomplete run count: {len(completed)}/250")
    update_execution_manifest(completed, executor_completed=True)
    experiment = read_json(EXPERIMENT_CONTRACT)
    experiment.update({
        "status": "UNIMODAL_EXECUTION_COMPLETE_PENDING_OOF_CONSOLIDATION",
        "unimodal_execution_completed_at_utc": now_utc(),
        "training_executed": True,
        "outer_test_predictions_generated": True,
        "canonical_oof_generated": False,
    })
    atomic_json(EXPERIMENT_CONTRACT, experiment)
    append_progress({"timestamp_utc": now_utc(), "event": "BATCH_COMPLETE", "completed_runs": 250, "total_runs": 250})
    print("PHASE 07 UNIMODAL BATCH CORE EXECUTION COMPLETE 250/250")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Phase 07 unimodal batch.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    STDOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        contract, _ = load_frozen_inputs()
        if args.dry_run:
            result = static_gate(contract)
            print(json.dumps(result, indent=2))
            return 0 if result["result"] == "PASS" else 1
        with STDOUT_PATH.open("a", encoding="utf-8") as stdout_file, STDERR_PATH.open("a", encoding="utf-8") as stderr_file:
            with redirect_stdout(Tee(sys.__stdout__, stdout_file)), redirect_stderr(Tee(sys.__stderr__, stderr_file)):
                execute_batch()
        return 0
    except Exception as error:
        failure = {
            "phase": "07",
            "stage": "unimodal_batch_execution",
            "timestamp_utc": now_utc(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "result": "FAIL",
        }
        atomic_json(PHASE_DIR / "audits" / "phase07_unimodal_execution_failure.json", failure)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
