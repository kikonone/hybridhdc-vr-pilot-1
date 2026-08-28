"""Resumable executor for exactly the 370 frozen Phase 08 model-runs."""

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
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer


PHASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE_DIR.parents[1]
PHASE07_SCRIPTS = PROJECT_ROOT / "experiments" / "phase_07_unimodal_contribution" / "scripts"
sys.path.insert(0, str(PHASE07_SCRIPTS))

from run_phase07_unimodal_batch import (  # noqa: E402
    classification_metrics,
    encode_values,
    fit_preprocessing,
    predict_hybrid,
    regression_metrics,
    train_hybrid,
)


P3 = PROJECT_ROOT / "experiments" / "phase_03_multimodal_dataset_labeling"
DATA_PATHS = {
    "primary": P3 / "data/primary_without_performance.csv",
    "with_performance": P3 / "data/auxiliary_with_performance.csv",
    "performance_only": P3 / "data/performance_only.csv",
    "folds": P3 / "data/fold_assignments.csv",
}
EXPECTED_HASHES = {
    "primary": "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44",
    "with_performance": "72977a2119e30e37996fb9f0e3404988c4977fb7d2b33992f87bf54bfe5decba",
    "performance_only": "d602282ae41153886d1306494515f2e41a5e7e89a2cec5c192d44b9ca87a07a4",
    "folds": "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f",
}
EXECUTION_MANIFEST = PHASE_DIR / "configs/phase08_execution_manifest.json"
MODEL_MATRIX = PHASE_DIR / "configs/phase08_model_matrix.json"
EXPERIMENT_CONTRACT = PHASE_DIR / "configs/phase08_experiment_contract.json"
FROZEN_CONTRACT = PHASE_DIR / "configs/phase08_frozen_contract.json"
FUSION_CONDITIONS = PHASE_DIR / "configs/phase08_fusion_conditions.json"
SHORTCUT_CONDITIONS = PHASE_DIR / "configs/phase08_shortcut_conditions.json"
FUSION_MANIFEST = PHASE_DIR / "manifests/phase08_fusion_feature_manifest.json"
PERFORMANCE_MANIFEST = PHASE_DIR / "manifests/phase08_performance_feature_manifest.json"
FLIGHT_PROVENANCE = PHASE_DIR / "manifests/phase08_flight_feature_provenance_manifest.json"
PRIMARY_MANIFEST = P3 / "manifests/primary_feature_manifest.json"
WITH_PERFORMANCE_MANIFEST = P3 / "manifests/with_performance_feature_manifest.json"
PERFORMANCE_ONLY_MANIFEST = P3 / "manifests/performance_only_feature_manifest.json"
PROGRESS_LOG = PHASE_DIR / "logs/phase08_batch_progress.jsonl"
STDOUT_LOG = PHASE_DIR / "logs/phase08_batch_stdout.log"
STDERR_LOG = PHASE_DIR / "logs/phase08_batch_stderr.log"
FAILURE_AUDIT = PHASE_DIR / "audits/phase08_batch_failure.json"
EXPECTED_TOTAL_RUNS = 370
EXPECTED_RAW_ROWS = 31_006


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_set_sha256(features: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(features)) + "\n").encode("utf-8")).hexdigest()


def run_key_set_sha256(run_keys: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(run_keys)) + "\n").encode("utf-8")).hexdigest()


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
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def payload_digest(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "checkpoint_payload_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_locked_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    required = [EXECUTION_MANIFEST, MODEL_MATRIX, EXPERIMENT_CONTRACT, FROZEN_CONTRACT, FUSION_CONDITIONS, SHORTCUT_CONDITIONS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing required frozen execution inputs: {missing}")
    hashes = {name: sha256(path) for name, path in DATA_PATHS.items()}
    if hashes != EXPECTED_HASHES:
        raise RuntimeError(f"Frozen data checksum mismatch: {hashes}")
    execution = read_json(EXECUTION_MANIFEST)
    matrix = read_json(MODEL_MATRIX)
    experiment = read_json(EXPERIMENT_CONTRACT)
    frozen = read_json(FROZEN_CONTRACT)
    if experiment.get("status") != "CONTRACT_FROZEN_NOT_TRAINED" and execution.get("status") not in {"EXECUTION_RUNNING", "EXECUTION_COMPLETE_PENDING_OOF_CONSOLIDATION"}:
        raise RuntimeError("Phase 08 execution is not authorized by the frozen contract")
    if frozen.get("status") != "CONTRACT_FROZEN_NOT_TRAINED" or matrix.get("status") != "CONTRACT_FROZEN_NOT_TRAINED":
        raise RuntimeError("Frozen Phase 08 contract/model matrix status mismatch")
    runs = list(execution.get("run_records", []))
    run_ids = [run["run_id"] for run in runs]
    if len(runs) != EXPECTED_TOTAL_RUNS or len(set(run_ids)) != EXPECTED_TOTAL_RUNS:
        raise RuntimeError("Execution manifest is not the frozen unique 370-run matrix")
    if any(run["condition"] == "FLIGHT_TASK_SETTING_ONLY" for run in runs):
        raise RuntimeError("Infeasible flight task-setting condition entered the execution queue")
    if any("phase09" in json.dumps(run).casefold() for run in runs):
        raise RuntimeError("Phase 09 task entered the Phase 08 execution queue")
    return execution, matrix, runs


def feature_sets() -> dict[str, list[str]]:
    fusion = read_json(FUSION_MANIFEST)["combinations"]
    primary = read_json(PRIMARY_MANIFEST)
    with_performance = read_json(WITH_PERFORMANCE_MANIFEST)
    performance_only = read_json(PERFORMANCE_ONLY_MANIFEST)
    provenance = read_json(FLIGHT_PROVENANCE)
    conditions = {
        "FUSION_PE": list(fusion["physiological_plus_eye"]["features"]),
        "FUSION_PEH": list(fusion["physiological_plus_eye_plus_head"]["features"]),
        "FUSION_PEHF": list(fusion["physiological_plus_eye_plus_head_plus_flight"]["features"]),
        "WITH_PERFORMANCE_AUXILIARY": list(with_performance["features"]),
        "PERFORMANCE_ONLY_AUXILIARY": list(performance_only["features"]),
        "FLIGHT_BEHAVIORAL_ONLY": [item["feature_name"] for item in provenance["features"] if item["semantic_category"] == "BEHAVIORAL_RESPONSE"],
        "FLIGHT_FULL": list(primary["feature_groups"]["flight_parameter_features"]),
    }
    expected = {"FUSION_PE": 649, "FUSION_PEH": 808, "FUSION_PEHF": 1134, "WITH_PERFORMANCE_AUXILIARY": 1235, "PERFORMANCE_ONLY_AUXILIARY": 59, "FLIGHT_BEHAVIORAL_ONLY": 323, "FLIGHT_FULL": 326}
    if {name: len(features) for name, features in conditions.items()} != expected:
        raise RuntimeError("Frozen condition feature counts changed")
    if any(len(features) != len(set(features)) for features in conditions.values()):
        raise RuntimeError("Duplicate feature found within a frozen condition")
    return conditions


def condition_data() -> dict[str, pd.DataFrame]:
    tables = {
        "primary": pd.read_csv(DATA_PATHS["primary"], low_memory=False),
        "with_performance": pd.read_csv(DATA_PATHS["with_performance"], low_memory=False),
        "performance_only": pd.read_csv(DATA_PATHS["performance_only"], low_memory=False),
    }
    for frame in tables.values():
        frame.sort_values("run_key", kind="stable", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    identity = ["run_key", "subject_id", "target_class", "target_score", "outer_fold"]
    reference = tables["primary"][identity].astype(str)
    for name in ("with_performance", "performance_only"):
        if not reference.equals(tables[name][identity].astype(str)):
            raise RuntimeError(f"Frozen dataset alignment mismatch: {name}")
    return {
        "FUSION_PE": tables["primary"], "FUSION_PEH": tables["primary"], "FUSION_PEHF": tables["primary"],
        "WITH_PERFORMANCE_AUXILIARY": tables["with_performance"], "PERFORMANCE_ONLY_AUXILIARY": tables["performance_only"],
        "FLIGHT_BEHAVIORAL_ONLY": tables["primary"], "FLIGHT_FULL": tables["primary"],
    }


def seed_token(run: dict[str, Any]) -> str:
    return f"seed_{run['seed']}" if run.get("seed") is not None else "canonical"


def checkpoint_path(run: dict[str, Any]) -> Path:
    return PHASE_DIR / "results/checkpoints" / run["condition"] / run["model_family"] / run["task"] / f"fold_{run['outer_fold']}" / seed_token(run) / "checkpoint.json"


def prediction_path(run: dict[str, Any]) -> Path:
    return PHASE_DIR / "results/predictions" / run["condition"] / run["model_family"] / run["task"] / f"fold_{run['outer_fold']}_{seed_token(run)}_predictions.csv"


def metrics_path(run: dict[str, Any]) -> Path:
    return PHASE_DIR / "results/fold_metrics" / run["condition"] / run["model_family"] / run["task"] / f"fold_{run['outer_fold']}_{seed_token(run)}_metrics.json"


def expected_test_keys(folds: pd.DataFrame, outer_fold: int) -> list[str]:
    return folds.loc[folds["outer_fold"] == outer_fold, "run_key"].astype(str).tolist()


def valid_checkpoint(run: dict[str, Any], folds: pd.DataFrame, matrix_sha: str) -> bool:
    path = checkpoint_path(run)
    try:
        checkpoint = read_json(path)
        if checkpoint.get("status") != "COMPLETE" or checkpoint.get("result") != "PASS":
            return False
        if checkpoint.get("checkpoint_payload_sha256") != payload_digest(checkpoint):
            return False
        for key in ("run_id", "condition", "model_family", "task", "outer_fold", "seed"):
            if checkpoint.get(key) != run.get(key):
                return False
        if checkpoint.get("model_matrix_sha256") != matrix_sha or checkpoint.get("frozen_fold_sha256") != EXPECTED_HASHES["folds"]:
            return False
        expected_keys = expected_test_keys(folds, int(run["outer_fold"]))
        if checkpoint.get("expected_test_run_key_sha256") != run_key_set_sha256(expected_keys):
            return False
        if checkpoint.get("leakage_audit_result") != "PASS" or checkpoint.get("checkpoint_integrity") != "PASS":
            return False
        prediction = prediction_path(run)
        metrics = metrics_path(run)
        hashes = checkpoint.get("artifact_hashes", {})
        if not prediction.is_file() or not metrics.is_file() or hashes.get("predictions") != sha256(prediction) or hashes.get("metrics") != sha256(metrics):
            return False
        frame = pd.read_csv(prediction, low_memory=False)
        if len(frame) != len(expected_keys) or set(frame["run_key"].astype(str)) != set(expected_keys) or frame["run_key"].nunique() != len(expected_keys):
            return False
        if run["task"] == "classification":
            values = frame["y_pred"].to_numpy(dtype=float)
            if not np.isfinite(values).all() or not set(values.astype(int)).issubset({0, 1, 2, 3}):
                return False
        else:
            raw = frame["y_pred_raw"].to_numpy(dtype=float)
            bounded = frame["y_pred_bounded"].to_numpy(dtype=float)
            if not np.isfinite(raw).all() or not np.isfinite(bounded).all() or not frame["y_pred_bounded"].between(1.0, 4.0).all():
                return False
        return True
    except Exception:
        return False


def frozen_hybrid_structure(matrix: dict[str, Any], outer_fold: int) -> dict[str, Any]:
    matches = [item for item in matrix["HDC"]["classification"]["fold_specific_structures"] if int(item["outer_fold"]) == outer_fold]
    if len(matches) != 1:
        raise RuntimeError(f"Missing frozen Hybrid structure for fold {outer_fold}")
    return json.loads(matches[0]["selected_structure_json"])


def fit_traditional_preprocessing(
    train_values: np.ndarray,
    test_values: np.ndarray,
    train_target: np.ndarray,
    feature_names: list[str],
    requested_k: int | str,
    task: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    train_imputed = imputer.fit_transform(train_values)
    test_imputed = imputer.transform(test_values)
    names = np.asarray(imputer.get_feature_names_out(feature_names), dtype=object)
    variance = VarianceThreshold(0.0)
    train_variable = variance.fit_transform(train_imputed)
    test_variable = variance.transform(test_imputed)
    variable_names = names[variance.get_support()]
    post_variance = int(train_variable.shape[1])
    effective_k = post_variance if requested_k == "all" else min(int(requested_k), post_variance)
    score_func = f_classif if task == "classification" else f_regression
    selector = SelectKBest(score_func=score_func, k=effective_k)
    with np.errstate(divide="ignore", invalid="ignore"):
        train_selected = selector.fit_transform(train_variable, train_target)
    test_selected = selector.transform(test_variable)
    if not np.isfinite(train_selected).all() or not np.isfinite(test_selected).all():
        raise RuntimeError("Traditional fold-local preprocessing left non-finite values")
    state = {
        "fit_scope": "outer-training only", "requested_feature_k": requested_k,
        "post_variance_feature_count": post_variance, "effective_feature_k": effective_k,
        "selected_feature_names": [str(name) for name in variable_names[selector.get_support()]],
        "missing_indicator_feature_names": [str(name) for name in names if str(name).startswith("missingindicator_")],
        "score_func": "f_classif" if task == "classification" else "f_regression",
        "preprocessing_seconds": time.perf_counter() - started,
    }
    return train_selected, test_selected, state


def common_prediction_columns(run: dict[str, Any], test: pd.DataFrame, model: str) -> dict[str, Any]:
    return {
        "run_key": test["run_key"].astype(str).to_numpy(), "subject_id": test["subject_id"].astype(str).to_numpy(),
        "outer_fold": np.full(len(test), int(run["outer_fold"])), "condition": np.full(len(test), run["condition"]),
        "model": np.full(len(test), model), "model_family": np.full(len(test), run["model_family"]),
        "task": np.full(len(test), run["task"]), "seed": np.full(len(test), run.get("seed") if run.get("seed") is not None else "canonical"),
    }


def base_checkpoint(
    run: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame, features: list[str],
    preprocessing: dict[str, Any], matrix_sha: str, model_name: str,
) -> dict[str, Any]:
    overlap = sorted(set(train["subject_id"].astype(str)) & set(test["subject_id"].astype(str)))
    if overlap:
        raise RuntimeError(f"Outer subject leakage: {run['run_id']}")
    expected_keys = test["run_key"].astype(str).tolist()
    return {
        "phase": "08", "stage": "frozen_batch_execution", "timestamp_utc": now_utc(), **run,
        "model": model_name, "model_matrix_sha256": matrix_sha,
        "frozen_contract_sha256": sha256(FROZEN_CONTRACT), "frozen_fold_sha256": EXPECTED_HASHES["folds"],
        "dataset_sha256": EXPECTED_HASHES["with_performance"] if run["condition"] == "WITH_PERFORMANCE_AUXILIARY" else EXPECTED_HASHES["performance_only"] if run["condition"] == "PERFORMANCE_ONLY_AUXILIARY" else EXPECTED_HASHES["primary"],
        "feature_count": len(features), "feature_set_sha256": feature_set_sha256(features),
        "train_rows": len(train), "test_rows": len(test), "prediction_row_count": len(test),
        "unique_test_run_key_count": test["run_key"].nunique(), "expected_test_run_key_sha256": run_key_set_sha256(expected_keys),
        "train_subject_count": train["subject_id"].nunique(), "test_subject_count": test["subject_id"].nunique(),
        "subject_overlap": overlap, "subject_overlap_count": len(overlap),
        "preprocessing": preprocessing, "frozen_config_matching": "PASS",
        "outer_test_feature_access_after_training_fit": True, "outer_test_used_for_fitting_or_selection": False,
        "inner_cv_executed": False, "hyperparameter_search_executed": False, "seed_selection_executed": False,
        "leakage_audit_result": "PASS",
    }


def write_artifacts(run: dict[str, Any], predictions: pd.DataFrame, metrics: dict[str, Any], checkpoint: dict[str, Any], folds: pd.DataFrame, matrix_sha: str) -> None:
    atomic_csv(prediction_path(run), predictions)
    atomic_json(metrics_path(run), metrics)
    checkpoint["artifact_hashes"] = {"predictions": sha256(prediction_path(run)), "metrics": sha256(metrics_path(run))}
    checkpoint.update({"checkpoint_integrity": "PASS", "status": "COMPLETE", "result": "PASS"})
    checkpoint["checkpoint_payload_sha256"] = payload_digest(checkpoint)
    atomic_json(checkpoint_path(run), checkpoint)
    if not valid_checkpoint(run, folds, matrix_sha):
        raise RuntimeError(f"Post-write checkpoint validation failed: {run['run_id']}")


def run_hdc_pair(
    condition: str, outer_fold: int, seed: int, matrix: dict[str, Any], features: list[str], data: pd.DataFrame,
    run_lookup: dict[tuple[str, str, str, int, int | None], dict[str, Any]], completed: set[str], folds: pd.DataFrame, matrix_sha: str,
) -> None:
    runs = [run_lookup[(condition, "HDC", task, outer_fold, seed)] for task in ("classification", "regression")]
    pending = [run for run in runs if run["run_id"] not in completed]
    if not pending:
        return
    train = data.loc[data["outer_fold"] != outer_fold].copy()
    test = data.loc[data["outer_fold"] == outer_fold].copy()
    processed_train, processed_test, preprocessing, preprocessing_timing = fit_preprocessing(
        train[features].to_numpy(float), test[features].to_numpy(float), train["target_class"].to_numpy(np.int64), features, 50,
    )
    preprocessing["input_feature_names"] = features
    train_hv, test_hv, encoding = encode_values(processed_train, processed_test, preprocessing["selected_feature_names"], 51, seed)
    for run in pending:
        if run["task"] == "classification":
            structure = frozen_hybrid_structure(matrix, outer_fold)
            started = time.perf_counter()
            model, model_info = train_hybrid(
                train_hv[:, :5000], train["target_class"].to_numpy(np.int64),
                centroids_per_class=int(structure["centroids_per_class"]), epochs=int(structure["epochs"]),
                learning_rate=float(structure["learning_rate"]), margin_threshold=float(structure["margin_threshold"]),
                seed=seed, stream_identifier=f"phase08|{condition}|fold={outer_fold}|seed={seed}|hybrid",
            )
            training_seconds = time.perf_counter() - started
            inference_started = time.perf_counter()
            predicted, scores = predict_hybrid(test_hv[:, :5000], model)
            inference_seconds = time.perf_counter() - inference_started
            metrics = classification_metrics(test["target_class"].to_numpy(np.int64), predicted)
            metrics.update({"training_seconds": training_seconds, "inference_seconds": inference_seconds, **preprocessing_timing, "encoding_seconds": encoding["encoding_seconds"]})
            predictions = pd.DataFrame({
                **common_prediction_columns(run, test, "HDC+OnlineHD Hybrid"),
                "y_true": test["target_class"].to_numpy(np.int64), "y_pred": predicted,
                **{f"class_score_{class_id}": scores[:, class_id] for class_id in range(4)},
            })
            checkpoint = base_checkpoint(run, train, test, features, {key: value for key, value in preprocessing.items() if key != "input_feature_names"}, matrix_sha, "HDC+OnlineHD Hybrid")
            checkpoint.update({"dimension": 5000, "levels": 51, "hybrid_structure": structure, "model_info": model_info, "classification_metrics": metrics, "encoding": encoding})
        else:
            from sklearn.linear_model import Ridge
            started = time.perf_counter()
            model = Ridge(alpha=0.01, fit_intercept=True, solver="lsqr")
            model.fit(train_hv.astype(np.float32, copy=False), train["target_score"].to_numpy(float))
            training_seconds = time.perf_counter() - started
            inference_started = time.perf_counter()
            raw = model.predict(test_hv.astype(np.float32, copy=False))
            inference_seconds = time.perf_counter() - inference_started
            metrics, bounded = regression_metrics(test["target_score"].to_numpy(float), raw)
            metrics.update({"training_seconds": training_seconds, "inference_seconds": inference_seconds, **preprocessing_timing, "encoding_seconds": encoding["encoding_seconds"]})
            predictions = pd.DataFrame({
                **common_prediction_columns(run, test, "COMMON_ENCODER_READOUT_BASELINE"),
                "y_true": test["target_score"].to_numpy(float), "y_pred_raw": raw, "y_pred_bounded": bounded,
            })
            checkpoint = base_checkpoint(run, train, test, features, {key: value for key, value in preprocessing.items() if key != "input_feature_names"}, matrix_sha, "COMMON_ENCODER_READOUT_BASELINE")
            checkpoint.update({"dimension": 10000, "levels": 51, "ridge_alpha": 0.01, "target_description": "bounded difficulty-induced workload proxy regression", "regression_metrics": metrics, "encoding": encoding})
        write_artifacts(run, predictions, metrics, checkpoint, folds, matrix_sha)
        completed.add(run["run_id"])
        update_manifest(completed, False)
        append_progress({"timestamp_utc": now_utc(), "event": "CHECKPOINT_COMPLETE", "run_id": run["run_id"], "completed_runs": len(completed), "total_runs": EXPECTED_TOTAL_RUNS, "checkpoint_sha256": sha256(checkpoint_path(run))})
        print(f"CHECKPOINT PASS {len(completed)}/{EXPECTED_TOTAL_RUNS} {run['run_id']}")


def run_traditional(
    run: dict[str, Any], matrix: dict[str, Any], features: list[str], data: pd.DataFrame,
    completed: set[str], folds: pd.DataFrame, matrix_sha: str,
) -> None:
    if run["run_id"] in completed:
        return
    outer_fold = int(run["outer_fold"])
    train = data.loc[data["outer_fold"] != outer_fold].copy()
    test = data.loc[data["outer_fold"] == outer_fold].copy()
    interface = matrix["traditional"][run["task"]]
    params = interface["fold_specific_parameters"][str(outer_fold)]
    if run["task"] == "classification":
        requested_k = params["selector__k"]
        processed_train, processed_test, preprocessing = fit_traditional_preprocessing(
            train[features].to_numpy(float), test[features].to_numpy(float), train["target_class"].to_numpy(np.int64), features, requested_k, "classification",
        )
        model = GradientBoostingClassifier(
            n_estimators=int(params["classifier__n_estimators"]), learning_rate=float(params["classifier__learning_rate"]),
            max_depth=int(params["classifier__max_depth"]), random_state=42,
        )
        started = time.perf_counter()
        model.fit(processed_train, train["target_class"].to_numpy(np.int64))
        training_seconds = time.perf_counter() - started
        inference_started = time.perf_counter()
        predicted = model.predict(processed_test)
        probabilities = model.predict_proba(processed_test)
        inference_seconds = time.perf_counter() - inference_started
        scores = np.zeros((len(test), 4), dtype=float)
        for index, class_value in enumerate(model.classes_.astype(int)):
            scores[:, class_value] = probabilities[:, index]
        metrics = classification_metrics(test["target_class"].to_numpy(np.int64), predicted)
        metrics.update({"training_seconds": training_seconds, "inference_seconds": inference_seconds})
        predictions = pd.DataFrame({
            **common_prediction_columns(run, test, "GradientBoostingClassifier"),
            "y_true": test["target_class"].to_numpy(np.int64), "y_pred": predicted,
            **{f"class_score_{class_id}": scores[:, class_id] for class_id in range(4)},
        })
        checkpoint = base_checkpoint(run, train, test, features, preprocessing, matrix_sha, "GradientBoostingClassifier")
        checkpoint.update({"frozen_fold_parameters": params, "random_state": 42, "classification_metrics": metrics})
    else:
        requested_k = params["feature_selection__k"]
        processed_train, processed_test, preprocessing = fit_traditional_preprocessing(
            train[features].to_numpy(float), test[features].to_numpy(float), train["target_score"].to_numpy(float), features, requested_k, "regression",
        )
        model = GradientBoostingRegressor(
            n_estimators=int(params["regressor__n_estimators"]), learning_rate=float(params["regressor__learning_rate"]),
            max_depth=int(params["regressor__max_depth"]), loss="squared_error", subsample=1.0,
            min_samples_leaf=1, max_features=None, random_state=42,
        )
        started = time.perf_counter()
        model.fit(processed_train, train["target_score"].to_numpy(float))
        training_seconds = time.perf_counter() - started
        inference_started = time.perf_counter()
        raw = model.predict(processed_test)
        inference_seconds = time.perf_counter() - inference_started
        metrics, bounded = regression_metrics(test["target_score"].to_numpy(float), raw)
        metrics.update({"training_seconds": training_seconds, "inference_seconds": inference_seconds})
        predictions = pd.DataFrame({
            **common_prediction_columns(run, test, "GradientBoostingRegressor"),
            "y_true": test["target_score"].to_numpy(float), "y_pred_raw": raw, "y_pred_bounded": bounded,
        })
        checkpoint = base_checkpoint(run, train, test, features, preprocessing, matrix_sha, "GradientBoostingRegressor")
        checkpoint.update({"frozen_fold_parameters": params, "random_state": 42, "target_description": "bounded difficulty-induced workload proxy regression", "regression_metrics": metrics})
    write_artifacts(run, predictions, metrics, checkpoint, folds, matrix_sha)
    completed.add(run["run_id"])
    update_manifest(completed, False)
    append_progress({"timestamp_utc": now_utc(), "event": "CHECKPOINT_COMPLETE", "run_id": run["run_id"], "completed_runs": len(completed), "total_runs": EXPECTED_TOTAL_RUNS, "checkpoint_sha256": sha256(checkpoint_path(run))})
    print(f"CHECKPOINT PASS {len(completed)}/{EXPECTED_TOTAL_RUNS} {run['run_id']}")


def update_manifest(completed: set[str], executor_completed: bool) -> None:
    manifest = read_json(EXECUTION_MANIFEST)
    lookup = set(completed)
    updated_records = []
    for record in manifest["run_records"]:
        item = dict(record)
        item["status"] = "COMPLETE" if item["run_id"] in lookup else "AUTHORIZED_NOT_EXECUTED"
        updated_records.append(item)
    counts = {
        "HDC_classification": sum("__HDC__classification__" in run_id for run_id in completed),
        "HDC_regression": sum("__HDC__regression__" in run_id for run_id in completed),
        "traditional_classification": sum("__TRADITIONAL__classification__" in run_id for run_id in completed),
        "traditional_regression": sum("__TRADITIONAL__regression__" in run_id for run_id in completed),
    }
    manifest.update({
        "status": "EXECUTION_COMPLETE_PENDING_OOF_CONSOLIDATION" if executor_completed else "EXECUTION_RUNNING",
        "last_updated_utc": now_utc(), "completed_runs": len(completed), "completion_counts": counts,
        "run_records": updated_records, "training_executed": bool(completed), "outer_test_predictions_generated": bool(completed),
        "final_oof_consolidation_executed": False, "phase09_executed": False, "executor_completed": executor_completed,
    })
    atomic_json(EXECUTION_MANIFEST, manifest)


def valid_completed_runs(runs: list[dict[str, Any]], folds: pd.DataFrame, matrix_sha: str) -> set[str]:
    return {run["run_id"] for run in runs if valid_checkpoint(run, folds, matrix_sha)}


def static_gate() -> dict[str, Any]:
    _, matrix, runs = load_locked_inputs()
    features = feature_sets()
    run_ids = [run["run_id"] for run in runs]
    path_tokens = [(str(checkpoint_path(run)), str(prediction_path(run)), str(metrics_path(run))) for run in runs]
    flat_paths = [path for paths in path_tokens for path in paths]
    counts = {
        "HDC_classification": sum(run["model_family"] == "HDC" and run["task"] == "classification" for run in runs),
        "HDC_regression": sum(run["model_family"] == "HDC" and run["task"] == "regression" for run in runs),
        "traditional_classification": sum(run["model_family"] == "TRADITIONAL" and run["task"] == "classification" for run in runs),
        "traditional_regression": sum(run["model_family"] == "TRADITIONAL" and run["task"] == "regression" for run in runs),
    }
    condition_counts = {condition: sum(run["condition"] == condition for run in runs) for condition in features}
    checks = {
        "unique_370_run_ids": len(runs) == len(set(run_ids)) == EXPECTED_TOTAL_RUNS,
        "frozen_model_task_counts": counts == {"HDC_classification": 150, "HDC_regression": 150, "traditional_classification": 35, "traditional_regression": 35},
        "condition_counts": condition_counts == {"FUSION_PE": 60, "FUSION_PEH": 60, "FUSION_PEHF": 60, "WITH_PERFORMANCE_AUXILIARY": 60, "PERFORMANCE_ONLY_AUXILIARY": 60, "FLIGHT_BEHAVIORAL_ONLY": 60, "FLIGHT_FULL": 10},
        "output_paths_unique": len(flat_paths) == len(set(flat_paths)),
        "task_setting_excluded": all(run["condition"] != "FLIGHT_TASK_SETTING_ONLY" for run in runs),
        "phase09_excluded": all("phase09" not in json.dumps(run).casefold() for run in runs),
        "model_interfaces_pass": all(value == "PASS" for value in matrix["interfaces"].values()),
        "dataset_checksums_pass": all(sha256(DATA_PATHS[name]) == EXPECTED_HASHES[name] for name in EXPECTED_HASHES),
    }
    audit = {"status": "PASS" if all(checks.values()) else "FAIL", "timestamp_utc": now_utc(), "checks": checks, "run_counts": counts, "condition_counts": condition_counts, "dry_run_unique_runs": len(set(run_ids)), "duplicate_run_identifiers": len(run_ids) - len(set(run_ids)), "output_path_collisions": len(flat_paths) - len(set(flat_paths)), "training_executed": False}
    atomic_json(PHASE_DIR / "audits/phase08_executor_validation_audit.json", audit)
    return audit


def execute_batch() -> None:
    _, matrix, runs = load_locked_inputs()
    gate = static_gate()
    if gate["status"] != "PASS":
        raise RuntimeError("Phase 08 executor static gate failed")
    features = feature_sets()
    datasets = condition_data()
    folds = pd.read_csv(DATA_PATHS["folds"], low_memory=False)
    matrix_sha = sha256(MODEL_MATRIX)
    completed = valid_completed_runs(runs, folds, matrix_sha)
    update_manifest(completed, False)
    append_progress({"timestamp_utc": now_utc(), "event": "RESUME_SCAN", "recovered_valid_checkpoints": len(completed), "total_runs": EXPECTED_TOTAL_RUNS})
    print(f"RESUME VALID CHECKPOINTS {len(completed)}/{EXPECTED_TOTAL_RUNS}")
    lookup = {(run["condition"], run["model_family"], run["task"], int(run["outer_fold"]), run.get("seed")): run for run in runs}
    hdc_conditions = ["FUSION_PE", "FUSION_PEH", "FUSION_PEHF", "WITH_PERFORMANCE_AUXILIARY", "PERFORMANCE_ONLY_AUXILIARY", "FLIGHT_BEHAVIORAL_ONLY"]
    for condition in hdc_conditions:
        for outer_fold in range(1, 6):
            for seed in [42, 43, 44, 45, 46]:
                run_hdc_pair(condition, outer_fold, seed, matrix, features[condition], datasets[condition], lookup, completed, folds, matrix_sha)
    for run in runs:
        if run["model_family"] == "TRADITIONAL":
            run_traditional(run, matrix, features[run["condition"]], datasets[run["condition"]], completed, folds, matrix_sha)
    completed = valid_completed_runs(runs, folds, matrix_sha)
    if len(completed) != EXPECTED_TOTAL_RUNS:
        raise RuntimeError(f"Execution incomplete after batch: {len(completed)}/{EXPECTED_TOTAL_RUNS}")
    raw_rows = sum(len(pd.read_csv(prediction_path(run), usecols=["run_key"])) for run in runs)
    if raw_rows != EXPECTED_RAW_ROWS:
        raise RuntimeError(f"Raw prediction row mismatch: {raw_rows}/{EXPECTED_RAW_ROWS}")
    update_manifest(completed, True)
    append_progress({"timestamp_utc": now_utc(), "event": "BATCH_COMPLETE", "completed_runs": len(completed), "total_runs": EXPECTED_TOTAL_RUNS, "raw_prediction_rows": raw_rows})
    print(f"PHASE 08 BATCH EXECUTION COMPLETE {len(completed)}/{EXPECTED_TOTAL_RUNS} rows={raw_rows}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exactly the frozen Phase 08 370-run batch.")
    parser.add_argument("--dry-run", action="store_true", help="Enumerate and validate only; do not train.")
    args = parser.parse_args()
    STDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.dry_run:
            audit = static_gate()
            print(json.dumps(audit, ensure_ascii=False, indent=2))
            return 0 if audit["status"] == "PASS" else 1
        with STDOUT_LOG.open("a", encoding="utf-8") as stdout_file, STDERR_LOG.open("a", encoding="utf-8") as stderr_file:
            with redirect_stdout(Tee(sys.__stdout__, stdout_file)), redirect_stderr(Tee(sys.__stderr__, stderr_file)):
                execute_batch()
        return 0
    except Exception as error:
        failure = {"timestamp_utc": now_utc(), "stage": "phase08_batch_execution", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "result": "FAIL"}
        atomic_json(FAILURE_AUDIT, failure)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
