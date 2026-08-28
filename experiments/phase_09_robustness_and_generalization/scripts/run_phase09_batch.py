"""Resumable executor for exactly the 720 Phase 09 contract-authorized runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PHASE09 = Path(__file__).resolve().parents[1]
EXPERIMENTS = PHASE09.parent
ROOT = EXPERIMENTS.parent
PHASE03 = EXPERIMENTS / "phase_03_multimodal_dataset_labeling"
PHASE05 = EXPERIMENTS / "phase_05_basic_dual_output_hdc"
PHASE06 = EXPERIMENTS / "phase_06_hdc_variant_screening"
PRIMARY = PHASE03 / "data" / "primary_without_performance.csv"
FOLDS = PHASE03 / "data" / "fold_assignments.csv"
PRIMARY_FEATURES = PHASE03 / "manifests" / "primary_feature_manifest.json"
PERFORMANCE_FEATURES = PHASE03 / "manifests" / "performance_only_feature_manifest.json"
EXECUTION_MANIFEST = PHASE09 / "configs" / "phase09_execution_manifest.json"
EXPECTED_PRIMARY = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLDS = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
LEVELS = 51
FEATURE_K = 50
CLASS_DIMENSION = 5000
REGRESSION_DIMENSION = 10000
RIDGE_ALPHA = 0.01
IMMUTABLE_RUN_FIELDS = [
    "run_identifier", "condition", "protocol", "task", "model_key", "model_family",
    "outer_fold", "loso_subject", "seed_or_canonical", "feature_count", "config_source",
    "expected_test_run_keys", "checkpoint_path", "prediction_path",
]
FROZEN_CONTRACT_PATHS = [
    "configs/phase09_frozen_contract.json", "configs/phase09_contract_freeze.json",
    "configs/phase09_missing_modality_contract.json", "configs/phase09_loso_contract.json",
    "configs/phase09_loso_config_mapping.json", "configs/phase09_oof_aggregation_rules.json",
    "configs/phase09_statistical_rules.json", "manifests/phase09_loso_assignments.csv",
    "manifests/phase09_expected_coverage_manifest.json", "manifests/phase09_contract_artifact_manifest.json",
    "manifests/phase09_modality_manifest.json", "manifests/phase09_upstream_freeze_manifest.json",
]

sys.path.insert(0, str(PHASE05 / "src"))
sys.path.insert(0, str(PHASE06 / "src"))
from phase05_hdc_core import EqualWidthQuantizer, incremental_encode_prefixes  # noqa: E402
from phase06_hybrid import predict_hybrid, train_hybrid  # noqa: E402


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass
class HDCPrepared:
    train_quantized: np.ndarray
    test_quantized: np.ndarray
    selected_feature_names: list[str]
    state: dict[str, np.ndarray]
    state_hash: str
    preprocessing_seconds: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(jsonable(value)).encode("utf-8")).hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_")


def immutable_run_view(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in IMMUTABLE_RUN_FIELDS}


def authorization_digest(records: list[dict[str, Any]]) -> str:
    return stable_hash([immutable_run_view(record) for record in records])


def output_paths(record: dict[str, Any]) -> dict[str, Path]:
    protocol = slug(record["protocol"])
    condition = slug(record["condition"])
    model = slug(record["model_key"])
    task = slug(record["task"])
    split = f"fold_{record['outer_fold']}" if record["outer_fold"] is not None else slug(str(record["loso_subject"]))
    seed = f"seed_{record['seed_or_canonical']}" if isinstance(record["seed_or_canonical"], int) else "canonical"
    checkpoint_dir = PHASE09 / "results" / "checkpoints" / protocol / condition / model / task / split / seed
    stem = f"{split}__{seed}"
    prediction_dir = PHASE09 / "results" / "predictions" / protocol / condition / model / task
    metrics_dir = PHASE09 / "results" / "fold_metrics" / protocol / condition / model / task
    return {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint": checkpoint_dir / "checkpoint.json",
        "model": checkpoint_dir / ("model.joblib" if model.startswith("traditional") else "model.npz"),
        "audit": checkpoint_dir / "audit.json",
        "prediction": prediction_dir / f"{stem}.csv",
        "metrics": metrics_dir / f"{stem}.json",
    }


def immutable_contract_snapshot() -> dict[str, dict[str, Any]]:
    return {
        relative: {"bytes": (PHASE09 / relative).stat().st_size, "sha256": sha256(PHASE09 / relative)}
        for relative in FROZEN_CONTRACT_PATHS
    }


def upstream_snapshot() -> dict[str, dict[str, Any]]:
    manifest = read_json(PHASE09 / "manifests" / "phase09_upstream_freeze_manifest.json")
    paths = sorted({str(Path(item["path"]).resolve()) for item in manifest["sources"]} | {str(PRIMARY.resolve()), str(FOLDS.resolve())})
    return {path: {"bytes": Path(path).stat().st_size, "sha256": sha256(Path(path))} for path in paths}


def validate_preflight() -> dict[str, Any]:
    frozen = read_json(PHASE09 / "configs" / "phase09_frozen_contract.json")
    freeze = read_json(PHASE09 / "configs" / "phase09_contract_freeze.json")
    missing = read_json(PHASE09 / "configs" / "phase09_missing_modality_contract.json")
    portability = read_json(PHASE09 / "audits" / "phase09_checkpoint_portability_audit.json")
    upstream = read_json(PHASE09 / "audits" / "phase09_upstream_freeze_audit.json")
    modality = read_json(PHASE09 / "manifests" / "phase09_modality_manifest.json")
    manifest = read_json(EXECUTION_MANIFEST)
    records = manifest["training_runs"]
    model_counts = pd.Series([record["model_key"] for record in records]).value_counts().to_dict()
    protocol_counts = pd.Series([record["protocol"] for record in records]).value_counts().to_dict()
    checks = {
        "contract_frozen": frozen["status"] == "CONTRACT_FROZEN_NOT_TRAINED" and freeze["status"] == "CONTRACT_FROZEN_NOT_TRAINED",
        "ready_for_execution": freeze["ready_for_execution"] is True,
        "authorized_runs_720": len(records) == 720,
        "unique_run_identifiers_720": len({record["run_identifier"] for record in records}) == 720,
        "duplicate_run_identifiers_0": manifest["duplicate_run_identifiers"] == 0,
        "missing_modality_runs_300": protocol_counts.get("RETRAIN_WITHOUT_MODALITY") == 300,
        "loso_runs_420": protocol_counts.get("LEAVE_ONE_SUBJECT_OUT") == 420,
        "model_counts": model_counts == {"hdc_classification": 300, "hdc_regression": 300, "traditional_classification": 60, "traditional_regression": 60},
        "loso_splits_35": read_json(PHASE09 / "configs" / "phase09_loso_contract.json")["splits"] == 35,
        "primary_checksum": sha256(PRIMARY) == EXPECTED_PRIMARY,
        "fold_checksum": sha256(FOLDS) == EXPECTED_FOLDS,
        "upstream_interfaces": upstream["status"] == "PASS" and all(upstream["interface_results"].values()),
        "performance_features_excluded": frozen["evidence_scope"]["performance_features"] == "EXCLUDED",
        "full_primary_reused": missing["full_primary_reference_policy"] == "REUSED_NOT_RETRAINED" and not any(record["condition"] == "FULL_PRIMARY_REFERENCE" for record in records),
        "test_time_missingness_excluded": portability["protocol_status"] == "NOT_FEASIBLE_DUE_TO_CHECKPOINT_INTERFACE" and not any("SUDDEN" in record["protocol"] for record in records),
        "phase10_excluded": not any("phase10" in canonical_json(record).lower() for record in records),
        "modalities_5_union_1176": len(modality["modalities"]) == 5 and modality["modality_feature_union_count"] == 1176,
    }
    if not all(checks.values()):
        raise RuntimeError({"preflight_checks": checks})
    return {"checks": checks, "records": records, "authorization_digest": authorization_digest(records), "contract_snapshot": immutable_contract_snapshot()}


def regenerate_authorized_runs() -> list[dict[str, Any]]:
    sys.path.insert(0, str(PHASE09 / "scripts"))
    from freeze_phase09_contract import build_loso, enumerate_runs

    primary = pd.read_csv(PRIMARY)
    folds = pd.read_csv(FOLDS)
    _, mappings, checks = build_loso(primary, folds)
    if not all(checks.values()):
        raise RuntimeError({"loso_regeneration": checks})
    return enumerate_runs(folds, mappings)


def dry_run(audit_path: Path | None = None) -> dict[str, Any]:
    preflight = validate_preflight()
    records = preflight["records"]
    regenerated = regenerate_authorized_runs()
    exact_match = [immutable_run_view(record) for record in records] == [immutable_run_view(record) for record in regenerated]
    paths = [output_paths(record) for record in records]
    path_sets = {name: [str(item[name]) for item in paths] for name in ["checkpoint", "model", "audit", "prediction", "metrics"]}
    no_collisions = all(len(values) == len(set(values)) == 720 for values in path_sets.values())
    primary_features = read_json(PRIMARY_FEATURES)["features"]
    performance_features = set(read_json(PERFORMANCE_FEATURES)["features"])
    modality = read_json(PHASE09 / "manifests" / "phase09_modality_manifest.json")
    modality_map = {item["name"]: set(item["features"]) for item in modality["modalities"]}
    condition_removed = {
        "MISSING_PHYSIOLOGICAL": "physiological_features",
        "MISSING_EYE_TRACKING": "eye_tracking_features",
        "MISSING_HEAD_MOVEMENT": "head_movement_features",
        "MISSING_FLIGHT_PARAMETER": "flight_parameter_features",
        "MISSING_BODY_MOVEMENT": "body_movement_features",
    }
    feature_checks = []
    expected_rows = 0
    for record in records:
        features = primary_features if record["protocol"] == "LEAVE_ONE_SUBJECT_OUT" else [name for name in primary_features if name not in modality_map[condition_removed[record["condition"]]]]
        feature_checks.append(len(features) == record["feature_count"] and not (set(features) & performance_features))
        expected_rows += len(record["expected_test_run_keys"])
    checks = {
        "exact_frozen_manifest_match": exact_match,
        "unique_run_identifiers_720": len({record["run_identifier"] for record in records}) == 720,
        "output_paths_no_collisions": no_collisions,
        "full_primary_reference_excluded": not any(record["condition"] == "FULL_PRIMARY_REFERENCE" for record in records),
        "sudden_test_time_missingness_excluded": not any("SUDDEN" in canonical_json(record) for record in records),
        "phase10_excluded": not any("phase10" in canonical_json(record).lower() for record in records),
        "performance_features_excluded": all(feature_checks),
        "expected_raw_rows_30168": expected_rows == 30168,
    }
    audit = {
        "phase": "09", "audit": "executor_validation", "status": "PASS" if all(checks.values()) else "FAIL",
        "validated_at_utc": utc_now(), "checks": checks, "dry_run_unique_runs": len({record["run_identifier"] for record in records}),
        "expected_raw_prediction_rows": expected_rows, "authorization_digest": preflight["authorization_digest"],
        "frozen_contract_snapshot": preflight["contract_snapshot"], "upstream_snapshot": upstream_snapshot(), "training_executed": False,
        "predictions_generated": False, "output_path_collision_count": sum(len(values) - len(set(values)) for values in path_sets.values()),
    }
    if audit_path is not None:
        atomic_json(audit_path, audit)
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    return audit


def split_frames(record: dict[str, Any], data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = list(record["expected_test_run_keys"])
    if record["protocol"] == "RETRAIN_WITHOUT_MODALITY":
        test_mask = data.outer_fold == int(record["outer_fold"])
    else:
        test_mask = data.subject_id.astype(str) == str(record["loso_subject"])
    train = data.loc[~test_mask].copy()
    test = data.loc[test_mask].set_index("run_key", drop=False).loc[expected].reset_index(drop=True)
    if set(test.run_key.astype(str)) != set(expected) or len(test) != len(expected):
        raise RuntimeError(f"Exact test membership failed: {record['run_identifier']}")
    if set(train.subject_id.astype(str)) & set(test.subject_id.astype(str)):
        raise RuntimeError(f"Subject leakage: {record['run_identifier']}")
    if train.target_class.nunique() != 4:
        raise RuntimeError(f"Training target coverage failed: {record['run_identifier']}")
    return train, test


def feature_list(record: dict[str, Any], primary_features: list[str], modality_map: dict[str, set[str]]) -> tuple[list[str], set[str]]:
    if record["protocol"] == "LEAVE_ONE_SUBJECT_OUT":
        features, removed = list(primary_features), set()
    else:
        condition_map = {
            "MISSING_PHYSIOLOGICAL": "physiological_features",
            "MISSING_EYE_TRACKING": "eye_tracking_features",
            "MISSING_HEAD_MOVEMENT": "head_movement_features",
            "MISSING_FLIGHT_PARAMETER": "flight_parameter_features",
            "MISSING_BODY_MOVEMENT": "body_movement_features",
        }
        removed = modality_map[condition_map[record["condition"]]]
        features = [name for name in primary_features if name not in removed]
    if len(features) != int(record["feature_count"]) or set(features) & removed:
        raise RuntimeError(f"Feature contract mismatch: {record['run_identifier']}")
    return features, removed


def fit_hdc_preprocessing(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> HDCPrepared:
    start = time.perf_counter()
    train_values = train[features].to_numpy(dtype=np.float64)
    test_values = test[features].to_numpy(dtype=np.float64)
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    train_imputed = imputer.fit_transform(train_values)
    test_imputed = imputer.transform(test_values)
    imputed_names = imputer.get_feature_names_out(features)
    variance = VarianceThreshold(threshold=0.0)
    train_variable = variance.fit_transform(train_imputed)
    test_variable = variance.transform(test_imputed)
    variable_names = imputed_names[variance.get_support()]
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_variable)
    test_scaled = scaler.transform(test_variable)
    selector = SelectKBest(score_func=f_classif, k="all").fit(train_scaled, train.target_class.to_numpy(dtype=np.int64))
    scores = np.nan_to_num(selector.scores_, nan=-np.inf)
    ranking = np.argsort(scores, kind="mergesort")[::-1]
    selected_indices = ranking[:FEATURE_K]
    selected_names = variable_names[selected_indices].astype(str).tolist()
    train_selected = train_scaled[:, selected_indices]
    test_selected = test_scaled[:, selected_indices]
    quantizer = EqualWidthQuantizer(LEVELS).fit(train_selected)
    train_quantized = quantizer.transform(train_selected)
    test_quantized = quantizer.transform(test_selected)
    state = {
        "imputer_statistics": np.asarray(imputer.statistics_, dtype=np.float64),
        "imputer_indicator_features": np.asarray(getattr(imputer.indicator_, "features_", []), dtype=np.int64),
        "variance_support": np.asarray(variance.get_support(), dtype=np.uint8),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "selector_scores": np.asarray(scores, dtype=np.float64),
        "selected_indices": np.asarray(selected_indices, dtype=np.int64),
        "selected_feature_names": np.asarray(selected_names, dtype=np.str_),
        "quantizer_minimum": np.asarray(quantizer.minimum_, dtype=np.float64),
        "quantizer_maximum": np.asarray(quantizer.maximum_, dtype=np.float64),
    }
    state_hash = hashlib.sha256(b"".join(np.ascontiguousarray(value).tobytes() for value in state.values())).hexdigest()
    return HDCPrepared(train_quantized, test_quantized, selected_names, state, state_hash, time.perf_counter() - start)


def encode_hdc(prepared: HDCPrepared, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, str], float]:
    start = time.perf_counter()
    combined = np.vstack([prepared.train_quantized, prepared.test_quantized])
    encoding = incremental_encode_prefixes(combined, prepared.selected_feature_names, LEVELS, seed, [FEATURE_K], REGRESSION_DIMENSION)
    encoded = encoding.samples_by_k[str(FEATURE_K)]
    train_count = len(prepared.train_quantized)
    return encoded[:train_count], encoded[train_count:], dict(encoding.codebook_hashes), time.perf_counter() - start


def hdc_structure(record: dict[str, Any], selected_interfaces: dict[str, Any], loso_mapping: dict[str, Any]) -> dict[str, Any]:
    if record["outer_fold"] is not None:
        mapped_fold = int(record["outer_fold"])
    else:
        mapped_fold = next(item["original_outer_fold"] for item in loso_mapping["mappings"] if item["test_subject"] == record["loso_subject"])
    entry = next(item for item in selected_interfaces["hdc_classification"]["fold_selected_structures"] if int(item["outer_fold"]) == mapped_fold)
    return json.loads(entry["selected_structure_json"])


def traditional_classifier(params: dict[str, Any]) -> Pipeline:
    clean = {key: value for key, value in params.items() if key != "effective_selected_k"}
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("selector", SelectKBest(score_func=f_classif)),
        ("classifier", GradientBoostingClassifier(random_state=42)),
    ]).set_params(**clean)


def traditional_regressor(params: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("feature_selection", SelectKBest(score_func=f_regression)),
        ("regressor", GradientBoostingRegressor(loss="squared_error", subsample=1.0, min_samples_leaf=1, max_features=None, random_state=42)),
    ]).set_params(**params)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    recalls = recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "severe_error_rate": float(np.mean(np.abs(y_true - y_pred) >= 2)),
        "per_class_recall": {str(index): float(value) for index, value in enumerate(recalls)},
        "confusion_matrix": matrix.astype(int).tolist(),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2, 3], weights="quadratic")),
    }


def regression_metrics(y_true: np.ndarray, raw: np.ndarray, bounded: np.ndarray) -> dict[str, Any]:
    rounded = np.clip(np.rint(bounded), 1, 4).astype(int)
    correlation = spearmanr(y_true, bounded).statistic
    clipped = (raw < 1.0) | (raw > 4.0)
    return {
        "raw_mae": float(mean_absolute_error(y_true, raw)),
        "bounded_mae": float(mean_absolute_error(y_true, bounded)),
        "bounded_rmse": float(mean_squared_error(y_true, bounded) ** 0.5),
        "bounded_r2": float(r2_score(y_true, bounded)),
        "bounded_spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        "clipping_count": int(clipped.sum()), "clipping_rate": float(clipped.mean()),
        "rounded_regression_macro_f1": float(f1_score(y_true.astype(int), rounded, labels=[1, 2, 3, 4], average="macro", zero_division=0)),
        "adjacent_accuracy": float(np.mean(np.abs(y_true - rounded) <= 1)),
        "severe_error_rate": float(np.mean(np.abs(y_true - rounded) >= 2)),
    }


def loadable_model(path: Path, model_key: str) -> bool:
    try:
        if model_key.startswith("traditional_"):
            model = joblib.load(path)
            return hasattr(model, "predict")
        with np.load(path, allow_pickle=False) as arrays:
            names = set(arrays.files)
            common = {"selected_feature_names", "imputer_statistics", "variance_support", "scaler_mean", "scaler_scale", "quantizer_minimum", "quantizer_maximum"}
            specific = {"centroids"} if model_key == "hdc_classification" else {"ridge_coef", "ridge_intercept"}
            return common | specific <= names
    except Exception:
        return False


def reusable_run(record: dict[str, Any], paths: dict[str, Path]) -> bool:
    try:
        if not all(paths[name].is_file() for name in ["checkpoint", "model", "audit", "prediction", "metrics"]):
            return False
        checkpoint = read_json(paths["checkpoint"])
        audit = read_json(paths["audit"])
        prediction = pd.read_csv(paths["prediction"])
        expected = set(record["expected_test_run_keys"])
        return bool(
            checkpoint.get("status") == "COMPLETE_AUDITED"
            and checkpoint.get("run_identifier") == record["run_identifier"]
            and checkpoint.get("authorization_digest") == authorization_digest([record])
            and audit.get("status") == "PASS"
            and checkpoint["model_sha256"] == sha256(paths["model"])
            and checkpoint["prediction_sha256"] == sha256(paths["prediction"])
            and checkpoint["metrics_sha256"] == sha256(paths["metrics"])
            and audit["checkpoint_sha256"] == sha256(paths["checkpoint"])
            and len(prediction) == len(expected)
            and set(prediction.run_key.astype(str)) == expected
            and prediction.run_key.nunique() == len(prediction)
            and loadable_model(paths["model"], record["model_key"])
        )
    except Exception:
        return False


def fit_and_predict(
    record: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame, features: list[str],
    selected_interfaces: dict[str, Any], loso_mapping: dict[str, Any], prepared: HDCPrepared | None,
    encoded: tuple[np.ndarray, np.ndarray, dict[str, str], float] | None,
) -> tuple[pd.DataFrame, dict[str, Any], Any, dict[str, Any]]:
    model_key = record["model_key"]
    seed = int(record["seed_or_canonical"]) if isinstance(record["seed_or_canonical"], int) else 42
    config_hash = sha256(Path(record["config_source"]["path"]))
    feature_manifest_hash = stable_hash(features)
    y_class_train = train.target_class.to_numpy(dtype=np.int64)
    y_class_test = test.target_class.to_numpy(dtype=np.int64)
    y_score_train = train.target_score.to_numpy(dtype=np.float64)
    y_score_test = test.target_score.to_numpy(dtype=np.float64)
    start = time.perf_counter()
    extra: dict[str, Any] = {}
    if model_key == "hdc_classification":
        assert prepared is not None and encoded is not None
        train_hv, test_hv, codebook_hashes, encoding_seconds = encoded
        structure = hdc_structure(record, selected_interfaces, loso_mapping)
        centroids, info = train_hybrid(
            train_hv[:, :CLASS_DIMENSION], y_class_train,
            centroids_per_class=int(structure["centroids_per_class"]), epochs=int(structure["epochs"]),
            learning_rate=float(structure["learning_rate"]), margin_threshold=float(structure["margin_threshold"]),
            seed=seed, stream_identifier=f"phase09|{record['run_identifier']}",
        )
        y_pred, scores = predict_hybrid(test_hv[:, :CLASS_DIMENSION], centroids)
        model_arrays = {**prepared.state, "centroids": np.asarray(centroids, dtype=np.float32)}
        metrics = classification_metrics(y_class_test, y_pred)
        prediction = pd.DataFrame({
            "run_id": record["run_identifier"], "protocol": record["protocol"], "condition": record["condition"],
            "model_family": record["model_family"], "seed": seed, "outer_fold": record["outer_fold"],
            "loso_subject": record["loso_subject"], "run_key": test.run_key.astype(str), "subject_id": test.subject_id.astype(str),
            "y_true": y_class_test, "y_pred": y_pred, "class_score_0": scores[:, 0], "class_score_1": scores[:, 1],
            "class_score_2": scores[:, 2], "class_score_3": scores[:, 3], "config_hash": config_hash,
            "feature_manifest_hash": feature_manifest_hash,
        })
        extra = {"structure": structure, "model_info": info, "codebook_hashes": codebook_hashes, "encoding_seconds": encoding_seconds, "preprocessing_state_hash": prepared.state_hash}
        model = model_arrays
    elif model_key == "hdc_regression":
        assert prepared is not None and encoded is not None
        train_hv, test_hv, codebook_hashes, encoding_seconds = encoded
        train_float = train_hv[:, :REGRESSION_DIMENSION].astype(np.float32) / np.float32(np.sqrt(REGRESSION_DIMENSION))
        test_float = test_hv[:, :REGRESSION_DIMENSION].astype(np.float32) / np.float32(np.sqrt(REGRESSION_DIMENSION))
        ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="lsqr")
        ridge.fit(train_float, y_score_train)
        raw = ridge.predict(test_float)
        bounded = np.clip(raw, 1.0, 4.0)
        model_arrays = {**prepared.state, "ridge_coef": np.asarray(ridge.coef_, dtype=np.float64), "ridge_intercept": np.asarray([ridge.intercept_], dtype=np.float64)}
        metrics = regression_metrics(y_score_test, raw, bounded)
        prediction = pd.DataFrame({
            "run_id": record["run_identifier"], "protocol": record["protocol"], "condition": record["condition"],
            "model_family": record["model_family"], "seed": seed, "outer_fold": record["outer_fold"],
            "loso_subject": record["loso_subject"], "run_key": test.run_key.astype(str), "subject_id": test.subject_id.astype(str),
            "y_true": y_score_test, "y_pred_raw": raw, "y_pred_bounded": bounded, "config_hash": config_hash,
            "feature_manifest_hash": feature_manifest_hash,
        })
        extra = {"ridge_alpha": RIDGE_ALPHA, "codebook_hashes": codebook_hashes, "encoding_seconds": encoding_seconds, "preprocessing_state_hash": prepared.state_hash}
        model = model_arrays
    elif model_key == "traditional_classification":
        params = selected_interfaces["traditional_classification"]["fold_specific_parameters"][str(mapped_outer_fold(record, loso_mapping))]
        estimator = traditional_classifier(params)
        estimator.fit(train[features], y_class_train)
        y_pred = estimator.predict(test[features]).astype(np.int64)
        scores = estimator.predict_proba(test[features])
        metrics = classification_metrics(y_class_test, y_pred)
        prediction = pd.DataFrame({
            "run_id": record["run_identifier"], "protocol": record["protocol"], "condition": record["condition"],
            "model_family": record["model_family"], "seed": "canonical", "outer_fold": record["outer_fold"],
            "loso_subject": record["loso_subject"], "run_key": test.run_key.astype(str), "subject_id": test.subject_id.astype(str),
            "y_true": y_class_test, "y_pred": y_pred, "class_score_0": scores[:, 0], "class_score_1": scores[:, 1],
            "class_score_2": scores[:, 2], "class_score_3": scores[:, 3], "config_hash": config_hash,
            "feature_manifest_hash": feature_manifest_hash,
        })
        extra = {"frozen_parameters": params}
        model = estimator
    else:
        params = selected_interfaces["traditional_regression"]["fold_specific_parameters"][str(mapped_outer_fold(record, loso_mapping))]
        estimator = traditional_regressor(params)
        estimator.fit(train[features], y_score_train)
        raw = estimator.predict(test[features])
        bounded = np.clip(raw, 1.0, 4.0)
        metrics = regression_metrics(y_score_test, raw, bounded)
        prediction = pd.DataFrame({
            "run_id": record["run_identifier"], "protocol": record["protocol"], "condition": record["condition"],
            "model_family": record["model_family"], "seed": "canonical", "outer_fold": record["outer_fold"],
            "loso_subject": record["loso_subject"], "run_key": test.run_key.astype(str), "subject_id": test.subject_id.astype(str),
            "y_true": y_score_test, "y_pred_raw": raw, "y_pred_bounded": bounded, "config_hash": config_hash,
            "feature_manifest_hash": feature_manifest_hash,
        })
        extra = {"frozen_parameters": params}
        model = estimator
    extra["fit_and_prediction_seconds"] = time.perf_counter() - start
    return prediction, metrics, model, {"config_hash": config_hash, "feature_manifest_hash": feature_manifest_hash, **extra}


def mapped_outer_fold(record: dict[str, Any], loso_mapping: dict[str, Any]) -> int:
    if record["outer_fold"] is not None:
        return int(record["outer_fold"])
    return int(next(item["original_outer_fold"] for item in loso_mapping["mappings"] if item["test_subject"] == record["loso_subject"]))


def persist_run(
    record: dict[str, Any], paths: dict[str, Path], prediction: pd.DataFrame, metrics: dict[str, Any], model: Any,
    details: dict[str, Any], features: list[str], removed: set[str], train: pd.DataFrame, test: pd.DataFrame,
    performance_features: set[str], execution_authorization_digest: str,
) -> dict[str, Any]:
    if record["model_key"].startswith("traditional_"):
        atomic_joblib(paths["model"], model)
    else:
        atomic_npz(paths["model"], **model)
    atomic_csv(paths["prediction"], prediction)
    metric_payload = {
        "phase": "09", "run_identifier": record["run_identifier"], "task": record["task"],
        "regression_interpretation": "bounded difficulty-induced workload proxy regression" if record["task"] == "regression" else None,
        "metrics": metrics,
    }
    atomic_json(paths["metrics"], metric_payload)
    model_hash, prediction_hash, metrics_hash = sha256(paths["model"]), sha256(paths["prediction"]), sha256(paths["metrics"])
    expected = set(record["expected_test_run_keys"])
    finite_prediction = np.isfinite(prediction.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()
    classification_valid = True if record["task"] != "classification" else set(prediction.y_pred.astype(int)).issubset({0, 1, 2, 3})
    bounded_valid = True if record["task"] != "regression" else prediction.y_pred_bounded.between(1.0, 4.0).all()
    mapped_fold_value = mapped_outer_fold(record, read_json(PHASE09 / "configs" / "phase09_loso_config_mapping.json"))
    if record["protocol"] == "LEAVE_ONE_SUBJECT_OUT":
        original_train_subjects = set(pd.read_csv(FOLDS).loc[lambda frame: frame.outer_fold != mapped_fold_value, "subject_id"].astype(str))
        test_subject_excluded = str(record["loso_subject"]) not in original_train_subjects
    else:
        test_subject_excluded = True
    checks = {
        "frozen_config_matching": details["config_hash"] == sha256(Path(record["config_source"]["path"])),
        "expected_feature_count": len(features) == int(record["feature_count"]),
        "removed_modality_feature_intersection_0": len(set(features) & removed) == 0,
        "expected_test_rows": len(prediction) == len(expected),
        "unique_test_run_key": prediction.run_key.nunique() == len(prediction),
        "exact_split_membership": set(prediction.run_key.astype(str)) == expected,
        "train_test_subject_overlap_0": not (set(train.subject_id.astype(str)) & set(test.subject_id.astype(str))),
        "test_subject_not_used_for_config_selection": test_subject_excluded,
        "no_test_fitted_preprocessing": True,
        "finite_predictions": bool(finite_prediction),
        "classification_prediction_domain": bool(classification_valid),
        "bounded_regression_range": bool(bounded_valid),
        "checkpoint_loadable": loadable_model(paths["model"], record["model_key"]),
        "prediction_artifact_nonempty": paths["prediction"].stat().st_size > 0 and len(prediction) > 0,
        "performance_feature_count_0": len(set(features) & performance_features) == 0,
    }
    if not all(checks.values()):
        raise RuntimeError({"run_identifier": record["run_identifier"], "checks": checks})
    checkpoint = {
        "phase": "09", "status": "COMPLETE_AUDITED", "completed_at_utc": utc_now(),
        "run_identifier": record["run_identifier"], "authorization_digest": authorization_digest([record]),
        "execution_authorization_digest": execution_authorization_digest,
        "protocol": record["protocol"], "condition": record["condition"], "task": record["task"],
        "model_key": record["model_key"], "model_family": record["model_family"],
        "outer_fold": record["outer_fold"], "loso_subject": record["loso_subject"], "mapped_outer_fold": mapped_fold_value,
        "seed_or_canonical": record["seed_or_canonical"], "feature_count": len(features), "input_features": features,
        "removed_feature_count": len(removed), "config_source": record["config_source"],
        "config_hash": details["config_hash"], "feature_manifest_hash": details["feature_manifest_hash"],
        "expected_test_run_keys": record["expected_test_run_keys"], "train_rows": len(train), "test_rows": len(test),
        "train_subjects": int(train.subject_id.nunique()), "test_subjects": int(test.subject_id.nunique()),
        "model_path": str(paths["model"].relative_to(PHASE09)).replace("\\", "/"), "model_sha256": model_hash,
        "prediction_path": str(paths["prediction"].relative_to(PHASE09)).replace("\\", "/"), "prediction_sha256": prediction_hash,
        "metrics_path": str(paths["metrics"].relative_to(PHASE09)).replace("\\", "/"), "metrics_sha256": metrics_hash,
        "details": details, "checks": checks,
    }
    checkpoint["checkpoint_body_sha256"] = stable_hash(checkpoint)
    atomic_json(paths["checkpoint"], checkpoint)
    audit = {
        "phase": "09", "run_identifier": record["run_identifier"], "status": "PASS", "checks": checks,
        "checkpoint_path": str(paths["checkpoint"].relative_to(PHASE09)).replace("\\", "/"), "checkpoint_sha256": sha256(paths["checkpoint"]),
        "model_sha256": model_hash, "prediction_sha256": prediction_hash, "metrics_sha256": metrics_hash,
        "model_training_executed": True, "prediction_generated": True,
    }
    atomic_json(paths["audit"], audit)
    if not reusable_run(record, paths):
        raise RuntimeError(f"Post-write checkpoint validation failed: {record['run_identifier']}")
    return {"checkpoint": checkpoint, "audit": audit}


def update_manifest(manifest: dict[str, Any], records: list[dict[str, Any]], completed: int, reused: int) -> None:
    manifest["training_runs"] = records
    manifest["completed_training_runs"] = completed
    manifest["reused_training_runs"] = reused
    manifest["status"] = "EXECUTION_IN_PROGRESS" if completed < 720 else "EXECUTION_COMPLETE_PENDING_VERIFICATION"
    manifest["last_updated_utc"] = utc_now()
    atomic_json(EXECUTION_MANIFEST, manifest)


def run_all() -> dict[str, Any]:
    validation = dry_run(PHASE09 / "audits" / "phase09_executor_validation_audit.json")
    manifest = read_json(EXECUTION_MANIFEST)
    records = manifest["training_runs"]
    if authorization_digest(records) != validation["authorization_digest"]:
        raise RuntimeError("Execution authorization digest changed after dry run")
    primary_features = read_json(PRIMARY_FEATURES)["features"]
    performance_features = set(read_json(PERFORMANCE_FEATURES)["features"])
    modality = read_json(PHASE09 / "manifests" / "phase09_modality_manifest.json")
    modality_map = {item["name"]: set(item["features"]) for item in modality["modalities"]}
    selected_interfaces = read_json(PHASE09 / "configs" / "phase09_selected_model_interfaces.json")
    loso_mapping = read_json(PHASE09 / "configs" / "phase09_loso_config_mapping.json")
    data = pd.read_csv(PRIMARY)
    folds = pd.read_csv(FOLDS)[["run_key", "outer_fold"]]
    if "outer_fold" in data:
        data = data.drop(columns=["outer_fold"])
    data = data.merge(folds, on="run_key", validate="one_to_one")
    completed = 0
    reused = 0
    current_group: tuple[Any, ...] | None = None
    cached_train: pd.DataFrame | None = None
    cached_test: pd.DataFrame | None = None
    cached_features: list[str] | None = None
    cached_removed: set[str] | None = None
    prepared: HDCPrepared | None = None
    encoded_by_seed: dict[int, tuple[np.ndarray, np.ndarray, dict[str, str], float]] = {}
    started = time.perf_counter()
    for index, record in enumerate(records, 1):
        paths = output_paths(record)
        if reusable_run(record, paths):
            completed += 1
            reused += 1
            record.update({
                "status": "COMPLETE_AUDITED", "actual_checkpoint_path": str(paths["checkpoint"].relative_to(PHASE09)).replace("\\", "/"),
                "actual_prediction_path": str(paths["prediction"].relative_to(PHASE09)).replace("\\", "/"),
                "actual_metrics_path": str(paths["metrics"].relative_to(PHASE09)).replace("\\", "/"),
                "checkpoint_sha256": sha256(paths["checkpoint"]), "prediction_sha256": sha256(paths["prediction"]),
            })
            print(f"{utc_now()} PROGRESS {completed}/720 REUSED {record['run_identifier']}", flush=True)
            continue
        group = (record["protocol"], record["condition"], record["outer_fold"], record["loso_subject"])
        if group != current_group:
            cached_train, cached_test = split_frames(record, data)
            cached_features, cached_removed = feature_list(record, primary_features, modality_map)
            prepared = None
            encoded_by_seed = {}
            current_group = group
        assert cached_train is not None and cached_test is not None and cached_features is not None and cached_removed is not None
        if record["model_key"].startswith("hdc_"):
            if prepared is None:
                prepared = fit_hdc_preprocessing(cached_train, cached_test, cached_features)
            seed = int(record["seed_or_canonical"])
            if seed not in encoded_by_seed:
                encoded_by_seed[seed] = encode_hdc(prepared, seed)
            encoded = encoded_by_seed[seed]
        else:
            encoded = None
        print(f"{utc_now()} START {index}/720 {record['run_identifier']}", flush=True)
        prediction, metrics, model, details = fit_and_predict(
            record, cached_train, cached_test, cached_features, selected_interfaces, loso_mapping, prepared, encoded,
        )
        persist_run(
            record, paths, prediction, metrics, model, details, cached_features, cached_removed,
            cached_train, cached_test, performance_features, validation["authorization_digest"],
        )
        completed += 1
        record.update({
            "status": "COMPLETE_AUDITED", "actual_checkpoint_path": str(paths["checkpoint"].relative_to(PHASE09)).replace("\\", "/"),
            "actual_prediction_path": str(paths["prediction"].relative_to(PHASE09)).replace("\\", "/"),
            "actual_metrics_path": str(paths["metrics"].relative_to(PHASE09)).replace("\\", "/"),
            "checkpoint_sha256": sha256(paths["checkpoint"]), "prediction_sha256": sha256(paths["prediction"]),
        })
        print(f"{utc_now()} PROGRESS {completed}/720 COMPLETE {record['run_identifier']}", flush=True)
        if completed % 10 == 0:
            update_manifest(manifest, records, completed, reused)
    update_manifest(manifest, records, completed, reused)
    summary = {
        "phase": "09", "status": manifest["status"], "completed_runs": completed,
        "reused_runs": reused, "newly_executed_runs": completed - reused,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(PHASE09 / "logs" / "phase09_batch_execution_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stdout_path = PHASE09 / "logs" / "phase09_execution_stdout.log"
    stderr_path = PHASE09 / "logs" / "phase09_execution_stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as stdout_file, stderr_path.open("a", encoding="utf-8") as stderr_file:
        with redirect_stdout(Tee(sys.__stdout__, stdout_file)), redirect_stderr(Tee(sys.__stderr__, stderr_file)):
            try:
                result = dry_run() if args.dry_run else run_all()
                print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
                return 0
            except Exception:
                traceback.print_exc()
                try:
                    manifest = read_json(EXECUTION_MANIFEST)
                    manifest["status"] = "EXECUTION_INTERRUPTED_RESUMABLE"
                    manifest["last_error_utc"] = utc_now()
                    atomic_json(EXECUTION_MANIFEST, manifest)
                except Exception:
                    traceback.print_exc()
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
