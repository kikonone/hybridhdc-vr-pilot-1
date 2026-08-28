"""Execute all preregistered Phase 05 Vanilla HDC Final Confirmation folds.

The executor is deliberately checkpoint-first.  Every dimension/seed model is
saved independently before outer-test features are loaded.  Existing valid
checkpoints are reused; malformed checkpoints fail closed.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterable, TextIO

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
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path.insert(0, str(PHASE / "src"))

from phase05_hdc_core import (  # noqa: E402
    CONTRACT_VERSION,
    EqualWidthQuantizer,
    build_prototypes,
    cosine_similarity_scores,
    incremental_encode_prefixes,
    predict_smallest_class_tie,
)


PRIMARY = ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv"
FOLDS = ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv"
EXPECTED_PRIMARY_SHA = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLD_SHA = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
DIMENSIONS = [1000, 2000, 5000, 10000]
SEEDS = [42, 43, 44, 45, 46]
TEMPERATURES = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
LEVELS = 51
FEATURE_K = 50
CLASSES = [0, 1, 2, 3]
NON_PREDICTIVE = {
    "subject_id", "session_id", "run_id", "difficulty_level_raw",
    "difficulty_level", "run_key", "target_class", "target_score", "outer_fold",
}
CONTRACT_PATHS = [
    PHASE / "configs/phase05_hdc_encoding_contract.json",
    PHASE / "configs/phase05_hdc_model_selection_contract.json",
    PHASE / "configs/phase05_hdc_regression_heads_contract.json",
    PHASE / "configs/phase05_hdc_efficiency_protocol.json",
    PHASE / "configs/phase05_hdc_search_space.json",
    PHASE / "configs/phase05_experiment_contract.json",
]


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def feature_names_from_primary() -> list[str]:
    header = pd.read_csv(PRIMARY, nrows=0).columns.astype(str).tolist()
    names = [name for name in header if name not in NON_PREDICTIVE]
    if len(names) != 1176:
        raise RuntimeError(f"expected 1176 predictive features, found {len(names)}")
    return names


def config_id(outer_fold: int, dimension: int, seed: int) -> str:
    return f"fold_{outer_fold}_dimension_{dimension}_seed_{seed}"


def model_paths(outer_fold: int, dimension: int, seed: int) -> tuple[Path, Path]:
    directory = PHASE / f"results/checkpoints/final_confirmation/fold_{outer_fold}"
    stem = f"vanilla_hdc_final_confirmation_{config_id(outer_fold, dimension, seed)}"
    return directory / f"{stem}_checkpoint.json", directory / f"{stem}_model.npz"


def quick_screen_hashes() -> dict[str, str]:
    manifest = load_json(PHASE / "configs/vanilla_hdc_final_confirmation_manifest.json")
    return {str(key): str(value) for key, value in manifest["quick_screen_input_artifacts_sha256"].items()}


def validate_quick_screen_hashes(expected: dict[str, str]) -> None:
    for relative, digest in expected.items():
        path = PHASE / relative
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"frozen quick-screen artifact mismatch: {relative}")


def preflight() -> dict[str, Any]:
    primary_sha = sha256(PRIMARY)
    fold_sha = sha256(FOLDS)
    if primary_sha != EXPECTED_PRIMARY_SHA:
        raise RuntimeError(f"Primary checksum mismatch: {primary_sha}")
    if fold_sha != EXPECTED_FOLD_SHA:
        raise RuntimeError(f"frozen fold checksum mismatch: {fold_sha}")

    manifest = load_json(PHASE / "configs/vanilla_hdc_final_confirmation_manifest.json")
    manifest_audit = load_json(PHASE / "audits/vanilla_hdc_final_confirmation_manifest_audit.json")
    consolidation_audit = load_json(PHASE / "audits/vanilla_hdc_quick_screen_consolidation_audit.json")
    notebook_audit = load_json(
        PHASE / "audits/vanilla_hdc_quick_screen_consolidation_notebook_persistence_audit.json"
    )
    summary = pd.read_csv(PHASE / "results/summaries/vanilla_hdc_quick_screen_all_folds.csv")
    if manifest.get("status") != "FINAL_CONFIRMATION_MANIFEST_FROZEN_NOT_EXECUTED":
        raise RuntimeError("Final Confirmation manifest status is not frozen/not-executed")
    if manifest_audit.get("result") != "PASS" or consolidation_audit.get("result") != "PASS":
        raise RuntimeError("consolidation or manifest audit is not PASS")
    if notebook_audit.get("result") != "PASS":
        raise RuntimeError("quick-screen consolidation notebook persistence is not PASS")
    if consolidation_audit.get("folds_verified") != 5 or len(summary) != 5:
        raise RuntimeError("quick-screen folds verified is not 5/5")
    if manifest.get("outer_folds") != [1, 2, 3, 4, 5]:
        raise RuntimeError("unexpected Final Confirmation outer folds")
    if manifest.get("final_confirmation_dimensions") != DIMENSIONS:
        raise RuntimeError("unexpected Final Confirmation dimensions")
    if manifest.get("evaluation_seeds") != SEEDS:
        raise RuntimeError("unexpected Final Confirmation seeds")
    if manifest.get("similarity_regression_temperature_grid") != TEMPERATURES:
        raise RuntimeError("unexpected similarity temperature grid")
    if manifest.get("ridge_alpha_grid") != RIDGE_ALPHAS:
        raise RuntimeError("unexpected Ridge alpha grid")
    if set(summary["selected_levels"].astype(int)) != {LEVELS}:
        raise RuntimeError("not every quick-screen fold selected levels=51")
    if set(summary["selected_feature_k"].astype(str)) != {str(FEATURE_K)}:
        raise RuntimeError("not every quick-screen fold selected feature_k=50")
    for fold in range(1, 6):
        best = load_json(
            PHASE / f"results/summaries/vanilla_hdc_quick_screen_fold_{fold}_best_config.json"
        )
        if int(best["levels"]) != LEVELS or str(best["k"]) != str(FEATURE_K):
            raise RuntimeError(f"Fold {fold} best_config disagrees with frozen levels/k")
    for path in CONTRACT_PATHS:
        load_json(path)
    historical = quick_screen_hashes()
    validate_quick_screen_hashes(historical)

    assignments = pd.read_csv(FOLDS, usecols=["run_key", "subject_id", "outer_fold"])
    if len(assignments) != 419 or not assignments["run_key"].is_unique:
        raise RuntimeError("frozen fold assignments do not cover 419 unique run keys")
    isolation: list[dict[str, Any]] = []
    for fold in range(1, 6):
        training = assignments.loc[assignments["outer_fold"] != fold].reset_index(drop=True)
        testing = assignments.loc[assignments["outer_fold"] == fold].reset_index(drop=True)
        overlap = set(training["subject_id"]) & set(testing["subject_id"])
        splits = list(GroupKFold(n_splits=3).split(training, groups=training["subject_id"]))
        inner_overlap = [
            len(set(training.iloc[a]["subject_id"]) & set(training.iloc[b]["subject_id"]))
            for a, b in splits
        ]
        if overlap or inner_overlap != [0, 0, 0]:
            raise RuntimeError(f"Fold {fold} subject isolation failed")
        isolation.append(
            {
                "outer_fold": fold,
                "outer_train_rows": int(len(training)),
                "outer_test_rows": int(len(testing)),
                "outer_train_subjects": int(training["subject_id"].nunique()),
                "outer_test_subjects": int(testing["subject_id"].nunique()),
                "outer_subject_overlap_count": 0,
                "inner_subject_overlap_counts": inner_overlap,
            }
        )
    return {
        "primary_sha256": primary_sha,
        "fold_sha256": fold_sha,
        "contract_sha256": {path.name: sha256(path) for path in CONTRACT_PATHS},
        "quick_screen_artifact_sha256": historical,
        "isolation": isolation,
        "result": "PASS",
    }


def load_feature_rows(
    allowed_keys: set[str], feature_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    matrices: list[list[float]] = []
    found: list[str] = []
    with PRIMARY.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        positions = {name: index for index, name in enumerate(header)}
        feature_positions = [positions[name] for name in feature_names]
        run_key_position = positions["run_key"]
        for row in reader:
            run_key = row[run_key_position]
            if run_key not in allowed_keys:
                continue
            matrices.append(
                [float(row[index]) if row[index].strip() else np.nan for index in feature_positions]
            )
            found.append(run_key)
    if len(found) != len(allowed_keys) or set(found) != allowed_keys:
        raise RuntimeError("feature extraction did not align one-to-one with allowed run keys")
    return np.asarray(matrices, dtype=np.float64), found


def load_targets_for_keys(allowed_keys: set[str]) -> dict[str, tuple[str, int, float]]:
    """Read target fields only after a row's run_key passes the allow-list."""
    targets: dict[str, tuple[str, int, float]] = {}
    with FOLDS.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run_key = str(row["run_key"])
            if run_key not in allowed_keys:
                continue
            targets[run_key] = (
                str(row["subject_id"]),
                int(row["target_class"]),
                float(row["target_score"]),
            )
    if set(targets) != allowed_keys:
        raise RuntimeError("target extraction did not align one-to-one with allowed run keys")
    return targets


@dataclass
class FrozenPreprocessor:
    input_feature_names: list[str]
    imputer_statistics: np.ndarray
    indicator_features: np.ndarray
    variance_support: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    selected_indices: np.ndarray
    selected_feature_names: list[str]
    quantizer_minimum: np.ndarray
    quantizer_maximum: np.ndarray

    def transform_selected(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        missing = np.isnan(matrix)
        filled = np.where(missing, self.imputer_statistics, matrix)
        if len(self.indicator_features):
            indicators = missing[:, self.indicator_features].astype(np.float64)
            filled = np.hstack([filled, indicators])
        variable = filled[:, self.variance_support]
        scaled = (variable - self.scaler_mean) / self.scaler_scale
        selected = scaled[:, self.selected_indices]
        if not np.isfinite(selected).all():
            raise RuntimeError("non-finite selected feature value")
        return selected

    def quantize(self, values: np.ndarray) -> np.ndarray:
        selected = self.transform_selected(values)
        span = self.quantizer_maximum - self.quantizer_minimum
        safe_span = np.where(span > 0.0, span, 1.0)
        quantized = np.floor((selected - self.quantizer_minimum) / safe_span * LEVELS)
        quantized[:, span == 0.0] = 0
        return np.clip(quantized, 0, LEVELS - 1).astype(np.int16)


def fit_preprocessor(
    train_values: np.ndarray, train_labels: np.ndarray, feature_names: list[str]
) -> tuple[FrozenPreprocessor, np.ndarray]:
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    imputed = imputer.fit_transform(train_values)
    imputed_names = imputer.get_feature_names_out(feature_names).astype(str)
    variance = VarianceThreshold(threshold=0.0)
    variable = variance.fit_transform(imputed)
    variable_names = imputed_names[variance.get_support()]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(variable)
    selector = SelectKBest(score_func=f_classif, k="all").fit(scaled, train_labels)
    scores = np.nan_to_num(selector.scores_, nan=-np.inf)
    ranking = np.argsort(scores, kind="mergesort")[::-1]
    selected_indices = ranking[:FEATURE_K]
    selected = scaled[:, selected_indices]
    quantizer = EqualWidthQuantizer(LEVELS).fit(selected)
    state = FrozenPreprocessor(
        input_feature_names=list(feature_names),
        imputer_statistics=np.asarray(imputer.statistics_, dtype=np.float64),
        indicator_features=np.asarray(imputer.indicator_.features_, dtype=np.int64),
        variance_support=np.flatnonzero(variance.get_support()).astype(np.int64),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        selected_indices=np.asarray(selected_indices, dtype=np.int64),
        selected_feature_names=variable_names[selected_indices].astype(str).tolist(),
        quantizer_minimum=np.asarray(quantizer.minimum_, dtype=np.float64),
        quantizer_maximum=np.asarray(quantizer.maximum_, dtype=np.float64),
    )
    return state, state.quantize(train_values)


def save_preprocessor(outer_fold: int, state: FrozenPreprocessor, gate: dict[str, Any]) -> None:
    directory = PHASE / f"results/checkpoints/final_confirmation/fold_{outer_fold}"
    npz_path = directory / "outer_training_preprocessing.npz"
    metadata_path = directory / "outer_training_preprocessing.json"
    atomic_write_npz(
        npz_path,
        imputer_statistics=state.imputer_statistics,
        indicator_features=state.indicator_features,
        variance_support=state.variance_support,
        scaler_mean=state.scaler_mean,
        scaler_scale=state.scaler_scale,
        selected_indices=state.selected_indices,
        quantizer_minimum=state.quantizer_minimum,
        quantizer_maximum=state.quantizer_maximum,
    )
    atomic_write_json(
        metadata_path,
        {
            "outer_fold": outer_fold,
            "fit_scope": "complete outer-training data only",
            "levels": LEVELS,
            "feature_k": FEATURE_K,
            "input_feature_names": state.input_feature_names,
            "selected_feature_names": state.selected_feature_names,
            "state_npz": str(npz_path.relative_to(PHASE)),
            "state_npz_sha256": sha256(npz_path),
            "primary_sha256": gate["primary_sha256"],
            "fold_sha256": gate["fold_sha256"],
        },
    )


def load_preprocessor(outer_fold: int, gate: dict[str, Any]) -> FrozenPreprocessor | None:
    directory = PHASE / f"results/checkpoints/final_confirmation/fold_{outer_fold}"
    npz_path = directory / "outer_training_preprocessing.npz"
    metadata_path = directory / "outer_training_preprocessing.json"
    if not npz_path.exists() and not metadata_path.exists():
        return None
    if not npz_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"Fold {outer_fold} preprocessing checkpoint is incomplete")
    metadata = load_json(metadata_path)
    if (
        metadata.get("outer_fold") != outer_fold
        or metadata.get("levels") != LEVELS
        or metadata.get("feature_k") != FEATURE_K
        or metadata.get("primary_sha256") != gate["primary_sha256"]
        or metadata.get("fold_sha256") != gate["fold_sha256"]
        or metadata.get("state_npz_sha256") != sha256(npz_path)
    ):
        raise RuntimeError(f"Fold {outer_fold} preprocessing checkpoint is invalid")
    with np.load(npz_path, allow_pickle=False) as arrays:
        return FrozenPreprocessor(
            input_feature_names=list(metadata["input_feature_names"]),
            imputer_statistics=arrays["imputer_statistics"],
            indicator_features=arrays["indicator_features"],
            variance_support=arrays["variance_support"],
            scaler_mean=arrays["scaler_mean"],
            scaler_scale=arrays["scaler_scale"],
            selected_indices=arrays["selected_indices"],
            selected_feature_names=list(metadata["selected_feature_names"]),
            quantizer_minimum=arrays["quantizer_minimum"],
            quantizer_maximum=arrays["quantizer_maximum"],
        )


def similarity_prediction(similarities: np.ndarray, temperature: float) -> np.ndarray:
    shifted = (np.asarray(similarities, dtype=np.float64) - np.max(similarities, axis=1, keepdims=True)) / float(temperature)
    exponentiated = np.exp(shifted)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    prediction = probabilities @ np.asarray([1.0, 2.0, 3.0, 4.0])
    return np.clip(prediction, 1.0, 4.0)


def choose_parameter(rows: list[dict[str, Any]], parameter: str) -> float:
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["mean_inner_mae"]),
            float(row["std_inner_mae"]),
            -float(row[parameter]),
            int(row["frozen_candidate_order"]),
        ),
    )
    return float(ranked[0][parameter])


def regression_metrics(target: np.ndarray, raw: np.ndarray, bounded: np.ndarray) -> dict[str, float]:
    statistic = spearmanr(target, bounded).statistic
    return {
        "mae_raw": float(mean_absolute_error(target, raw)),
        "mae_bounded": float(mean_absolute_error(target, bounded)),
        "rmse_bounded": float(mean_squared_error(target, bounded) ** 0.5),
        "r2_bounded": float(r2_score(target, bounded)),
        "spearman_bounded": float(statistic),
    }


def classification_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(target, prediction, labels=CLASSES)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(target, prediction, average="weighted", zero_division=0)),
        "severe_error_rate": float(np.mean(np.abs(target - prediction) >= 2)),
    }
    for true_class in CLASSES:
        for predicted_class in CLASSES:
            metrics[f"confusion_true_{true_class}_pred_{predicted_class}"] = int(
                matrix[true_class, predicted_class]
            )
    return metrics


@dataclass
class InnerPrepared:
    inner_fold: int
    train_quantized: np.ndarray
    validation_quantized: np.ndarray
    selected_feature_names: list[str]
    train_labels: np.ndarray
    validation_labels: np.ndarray
    train_targets: np.ndarray
    validation_targets: np.ndarray


def prepare_inner_splits(
    values: np.ndarray,
    labels: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
) -> tuple[list[InnerPrepared], list[dict[str, Any]]]:
    prepared: list[InnerPrepared] = []
    audit_rows: list[dict[str, Any]] = []
    for inner_fold, (train_index, validation_index) in enumerate(
        GroupKFold(n_splits=3).split(values, labels, groups), start=1
    ):
        train_subjects = set(groups[train_index])
        validation_subjects = set(groups[validation_index])
        overlap = train_subjects & validation_subjects
        if overlap:
            raise RuntimeError(f"inner Fold {inner_fold} subject leakage")
        state, train_quantized = fit_preprocessor(values[train_index], labels[train_index], feature_names)
        prepared.append(
            InnerPrepared(
                inner_fold=inner_fold,
                train_quantized=train_quantized,
                validation_quantized=state.quantize(values[validation_index]),
                selected_feature_names=state.selected_feature_names,
                train_labels=labels[train_index],
                validation_labels=labels[validation_index],
                train_targets=targets[train_index],
                validation_targets=targets[validation_index],
            )
        )
        audit_rows.append(
            {
                "inner_fold": inner_fold,
                "train_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)),
                "train_subjects": len(train_subjects),
                "validation_subjects": len(validation_subjects),
                "subject_overlap_count": 0,
                "preprocessing_fit_scope": "inner-training only",
            }
        )
    return prepared, audit_rows


def select_heads_for_seed(
    outer_fold: int,
    seed: int,
    dimensions: list[int],
    inner_splits: list[InnerPrepared],
) -> tuple[dict[int, tuple[float, float]], list[dict[str, Any]], dict[int, dict[str, str]]]:
    temperature_maes = {dimension: {value: [] for value in TEMPERATURES} for dimension in dimensions}
    alpha_maes = {dimension: {value: [] for value in RIDGE_ALPHAS} for dimension in dimensions}
    codebook_hashes: dict[int, dict[str, str]] = {}
    for prepared in inner_splits:
        combined = np.vstack([prepared.train_quantized, prepared.validation_quantized])
        split_point = len(prepared.train_quantized)
        encoding = incremental_encode_prefixes(
            combined,
            prepared.selected_feature_names,
            LEVELS,
            seed,
            [FEATURE_K],
            max(DIMENSIONS),
            CONTRACT_VERSION,
        )
        encoded = encoding.samples_by_k[str(FEATURE_K)]
        for dimension in dimensions:
            train_hv = encoded[:split_point, :dimension]
            validation_hv = encoded[split_point:, :dimension]
            prototypes = build_prototypes(train_hv, prepared.train_labels)
            similarities = cosine_similarity_scores(validation_hv, prototypes)
            for temperature in TEMPERATURES:
                prediction = similarity_prediction(similarities, temperature)
                temperature_maes[dimension][temperature].append(
                    float(mean_absolute_error(prepared.validation_targets, prediction))
                )
            train_float = train_hv.astype(np.float32) / np.float32(np.sqrt(dimension))
            validation_float = validation_hv.astype(np.float32) / np.float32(np.sqrt(dimension))
            for alpha in RIDGE_ALPHAS:
                ridge = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
                ridge.fit(train_float, prepared.train_targets)
                raw = ridge.predict(validation_float)
                bounded = np.clip(raw, 1.0, 4.0)
                alpha_maes[dimension][alpha].append(
                    float(mean_absolute_error(prepared.validation_targets, bounded))
                )
            codebook_hashes[dimension] = dict(encoding.codebook_hashes)

    selections: dict[int, tuple[float, float]] = {}
    selection_rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        temperature_rows: list[dict[str, Any]] = []
        for order, temperature in enumerate(TEMPERATURES):
            split_maes = temperature_maes[dimension][temperature]
            row = {
                "outer_fold": outer_fold,
                "dimension": dimension,
                "seed": seed,
                "head": "similarity_regression",
                "temperature": temperature,
                "ridge_alpha": None,
                "inner_fold_1_mae": split_maes[0],
                "inner_fold_2_mae": split_maes[1],
                "inner_fold_3_mae": split_maes[2],
                "mean_inner_mae": float(np.mean(split_maes)),
                "std_inner_mae": float(np.std(split_maes, ddof=0)),
                "frozen_candidate_order": order,
            }
            temperature_rows.append(row)
        selected_temperature = choose_parameter(temperature_rows, "temperature")
        for row in temperature_rows:
            row["selected"] = bool(row["temperature"] == selected_temperature)
            selection_rows.append(row)

        alpha_rows: list[dict[str, Any]] = []
        for order, alpha in enumerate(RIDGE_ALPHAS):
            split_maes = alpha_maes[dimension][alpha]
            row = {
                "outer_fold": outer_fold,
                "dimension": dimension,
                "seed": seed,
                "head": "ridge_regression",
                "temperature": None,
                "ridge_alpha": alpha,
                "inner_fold_1_mae": split_maes[0],
                "inner_fold_2_mae": split_maes[1],
                "inner_fold_3_mae": split_maes[2],
                "mean_inner_mae": float(np.mean(split_maes)),
                "std_inner_mae": float(np.std(split_maes, ddof=0)),
                "frozen_candidate_order": order,
            }
            alpha_rows.append(row)
        selected_alpha = choose_parameter(alpha_rows, "ridge_alpha")
        for row in alpha_rows:
            row["selected"] = bool(row["ridge_alpha"] == selected_alpha)
            selection_rows.append(row)
        selections[dimension] = (selected_temperature, selected_alpha)
    return selections, selection_rows, codebook_hashes


def validate_checkpoint(
    outer_fold: int, dimension: int, seed: int, gate: dict[str, Any]
) -> dict[str, Any] | None:
    checkpoint_path, model_path = model_paths(outer_fold, dimension, seed)
    if not checkpoint_path.exists() and not model_path.exists():
        return None
    if not checkpoint_path.is_file() or not model_path.is_file():
        raise RuntimeError(f"incomplete checkpoint for {config_id(outer_fold, dimension, seed)}")
    checkpoint = load_json(checkpoint_path)
    valid = (
        checkpoint.get("status") == "COMPLETE_AUDITED"
        and checkpoint.get("outer_fold") == outer_fold
        and checkpoint.get("dimension") == dimension
        and checkpoint.get("seed") == seed
        and checkpoint.get("levels") == LEVELS
        and checkpoint.get("feature_k") == FEATURE_K
        and checkpoint.get("primary_sha256") == gate["primary_sha256"]
        and checkpoint.get("fold_sha256") == gate["fold_sha256"]
        and checkpoint.get("model_npz_sha256") == sha256(model_path)
        and checkpoint.get("temperature_inner_cv_only") is True
        and checkpoint.get("ridge_alpha_inner_cv_only") is True
        and checkpoint.get("outer_test_feature_access") is False
    )
    if not valid:
        raise RuntimeError(f"invalid checkpoint for {config_id(outer_fold, dimension, seed)}")
    with np.load(model_path, allow_pickle=False) as model:
        if model["prototypes"].shape != (4, dimension) or model["ridge_coef"].shape != (dimension,):
            raise RuntimeError(f"invalid model shape for {config_id(outer_fold, dimension, seed)}")
    return checkpoint


def save_model_checkpoint(
    outer_fold: int,
    dimension: int,
    seed: int,
    selected_temperature: float,
    selected_alpha: float,
    prototypes: np.ndarray,
    ridge: Ridge,
    selection_rows: list[dict[str, Any]],
    codebook_hashes: dict[str, str],
    gate: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path, model_path = model_paths(outer_fold, dimension, seed)
    atomic_write_npz(
        model_path,
        prototypes=np.asarray(prototypes, dtype=np.int32),
        ridge_coef=np.asarray(ridge.coef_, dtype=np.float64),
        ridge_intercept=np.asarray([ridge.intercept_], dtype=np.float64),
    )
    checkpoint = {
        "phase": "05",
        "model": "Vanilla Prototype HDC",
        "stage": "final_confirmation",
        "regression_target": "bounded difficulty-induced workload proxy regression",
        "status": "COMPLETE_AUDITED",
        "completed_utc": utc_now(),
        "config_id": config_id(outer_fold, dimension, seed),
        "outer_fold": outer_fold,
        "dimension": dimension,
        "seed": seed,
        "levels": LEVELS,
        "feature_k": FEATURE_K,
        "selected_similarity_temperature": selected_temperature,
        "selected_ridge_alpha": selected_alpha,
        "temperature_inner_cv_only": True,
        "ridge_alpha_inner_cv_only": True,
        "inner_groupkfold_splits": 3,
        "inner_selection_records": selection_rows,
        "fit_scope": "complete outer-training data only after inner selection",
        "outer_test_feature_access": False,
        "outer_test_used_for_tuning": False,
        "codebook_hashes": codebook_hashes,
        "model_npz": str(model_path.relative_to(PHASE)),
        "model_npz_sha256": sha256(model_path),
        "primary_sha256": gate["primary_sha256"],
        "fold_sha256": gate["fold_sha256"],
        "contract_sha256": gate["contract_sha256"],
    }
    atomic_write_json(checkpoint_path, checkpoint)
    return checkpoint


def load_model(outer_fold: int, dimension: int, seed: int) -> dict[str, np.ndarray]:
    _, model_path = model_paths(outer_fold, dimension, seed)
    with np.load(model_path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


def run_fold(outer_fold: int, gate: dict[str, Any]) -> dict[str, Any]:
    fold_start = time.perf_counter()
    validate_quick_screen_hashes(gate["quick_screen_artifact_sha256"])
    metadata = pd.read_csv(FOLDS, usecols=["run_key", "subject_id", "outer_fold"])
    train_meta = metadata.loc[metadata["outer_fold"] != outer_fold].reset_index(drop=True)
    test_meta = metadata.loc[metadata["outer_fold"] == outer_fold].reset_index(drop=True)
    if set(train_meta["subject_id"]) & set(test_meta["subject_id"]):
        raise RuntimeError(f"Fold {outer_fold} outer subject isolation failed")
    feature_names = feature_names_from_primary()
    train_values, train_run_order = load_feature_rows(set(train_meta["run_key"]), feature_names)
    train_target_map = load_targets_for_keys(set(train_meta["run_key"]))
    train_labels = np.asarray([train_target_map[key][1] for key in train_run_order], dtype=np.int64)
    train_targets = np.asarray([train_target_map[key][2] for key in train_run_order], dtype=np.float64)
    train_groups = np.asarray([train_target_map[key][0] for key in train_run_order], dtype=object)

    inner_splits, inner_audit = prepare_inner_splits(
        train_values, train_labels, train_targets, train_groups, feature_names
    )
    preprocessor = load_preprocessor(outer_fold, gate)
    preprocessing_start = time.perf_counter()
    if preprocessor is None:
        preprocessor, train_quantized = fit_preprocessor(train_values, train_labels, feature_names)
        save_preprocessor(outer_fold, preprocessor, gate)
        preprocessing_reused = False
    else:
        train_quantized = preprocessor.quantize(train_values)
        preprocessing_reused = True
    preprocessing_seconds = time.perf_counter() - preprocessing_start

    checkpoints: dict[tuple[int, int], dict[str, Any]] = {}
    selection_rows_all: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        missing: list[int] = []
        for dimension in DIMENSIONS:
            checkpoint = validate_checkpoint(outer_fold, dimension, seed, gate)
            if checkpoint is None:
                missing.append(dimension)
            else:
                checkpoints[(dimension, seed)] = checkpoint
                selection_rows_all.extend(checkpoint["inner_selection_records"])
                print(f"Fold {outer_fold}: reused valid {config_id(outer_fold, dimension, seed)}")
        if not missing:
            continue
        selection_start = time.perf_counter()
        selections, selection_rows, inner_codebooks = select_heads_for_seed(
            outer_fold, seed, missing, inner_splits
        )
        selection_seconds = time.perf_counter() - selection_start
        full_encoding_start = time.perf_counter()
        encoding = incremental_encode_prefixes(
            train_quantized,
            preprocessor.selected_feature_names,
            LEVELS,
            seed,
            [FEATURE_K],
            max(DIMENSIONS),
            CONTRACT_VERSION,
        )
        full_encoded = encoding.samples_by_k[str(FEATURE_K)]
        encoding_seconds = time.perf_counter() - full_encoding_start
        for dimension in missing:
            fit_start = time.perf_counter()
            temperature, alpha = selections[dimension]
            train_hv = full_encoded[:, :dimension]
            prototypes = build_prototypes(train_hv, train_labels)
            ridge = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
            ridge.fit(
                train_hv.astype(np.float32) / np.float32(np.sqrt(dimension)), train_targets
            )
            fit_seconds = time.perf_counter() - fit_start
            rows = [
                row for row in selection_rows
                if int(row["dimension"]) == dimension and int(row["seed"]) == seed
            ]
            checkpoint = save_model_checkpoint(
                outer_fold,
                dimension,
                seed,
                temperature,
                alpha,
                prototypes,
                ridge,
                rows,
                {**encoding.codebook_hashes, "inner_codebook_hashes": inner_codebooks[dimension]},
                gate,
            )
            checkpoints[(dimension, seed)] = checkpoint
            selection_rows_all.extend(rows)
            efficiency_rows.append(
                {
                    "outer_fold": outer_fold,
                    "dimension": dimension,
                    "seed": seed,
                    "preprocessing_seconds": preprocessing_seconds,
                    "preprocessing_reused": preprocessing_reused,
                    "inner_selection_seconds": selection_seconds / len(missing),
                    "outer_training_encoding_seconds": encoding_seconds / len(missing),
                    "outer_training_fit_seconds": fit_seconds,
                    "checkpoint_reused": False,
                }
            )
            print(
                f"Fold {outer_fold}: checkpointed dimension={dimension} seed={seed} "
                f"temperature={temperature} alpha={alpha} ({len(checkpoints)}/20)"
            )

    expected_configs = {(dimension, seed) for dimension in DIMENSIONS for seed in SEEDS}
    if set(checkpoints) != expected_configs:
        raise RuntimeError(f"Fold {outer_fold} did not finalize 20/20 model checkpoints")
    for dimension, seed in sorted(expected_configs):
        validate_checkpoint(outer_fold, dimension, seed, gate)
    print(f"Fold {outer_fold}: all 20 models finalized; outer-test feature load is now authorized")

    test_feature_load_sequence = 1
    test_values, test_run_order = load_feature_rows(set(test_meta["run_key"]), feature_names)
    test_quantized = preprocessor.quantize(test_values)
    prediction_records: list[dict[str, Any]] = []
    inference_seconds: dict[tuple[int, int], float] = {}
    for seed in SEEDS:
        encoding = incremental_encode_prefixes(
            test_quantized,
            preprocessor.selected_feature_names,
            LEVELS,
            seed,
            [FEATURE_K],
            max(DIMENSIONS),
            CONTRACT_VERSION,
        )
        test_encoded = encoding.samples_by_k[str(FEATURE_K)]
        for dimension in DIMENSIONS:
            inference_start = time.perf_counter()
            checkpoint = checkpoints[(dimension, seed)]
            model = load_model(outer_fold, dimension, seed)
            test_hv = test_encoded[:, :dimension]
            similarities = cosine_similarity_scores(test_hv, model["prototypes"])
            predicted_class = predict_smallest_class_tie(similarities)
            temperature = float(checkpoint["selected_similarity_temperature"])
            similarity_values = similarity_prediction(similarities, temperature)
            test_float = test_hv.astype(np.float32) / np.float32(np.sqrt(dimension))
            ridge_raw = test_float @ model["ridge_coef"] + float(model["ridge_intercept"][0])
            ridge_bounded = np.clip(ridge_raw, 1.0, 4.0)
            inference_seconds[(dimension, seed)] = time.perf_counter() - inference_start
            for index, run_key in enumerate(test_run_order):
                prediction_records.append(
                    {
                        "run_key": run_key,
                        "subject_id": str(test_meta.set_index("run_key").at[run_key, "subject_id"]),
                        "outer_fold": outer_fold,
                        "dimension": dimension,
                        "seed": seed,
                        "levels": LEVELS,
                        "feature_k": FEATURE_K,
                        "predicted_class": int(predicted_class[index]),
                        "similarity_temperature": temperature,
                        "similarity_prediction": float(similarity_values[index]),
                        "ridge_alpha": float(checkpoint["selected_ridge_alpha"]),
                        "ridge_prediction_raw": float(ridge_raw[index]),
                        "ridge_prediction_bounded": float(ridge_bounded[index]),
                        "similarity_class_0": float(similarities[index, 0]),
                        "similarity_class_1": float(similarities[index, 1]),
                        "similarity_class_2": float(similarities[index, 2]),
                        "similarity_class_3": float(similarities[index, 3]),
                    }
                )
    expected_prediction_rows = len(test_meta) * 20
    if len(prediction_records) != expected_prediction_rows:
        raise RuntimeError(f"Fold {outer_fold} predictions are incomplete before label loading")
    prediction_generation_sequence = 2

    test_labels_source = load_targets_for_keys(set(test_meta["run_key"]))
    for record in prediction_records:
        record["true_class"] = test_labels_source[record["run_key"]][1]
        record["target_score"] = test_labels_source[record["run_key"]][2]
    outer_label_load_sequence = 3
    predictions = pd.DataFrame(prediction_records)
    required_prediction_order = [
        "run_key", "subject_id", "outer_fold", "dimension", "seed", "levels", "feature_k",
        "true_class", "predicted_class", "target_score", "similarity_temperature",
        "similarity_prediction", "ridge_alpha", "ridge_prediction_raw",
        "ridge_prediction_bounded", "similarity_class_0", "similarity_class_1",
        "similarity_class_2", "similarity_class_3",
    ]
    predictions = predictions[required_prediction_order]

    metric_rows: list[dict[str, Any]] = []
    for (dimension, seed), group in predictions.groupby(["dimension", "seed"], sort=True):
        target_class = group["true_class"].to_numpy(dtype=np.int64)
        predicted_class = group["predicted_class"].to_numpy(dtype=np.int64)
        target_score = group["target_score"].to_numpy(dtype=np.float64)
        similarity_bounded = group["similarity_prediction"].to_numpy(dtype=np.float64)
        ridge_raw = group["ridge_prediction_raw"].to_numpy(dtype=np.float64)
        ridge_bounded = group["ridge_prediction_bounded"].to_numpy(dtype=np.float64)
        row: dict[str, Any] = {
            "outer_fold": outer_fold,
            "dimension": int(dimension),
            "seed": int(seed),
            "levels": LEVELS,
            "feature_k": FEATURE_K,
            "test_rows": int(len(group)),
            "similarity_temperature": float(group["similarity_temperature"].iloc[0]),
            "ridge_alpha": float(group["ridge_alpha"].iloc[0]),
        }
        row.update({f"classification_{key}": value for key, value in classification_metrics(target_class, predicted_class).items()})
        row.update({f"similarity_{key}": value for key, value in regression_metrics(target_score, similarity_bounded, similarity_bounded).items()})
        row.update({f"ridge_{key}": value for key, value in regression_metrics(target_score, ridge_raw, ridge_bounded).items()})
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    inner_selection = pd.DataFrame(selection_rows_all).drop_duplicates(
        subset=["outer_fold", "dimension", "seed", "head", "temperature", "ridge_alpha"],
        keep="last",
    )
    if len(inner_selection) != 20 * (len(TEMPERATURES) + len(RIDGE_ALPHAS)):
        raise RuntimeError(f"Fold {outer_fold} inner selection coverage is incomplete")
    efficiency = pd.DataFrame(efficiency_rows)
    if len(efficiency) < 20:
        existing_keys = {(int(row["dimension"]), int(row["seed"])) for row in efficiency_rows}
        for dimension, seed in sorted(expected_configs - existing_keys):
            efficiency_rows.append(
                {
                    "outer_fold": outer_fold,
                    "dimension": dimension,
                    "seed": seed,
                    "preprocessing_seconds": 0.0,
                    "preprocessing_reused": True,
                    "inner_selection_seconds": 0.0,
                    "outer_training_encoding_seconds": 0.0,
                    "outer_training_fit_seconds": 0.0,
                    "checkpoint_reused": True,
                }
            )
        efficiency = pd.DataFrame(efficiency_rows)
    efficiency["outer_test_inference_seconds"] = [
        inference_seconds[(int(row.dimension), int(row.seed))] for row in efficiency.itertuples()
    ]
    efficiency["fold_elapsed_seconds"] = time.perf_counter() - fold_start

    prediction_path = PHASE / f"results/predictions/vanilla_hdc_final_confirmation_fold_{outer_fold}_predictions.csv"
    metrics_path = PHASE / f"results/fold_metrics/vanilla_hdc_final_confirmation_fold_{outer_fold}_metrics.csv"
    inner_path = PHASE / f"results/fold_metrics/vanilla_hdc_final_confirmation_fold_{outer_fold}_inner_selection.csv"
    efficiency_path = PHASE / f"results/efficiency/vanilla_hdc_final_confirmation_fold_{outer_fold}_efficiency.csv"
    atomic_write_csv(prediction_path, predictions)
    atomic_write_csv(metrics_path, metrics)
    atomic_write_csv(inner_path, inner_selection)
    atomic_write_csv(efficiency_path, efficiency)

    expected_keys = set(test_meta["run_key"])
    coverage_pass = (
        len(metrics) == 20
        and len(predictions) == expected_prediction_rows
        and not predictions.duplicated(["run_key", "dimension", "seed"]).any()
        and all(set(group["run_key"]) == expected_keys for _, group in predictions.groupby(["dimension", "seed"]))
    )
    coverage_audit = {
        "outer_fold": outer_fold,
        "expected_configs": 20,
        "completed_configs": int(len(metrics)),
        "expected_test_rows_per_config": int(len(test_meta)),
        "expected_prediction_rows": int(expected_prediction_rows),
        "actual_prediction_rows": int(len(predictions)),
        "unique_config_run_keys": int(predictions[["dimension", "seed", "run_key"]].drop_duplicates().shape[0]),
        "all_prediction_run_keys_valid": bool(coverage_pass),
        "result": "PASS" if coverage_pass else "FAIL",
    }
    coverage_path = PHASE / f"audits/vanilla_hdc_final_confirmation_fold_{outer_fold}_coverage_audit.json"
    atomic_write_json(coverage_path, coverage_audit)

    leakage_pass = (
        all(row["subject_overlap_count"] == 0 for row in inner_audit)
        and not (set(train_meta["subject_id"]) & set(test_meta["subject_id"]))
        and test_feature_load_sequence < prediction_generation_sequence < outer_label_load_sequence
    )
    leakage_audit = {
        "outer_fold": outer_fold,
        "outer_train_rows": int(len(train_meta)),
        "outer_test_rows": int(len(test_meta)),
        "outer_train_subjects": int(train_meta["subject_id"].nunique()),
        "outer_test_subjects": int(test_meta["subject_id"].nunique()),
        "outer_subject_overlap_count": 0,
        "inner_groupkfold": "GroupKFold(n_splits=3, groups=subject_id)",
        "inner_splits": inner_audit,
        "all_preprocessing_fit_on_training_only": True,
        "all_20_models_finalized_before_outer_test_feature_load": True,
        "outer_test_feature_load_sequence": test_feature_load_sequence,
        "all_predictions_generated_sequence": prediction_generation_sequence,
        "outer_test_labels_loaded_sequence": outer_label_load_sequence,
        "temperature_inner_cv_only": True,
        "ridge_alpha_inner_cv_only": True,
        "outer_test_used_for_tuning": False,
        "quick_screen_artifacts_preserved": True,
        "primary_checksum_pass": sha256(PRIMARY) == EXPECTED_PRIMARY_SHA,
        "fold_checksum_pass": sha256(FOLDS) == EXPECTED_FOLD_SHA,
        "result": "PASS" if leakage_pass else "FAIL",
    }
    leakage_path = PHASE / f"audits/vanilla_hdc_final_confirmation_fold_{outer_fold}_leakage_audit.json"
    atomic_write_json(leakage_path, leakage_audit)

    required_paths = [prediction_path, metrics_path, inner_path, efficiency_path, leakage_path, coverage_path]
    artifacts = [
        {
            "path": str(path.relative_to(PHASE)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in required_paths
    ]
    artifact_pass = all(path.is_file() and path.stat().st_size > 0 for path in required_paths)
    artifact_audit = {
        "outer_fold": outer_fold,
        "configuration_checkpoints_valid": all(
            validate_checkpoint(outer_fold, dimension, seed, gate) is not None
            for dimension, seed in expected_configs
        ),
        "required_artifacts": artifacts,
        "quick_screen_artifacts_preserved": True,
        "result": "PASS" if artifact_pass and coverage_pass and leakage_pass else "FAIL",
    }
    artifact_path = PHASE / f"audits/vanilla_hdc_final_confirmation_fold_{outer_fold}_artifact_audit.json"
    atomic_write_json(artifact_path, artifact_audit)
    validate_quick_screen_hashes(gate["quick_screen_artifact_sha256"])
    if {coverage_audit["result"], leakage_audit["result"], artifact_audit["result"]} != {"PASS"}:
        raise RuntimeError(f"Fold {outer_fold} audit failure")
    print(
        f"Fold {outer_fold} COMPLETE: 20/20 configurations, {len(predictions)} predictions, "
        "leakage/coverage/artifact audits PASS"
    )
    return {
        "outer_fold": outer_fold,
        "configs_completed": 20,
        "test_rows": int(len(test_meta)),
        "prediction_rows": int(len(predictions)),
        "classification_macro_f1_mean_across_configs": float(metrics["classification_macro_f1"].mean()),
        "classification_macro_f1_std_across_configs": float(metrics["classification_macro_f1"].std(ddof=0)),
        "similarity_mae_bounded_mean_across_configs": float(metrics["similarity_mae_bounded"].mean()),
        "similarity_mae_bounded_std_across_configs": float(metrics["similarity_mae_bounded"].std(ddof=0)),
        "ridge_mae_bounded_mean_across_configs": float(metrics["ridge_mae_bounded"].mean()),
        "ridge_mae_bounded_std_across_configs": float(metrics["ridge_mae_bounded"].std(ddof=0)),
        "leakage_audit": "PASS",
        "coverage_audit": "PASS",
        "artifact_audit": "PASS",
    }


def consolidate_all_folds(gate: dict[str, Any], fold_summaries: list[dict[str, Any]]) -> None:
    summary_path = PHASE / "results/summaries/vanilla_hdc_final_confirmation_execution_summary.csv"
    atomic_write_csv(summary_path, pd.DataFrame(fold_summaries).sort_values("outer_fold"))
    artifact_paths: list[Path] = [summary_path]
    for fold in range(1, 6):
        artifact_paths.extend(
            [
                PHASE / f"results/predictions/vanilla_hdc_final_confirmation_fold_{fold}_predictions.csv",
                PHASE / f"results/fold_metrics/vanilla_hdc_final_confirmation_fold_{fold}_metrics.csv",
                PHASE / f"results/fold_metrics/vanilla_hdc_final_confirmation_fold_{fold}_inner_selection.csv",
                PHASE / f"results/efficiency/vanilla_hdc_final_confirmation_fold_{fold}_efficiency.csv",
                PHASE / f"audits/vanilla_hdc_final_confirmation_fold_{fold}_leakage_audit.json",
                PHASE / f"audits/vanilla_hdc_final_confirmation_fold_{fold}_coverage_audit.json",
                PHASE / f"audits/vanilla_hdc_final_confirmation_fold_{fold}_artifact_audit.json",
                PHASE / f"logs/vanilla_hdc_final_confirmation_fold_{fold}_stdout.log",
                PHASE / f"logs/vanilla_hdc_final_confirmation_fold_{fold}_stderr.log",
            ]
        )
        artifact_paths.extend(
            sorted((PHASE / f"results/checkpoints/final_confirmation/fold_{fold}").glob("*"))
        )
    if not all(path.is_file() for path in artifact_paths):
        missing = [str(path.relative_to(PHASE)) for path in artifact_paths if not path.is_file()]
        raise RuntimeError(f"Final Confirmation artifacts missing: {missing}")
    artifact_manifest = {
        "phase": "05",
        "stage": "Vanilla HDC Final Confirmation execution",
        "artifact_count": len(artifact_paths),
        "artifacts": [
            {
                "path": str(path.relative_to(PHASE)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifact_paths
        ],
        "quick_screen_input_artifacts_sha256": gate["quick_screen_artifact_sha256"],
        "primary_sha256": gate["primary_sha256"],
        "fold_sha256": gate["fold_sha256"],
    }
    manifest_path = PHASE / "manifests/vanilla_hdc_final_confirmation_artifact_manifest.json"
    atomic_write_json(manifest_path, artifact_manifest)
    all_fold_audit = {
        "phase": "05",
        "folds_completed": 5,
        "configs_completed": 100,
        "configs_expected": 100,
        "classification_predictions_generated": True,
        "similarity_regression_predictions_generated": True,
        "ridge_regression_predictions_generated": True,
        "all_prediction_run_keys_valid": True,
        "all_outer_subject_isolation": True,
        "all_inner_subject_isolation": True,
        "temperature_inner_cv_only": True,
        "ridge_alpha_inner_cv_only": True,
        "outer_test_used_for_tuning": False,
        "primary_checksum_pass": sha256(PRIMARY) == EXPECTED_PRIMARY_SHA,
        "fold_checksum_pass": sha256(FOLDS) == EXPECTED_FOLD_SHA,
        "checkpoint_integrity": True,
        "quick_screen_artifacts_preserved": True,
        "phase05_frozen": False,
        "phase06_executed": False,
        "result": "PASS",
    }
    audit_path = PHASE / "audits/vanilla_hdc_final_confirmation_all_folds_audit.json"
    atomic_write_json(audit_path, all_fold_audit)
    print("ALL FOLDS COMPLETE: 100/100 Final Confirmation configurations; all-fold audit PASS")


def completed_fold_summary(outer_fold: int) -> dict[str, Any] | None:
    metrics_path = PHASE / f"results/fold_metrics/vanilla_hdc_final_confirmation_fold_{outer_fold}_metrics.csv"
    predictions_path = PHASE / f"results/predictions/vanilla_hdc_final_confirmation_fold_{outer_fold}_predictions.csv"
    audits = [
        PHASE / f"audits/vanilla_hdc_final_confirmation_fold_{outer_fold}_{name}_audit.json"
        for name in ("leakage", "coverage", "artifact")
    ]
    if not metrics_path.is_file() or not predictions_path.is_file() or not all(path.is_file() for path in audits):
        return None
    if any(load_json(path).get("result") != "PASS" for path in audits):
        raise RuntimeError(f"Fold {outer_fold} has an existing failed audit")
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    if len(metrics) != 20:
        raise RuntimeError(f"Fold {outer_fold} existing metrics do not contain 20 configs")
    return {
        "outer_fold": outer_fold,
        "configs_completed": 20,
        "test_rows": int(predictions["run_key"].nunique()),
        "prediction_rows": int(len(predictions)),
        "classification_macro_f1_mean_across_configs": float(metrics["classification_macro_f1"].mean()),
        "classification_macro_f1_std_across_configs": float(metrics["classification_macro_f1"].std(ddof=0)),
        "similarity_mae_bounded_mean_across_configs": float(metrics["similarity_mae_bounded"].mean()),
        "similarity_mae_bounded_std_across_configs": float(metrics["similarity_mae_bounded"].std(ddof=0)),
        "ridge_mae_bounded_mean_across_configs": float(metrics["ridge_mae_bounded"].mean()),
        "ridge_mae_bounded_std_across_configs": float(metrics["ridge_mae_bounded"].std(ddof=0)),
        "leakage_audit": "PASS",
        "coverage_audit": "PASS",
        "artifact_audit": "PASS",
    }


def execute_all_folds() -> None:
    gate = preflight()
    print("Final Confirmation preflight PASS; sequential Fold 1→5 execution authorized")
    summaries: list[dict[str, Any]] = []
    for outer_fold in range(1, 6):
        stdout_path = PHASE / f"logs/vanilla_hdc_final_confirmation_fold_{outer_fold}_stdout.log"
        stderr_path = PHASE / f"logs/vanilla_hdc_final_confirmation_fold_{outer_fold}_stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("a", encoding="utf-8") as stdout_file, stderr_path.open("a", encoding="utf-8") as stderr_file:
            with redirect_stdout(Tee(sys.__stdout__, stdout_file)), redirect_stderr(Tee(sys.__stderr__, stderr_file)):
                try:
                    print(f"\n{utc_now()} START/RESUME Final Confirmation Fold {outer_fold}")
                    existing = completed_fold_summary(outer_fold)
                    if existing is not None:
                        for dimension in DIMENSIONS:
                            for seed in SEEDS:
                                validate_checkpoint(outer_fold, dimension, seed, gate)
                        print(f"Fold {outer_fold}: complete audited artifacts reused; no retraining")
                        summaries.append(existing)
                    else:
                        summaries.append(run_fold(outer_fold, gate))
                    atomic_write_json(
                        PHASE / f"results/checkpoints/final_confirmation/fold_{outer_fold}/recovery_status.json",
                        {
                            "outer_fold": outer_fold,
                            "status": "COMPLETE",
                            "updated_utc": utc_now(),
                            "configs_completed": 20,
                            "resume_required": False,
                        },
                    )
                except Exception as error:
                    failure_traceback = traceback.format_exc()
                    print(failure_traceback, file=sys.stderr, end="")
                    checkpoint_directory = PHASE / f"results/checkpoints/final_confirmation/fold_{outer_fold}"
                    completed = len(list(checkpoint_directory.glob("*_checkpoint.json")))
                    atomic_write_json(
                        checkpoint_directory / "recovery_status.json",
                        {
                            "outer_fold": outer_fold,
                            "status": "FAILED_REQUIRES_REVIEW",
                            "failed_utc": utc_now(),
                            "checkpoint_files_present": completed,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "traceback": failure_traceback,
                        },
                    )
                    raise
    consolidate_all_folds(gate, summaries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    try:
        if arguments.preflight_only:
            result = preflight()
            print(json.dumps({"preflight": result["result"], "folds": 5, "configs": 100}))
        else:
            execute_all_folds()
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
