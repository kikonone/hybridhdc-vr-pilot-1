"""Complete Phase 05 diagnostics and inference efficiency without model fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    recall_score,
)


PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
SRC = PHASE / "src"
sys.path.insert(0, str(SRC))

from phase05_hdc_core import (  # noqa: E402
    CONTRACT_VERSION,
    MAX_CODEBOOK_DIMENSION,
    cosine_similarity_scores,
    ordered_level_codebook,
    predict_smallest_class_tie,
    stable_bipolar_vector,
)


PRIMARY = ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv"
FOLDS = ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv"
EXPECTED_PRIMARY_SHA = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLD_SHA = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
DIMENSIONS = [1000, 2000, 5000, 10000]
SEEDS = [42, 43, 44, 45, 46]
LEVELS = 51
FEATURE_K = 50
WARMUPS = 5
REPETITIONS = 30
NON_PREDICTIVE = {
    "subject_id", "session_id", "run_id", "difficulty_level_raw", "difficulty_level",
    "run_key", "target_class", "target_score", "outer_fold",
}
SNAPSHOT_PATH = PHASE / "audits/phase05_no_retraining_pre_amendment_snapshot.json"
AMENDMENT_PATH = PHASE / "configs/phase05_no_retraining_completion_amendment.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return str(path.relative_to(PHASE)).replace("/", "\\")


def snapshot() -> None:
    if SNAPSHOT_PATH.exists() or AMENDMENT_PATH.exists():
        raise RuntimeError("No-retraining amendment snapshot already exists")
    freeze_path = PHASE / "configs/phase05_freeze.json"
    manifest_path = PHASE / "manifests/phase05_final_artifact_manifest.json"
    freeze = load_json(freeze_path)
    manifest = load_json(manifest_path)
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("Phase 05 is not frozen before amendment")
    if freeze.get("final_artifact_manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("Pre-amendment freeze/manifest mismatch")

    amendable = {
        "README.md",
        "Phase_05_Basic_Dual_Output_HDC.ipynb",
        "reports\\phase05_final_summary.md",
        "audits\\phase05_final_notebook_persistence_audit.json",
        "audits\\phase05_final_reproducibility_audit.json",
        "audits\\phase05_upstream_freeze_integrity_audit.json",
    }
    records = {item["relative_path"]: item["sha256"] for item in manifest["artifacts"]}
    immutable = {key: value for key, value in records.items() if key not in amendable}
    notebook = json.loads((PHASE / "Phase_05_Basic_Dual_Output_HDC.ipynb").read_text(encoding="utf-8"))
    cell_hashes = [
        hashlib.sha256(json.dumps(cell, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        for cell in notebook["cells"]
    ]
    payload = {
        "phase": "05",
        "stage": "pre-amendment immutable snapshot",
        "created_utc": utc_now(),
        "authorization": "complete missing Phase 05 content without retraining",
        "primary_sha256": sha256(PRIMARY),
        "fold_sha256": sha256(FOLDS),
        "pre_amendment_freeze_sha256": sha256(freeze_path),
        "pre_amendment_manifest_sha256": sha256(manifest_path),
        "pre_amendment_notebook_sha256": sha256(PHASE / "Phase_05_Basic_Dual_Output_HDC.ipynb"),
        "pre_amendment_notebook_cell_count": len(notebook["cells"]),
        "pre_amendment_notebook_cell_sha256": cell_hashes,
        "immutable_artifact_count": len(immutable),
        "immutable_artifact_sha256": immutable,
        "intentionally_amendable_artifacts": sorted(amendable),
        "result": "PASS" if sha256(PRIMARY) == EXPECTED_PRIMARY_SHA and sha256(FOLDS) == EXPECTED_FOLD_SHA else "FAIL",
    }
    if payload["result"] != "PASS":
        raise RuntimeError("Frozen input checksum mismatch")
    atomic_json(SNAPSHOT_PATH, payload)
    amendment = {
        "phase": "05",
        "amendment": "NO_RETRAINING_DIAGNOSTIC_AND_EFFICIENCY_COMPLETION",
        "status": "AUTHORIZED_IN_PROGRESS",
        "created_utc": utc_now(),
        "allowed": [
            "derive diagnostics from frozen OOF predictions",
            "load saved fitted model and preprocessing artifacts",
            "repeat inference for timing and memory measurement while discarding outputs",
            "append Notebook/report evidence",
            "refresh final manifest, audits, and freeze",
        ],
        "prohibited": [
            "model fitting", "checkpoint replacement", "prediction artifact replacement",
            "hyperparameter tuning", "canonical seed or dimension selection", "Phase 06 execution",
        ],
        "warmups": WARMUPS,
        "timed_repetitions": REPETITIONS,
        "clock": "time.perf_counter_ns",
        "pre_amendment_snapshot": relative(SNAPSHOT_PATH),
        "pre_amendment_snapshot_sha256": sha256(SNAPSHOT_PATH),
    }
    atomic_json(AMENDMENT_PATH, amendment)
    print(json.dumps({"snapshot": "PASS", "immutable_artifacts": len(immutable), "notebook_cells": len(cell_hashes)}))


def softmax_diagnostics(similarities: np.ndarray, temperatures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = (similarities - similarities.max(axis=1, keepdims=True)) / temperatures[:, None]
    exponentiated = np.exp(shifted)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    raw = probabilities @ np.asarray([1.0, 2.0, 3.0, 4.0])
    return probabilities, raw


def rounded_class(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values).astype(np.int64) - 1, 0, 3)


def regression_diagnostic_row(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    rounded = rounded_class(prediction)
    true_class = target.astype(np.int64) - 1
    row: dict[str, Any] = {
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction, ddof=0)),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
        "prediction_range": float(np.ptp(prediction)),
        "mean_abs_distance_from_midpoint_2_5": float(np.mean(np.abs(prediction - 2.5))),
        "rounded_unique_levels": int(np.unique(rounded).size),
        "adjacent_accuracy": float(np.mean(np.abs(target - prediction) <= 1.0)),
        "severe_error_rate": float(np.mean(np.abs(target - prediction) >= 2.0)),
        "rounded_macro_f1": float(f1_score(true_class, rounded, average="macro", zero_division=0)),
        "rounded_quadratic_weighted_kappa": float(cohen_kappa_score(true_class, rounded, weights="quadratic")),
    }
    for level in [1, 2, 3, 4]:
        mask = target == float(level)
        row[f"true_level_{level}_rows"] = int(mask.sum())
        row[f"true_level_{level}_mae"] = float(mean_absolute_error(target[mask], prediction[mask]))
        row[f"true_level_{level}_prediction_mean"] = float(np.mean(prediction[mask]))
        row[f"true_level_{level}_prediction_std"] = float(np.std(prediction[mask], ddof=0))
    return row


def diagnostics() -> None:
    if not SNAPSHOT_PATH.is_file() or load_json(AMENDMENT_PATH).get("status") != "AUTHORIZED_IN_PROGRESS":
        raise RuntimeError("Amendment snapshot gate is not ready")
    oof_path = PHASE / "results/oof/vanilla_hdc_final_confirmation_oof_long.csv"
    oof_sha_before = sha256(oof_path)
    oof = pd.read_csv(oof_path)
    similarities = oof[[f"similarity_class_{index}" for index in range(4)]].to_numpy(dtype=np.float64)
    temperatures = oof["similarity_temperature"].to_numpy(dtype=np.float64)
    probabilities, similarity_raw = softmax_diagnostics(similarities, temperatures)
    similarity_bounded = np.clip(similarity_raw, 1.0, 4.0)
    if not np.allclose(similarity_bounded, oof["similarity_prediction"].to_numpy(float), rtol=0.0, atol=1e-12):
        raise RuntimeError("Derived similarity prediction does not reproduce frozen OOF")
    if not np.array_equal(predict_smallest_class_tie(similarities), oof["predicted_class"].to_numpy(int)):
        raise RuntimeError("Saved class prediction does not match saved similarities")
    sorted_similarity = np.sort(similarities, axis=1)
    margin = sorted_similarity[:, -1] - sorted_similarity[:, -2]
    target = oof["target_score"].to_numpy(float)
    true_class = oof["true_class"].to_numpy(int)
    predicted_class = oof["predicted_class"].to_numpy(int)
    ridge = oof["ridge_prediction_bounded"].to_numpy(float)
    similarity_rounded = rounded_class(similarity_bounded)
    ridge_rounded = rounded_class(ridge)

    diagnostic = oof[["run_key", "subject_id", "outer_fold", "dimension", "seed", "true_class", "predicted_class", "target_score"]].copy()
    for index in range(4):
        diagnostic[f"similarity_probability_{index}"] = probabilities[:, index]
    diagnostic["similarity_margin"] = margin
    diagnostic["similarity_prediction_raw"] = similarity_raw
    diagnostic["similarity_prediction_bounded"] = similarity_bounded
    diagnostic["similarity_rounded_class"] = similarity_rounded
    diagnostic["ridge_rounded_class"] = ridge_rounded
    diagnostic["classification_similarity_agreement"] = (predicted_class == similarity_rounded).astype(np.int8)
    diagnostic["classification_ridge_agreement"] = (predicted_class == ridge_rounded).astype(np.int8)
    diagnostic["classification_similarity_absolute_difference"] = np.abs((predicted_class + 1) - similarity_bounded)
    diagnostic["classification_ridge_absolute_difference"] = np.abs((predicted_class + 1) - ridge)
    diagnostic["similarity_distance_to_nearest_integer"] = np.abs(similarity_bounded - np.rint(similarity_bounded))
    diagnostic["ridge_distance_to_nearest_integer"] = np.abs(ridge - np.rint(ridge))
    diagnostic["classification_margin_minus_similarity_integer_distance"] = margin - diagnostic["similarity_distance_to_nearest_integer"]
    diagnostic["classification_margin_minus_ridge_integer_distance"] = margin - diagnostic["ridge_distance_to_nearest_integer"]
    diagnostic = diagnostic.sort_values(["dimension", "seed", "outer_fold", "subject_id", "run_key"]).reset_index(drop=True)
    atomic_csv(PHASE / "results/oof/vanilla_hdc_final_confirmation_diagnostics.csv", diagnostic)

    class_rows: list[dict[str, Any]] = []
    similarity_rows: list[dict[str, Any]] = []
    ridge_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    for (dimension, seed), index_values in oof.groupby(["dimension", "seed"], sort=True).groups.items():
        indices = np.asarray(list(index_values), dtype=np.int64)
        y = true_class[indices]
        yp = predicted_class[indices]
        recalls = recall_score(y, yp, labels=[0, 1, 2, 3], average=None, zero_division=0)
        class_row: dict[str, Any] = {"dimension": dimension, "seed": seed, "oof_rows": len(indices)}
        class_row.update({f"recall_class_{i}": float(recalls[i]) for i in range(4)})
        class_row.update({
            "similarity_margin_mean": float(np.mean(margin[indices])),
            "similarity_margin_std": float(np.std(margin[indices], ddof=0)),
            "similarity_margin_min": float(np.min(margin[indices])),
            "similarity_margin_max": float(np.max(margin[indices])),
        })
        class_rows.append(class_row)
        similarity_rows.append({"dimension": dimension, "seed": seed, "oof_rows": len(indices), **regression_diagnostic_row(target[indices], similarity_bounded[indices])})
        ridge_rows.append({"dimension": dimension, "seed": seed, "oof_rows": len(indices), **regression_diagnostic_row(target[indices], ridge[indices])})
        cross_rows.append({
            "dimension": dimension,
            "seed": seed,
            "oof_rows": len(indices),
            "classification_similarity_agreement_rate": float(np.mean(predicted_class[indices] == similarity_rounded[indices])),
            "classification_ridge_agreement_rate": float(np.mean(predicted_class[indices] == ridge_rounded[indices])),
            "classification_similarity_mean_absolute_difference": float(np.mean(np.abs((predicted_class[indices] + 1) - similarity_bounded[indices]))),
            "classification_ridge_mean_absolute_difference": float(np.mean(np.abs((predicted_class[indices] + 1) - ridge[indices]))),
            "classification_margin_mean": float(np.mean(margin[indices])),
            "similarity_integer_distance_mean": float(np.mean(np.abs(similarity_bounded[indices] - np.rint(similarity_bounded[indices])))),
            "ridge_integer_distance_mean": float(np.mean(np.abs(ridge[indices] - np.rint(ridge[indices])))),
            "classification_margin_minus_similarity_integer_distance_mean": float(np.mean(margin[indices] - np.abs(similarity_bounded[indices] - np.rint(similarity_bounded[indices])))),
            "classification_margin_minus_ridge_integer_distance_mean": float(np.mean(margin[indices] - np.abs(ridge[indices] - np.rint(ridge[indices])))),
        })

    outputs = {
        "results/summaries/vanilla_hdc_classification_diagnostics_by_config.csv": pd.DataFrame(class_rows),
        "results/summaries/vanilla_hdc_similarity_regression_diagnostics_by_config.csv": pd.DataFrame(similarity_rows),
        "results/summaries/vanilla_hdc_ridge_regression_diagnostics_by_config.csv": pd.DataFrame(ridge_rows),
        "results/summaries/vanilla_hdc_cross_task_consistency_by_config.csv": pd.DataFrame(cross_rows),
    }
    for path, frame in outputs.items():
        atomic_csv(PHASE / path, frame.sort_values(["dimension", "seed"]).reset_index(drop=True))
    if sha256(oof_path) != oof_sha_before:
        raise RuntimeError("Frozen OOF changed during diagnostic derivation")
    audit = {
        "phase": "05", "audit": "no_retraining_diagnostic_completion", "created_utc": utc_now(),
        "source_oof": relative(oof_path), "source_oof_sha256_before": oof_sha_before,
        "source_oof_sha256_after": sha256(oof_path), "source_oof_unchanged": True,
        "rows": len(diagnostic), "configs": len(class_rows), "probability_row_sum_max_abs_error": float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))),
        "similarity_prediction_max_abs_error": float(np.max(np.abs(similarity_bounded - oof["similarity_prediction"].to_numpy(float)))),
        "classification_prediction_mismatches": int(np.sum(predict_smallest_class_tie(similarities) != predicted_class)),
        "model_fitting_executed": False, "prediction_artifact_replaced": False,
        "result": "PASS",
    }
    atomic_json(PHASE / "audits/phase05_no_retraining_diagnostic_completion_audit.json", audit)
    print(json.dumps({"diagnostics": "PASS", "rows": len(diagnostic), "configs": len(class_rows)}))


class ReadOnlyPreprocessor:
    def __init__(self, fold: int):
        directory = PHASE / f"results/checkpoints/final_confirmation/fold_{fold}"
        metadata = load_json(directory / "outer_training_preprocessing.json")
        state_path = directory / "outer_training_preprocessing.npz"
        if metadata["state_npz_sha256"] != sha256(state_path):
            raise RuntimeError(f"Fold {fold} preprocessing hash mismatch")
        with np.load(state_path, allow_pickle=False) as arrays:
            self.arrays = {name: arrays[name].copy() for name in arrays.files}
        self.selected_feature_names = list(metadata["selected_feature_names"])

    def quantize(self, values: np.ndarray) -> np.ndarray:
        a = self.arrays
        missing = np.isnan(values)
        filled = np.where(missing, a["imputer_statistics"], values)
        if len(a["indicator_features"]):
            filled = np.hstack([filled, missing[:, a["indicator_features"]].astype(np.float64)])
        variable = filled[:, a["variance_support"]]
        scaled = (variable - a["scaler_mean"]) / a["scaler_scale"]
        selected = scaled[:, a["selected_indices"]]
        span = a["quantizer_maximum"] - a["quantizer_minimum"]
        safe_span = np.where(span > 0.0, span, 1.0)
        quantized = np.floor((selected - a["quantizer_minimum"]) / safe_span * LEVELS)
        quantized[:, span == 0.0] = 0
        return np.clip(quantized, 0, LEVELS - 1).astype(np.int16)

    @property
    def nbytes(self) -> int:
        return int(sum(value.nbytes for value in self.arrays.values()))


def feature_names() -> list[str]:
    names = [name for name in pd.read_csv(PRIMARY, nrows=0).columns.astype(str) if name not in NON_PREDICTIVE]
    if len(names) != 1176:
        raise RuntimeError("Primary feature interface is not 1,176 columns")
    return names


def load_test_values(fold: int, names: list[str]) -> tuple[np.ndarray, list[str]]:
    assignments = pd.read_csv(FOLDS, usecols=["run_key", "outer_fold"])
    allowed = set(assignments.loc[assignments["outer_fold"] == fold, "run_key"].astype(str))
    rows: list[list[float]] = []
    keys: list[str] = []
    with PRIMARY.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        positions = {name: index for index, name in enumerate(header)}
        indices = [positions[name] for name in names]
        run_index = positions["run_key"]
        for row in reader:
            key = row[run_index]
            if key in allowed:
                rows.append([float(row[index]) if row[index].strip() else np.nan for index in indices])
                keys.append(key)
    if set(keys) != allowed or len(keys) != len(allowed):
        raise RuntimeError(f"Fold {fold} test feature alignment failed")
    return np.asarray(rows, dtype=np.float64), keys


def prepare_codebooks(seed: int, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    level = ordered_level_codebook(seed, LEVELS, MAX_CODEBOOK_DIMENSION, CONTRACT_VERSION)
    tie = stable_bipolar_vector(seed, "bundle_tie", "shared", MAX_CODEBOOK_DIMENSION, CONTRACT_VERSION)
    identities = np.empty((len(names), MAX_CODEBOOK_DIMENSION), dtype=np.int8)
    digest = hashlib.sha256()
    for index, name in enumerate(names):
        vector = stable_bipolar_vector(seed, "feature_identity", name, MAX_CODEBOOK_DIMENSION, CONTRACT_VERSION)
        identities[index] = vector
        digest.update(name.encode("utf-8"))
        digest.update(vector.tobytes())
    hashes = {
        "identity_codebook_sha256": digest.hexdigest(),
        "level_codebook_sha256": hashlib.sha256(level.tobytes()).hexdigest(),
        "tie_vector_sha256": hashlib.sha256(tie.tobytes()).hexdigest(),
    }
    return identities, level, tie, hashes


def encode_precomputed(quantized: np.ndarray, identities: np.ndarray, levels: np.ndarray, tie: np.ndarray, dimension: int) -> np.ndarray:
    accumulator = np.zeros((len(quantized), dimension), dtype=np.int32)
    level_work = levels[:, :dimension]
    for index in range(quantized.shape[1]):
        selected = level_work[quantized[:, index]]
        accumulator += np.multiply(selected, identities[index, :dimension], dtype=np.int8).astype(np.int32, copy=False)
    return np.where(accumulator > 0, 1, np.where(accumulator < 0, -1, tie[:dimension])).astype(np.int8)


def load_fitted(fold: int, dimension: int, seed: int) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    directory = PHASE / f"results/checkpoints/final_confirmation/fold_{fold}"
    stem = f"vanilla_hdc_final_confirmation_fold_{fold}_dimension_{dimension}_seed_{seed}"
    checkpoint_path = directory / f"{stem}_checkpoint.json"
    model_path = directory / f"{stem}_model.npz"
    checkpoint = load_json(checkpoint_path)
    if checkpoint["model_npz_sha256"] != sha256(model_path):
        raise RuntimeError(f"Saved model hash mismatch: {stem}")
    with np.load(model_path, allow_pickle=False) as arrays:
        model = {name: arrays[name].copy() for name in arrays.files}
    return checkpoint, model, model_path


def heads(encoded: np.ndarray, model: dict[str, np.ndarray], temperature: float, dimension: int) -> tuple[np.ndarray, ...]:
    similarity = cosine_similarity_scores(encoded, model["prototypes"])
    predicted = predict_smallest_class_tie(similarity)
    shifted = (similarity.astype(np.float64) - similarity.max(axis=1, keepdims=True)) / temperature
    exponentiated = np.exp(shifted)
    probability = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    similarity_prediction = np.clip(probability @ np.asarray([1.0, 2.0, 3.0, 4.0]), 1.0, 4.0)
    normalized = encoded.astype(np.float32) / np.float32(np.sqrt(dimension))
    ridge_raw = normalized @ model["ridge_coef"] + float(model["ridge_intercept"][0])
    ridge_bounded = np.clip(ridge_raw, 1.0, 4.0)
    return similarity, predicted, similarity_prediction, ridge_raw, ridge_bounded


def inference_once(
    values: np.ndarray,
    preprocessor: ReadOnlyPreprocessor,
    identities: np.ndarray,
    levels: np.ndarray,
    tie: np.ndarray,
    model: dict[str, np.ndarray],
    temperature: float,
    dimension: int,
) -> tuple[tuple[np.ndarray, ...], tuple[int, int, int, int]]:
    start = time.perf_counter_ns()
    quantized = preprocessor.quantize(values)
    after_preprocessing = time.perf_counter_ns()
    encoded = encode_precomputed(quantized, identities, levels, tie, dimension)
    after_encoding = time.perf_counter_ns()
    output = heads(encoded, model, temperature, dimension)
    end = time.perf_counter_ns()
    return output, (after_preprocessing - start, after_encoding - after_preprocessing, end - after_encoding, end - start)


def verify_output(output: tuple[np.ndarray, ...], keys: list[str], frozen: pd.DataFrame) -> float:
    similarity, predicted, similarity_prediction, ridge_raw, ridge_bounded = output
    expected = frozen.set_index("run_key").loc[keys]
    mismatch = int(np.sum(predicted != expected["predicted_class"].to_numpy(int)))
    differences = [
        np.max(np.abs(similarity - expected[[f"similarity_class_{i}" for i in range(4)]].to_numpy(float))),
        np.max(np.abs(similarity_prediction - expected["similarity_prediction"].to_numpy(float))),
        np.max(np.abs(ridge_raw - expected["ridge_prediction_raw"].to_numpy(float))),
        np.max(np.abs(ridge_bounded - expected["ridge_prediction_bounded"].to_numpy(float))),
    ]
    maximum = float(np.max(differences))
    if mismatch or maximum > 1e-6:
        raise RuntimeError(f"Repeated inference disagrees with frozen predictions: classes={mismatch}, max={maximum}")
    return maximum


def efficiency() -> None:
    diagnostic_audit = load_json(PHASE / "audits/phase05_no_retraining_diagnostic_completion_audit.json")
    if diagnostic_audit.get("result") != "PASS":
        raise RuntimeError("Diagnostic completion gate failed")
    snapshot_record = load_json(SNAPSHOT_PATH)
    immutable_before = snapshot_record["immutable_artifact_sha256"]
    names = feature_names()
    frozen_predictions = {
        fold: pd.read_csv(PHASE / f"results/predictions/vanilla_hdc_final_confirmation_fold_{fold}_predictions.csv")
        for fold in range(1, 6)
    }
    rows: list[dict[str, Any]] = []
    max_prediction_difference = 0.0
    for fold in range(1, 6):
        preprocessor = ReadOnlyPreprocessor(fold)
        values, keys = load_test_values(fold, names)
        if len(preprocessor.selected_feature_names) != FEATURE_K:
            raise RuntimeError(f"Fold {fold} selected feature count mismatch")
        for seed in SEEDS:
            identities, level_codebook, tie_vector, codebook_hashes = prepare_codebooks(seed, preprocessor.selected_feature_names)
            for dimension in DIMENSIONS:
                checkpoint, model, model_path = load_fitted(fold, dimension, seed)
                for key, expected in codebook_hashes.items():
                    if checkpoint["codebook_hashes"][key] != expected:
                        raise RuntimeError(f"Codebook hash mismatch for fold={fold}, dimension={dimension}, seed={seed}, key={key}")
                temperature = float(checkpoint["selected_similarity_temperature"])
                untimed, _ = inference_once(values, preprocessor, identities, level_codebook, tie_vector, model, temperature, dimension)
                group = frozen_predictions[fold].loc[
                    (frozen_predictions[fold]["dimension"] == dimension) & (frozen_predictions[fold]["seed"] == seed)
                ]
                max_prediction_difference = max(max_prediction_difference, verify_output(untimed, keys, group))
                for _ in range(WARMUPS):
                    inference_once(values, preprocessor, identities, level_codebook, tie_vector, model, temperature, dimension)
                component_times: list[tuple[int, int, int, int]] = []
                output_digest = hashlib.sha256()
                for _ in range(REPETITIONS):
                    output, measured = inference_once(values, preprocessor, identities, level_codebook, tie_vector, model, temperature, dimension)
                    component_times.append(measured)
                    for array in output:
                        output_digest.update(np.asarray(array).tobytes())
                tracemalloc.start()
                inference_once(values, preprocessor, identities, level_codebook, tie_vector, model, temperature, dimension)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                components = np.asarray(component_times, dtype=np.int64)
                preprocessing_median = int(np.median(components[:, 0]))
                encoding_median = int(np.median(components[:, 1]))
                heads_median = int(np.median(components[:, 2]))
                total_median = int(np.median(components[:, 3]))
                identity_bytes = int(FEATURE_K * dimension * np.dtype(np.int8).itemsize)
                level_bytes = int(LEVELS * dimension * np.dtype(np.int8).itemsize)
                tie_bytes = int(dimension * np.dtype(np.int8).itemsize)
                prototype_bytes = int(model["prototypes"].nbytes)
                ridge_bytes = int(model["ridge_coef"].nbytes + model["ridge_intercept"].nbytes)
                model_memory = identity_bytes + level_bytes + tie_bytes + prototype_bytes + ridge_bytes + preprocessor.nbytes
                rows.append({
                    "outer_fold": fold, "dimension": dimension, "seed": seed, "test_rows": len(values),
                    "warmups": WARMUPS, "timed_repetitions": REPETITIONS, "clock": "time.perf_counter_ns",
                    "preprocessing_median_batch_ns": preprocessing_median,
                    "encoding_median_batch_ns": encoding_median,
                    "dual_heads_median_batch_ns": heads_median,
                    "complete_inference_median_batch_ns": total_median,
                    "complete_inference_median_per_sample_ns": float(total_median / len(values)),
                    "encoding_throughput_rows_per_second": float(len(values) / (encoding_median / 1e9)),
                    "identity_hv_bytes": identity_bytes, "level_hv_bytes": level_bytes,
                    "tie_vector_bytes": tie_bytes, "prototype_bytes": prototype_bytes,
                    "ridge_readout_bytes": ridge_bytes, "preprocessing_state_bytes": preprocessor.nbytes,
                    "total_model_component_bytes": model_memory,
                    "model_artifact_file_bytes": model_path.stat().st_size,
                    "inference_python_peak_allocated_bytes": int(peak),
                    "repeated_outputs_discarded": True,
                    "repeated_output_digest": output_digest.hexdigest(),
                    "frozen_prediction_max_abs_difference": verify_output(untimed, keys, group),
                })
            print(f"Measured fold={fold} seed={seed}: {len(rows)}/100 configurations", flush=True)

    detailed = pd.DataFrame(rows).sort_values(["outer_fold", "dimension", "seed"]).reset_index(drop=True)
    atomic_csv(PHASE / "results/efficiency/vanilla_hdc_final_confirmation_protocol_completion_by_fold_config.csv", detailed)
    config_rows: list[dict[str, Any]] = []
    for (dimension, seed), group in detailed.groupby(["dimension", "seed"], sort=True):
        total_ns = int(group["complete_inference_median_batch_ns"].sum())
        encoding_ns = int(group["encoding_median_batch_ns"].sum())
        config_rows.append({
            "dimension": dimension, "seed": seed, "folds": 5, "oof_rows": int(group["test_rows"].sum()),
            "warmups_per_fold": WARMUPS, "timed_repetitions_per_fold": REPETITIONS,
            "complete_oof_inference_seconds_sum_of_fold_medians": total_ns / 1e9,
            "complete_inference_nanoseconds_per_sample": total_ns / int(group["test_rows"].sum()),
            "encoding_throughput_rows_per_second": int(group["test_rows"].sum()) / (encoding_ns / 1e9),
            "maximum_total_model_component_bytes_across_folds": int(group["total_model_component_bytes"].max()),
            "maximum_inference_python_peak_allocated_bytes_across_folds": int(group["inference_python_peak_allocated_bytes"].max()),
        })
    config = pd.DataFrame(config_rows)
    atomic_csv(PHASE / "results/summaries/vanilla_hdc_inference_efficiency_protocol_by_config.csv", config)
    aggregate_rows: list[dict[str, Any]] = []
    metric_names = [
        "complete_oof_inference_seconds_sum_of_fold_medians",
        "complete_inference_nanoseconds_per_sample",
        "encoding_throughput_rows_per_second",
        "maximum_total_model_component_bytes_across_folds",
        "maximum_inference_python_peak_allocated_bytes_across_folds",
    ]
    for dimension, group in config.groupby("dimension", sort=True):
        row: dict[str, Any] = {"dimension": dimension, "seed_count": len(group)}
        for metric in metric_names:
            values = group[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sample_sd"] = float(np.std(values, ddof=1))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        aggregate_rows.append(row)
    atomic_csv(PHASE / "results/summaries/vanilla_hdc_inference_efficiency_protocol_seed_aggregate_by_dimension.csv", pd.DataFrame(aggregate_rows))

    immutable_mismatches = []
    for path_text, expected in immutable_before.items():
        path = PHASE / path_text
        if not path.exists() or sha256(path) != expected:
            immutable_mismatches.append(path_text)
    audit = {
        "phase": "05", "audit": "no_retraining_efficiency_protocol_completion", "created_utc": utc_now(),
        "configurations_measured": len(detailed), "expected_configurations": 100,
        "warmups": WARMUPS, "timed_repetitions": REPETITIONS, "clock": "time.perf_counter_ns",
        "codebook_prepared_outside_timed_repetitions": True,
        "complete_batch_components": ["frozen preprocessing transform", "record encoding", "classification head", "similarity regression head", "Ridge readout head"],
        "model_fitting_executed": False, "training_timing_remeasurement": "NOT_PERFORMED_RETRAINING_PROHIBITED",
        "prediction_artifact_replaced": False, "repeated_outputs_discarded": True,
        "maximum_frozen_prediction_abs_difference": max_prediction_difference,
        "immutable_artifact_mismatches": immutable_mismatches,
        "result": "PASS" if len(detailed) == 100 and not immutable_mismatches and max_prediction_difference <= 1e-6 else "FAIL",
    }
    atomic_json(PHASE / "audits/phase05_no_retraining_efficiency_protocol_completion_audit.json", audit)
    if audit["result"] != "PASS":
        raise RuntimeError("Efficiency protocol completion audit failed")
    print(json.dumps({"efficiency": "PASS", "configurations": len(detailed), "max_prediction_difference": max_prediction_difference}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["snapshot", "diagnostics", "efficiency"])
    args = parser.parse_args()
    {"snapshot": snapshot, "diagnostics": diagnostics, "efficiency": efficiency}[args.mode]()


if __name__ == "__main__":
    main()
