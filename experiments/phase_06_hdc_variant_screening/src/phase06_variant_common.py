"""Shared Phase 06 interfaces built on direct imports from the frozen Phase 05 encoder."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray


PHASE06 = Path(__file__).resolve().parents[1]
ROOT = PHASE06.parents[1]
PHASE05 = ROOT / "experiments" / "phase_05_basic_dual_output_hdc"
sys.path.insert(0, str(PHASE05 / "src"))
sys.path.insert(0, str(PHASE05 / "scripts"))

from phase05_hdc_core import (  # noqa: E402
    CONTRACT_VERSION as PHASE05_ENCODER_CONTRACT_VERSION,
    EqualWidthQuantizer,
    cosine_similarity_scores,
    incremental_encode_prefixes,
    stable_rng,
)
from run_vanilla_hdc_quick_screen import (  # noqa: E402
    classification_metrics,
    fitted_preprocessing,
    load_outer_training_features,
)


PHASE06_CONTRACT_VERSION = "phase06_hdc_variant_contract_v1"
CLASSES = np.asarray([0, 1, 2, 3], dtype=np.int64)
Float32Array = NDArray[np.float32]
Int8Array = NDArray[np.int8]


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_identifier_seed(seed: int, role: str, identifier: str) -> int:
    payload = canonical_json(
        {"contract_version": PHASE06_CONTRACT_VERSION, "seed": int(seed), "role": role, "identifier": identifier}
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def stable_pcg64(seed: int, role: str, identifier: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(stable_identifier_seed(seed, role, identifier)))


def stable_kmeans_random_state(seed: int, identifier: str) -> int:
    return int(stable_identifier_seed(seed, "kmeans", identifier) % (2**31 - 1))


def normalize_rows(values: NDArray[np.number]) -> Float32Array:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("Cannot normalize a zero vector")
    return np.asarray(matrix / norms, dtype=np.float32)


def class_prototypes(sample_hv: Int8Array, labels: NDArray[np.integer]) -> Float32Array:
    vectors = np.asarray(sample_hv, dtype=np.int8)
    targets = np.asarray(labels, dtype=np.int64)
    prototypes = np.empty((len(CLASSES), vectors.shape[1]), dtype=np.float32)
    for index, class_id in enumerate(CLASSES):
        mask = targets == class_id
        if not mask.any():
            raise ValueError(f"No training samples for class {class_id}")
        prototypes[index] = vectors[mask].sum(axis=0, dtype=np.float32)
    return normalize_rows(prototypes)


def smallest_index_argmax(scores: NDArray[np.floating]) -> NDArray[np.int64]:
    return np.argmax(np.asarray(scores), axis=1).astype(np.int64)


def runner_up_indices(scores: NDArray[np.floating], true_labels: NDArray[np.integer]) -> NDArray[np.int64]:
    values = np.asarray(scores, dtype=np.float32).copy()
    labels = np.asarray(true_labels, dtype=np.int64)
    values[np.arange(len(labels)), labels] = -np.inf
    return smallest_index_argmax(values)


def margins_and_runner_up(
    scores: NDArray[np.floating], true_labels: NDArray[np.integer]
) -> tuple[Float32Array, NDArray[np.int64]]:
    labels = np.asarray(true_labels, dtype=np.int64)
    runners = runner_up_indices(scores, labels)
    margins = np.asarray(scores[np.arange(len(labels)), labels] - scores[np.arange(len(labels)), runners], dtype=np.float32)
    return margins, runners


def candidate_grid(variant: str) -> list[dict[str, Any]]:
    dimensions = [2000, 5000]
    candidates: list[dict[str, Any]] = []
    if variant == "onlinehd":
        for dimension in dimensions:
            for epochs in [1, 3, 5]:
                for learning_rate in [0.05, 0.1]:
                    for margin_threshold in [0.0, 0.1]:
                        candidates.append({"variant": variant, "dimension": dimension, "epochs": epochs, "learning_rate": learning_rate, "margin_threshold": margin_threshold, "levels": 51, "feature_k": 50, "seed": 42})
    elif variant == "multicentroid":
        for dimension in dimensions:
            for centers in [2, 3, 4]:
                candidates.append({"variant": variant, "dimension": dimension, "centroids_per_class": centers, "levels": 51, "feature_k": 50, "seed": 42})
    elif variant == "hybrid":
        for dimension in dimensions:
            for centers in [2, 3]:
                for epochs in [1, 3]:
                    for learning_rate in [0.05, 0.1]:
                        for margin_threshold in [0.0, 0.1]:
                            candidates.append({"variant": variant, "dimension": dimension, "centroids_per_class": centers, "epochs": epochs, "learning_rate": learning_rate, "margin_threshold": margin_threshold, "levels": 51, "feature_k": 50, "seed": 42})
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return candidates


def expected_candidate_count(variant: str) -> int:
    return {"onlinehd": 24, "multicentroid": 6, "hybrid": 32}[variant]


def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["mean_macro_f1"]),
        float(row["std_macro_f1_sample"]),
        -float(row["mean_balanced_accuracy"]),
        float(row["mean_severe_error_rate"]),
        int(row["dimension"]),
        int(row.get("epochs", 0)),
        int(row.get("centroids_per_class", 0)),
        float(row.get("learning_rate", 0.0)),
        float(row.get("margin_threshold", 0.0)),
        str(row["canonical_config_json"]),
    )


def best_candidate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    candidates = list(rows)
    if not candidates:
        raise ValueError("No completed candidates")
    return min(candidates, key=selection_key)
