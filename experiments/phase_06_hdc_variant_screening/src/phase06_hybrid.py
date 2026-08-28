"""Frozen Multi-centroid plus OnlineHD-style Hybrid for Phase 06."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from phase06_multicentroid import multicentroid_scores, train_multicentroid
from phase06_onlinehd import apply_online_update
from phase06_variant_common import normalize_rows, stable_pcg64


def hybrid_update_targets(similarities: NDArray[np.floating], true_class: int) -> tuple[int, int, int]:
    """Return true centroid plus the globally strongest non-true centroid."""
    values = np.asarray(similarities, dtype=np.float32)
    true_centroid = int(np.argmax(values[true_class]))
    non_true = values.copy()
    non_true[true_class, :] = -np.inf
    flat_runner = int(np.argmax(non_true))
    runner_class, runner_centroid = np.unravel_index(flat_runner, non_true.shape)
    return true_centroid, int(runner_class), int(runner_centroid)


def train_hybrid(
    sample_hv: NDArray[np.integer],
    labels: NDArray[np.integer],
    *,
    centroids_per_class: int,
    epochs: int,
    learning_rate: float,
    margin_threshold: float,
    seed: int,
    stream_identifier: str,
    initial_centroids: NDArray[np.floating] | None = None,
    initialization_info: dict[str, Any] | None = None,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    vectors = np.asarray(sample_hv, dtype=np.int8)
    targets = np.asarray(labels, dtype=np.int64)
    if initial_centroids is None:
        centroids, initialization = train_multicentroid(
            vectors, targets, centroids_per_class=centroids_per_class, seed=seed, stream_identifier=stream_identifier
        )
    else:
        centroids = np.asarray(initial_centroids, dtype=np.float32).copy()
        initialization = dict(initialization_info or {})
    updates = 0
    for epoch in range(int(epochs)):
        order = stable_pcg64(seed, "hybrid_order", f"{stream_identifier}|epoch={epoch}").permutation(len(targets))
        for sample_index in order:
            sample = normalize_rows(vectors[sample_index : sample_index + 1])[0]
            similarities = np.einsum("ckd,d->ck", centroids, sample, optimize=True)
            true_class = int(targets[sample_index])
            true_centroid, runner_class, runner_centroid = hybrid_update_targets(similarities, true_class)
            similarity_true = float(similarities[true_class, true_centroid])
            similarity_runner = float(similarities[runner_class, runner_centroid])
            class_scores = similarities.max(axis=1)
            predicted = int(np.argmax(class_scores))
            margin = similarity_true - similarity_runner
            if predicted != true_class or margin < float(margin_threshold):
                apply_online_update(
                    centroids.reshape(-1, centroids.shape[-1]),
                    true_index=true_class * centroids.shape[1] + true_centroid,
                    runner_up_index=runner_class * centroids.shape[1] + runner_centroid,
                    sample=sample,
                    similarity_true=similarity_true,
                    similarity_runner_up=similarity_runner,
                    learning_rate=learning_rate,
                )
                updates += 1
        for class_id in range(centroids.shape[0]):
            centroids[class_id] = normalize_rows(centroids[class_id])
    return centroids.astype(np.float32, copy=False), {"initialization": initialization, "updates": updates, "epochs": int(epochs)}


def predict_hybrid(sample_hv: NDArray[np.integer], centroids: NDArray[np.floating]) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    scores = multicentroid_scores(sample_hv, centroids)
    return np.argmax(scores, axis=1).astype(np.int64), scores
