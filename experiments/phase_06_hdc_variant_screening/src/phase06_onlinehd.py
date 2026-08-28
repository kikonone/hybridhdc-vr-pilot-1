"""Frozen OnlineHD-style update rule for Phase 06."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from phase06_variant_common import (
    class_prototypes,
    cosine_similarity_scores,
    margins_and_runner_up,
    normalize_rows,
    smallest_index_argmax,
    stable_pcg64,
)


def apply_online_update(
    prototypes: NDArray[np.float32],
    *,
    true_index: int,
    runner_up_index: int,
    sample: NDArray[np.integer] | NDArray[np.floating],
    similarity_true: float,
    similarity_runner_up: float,
    learning_rate: float,
) -> None:
    """Apply the frozen attractive/repulsive update in place to exactly two rows."""
    vector = np.asarray(sample, dtype=np.float32)
    prototypes[true_index] += np.float32(learning_rate * (1.0 - similarity_true)) * vector
    prototypes[runner_up_index] -= np.float32(learning_rate * max(0.0, similarity_runner_up)) * vector


def train_onlinehd(
    sample_hv: NDArray[np.integer],
    labels: NDArray[np.integer],
    *,
    epochs: int,
    learning_rate: float,
    margin_threshold: float,
    seed: int,
    stream_identifier: str,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    vectors = np.asarray(sample_hv, dtype=np.int8)
    targets = np.asarray(labels, dtype=np.int64)
    prototypes = class_prototypes(vectors, targets)
    updates = 0
    for epoch in range(int(epochs)):
        order = stable_pcg64(seed, "onlinehd_order", f"{stream_identifier}|epoch={epoch}").permutation(len(targets))
        for sample_index in order:
            sample = vectors[sample_index : sample_index + 1]
            scores = cosine_similarity_scores(sample, prototypes)[0]
            true_class = int(targets[sample_index])
            runner_scores = scores.copy()
            runner_scores[true_class] = -np.inf
            runner_up = int(np.argmax(runner_scores))
            margin = float(scores[true_class] - scores[runner_up])
            predicted = int(np.argmax(scores))
            if predicted != true_class or margin < float(margin_threshold):
                apply_online_update(
                    prototypes, true_index=true_class, runner_up_index=runner_up, sample=sample[0],
                    similarity_true=float(scores[true_class]), similarity_runner_up=float(scores[runner_up]),
                    learning_rate=learning_rate,
                )
                updates += 1
        prototypes = normalize_rows(prototypes)
    return prototypes.astype(np.float32, copy=False), {"updates": updates, "epochs": int(epochs)}


def predict_onlinehd(sample_hv: NDArray[np.integer], prototypes: NDArray[np.floating]) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    scores = cosine_similarity_scores(sample_hv, prototypes)
    return smallest_index_argmax(scores), scores
