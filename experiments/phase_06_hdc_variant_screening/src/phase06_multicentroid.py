"""Frozen class-wise deterministic Multi-centroid HDC for Phase 06."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from phase06_variant_common import CLASSES, normalize_rows, stable_kmeans_random_state


def train_multicentroid(
    sample_hv: NDArray[np.integer],
    labels: NDArray[np.integer],
    *,
    centroids_per_class: int,
    seed: int,
    stream_identifier: str,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    vectors = normalize_rows(np.asarray(sample_hv, dtype=np.int8))
    targets = np.asarray(labels, dtype=np.int64)
    centers = int(centroids_per_class)
    output = np.empty((len(CLASSES), centers, vectors.shape[1]), dtype=np.float32)
    class_counts: dict[str, int] = {}
    cluster_counts: dict[str, list[int]] = {}
    for class_id in CLASSES:
        class_vectors = vectors[targets == class_id]
        class_counts[str(class_id)] = int(len(class_vectors))
        if len(class_vectors) < centers:
            raise ValueError(f"Class {class_id} has fewer samples than requested centroids")
        model = KMeans(
            n_clusters=centers,
            n_init=10,
            max_iter=300,
            algorithm="lloyd",
            random_state=stable_kmeans_random_state(seed, f"{stream_identifier}|class={class_id}|k={centers}"),
        )
        assignments = model.fit_predict(class_vectors)
        counts = np.bincount(assignments, minlength=centers)
        if np.any(counts == 0) or len(np.unique(assignments)) != centers:
            raise RuntimeError(f"Class {class_id} produced an empty cluster")
        cluster_counts[str(class_id)] = counts.astype(int).tolist()
        for cluster_id in range(centers):
            output[class_id, cluster_id] = class_vectors[assignments == cluster_id].mean(axis=0, dtype=np.float32)
        output[class_id] = normalize_rows(output[class_id])
    return output, {"class_counts": class_counts, "cluster_counts": cluster_counts, "centroids_per_class": centers}


def multicentroid_scores(sample_hv: NDArray[np.integer], centroids: NDArray[np.floating]) -> NDArray[np.float32]:
    vectors = normalize_rows(np.asarray(sample_hv, dtype=np.int8))
    centers = np.asarray(centroids, dtype=np.float32)
    if centers.ndim != 3 or centers.shape[0] != len(CLASSES):
        raise ValueError("Centroids must have shape (4, k, dimension)")
    flat = centers.reshape(-1, centers.shape[-1])
    similarities = vectors @ flat.T
    return similarities.reshape(len(vectors), len(CLASSES), centers.shape[1]).max(axis=2).astype(np.float32)


def predict_multicentroid(sample_hv: NDArray[np.integer], centroids: NDArray[np.floating]) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    scores = multicentroid_scores(sample_hv, centroids)
    return np.argmax(scores, axis=1).astype(np.int64), scores
