"""Leakage-safe Vanilla Prototype HDC primitives for Phase 05.

The module implements the frozen bipolar record-encoding contract.  It has no
knowledge of outer-test data and performs no work at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.signedinteger]
FloatArray = NDArray[np.floating]
CONTRACT_VERSION = "phase05_basic_dual_output_hdc_v1"
MAX_CODEBOOK_DIMENSION = 10_000


def _seed_material(seed: int, role: str, identifier: str, contract_version: str) -> int:
    payload = json.dumps(
        {
            "contract_version": contract_version,
            "experiment_seed": int(seed),
            "identifier": str(identifier),
            "vector_role": str(role),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big", signed=False)


def stable_rng(
    seed: int,
    role: str,
    identifier: str,
    contract_version: str = CONTRACT_VERSION,
) -> np.random.Generator:
    """Return a PCG64 generator from a stable SHA-256-derived seed."""
    return np.random.Generator(
        np.random.PCG64(_seed_material(seed, role, identifier, contract_version))
    )


def stable_bipolar_vector(
    seed: int,
    role: str,
    identifier: str,
    dimension: int = MAX_CODEBOOK_DIMENSION,
    contract_version: str = CONTRACT_VERSION,
) -> NDArray[np.int8]:
    """Generate a deterministic bipolar vector using the frozen random stream."""
    if dimension <= 0 or dimension > MAX_CODEBOOK_DIMENSION:
        raise ValueError(f"dimension must be in [1, {MAX_CODEBOOK_DIMENSION}]")
    rng = stable_rng(seed, role, identifier, contract_version)
    return (rng.integers(0, 2, size=dimension, dtype=np.int8) * 2 - 1).astype(
        np.int8, copy=False
    )


def ordered_level_codebook(
    seed: int,
    levels: int,
    dimension: int = MAX_CODEBOOK_DIMENSION,
    contract_version: str = CONTRACT_VERSION,
) -> NDArray[np.int8]:
    """Build the frozen ordered correlated bipolar level codebook."""
    if levels < 2:
        raise ValueError("levels must be at least 2")
    base = stable_bipolar_vector(
        seed, "level_base", str(levels), dimension, contract_version
    )
    permutation_rng = stable_rng(seed, "level_permutation", str(levels), contract_version)
    changed_coordinates = permutation_rng.permutation(dimension)[: dimension // 2]
    transition_groups = np.array_split(changed_coordinates, levels - 1)
    codebook = np.empty((levels, dimension), dtype=np.int8)
    codebook[0] = base
    for level_index, coordinates in enumerate(transition_groups, start=1):
        codebook[level_index] = codebook[level_index - 1]
        codebook[level_index, coordinates] *= np.int8(-1)
    return codebook


def bind(identity_hv: IntArray, level_hv: IntArray) -> NDArray[np.int8]:
    """Bipolar elementwise multiplication binding."""
    if identity_hv.shape != level_hv.shape:
        raise ValueError("identity and level hypervectors must have identical shapes")
    return np.multiply(identity_hv, level_hv, dtype=np.int8)


def sign_with_tie(bundle: IntArray, tie_vector: IntArray) -> NDArray[np.int8]:
    """Apply sign bundling and resolve zero coordinates with a fixed tie vector."""
    if bundle.shape[-1] != tie_vector.shape[-1]:
        raise ValueError("tie vector dimension does not match bundle")
    result = np.where(bundle > 0, 1, np.where(bundle < 0, -1, tie_vector))
    return result.astype(np.int8, copy=False)


def bundle_bound_features(
    bound_features: Sequence[IntArray], tie_vector: IntArray
) -> NDArray[np.int8]:
    """Naively sum bound features and apply deterministic sign/tie bundling."""
    if not bound_features:
        raise ValueError("at least one bound feature is required")
    accumulator = np.zeros_like(bound_features[0], dtype=np.int32)
    for bound_feature in bound_features:
        accumulator += np.asarray(bound_feature, dtype=np.int32)
    return sign_with_tie(accumulator, tie_vector)


@dataclass
class EqualWidthQuantizer:
    """Fold-local equal-width quantizer with explicit fitted state."""

    levels: int
    minimum_: NDArray[np.float64] | None = None
    maximum_: NDArray[np.float64] | None = None

    def fit(self, values: FloatArray) -> "EqualWidthQuantizer":
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("fit values must be a non-empty 2D matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("non-finite value encountered before quantizer fit")
        if self.levels < 2:
            raise ValueError("levels must be at least 2")
        self.minimum_ = matrix.min(axis=0)
        self.maximum_ = matrix.max(axis=0)
        return self

    def transform(self, values: FloatArray) -> NDArray[np.int16]:
        if self.minimum_ is None or self.maximum_ is None:
            raise RuntimeError("quantizer is not fitted")
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.minimum_.shape[0]:
            raise ValueError("transform matrix has incompatible shape")
        if not np.isfinite(matrix).all():
            raise ValueError("non-finite value encountered before encoding")
        span = self.maximum_ - self.minimum_
        safe_span = np.where(span > 0.0, span, 1.0)
        scaled = (matrix - self.minimum_) / safe_span
        quantized = np.floor(scaled * self.levels).astype(np.int64)
        quantized[:, span == 0.0] = 0
        return np.clip(quantized, 0, self.levels - 1).astype(np.int16)

    def fit_transform(self, values: FloatArray) -> NDArray[np.int16]:
        return self.fit(values).transform(values)

    def state_digest(self) -> str:
        if self.minimum_ is None or self.maximum_ is None:
            raise RuntimeError("quantizer is not fitted")
        digest = hashlib.sha256()
        digest.update(self.minimum_.tobytes())
        digest.update(self.maximum_.tobytes())
        digest.update(str(self.levels).encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True)
class IncrementalEncodingResult:
    samples_by_k: Mapping[str, NDArray[np.int8]]
    codebook_hashes: Mapping[str, str]
    feature_completion_seconds: Mapping[str, float]


def incremental_encode_prefixes(
    quantized_values: NDArray[np.integer],
    ranked_feature_names: Sequence[str],
    levels: int,
    seed: int,
    snapshot_k: Iterable[int | str],
    work_dimension: int,
    contract_version: str = CONTRACT_VERSION,
) -> IncrementalEncodingResult:
    """Encode feature-rank prefixes using one 2D int32 bundle accumulator.

    No samples×features×dimension array is created.  All item and level vectors
    originate at 10,000 dimensions; the work array uses their first D entries.
    """
    import time

    matrix = np.asarray(quantized_values)
    if matrix.ndim != 2 or matrix.shape[1] != len(ranked_feature_names):
        raise ValueError("quantized matrix and ranked feature names do not align")
    if work_dimension <= 0 or work_dimension > MAX_CODEBOOK_DIMENSION:
        raise ValueError("invalid work dimension")
    requested = list(snapshot_k)
    resolved: dict[int, str] = {}
    for candidate in requested:
        prefix = matrix.shape[1] if candidate == "all" else int(candidate)
        if prefix <= 0 or prefix > matrix.shape[1]:
            raise ValueError(f"invalid feature prefix {candidate!r}")
        resolved[prefix] = str(candidate)

    level_full = ordered_level_codebook(
        seed, levels, MAX_CODEBOOK_DIMENSION, contract_version
    )
    level_work = level_full[:, :work_dimension]
    tie_work = stable_bipolar_vector(
        seed, "bundle_tie", "shared", MAX_CODEBOOK_DIMENSION, contract_version
    )[:work_dimension]
    accumulator = np.zeros((matrix.shape[0], work_dimension), dtype=np.int32)
    snapshots: dict[str, NDArray[np.int8]] = {}
    completion_times: dict[str, float] = {}
    identity_digest = hashlib.sha256()
    start = time.perf_counter()

    for feature_index, feature_name in enumerate(ranked_feature_names, start=1):
        identity_full = stable_bipolar_vector(
            seed, "feature_identity", feature_name, MAX_CODEBOOK_DIMENSION, contract_version
        )
        identity_work = identity_full[:work_dimension]
        identity_digest.update(feature_name.encode("utf-8"))
        identity_digest.update(identity_full.tobytes())
        selected_levels = level_work[matrix[:, feature_index - 1]]
        accumulator += np.multiply(selected_levels, identity_work, dtype=np.int8).astype(
            np.int32, copy=False
        )
        if feature_index in resolved:
            label = resolved[feature_index]
            snapshots[label] = sign_with_tie(accumulator, tie_work)
            completion_times[label] = time.perf_counter() - start

    hashes = {
        "identity_codebook_sha256": identity_digest.hexdigest(),
        "level_codebook_sha256": hashlib.sha256(level_full.tobytes()).hexdigest(),
        "tie_vector_sha256": hashlib.sha256(tie_work.tobytes()).hexdigest(),
    }
    return IncrementalEncodingResult(snapshots, hashes, completion_times)


def naive_encode(
    quantized_values: NDArray[np.integer],
    feature_names: Sequence[str],
    levels: int,
    seed: int,
    dimension: int,
    contract_version: str = CONTRACT_VERSION,
) -> NDArray[np.int8]:
    """Small reference implementation used only for correctness tests."""
    matrix = np.asarray(quantized_values)
    level_codebook = ordered_level_codebook(
        seed, levels, MAX_CODEBOOK_DIMENSION, contract_version
    )[:, :dimension]
    tie_vector = stable_bipolar_vector(
        seed, "bundle_tie", "shared", MAX_CODEBOOK_DIMENSION, contract_version
    )[:dimension]
    output = np.empty((matrix.shape[0], dimension), dtype=np.int8)
    for sample_index in range(matrix.shape[0]):
        bound: list[NDArray[np.int8]] = []
        for feature_index, feature_name in enumerate(feature_names):
            identity = stable_bipolar_vector(
                seed,
                "feature_identity",
                feature_name,
                MAX_CODEBOOK_DIMENSION,
                contract_version,
            )[:dimension]
            bound.append(bind(identity, level_codebook[matrix[sample_index, feature_index]]))
        output[sample_index] = bundle_bound_features(bound, tie_vector)
    return output


def build_prototypes(
    sample_hv: IntArray, labels: NDArray[np.integer], classes: Sequence[int] = (0, 1, 2, 3)
) -> NDArray[np.int32]:
    """Accumulate one int32 prototype per class from training samples only."""
    vectors = np.asarray(sample_hv, dtype=np.int8)
    targets = np.asarray(labels)
    prototypes = np.zeros((len(classes), vectors.shape[1]), dtype=np.int32)
    for prototype_index, class_value in enumerate(classes):
        class_mask = targets == class_value
        if not class_mask.any():
            raise ValueError(f"training data contains no samples for class {class_value}")
        prototypes[prototype_index] = vectors[class_mask].sum(axis=0, dtype=np.int32)
    return prototypes


def cosine_similarity_scores(sample_hv: IntArray, prototypes: IntArray) -> NDArray[np.float32]:
    """Return finite float32 cosine similarities in [-1, 1]."""
    vectors = np.asarray(sample_hv, dtype=np.float32)
    prototype_values = np.asarray(prototypes, dtype=np.float32)
    sample_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    prototype_norm = np.linalg.norm(prototype_values, axis=1, keepdims=True).T
    if np.any(sample_norm == 0.0) or np.any(prototype_norm == 0.0):
        raise ValueError("cosine similarity received a zero vector")
    scores = (vectors @ prototype_values.T) / (sample_norm * prototype_norm)
    scores = np.clip(scores, -1.0, 1.0).astype(np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("cosine similarity produced a non-finite value")
    return scores


def predict_smallest_class_tie(
    similarities: FloatArray, classes: Sequence[int] = (0, 1, 2, 3)
) -> NDArray[np.int64]:
    """Predict maximum similarity; np.argmax fixes exact ties to lowest class."""
    class_values = np.asarray(classes, dtype=np.int64)
    return class_values[np.argmax(np.asarray(similarities), axis=1)]
