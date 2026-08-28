from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "src"))
sys.path.insert(0, str(PHASE / "scripts"))

from phase06_hybrid import hybrid_update_targets, predict_hybrid, train_hybrid  # noqa: E402
from phase06_multicentroid import predict_multicentroid, train_multicentroid  # noqa: E402
from phase06_onlinehd import apply_online_update, predict_onlinehd, train_onlinehd  # noqa: E402
from phase06_variant_common import (  # noqa: E402
    candidate_grid,
    fitted_preprocessing,
    margins_and_runner_up,
    normalize_rows,
    runner_up_indices,
    stable_pcg64,
)
from run_phase06_quick_screen import load_outer_training_features, valid_checkpoint, write_json  # noqa: E402


def toy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(4, dtype=np.int64), 8)
    vectors = np.where(rng.normal(size=(32, 24)) >= 0, 1, -1).astype(np.int8)
    for class_id in range(4):
        vectors[labels == class_id, class_id * 4 : class_id * 4 + 4] = 1
        vectors[labels != class_id, class_id * 4 : class_id * 4 + 4] = -1
    return vectors, labels


def test_random_streams_are_reproducible_and_seed_distinct() -> None:
    first = stable_pcg64(42, "test", "same").integers(0, 2**31, size=16)
    second = stable_pcg64(42, "test", "same").integers(0, 2**31, size=16)
    different = stable_pcg64(43, "test", "same").integers(0, 2**31, size=16)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def test_margin_runner_up_and_update_direction() -> None:
    scores = np.asarray([[0.7, 0.2, 0.4, 0.1], [0.5, 0.5, 0.2, 0.1]], dtype=np.float32)
    labels = np.asarray([0, 1])
    margins, runners = margins_and_runner_up(scores, labels)
    assert np.allclose(margins, [0.3, 0.0])
    assert np.array_equal(runners, [2, 0])
    prototypes = np.zeros((4, 3), dtype=np.float32)
    sample = np.asarray([1, -1, 1], dtype=np.float32)
    apply_online_update(prototypes, true_index=1, runner_up_index=2, sample=sample, similarity_true=0.25, similarity_runner_up=0.5, learning_rate=0.1)
    assert float(prototypes[1] @ sample) > 0.0
    assert float(prototypes[2] @ sample) < 0.0
    assert np.all(prototypes[[0, 3]] == 0.0)


def test_onlinehd_determinism_normalization_dtype_and_no_input_mutation() -> None:
    vectors, labels = toy_data()
    before = vectors.copy()
    first, _ = train_onlinehd(vectors, labels, epochs=3, learning_rate=0.1, margin_threshold=0.1, seed=42, stream_identifier="unit")
    second, _ = train_onlinehd(vectors, labels, epochs=3, learning_rate=0.1, margin_threshold=0.1, seed=42, stream_identifier="unit")
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    assert first.dtype == np.float32 and vectors.dtype == np.int8
    assert np.array_equal(vectors, before)


def test_multicentroid_count_class_isolation_determinism_and_classification() -> None:
    vectors, labels = toy_data()
    before = vectors.copy()
    first, info = train_multicentroid(vectors, labels, centroids_per_class=2, seed=42, stream_identifier="unit")
    second, _ = train_multicentroid(vectors, labels, centroids_per_class=2, seed=42, stream_identifier="unit")
    predictions, scores = predict_multicentroid(vectors, first)
    assert first.shape == (4, 2, 24)
    assert all(count == 8 for count in info["class_counts"].values())
    assert all(len(counts) == 2 and all(value > 0 for value in counts) for counts in info["cluster_counts"].values())
    assert np.array_equal(first, second)
    assert np.mean(predictions == labels) >= 0.75
    assert scores.dtype == np.float32 and first.dtype == np.float32
    assert np.array_equal(vectors, before)


def test_hybrid_targets_determinism_and_no_input_mutation() -> None:
    similarities = np.asarray([[0.2, 0.5], [0.6, 0.4], [0.9, 0.1], [0.3, 0.2]], dtype=np.float32)
    assert hybrid_update_targets(similarities, 1) == (0, 2, 0)
    vectors, labels = toy_data()
    before = vectors.copy()
    first, _ = train_hybrid(vectors, labels, centroids_per_class=2, epochs=1, learning_rate=0.05, margin_threshold=0.0, seed=42, stream_identifier="unit")
    second, _ = train_hybrid(vectors, labels, centroids_per_class=2, epochs=1, learning_rate=0.05, margin_threshold=0.0, seed=42, stream_identifier="unit")
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=2), 1.0, atol=1e-6)
    assert np.array_equal(vectors, before)


def test_tied_predictions_choose_smallest_class() -> None:
    sample = np.ones((1, 8), dtype=np.int8)
    prototypes = normalize_rows(np.ones((4, 8), dtype=np.float32))
    online_prediction, _ = predict_onlinehd(sample, prototypes)
    centroids = np.repeat(prototypes[:, None, :], 2, axis=1)
    multi_prediction, _ = predict_multicentroid(sample, centroids)
    hybrid_prediction, _ = predict_hybrid(sample, centroids)
    assert online_prediction.tolist() == multi_prediction.tolist() == hybrid_prediction.tolist() == [0]


def test_subject_isolation_and_preprocessing_scope() -> None:
    input_audit = json.loads((PHASE / "audits" / "phase06_input_and_fold_audit.json").read_text(encoding="utf-8"))
    assert input_audit["checks"]["outer_subject_isolation"] is True
    assert input_audit["checks"]["inner_groupkfold_3_feasible"] is True
    rng = np.random.default_rng(1)
    train = rng.normal(size=(20, 6))
    validation_a = rng.normal(size=(5, 6))
    validation_b = validation_a + 10000.0
    labels = np.tile(np.arange(4), 5)
    names = [f"x{i}" for i in range(6)]
    train_a, _, ranked_a, _ = fitted_preprocessing(train, validation_a, labels, names)
    train_b, _, ranked_b, _ = fitted_preprocessing(train, validation_b, labels, names)
    assert np.array_equal(train_a, train_b)
    assert ranked_a == ranked_b


def test_outer_test_feature_guard_precedes_numeric_conversion() -> None:
    source = inspect.getsource(load_outer_training_features)
    guard = source.index("if run_key in forbidden_test_keys")
    conversion = source.index("values = [float")
    assert guard < conversion


def test_checkpoint_recovery_validation(tmp_path: Path) -> None:
    config = candidate_grid("onlinehd")[0]
    hashes = {"contract": "abc"}
    checkpoint = {
        "status": "COMPLETE", "canonical_config_json": json.dumps(config, sort_keys=True, separators=(",", ":")),
        "frozen_hashes": hashes, "inner_metrics": [{}, {}, {}],
    }
    path = tmp_path / "checkpoint.json"
    write_json(path, checkpoint)
    assert valid_checkpoint(path, config, hashes)


def test_frozen_candidate_counts_and_dtypes() -> None:
    assert len(candidate_grid("onlinehd")) == 24
    assert len(candidate_grid("multicentroid")) == 6
    assert len(candidate_grid("hybrid")) == 32
    assert {item["dimension"] for item in candidate_grid("onlinehd")} == {2000, 5000}
