from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, ParameterGrid


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "src"))

from phase05_hdc_core import (  # noqa: E402
    EqualWidthQuantizer,
    bind,
    build_prototypes,
    bundle_bound_features,
    cosine_similarity_scores,
    incremental_encode_prefixes,
    naive_encode,
    ordered_level_codebook,
    predict_smallest_class_tie,
    stable_bipolar_vector,
)


def test_same_seed_and_configuration_are_exactly_reproducible() -> None:
    first = stable_bipolar_vector(42, "feature_identity", "feature_a")
    second = stable_bipolar_vector(42, "feature_identity", "feature_a")
    assert np.array_equal(first, second)


def test_different_seeds_create_different_codebooks() -> None:
    assert not np.array_equal(
        ordered_level_codebook(42, 21), ordered_level_codebook(43, 21)
    )


def test_identity_level_and_sample_hvs_are_bipolar() -> None:
    identity = stable_bipolar_vector(42, "feature_identity", "x", 64)
    levels = ordered_level_codebook(42, 5, 64)
    sample = bundle_bound_features([bind(identity, levels[0]), bind(identity, levels[1])], identity)
    for array in (identity, levels, sample):
        assert set(np.unique(array)) <= {-1, 1}


def test_ordered_levels_have_higher_adjacent_than_endpoint_similarity() -> None:
    levels = ordered_level_codebook(42, 21, 2_000).astype(np.float32)
    adjacent = np.mean(np.sum(levels[:-1] * levels[1:], axis=1) / 2_000)
    endpoint = float(np.sum(levels[0] * levels[-1]) / 2_000)
    assert adjacent > endpoint


def test_binding_and_bundling_follow_frozen_contract() -> None:
    identity = np.array([1, -1, 1, -1], dtype=np.int8)
    level = np.array([-1, -1, 1, 1], dtype=np.int8)
    assert np.array_equal(bind(identity, level), np.array([-1, 1, 1, -1], dtype=np.int8))
    tie = np.array([-1, 1, -1, 1], dtype=np.int8)
    bundled = bundle_bound_features(
        [np.array([1, 1, -1, -1], dtype=np.int8), np.array([1, -1, -1, 1], dtype=np.int8)],
        tie,
    )
    assert np.array_equal(bundled, np.array([1, 1, -1, 1], dtype=np.int8))


def test_zero_bundle_uses_fixed_tie_vector() -> None:
    vector = np.array([1, -1, 1, -1], dtype=np.int8)
    tie = np.array([-1, -1, 1, 1], dtype=np.int8)
    result = bundle_bound_features([vector, -vector], tie)
    assert np.array_equal(result, tie)


def test_2000_dimension_is_prefix_of_same_10000_codebook() -> None:
    full = stable_bipolar_vector(42, "feature_identity", "x", 10_000)
    prefix = stable_bipolar_vector(42, "feature_identity", "x", 2_000)
    assert np.array_equal(prefix, full[:2_000])


def test_quantizer_uses_fit_minimum_and_maximum_only() -> None:
    train = np.array([[0.0, 10.0], [2.0, 20.0]])
    validation = np.array([[-100.0, 1_000.0]])
    quantizer = EqualWidthQuantizer(5).fit(train)
    assert np.array_equal(quantizer.minimum_, np.array([0.0, 10.0]))
    assert np.array_equal(quantizer.maximum_, np.array([2.0, 20.0]))
    quantizer.transform(validation)
    assert np.array_equal(quantizer.minimum_, np.array([0.0, 10.0]))
    assert np.array_equal(quantizer.maximum_, np.array([2.0, 20.0]))


def test_validation_values_outside_training_range_clip() -> None:
    quantizer = EqualWidthQuantizer(5).fit(np.array([[0.0], [4.0]]))
    transformed = quantizer.transform(np.array([[-1.0], [5.0]]))
    assert transformed[:, 0].tolist() == [0, 4]


def test_transform_does_not_change_fitted_state() -> None:
    quantizer = EqualWidthQuantizer(5).fit(np.array([[0.0], [4.0]]))
    before = quantizer.state_digest()
    quantizer.transform(np.array([[2.0], [3.0]]))
    assert quantizer.state_digest() == before


def test_incremental_encoding_equals_naive_reference() -> None:
    quantized = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int16)
    names = ["a", "b", "c"]
    cached = incremental_encode_prefixes(quantized, names, 3, 42, ["all"], 64)
    naive = naive_encode(quantized, names, 3, 42, 64)
    assert np.array_equal(cached.samples_by_k["all"], naive)


def test_frozen_candidate_count_is_16() -> None:
    candidates = list(ParameterGrid({"dimension": [2_000, 5_000], "levels": [21, 51], "k": [50, 100, 200, "all"], "seed": [42]}))
    assert len(candidates) == 16


def test_prototypes_depend_only_on_supplied_training_samples() -> None:
    train_hv = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]], dtype=np.int8)
    labels = np.array([0, 1, 2, 3])
    expected = train_hv.astype(np.int32)
    assert np.array_equal(build_prototypes(train_hv, labels), expected)


def test_inner_train_validation_subject_overlap_is_zero() -> None:
    groups = np.repeat(np.arange(9), 2)
    rows = np.zeros((len(groups), 1))
    for train_index, validation_index in GroupKFold(3).split(rows, groups=groups):
        assert not (set(groups[train_index]) & set(groups[validation_index]))


def test_classification_tie_selects_smallest_class_index() -> None:
    similarities = np.array([[0.5, 0.5, 0.2, -0.1]], dtype=np.float32)
    assert predict_smallest_class_tie(similarities).tolist() == [0]


def test_cosine_similarity_is_finite_and_bounded() -> None:
    samples = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]], dtype=np.int8)
    prototypes = np.array([[1, -1, 1, -1], [-1, 1, -1, 1], [1, 1, -1, -1], [-1, -1, 1, 1]], dtype=np.int32)
    scores = cosine_similarity_scores(samples, prototypes)
    assert np.isfinite(scores).all()
    assert np.all(scores >= -1.0) and np.all(scores <= 1.0)
