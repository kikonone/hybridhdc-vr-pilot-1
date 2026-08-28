from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from run_phase07_unimodal_batch import (  # noqa: E402
    CONTRACT_PATH,
    EXPECTED_FOLDS,
    EXPECTED_PRIMARY,
    FOLDS,
    PRIMARY,
    checkpoint_path,
    enumerate_runs,
    fit_preprocessing,
    payload_digest,
    sha256,
)


def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_import_has_no_training_side_effect() -> None:
    assert contract()["status"] == "CONTRACT_FROZEN_NOT_TRAINED"


def test_frozen_checksums() -> None:
    assert sha256(PRIMARY) == EXPECTED_PRIMARY
    assert sha256(FOLDS) == EXPECTED_FOLDS


def test_dry_run_enumeration_is_exact_and_unique() -> None:
    runs = enumerate_runs(contract())
    run_ids = [run["run_id"] for run in runs]
    assert len(runs) == len(set(run_ids)) == 250
    assert sum(run["task"] == "classification" for run in runs) == 125
    assert sum(run["task"] == "regression" for run in runs) == 125
    assert all(sum(run["modality"] == modality for run in runs) == 50 for modality in [item["name"] for item in contract()["modalities"]])


def test_checkpoint_paths_have_frozen_granularity() -> None:
    run = enumerate_runs(contract())[0]
    path = checkpoint_path(run)
    assert run["modality"] in path.parts and run["task"] in path.parts
    assert f"fold_{run['outer_fold']}" in path.parts and f"seed_{run['seed']}" in path.parts


def test_checkpoint_payload_digest_excludes_only_digest_field() -> None:
    base = {"run_id": "x", "status": "COMPLETE"}
    with_digest = {**base, "checkpoint_payload_sha256": "ignored"}
    assert payload_digest(base) == payload_digest(with_digest)
    assert payload_digest({**base, "status": "FAIL"}) != payload_digest(base)


def test_fold_local_preprocessing_handles_missing_and_effective_k() -> None:
    rng = np.random.default_rng(42)
    train = rng.normal(size=(40, 42))
    test = rng.normal(size=(8, 42))
    train[:5, 0] = np.nan
    test[:, 0] = np.nan
    labels = np.tile(np.arange(4), 10)
    train_out, test_out, state, timing = fit_preprocessing(train, test, labels, [f"f{i}" for i in range(42)], 50)
    assert state["effective_feature_k"] <= 50
    assert state["effective_feature_k"] == state["post_variance_feature_count"]
    assert state["missing_indicator_feature_names"] == ["missingindicator_f0"]
    assert train_out.shape[1] == test_out.shape[1] == state["effective_feature_k"]
    assert np.isfinite(train_out).all() and np.isfinite(test_out).all()
    assert timing["preprocessing_seconds"] >= 0.0


def test_contract_contains_only_frozen_models_and_parameters() -> None:
    frozen = contract()
    assert frozen["classification"]["model"] == "HDC+OnlineHD Hybrid"
    assert frozen["classification"]["dimension"] == 5000
    assert frozen["regression"]["head"] == "COMMON_ENCODER_READOUT_BASELINE"
    assert frozen["regression"]["dimension"] == 10000
    assert frozen["regression"]["ridge_alpha"] == 0.01
    assert frozen["randomness"]["seeds"] == [42, 43, 44, 45, 46]
    assert len(frozen["classification"]["fold_structures"]) == 5
