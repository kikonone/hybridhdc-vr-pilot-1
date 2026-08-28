from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from components import data_access

UI = Path(__file__).resolve().parents[1]
DATA = UI / "data"


def _rows() -> list[dict[str, str]]:
    with (DATA / "frozen_dual_task_oof.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_manifest_sources_and_outputs_match_recorded_hashes() -> None:
    manifest = json.loads((DATA / "demo_data_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["source_files"] + manifest["output_files"]:
        path = Path(entry["path"])
        assert path.is_file(), path
        assert path.stat().st_size == entry["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_package_verification_passes() -> None:
    assert data_access.verify_package() == (True, "Local demo package verified")


def test_schema_is_exact_and_identifier_free() -> None:
    rows = _rows()
    assert tuple(rows[0]) == data_access.EXPECTED_COLUMNS
    forbidden = ("run_key", "subject", "session", "participant")
    assert not any(token in column.lower() for column in rows[0] for token in forbidden)


def test_both_tasks_cover_all_419_aligned_records() -> None:
    rows = _rows()
    assert len(rows) == 419
    assert [row["demo_id"] for row in rows] == [f"DEMO-{index:04d}" for index in range(1, 420)]
    classification_traversal = [(row["demo_id"], row["true_difficulty"], row["predicted_difficulty"]) for row in rows]
    regression_traversal = [(row["demo_id"], row["true_difficulty_score"], row["bounded_frozen_prediction"]) for row in rows]
    assert len(classification_traversal) == len(regression_traversal) == 419
    assert [item[0] for item in classification_traversal] == [item[0] for item in regression_traversal]
    assert {int(row["fold"]) for row in rows} == {1, 2, 3, 4, 5}


def test_no_missing_target_prediction_or_score_values() -> None:
    checked_fields = data_access.EXPECTED_COLUMNS[2:]
    for row in _rows():
        assert all(row[field].strip() for field in checked_fields)
        numeric = [float(row[field]) for field in checked_fields if field != "classification_correct"]
        assert all(math.isfinite(value) for value in numeric)


def test_classification_values_and_correctness() -> None:
    for row in _rows():
        true_value = int(row["true_difficulty"])
        prediction = int(row["predicted_difficulty"])
        assert true_value in range(1, 5)
        assert prediction in range(1, 5)
        assert row["classification_correct"] == str(true_value == prediction).lower()
        assert len([float(row[f"difficulty_{index}_cosine"]) for index in range(1, 5)]) == 4


def test_regression_bounded_values_are_canonical_and_absolute_error_is_derived() -> None:
    for row in _rows():
        target = float(row["true_difficulty_score"])
        bounded = float(row["bounded_frozen_prediction"])
        assert 1.0 <= target <= 4.0
        assert 1.0 <= bounded <= 4.0
        assert math.isclose(float(row["absolute_error"]), abs(target - bounded), rel_tol=0.0, abs_tol=1e-12)


def test_frozen_model_contract() -> None:
    model = data_access.load_model()
    assert model["classification"] == {
        "model": "HDC+OnlineHD Hybrid",
        "variant": "hybrid",
        "dimension": 5000,
        "feature_k": 50,
        "levels": 51,
        "aggregation": "arithmetic mean of the five seed class scores, then argmax",
        "score_semantics": "cosine similarities, not probabilities",
    }
    assert model["regression"]["model"] == "COMMON_ENCODER_READOUT_BASELINE"
    assert model["regression"]["variant"] == "common_ridge"
    assert model["regression"]["dimension"] == 10000
    assert model["regression"]["feature_k"] == 50
    assert model["regression"]["levels"] == 51
    assert model["regression"]["ridge_alpha"] == 0.01


def test_manifest_records_alignment_before_anonymization() -> None:
    manifest = json.loads((DATA / "demo_data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_counts"] == {"classification": 419, "regression": 419, "aligned": 419}
    assert manifest["alignment"]["performed_before_anonymization"] is True
    assert manifest["alignment"]["real_key_sets_equal"] is True
    assert manifest["alignment"]["stable_demo_id_range"] == ["DEMO-0001", "DEMO-0419"]
    assert manifest["ui_clipping_executed"] is False
    assert manifest["training_executed"] is False
    assert manifest["new_predictions_generated"] is False
    assert manifest["statistics_recomputed"] is False


def test_negative_missing_row_fails_closed() -> None:
    with pytest.raises(data_access.DataContractError, match="Expected 419"):
        data_access._validate_rows(_rows()[:-1])


def test_negative_duplicate_id_fails_closed() -> None:
    rows = _rows()
    rows[1] = {**rows[1], "demo_id": rows[0]["demo_id"]}
    with pytest.raises(data_access.DataContractError, match="Anonymous IDs"):
        data_access._validate_rows(rows)


def test_negative_missing_value_fails_closed() -> None:
    rows = _rows()
    rows[0] = {**rows[0], "raw_frozen_prediction": ""}
    with pytest.raises(data_access.DataContractError, match="Missing value"):
        data_access._validate_rows(rows)


def test_negative_null_value_fails_closed() -> None:
    rows = _rows()
    rows[0] = {**rows[0], "bounded_frozen_prediction": None}
    with pytest.raises(data_access.DataContractError, match="Missing value"):
        data_access._validate_rows(rows)


def test_negative_nan_fails_closed() -> None:
    rows = _rows()
    rows[0] = {**rows[0], "raw_frozen_prediction": "NaN"}
    with pytest.raises(data_access.DataContractError, match="Non-finite"):
        data_access._validate_rows(rows)


def test_negative_illegal_fold_fails_closed() -> None:
    rows = _rows()
    rows[0] = {**rows[0], "fold": "9"}
    with pytest.raises(data_access.DataContractError, match="Fold coverage"):
        data_access._validate_rows(rows)


def test_negative_illegal_difficulty_fails_closed() -> None:
    rows = _rows()
    rows[0] = {**rows[0], "true_difficulty": "5", "classification_correct": "false"}
    with pytest.raises(data_access.DataContractError, match="outside Difficulty 1-4"):
        data_access._validate_rows(rows)


def test_negative_illegal_numeric_type_fails_closed() -> None:
    rows = _rows()
    rows[0] = {**rows[0], "raw_frozen_prediction": "not-a-number"}
    with pytest.raises(data_access.DataContractError, match="Invalid numeric value"):
        data_access._validate_rows(rows)


def test_negative_out_of_range_bounded_value_fails_closed() -> None:
    rows = _rows()
    rows[0] = {**rows[0], "bounded_frozen_prediction": "4.1"}
    with pytest.raises(data_access.DataContractError, match="outside 1-4"):
        data_access._validate_rows(rows)


def test_negative_wrong_absolute_error_fails_closed() -> None:
    rows = _rows()
    rows[0] = {**rows[0], "absolute_error": "999"}
    with pytest.raises(data_access.DataContractError, match="absolute error mismatch"):
        data_access._validate_rows(rows)
