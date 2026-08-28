from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from consolidate_phase09_oof import aggregate_group  # noqa: E402


def test_hdc_classification_averages_scores_then_argmax() -> None:
    frame = pd.DataFrame({
        "run_key": ["r"] * 5, "subject_id": ["s"] * 5, "seed": [42, 43, 44, 45, 46],
        "y_true": [2] * 5, "class_score_0": [0.1] * 5, "class_score_1": [0.2] * 5,
        "class_score_2": [0.4, 0.4, 0.1, 0.1, 0.1], "class_score_3": [0.3, 0.3, 0.6, 0.6, 0.6],
    })
    result = aggregate_group(frame, "hdc_classification")
    assert result["seed_count"] == 5
    assert result["y_pred"] == 3
    assert result["aggregation"] == "FIVE_SEED_CLASS_SCORE_MEAN_ARGMAX"


def test_hdc_regression_averages_raw_then_clips() -> None:
    frame = pd.DataFrame({
        "run_key": ["r"] * 5, "subject_id": ["s"] * 5, "seed": [42, 43, 44, 45, 46],
        "y_true": [4.0] * 5, "y_pred_raw": [4.5, 4.6, 4.7, 4.8, 4.9],
    })
    result = aggregate_group(frame, "hdc_regression")
    assert result["y_pred_raw"] == pytest.approx(4.7)
    assert result["y_pred_bounded"] == 4.0
    assert result["clipped"] is True


def test_hdc_requires_exact_five_seed_set() -> None:
    frame = pd.DataFrame({
        "run_key": ["r"] * 5, "subject_id": ["s"] * 5, "seed": [42, 43, 44, 45, 45],
        "y_true": [1] * 5, "class_score_0": [1.0] * 5, "class_score_1": [0.0] * 5,
        "class_score_2": [0.0] * 5, "class_score_3": [0.0] * 5,
    })
    with pytest.raises(RuntimeError, match="Incomplete five-seed group"):
        aggregate_group(frame, "hdc_classification")
