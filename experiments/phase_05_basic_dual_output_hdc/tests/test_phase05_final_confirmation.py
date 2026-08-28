from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))

from run_vanilla_hdc_final_confirmation import (  # noqa: E402
    choose_parameter,
    config_id,
    regression_metrics,
    similarity_prediction,
)


def test_similarity_prediction_is_bounded_and_ordered() -> None:
    similarities = np.asarray([[1.0, 0.0, -1.0, -2.0], [-2.0, -1.0, 0.0, 1.0]])
    predictions = similarity_prediction(similarities, 0.5)
    assert np.all((predictions >= 1.0) & (predictions <= 4.0))
    assert predictions[0] < predictions[1]


def test_parameter_tie_break_prefers_larger_value() -> None:
    rows = [
        {"mean_inner_mae": 0.2, "std_inner_mae": 0.1, "temperature": 0.1, "frozen_candidate_order": 0},
        {"mean_inner_mae": 0.2, "std_inner_mae": 0.1, "temperature": 2.0, "frozen_candidate_order": 1},
    ]
    assert choose_parameter(rows, "temperature") == 2.0


def test_regression_metrics_reports_raw_and_bounded() -> None:
    target = np.asarray([1.0, 2.0, 3.0, 4.0])
    raw = np.asarray([0.0, 2.0, 3.0, 5.0])
    bounded = np.clip(raw, 1.0, 4.0)
    metrics = regression_metrics(target, raw, bounded)
    assert metrics["mae_raw"] == 0.5
    assert metrics["mae_bounded"] == 0.0
    assert metrics["rmse_bounded"] == 0.0


def test_config_id_is_independent_per_fold_dimension_seed() -> None:
    assert config_id(3, 5000, 44) == "fold_3_dimension_5000_seed_44"
