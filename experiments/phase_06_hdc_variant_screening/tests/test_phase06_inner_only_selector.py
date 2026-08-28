from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/select_phase06_models_from_inner_evidence.py"
SPEC = importlib.util.spec_from_file_location("phase06_inner_selector", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_outer_oof_is_rejected() -> None:
    with pytest.raises(PermissionError):
        MODULE.assert_allowed(MODULE.PHASE / "results/oof/phase06_onlinehd_final_oof.csv")


def test_outer_prediction_is_rejected() -> None:
    with pytest.raises(PermissionError):
        MODULE.assert_allowed(MODULE.PHASE / "results/predictions/onlinehd_final_confirmation_fold_1_predictions.csv")


def test_allowlisted_inner_selection_is_accepted() -> None:
    MODULE.assert_allowed(MODULE.PHASE / "results/fold_metrics/onlinehd_final_confirmation_fold_1_inner_selection.csv")


def test_pareto_is_deterministic() -> None:
    frame = pd.DataFrame({"score": [0.8, 0.7, 0.9], "mean_inner_measured_runtime_seconds": [2.0, 1.0, 3.0], "mean_model_bytes": [20.0, 10.0, 30.0]})
    first = MODULE.nondominated(frame, "score", True).to_numpy()
    second = MODULE.nondominated(frame, "score", True).to_numpy()
    assert np.array_equal(first, second)
