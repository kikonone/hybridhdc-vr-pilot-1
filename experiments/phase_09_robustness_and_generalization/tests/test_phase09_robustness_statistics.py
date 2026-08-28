from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from analyze_phase09_robustness import bootstrap_ci, holm_adjust, rank_biserial_signed  # noqa: E402


def test_holm_adjustment_is_monotone_in_sorted_p_order() -> None:
    raw = [0.04, 0.001, 0.03, 0.20, 0.05]
    adjusted = holm_adjust(raw)
    ordered = np.argsort(raw)
    assert np.all(np.diff(np.asarray(adjusted)[ordered]) >= 0)
    assert adjusted[1] == pytest.approx(0.005)
    assert all(0 <= value <= 1 for value in adjusted)


def test_signed_rank_biserial_direction() -> None:
    assert rank_biserial_signed(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert rank_biserial_signed(np.array([-1.0, -2.0, -3.0])) == pytest.approx(-1.0)
    assert rank_biserial_signed(np.zeros(3)) == 0.0


def test_bootstrap_is_deterministic_and_contains_constant() -> None:
    first = bootstrap_ci(np.full(35, 2.5), seed_offset=7)
    second = bootstrap_ci(np.full(35, 2.5), seed_offset=7)
    assert first == second == pytest.approx((2.5, 2.5))
