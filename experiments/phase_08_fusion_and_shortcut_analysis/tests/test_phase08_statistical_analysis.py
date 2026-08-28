"""Unit tests for frozen Phase 08 subject-level statistical rules."""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_phase08_conditions import bootstrap_ci, holm, rank_biserial  # noqa: E402


class Phase08StatisticsTests(unittest.TestCase):
    def test_holm_is_monotone_in_sorted_p_order(self) -> None:
        raw = [0.04, 0.001, 0.02]
        adjusted = holm(raw)
        ordered = [adjusted[i] for i in np.argsort(raw)]
        self.assertTrue(all(a <= b for a, b in zip(ordered, ordered[1:])))
        self.assertTrue(all(a >= b for a, b in zip(adjusted, raw)))

    def test_rank_biserial_direction(self) -> None:
        self.assertAlmostEqual(rank_biserial(np.array([1, 2, 3], float)), 1.0)
        self.assertAlmostEqual(rank_biserial(np.array([-1, -2, -3], float)), -1.0)
        self.assertEqual(rank_biserial(np.zeros(4)), 0.0)

    def test_bootstrap_is_subject_level_and_deterministic(self) -> None:
        values = np.arange(35, dtype=float)
        first = bootstrap_ci(values, np.random.default_rng(42))
        second = bootstrap_ci(values, np.random.default_rng(42))
        self.assertEqual(first, second)
        self.assertLess(first[0], values.mean())
        self.assertGreater(first[1], values.mean())


if __name__ == "__main__":
    unittest.main()
