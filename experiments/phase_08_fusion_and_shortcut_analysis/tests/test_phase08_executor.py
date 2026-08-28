"""Static executor tests; no model fit or prediction is invoked."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_phase08_batch import (  # noqa: E402
    EXPECTED_TOTAL_RUNS,
    checkpoint_path,
    feature_sets,
    fit_traditional_preprocessing,
    load_locked_inputs,
    metrics_path,
    payload_digest,
    prediction_path,
)


class Phase08ExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.execution, cls.matrix, cls.runs = load_locked_inputs()

    def test_import_has_no_training_side_effect(self) -> None:
        self.assertIn(self.execution["status"], {"AUTHORIZED_NOT_EXECUTED", "EXECUTION_RUNNING", "EXECUTION_COMPLETE_PENDING_OOF_CONSOLIDATION", "ANALYSIS_COMPLETE_PENDING_FREEZE", "FROZEN"})

    def test_exact_unique_run_matrix(self) -> None:
        ids = [run["run_id"] for run in self.runs]
        self.assertEqual(len(ids), EXPECTED_TOTAL_RUNS)
        self.assertEqual(len(set(ids)), EXPECTED_TOTAL_RUNS)
        counts = {
            "hc": sum(run["model_family"] == "HDC" and run["task"] == "classification" for run in self.runs),
            "hr": sum(run["model_family"] == "HDC" and run["task"] == "regression" for run in self.runs),
            "tc": sum(run["model_family"] == "TRADITIONAL" and run["task"] == "classification" for run in self.runs),
            "tr": sum(run["model_family"] == "TRADITIONAL" and run["task"] == "regression" for run in self.runs),
        }
        self.assertEqual(counts, {"hc": 150, "hr": 150, "tc": 35, "tr": 35})

    def test_condition_counts_and_infeasible_exclusion(self) -> None:
        expected = {"FUSION_PE": 60, "FUSION_PEH": 60, "FUSION_PEHF": 60, "WITH_PERFORMANCE_AUXILIARY": 60, "PERFORMANCE_ONLY_AUXILIARY": 60, "FLIGHT_BEHAVIORAL_ONLY": 60, "FLIGHT_FULL": 10}
        self.assertEqual({condition: sum(run["condition"] == condition for run in self.runs) for condition in expected}, expected)
        self.assertFalse(any(run["condition"] == "FLIGHT_TASK_SETTING_ONLY" for run in self.runs))
        self.assertFalse(any("phase09" in json.dumps(run).casefold() for run in self.runs))

    def test_feature_sets_are_frozen_and_unique(self) -> None:
        features = feature_sets()
        self.assertEqual({name: len(value) for name, value in features.items()}, {"FUSION_PE": 649, "FUSION_PEH": 808, "FUSION_PEHF": 1134, "WITH_PERFORMANCE_AUXILIARY": 1235, "PERFORMANCE_ONLY_AUXILIARY": 59, "FLIGHT_BEHAVIORAL_ONLY": 323, "FLIGHT_FULL": 326})
        self.assertTrue(all(len(value) == len(set(value)) for value in features.values()))

    def test_output_paths_are_collision_free_and_granular(self) -> None:
        paths = []
        for run in self.runs:
            current = [checkpoint_path(run), prediction_path(run), metrics_path(run)]
            paths.extend(map(str, current))
            self.assertIn(run["condition"], current[0].parts)
            self.assertIn(run["model_family"], current[0].parts)
            self.assertIn(run["task"], current[0].parts)
            self.assertIn(f"fold_{run['outer_fold']}", current[0].parts)
        self.assertEqual(len(paths), len(set(paths)))

    def test_payload_digest_excludes_only_digest(self) -> None:
        base = {"run_id": "x", "status": "COMPLETE"}
        self.assertEqual(payload_digest(base), payload_digest({**base, "checkpoint_payload_sha256": "ignored"}))
        self.assertNotEqual(payload_digest(base), payload_digest({**base, "status": "FAIL"}))

    def test_traditional_effective_k_is_fold_local_and_safe(self) -> None:
        rng = np.random.default_rng(42)
        train = rng.normal(size=(40, 20))
        test = rng.normal(size=(8, 20))
        train[:5, 0] = np.nan
        test[:, 0] = np.nan
        labels = np.tile(np.arange(4), 10)
        train_out, test_out, state = fit_traditional_preprocessing(train, test, labels, [f"f{i}" for i in range(20)], 200, "classification")
        self.assertEqual(state["effective_feature_k"], state["post_variance_feature_count"])
        self.assertLessEqual(state["effective_feature_k"], 200)
        self.assertTrue(np.isfinite(train_out).all() and np.isfinite(test_out).all())

    def test_frozen_model_interfaces(self) -> None:
        hdc = self.matrix["HDC"]
        traditional = self.matrix["traditional"]
        self.assertEqual((hdc["classification"]["dimension"], hdc["classification"]["levels"], hdc["classification"]["seeds"]), (5000, 51, [42, 43, 44, 45, 46]))
        self.assertEqual((hdc["regression"]["dimension"], hdc["regression"]["ridge_alpha"]), (10000, 0.01))
        self.assertEqual(traditional["classification"]["model"], "Gradient Boosting")
        self.assertEqual(traditional["regression"]["model"], "Gradient Boosting Regressor")
        self.assertFalse(traditional["classification"]["parameter_search_authorized"])
        self.assertFalse(traditional["regression"]["parameter_search_authorized"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
