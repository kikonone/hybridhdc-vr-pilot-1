"""Expected coverage tests that are safe before and after execution."""

from __future__ import annotations

import collections
import json
import unittest
from pathlib import Path


PHASE09 = Path(__file__).resolve().parents[1]


class Phase09ExecutionCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((PHASE09 / "configs" / "phase09_execution_manifest.json").read_text(encoding="utf-8"))
        cls.records = cls.manifest["training_runs"]

    def test_run_counts(self) -> None:
        protocols = collections.Counter(record["protocol"] for record in self.records)
        models = collections.Counter(record["model_key"] for record in self.records)
        self.assertEqual(protocols, {"RETRAIN_WITHOUT_MODALITY": 300, "LEAVE_ONE_SUBJECT_OUT": 420})
        self.assertEqual(models, {"hdc_classification": 300, "hdc_regression": 300, "traditional_classification": 60, "traditional_regression": 60})

    def test_expected_raw_rows(self) -> None:
        self.assertEqual(sum(len(record["expected_test_run_keys"]) for record in self.records), 30168)
        self.assertEqual(len({record["run_identifier"] for record in self.records}), 720)

    def test_split_coverage(self) -> None:
        missing_conditions = {record["condition"] for record in self.records if record["protocol"] == "RETRAIN_WITHOUT_MODALITY"}
        loso_subjects = {record["loso_subject"] for record in self.records if record["protocol"] == "LEAVE_ONE_SUBJECT_OUT"}
        self.assertEqual(len(missing_conditions), 5)
        self.assertEqual(len(loso_subjects), 35)


if __name__ == "__main__":
    unittest.main()
