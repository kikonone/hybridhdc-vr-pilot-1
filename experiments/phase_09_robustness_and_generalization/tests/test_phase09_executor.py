"""Pre-training tests for the Phase 09 executor contract."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PHASE09 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE09 / "scripts"))

from run_phase09_batch import dry_run, output_paths, read_json  # noqa: E402


class Phase09ExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.audit = dry_run(Path(cls.temp_dir.name) / "phase09_executor_validation_audit.json")
        cls.records = read_json(PHASE09 / "configs" / "phase09_execution_manifest.json")["training_runs"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_dry_run_exactly_720(self) -> None:
        self.assertEqual(self.audit["status"], "PASS")
        self.assertEqual(self.audit["dry_run_unique_runs"], 720)
        self.assertEqual(self.audit["expected_raw_prediction_rows"], 30168)

    def test_forbidden_protocols_absent(self) -> None:
        self.assertFalse(any(record["condition"] == "FULL_PRIMARY_REFERENCE" for record in self.records))
        self.assertFalse(any("SUDDEN" in str(record) for record in self.records))
        self.assertFalse(any("phase10" in str(record).lower() for record in self.records))

    def test_output_paths_are_unique_and_nested(self) -> None:
        outputs = [output_paths(record) for record in self.records]
        for role in ["checkpoint", "model", "audit", "prediction", "metrics"]:
            values = [str(item[role]) for item in outputs]
            self.assertEqual(len(values), len(set(values)))
        self.assertTrue(all("results\\checkpoints" in str(item["checkpoint"]) for item in outputs))


if __name__ == "__main__":
    unittest.main()
