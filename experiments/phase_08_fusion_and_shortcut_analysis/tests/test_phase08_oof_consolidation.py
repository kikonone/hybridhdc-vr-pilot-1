"""Read-only tests for canonical Phase 08 OOF consolidation."""

import sys
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from consolidate_phase08_oof import EXPECTED_CANONICAL_ROWS, aggregate_group, preflight, read_json  # noqa: E402


class Phase08OOFTests(unittest.TestCase):
    def test_preflight_recomputes_all_prediction_checkpoint_hashes(self) -> None:
        gate = preflight()
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["artifact_hashes_recomputed"], 740)
        self.assertEqual(gate["raw_prediction_rows"], 31006)

    def test_authorized_combinations_aggregate_to_419(self) -> None:
        runs = read_json(ROOT / "configs/phase08_execution_manifest.json")["run_records"]
        grouped = defaultdict(list)
        for run in runs:
            grouped[(run["condition"], run["model_family"], run["task"])].append(run)
        self.assertEqual(len(grouped), 26)
        for records in grouped.values():
            frame = aggregate_group(records)
            self.assertEqual((len(frame), frame.run_key.nunique()), (419, 419))
        self.assertEqual(len(grouped) * 419, EXPECTED_CANONICAL_ROWS)

    def test_task_setting_condition_is_not_fabricated(self) -> None:
        runs = read_json(ROOT / "configs/phase08_execution_manifest.json")["run_records"]
        self.assertNotIn("FLIGHT_TASK_SETTING_ONLY", {x["condition"] for x in runs})


if __name__ == "__main__":
    unittest.main()
