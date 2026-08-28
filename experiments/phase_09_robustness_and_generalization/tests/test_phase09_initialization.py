"""Static tests for the Phase 09 audit-only initialization contract."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PHASE09 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE09 / "scripts"))

from initialize_phase09 import EXPECTED_FOLD_SHA256  # noqa: E402


class Phase09InitializationTests(unittest.TestCase):
    @staticmethod
    def load(relative_path: str) -> dict:
        return json.loads((PHASE09 / relative_path).read_text(encoding="utf-8"))

    def test_input_and_frozen_fold(self) -> None:
        audit = self.load("audits/phase09_input_and_fold_audit.json")
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["actual"]["fold_sha256"], EXPECTED_FOLD_SHA256)
        self.assertTrue(audit["checks"]["outer_subject_isolation"])

    def test_all_upstream_freeze_interfaces(self) -> None:
        audit = self.load("audits/phase09_upstream_freeze_audit.json")
        self.assertEqual(audit["status"], "PASS")
        self.assertTrue(all(audit["interface_results"].values()))

    def test_modality_partition(self) -> None:
        audit = self.load("audits/phase09_modality_coverage_audit.json")
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["union_count"], 1176)
        self.assertEqual(audit["overlap_count"], 0)

    def test_loso_feasibility(self) -> None:
        audit = self.load("audits/phase09_loso_feasibility_audit.json")
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["split_count"], 35)
        self.assertTrue(audit["checks"]["subject_isolation"])
        self.assertTrue(audit["checks"]["training_four_class_coverage"])
        self.assertTrue(audit["checks"]["test_four_class_coverage"])

    def test_scope_and_prohibited_operations(self) -> None:
        scope = self.load("configs/phase09_generalization_scope.json")
        contract = self.load("configs/phase09_experiment_contract_draft.json")
        self.assertEqual(scope["unseen_scenario_generalization"], "NOT_FEASIBLE_DUE_TO_METADATA")
        self.assertEqual(scope["flight_generalizable_behavior_claim"], "INCONCLUSIVE_DUE_TO_METADATA")
        self.assertEqual(contract["status"], "PENDING_CONTRACT_FREEZE")
        self.assertFalse(contract["training_authorized"])
        self.assertFalse(contract["prediction_generation_authorized"])


if __name__ == "__main__":
    unittest.main()
