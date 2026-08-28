"""Static Contract Freeze tests; no estimator is imported or executed."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


PHASE09 = Path(__file__).resolve().parents[1]
class Phase09ContractTests(unittest.TestCase):
    @staticmethod
    def load(relative_path: str) -> dict:
        return json.loads((PHASE09 / relative_path).read_text(encoding="utf-8"))

    def test_initialization_and_upstream_gate(self) -> None:
        audit = self.load("audits/phase09_contract_freeze_audit.json")
        self.assertEqual(audit["status"], "PASS")
        self.assertTrue(audit["checks"]["primary_checksum"])
        self.assertTrue(audit["checks"]["fold_checksum"])
        self.assertTrue(audit["checks"]["interfaces_all_pass"])
        self.assertTrue(audit["checks"]["upstream_files_modified_0"])

    def test_missing_modality_contract(self) -> None:
        contract = self.load("configs/phase09_missing_modality_contract.json")
        audit = self.load("audits/phase09_missing_modality_contract_audit.json")
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(len(contract["conditions"]), 6)
        self.assertEqual(contract["new_training_runs"], 300)
        self.assertEqual(contract["full_primary_reference_policy"], "REUSED_NOT_RETRAINED")

    def test_checkpoint_portability_is_not_fabricated(self) -> None:
        audit = self.load("audits/phase09_checkpoint_portability_audit.json")
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["protocol_status"], "NOT_FEASIBLE_DUE_TO_CHECKPOINT_INTERFACE")
        self.assertFalse(audit["training_executed"])
        self.assertFalse(audit["predictions_generated"])

    def test_loso_assignment_and_config_mapping(self) -> None:
        assignment = self.load("audits/phase09_loso_assignment_audit.json")
        leakage = self.load("audits/phase09_config_mapping_leakage_audit.json")
        mapping = self.load("configs/phase09_loso_config_mapping.json")
        self.assertEqual(assignment["status"], "PASS")
        self.assertEqual(assignment["splits"], 35)
        self.assertEqual(assignment["assignment_rows"], 419)
        self.assertEqual(leakage["status"], "PASS")
        self.assertEqual(len(mapping["mappings"]), 35)
        self.assertTrue(all(item["test_subject_excluded_from_original_config_selection_evidence"] for item in mapping["mappings"]))

    def test_dynamic_720_run_matrix(self) -> None:
        execution = self.load("configs/phase09_execution_manifest.json")
        audit = self.load("audits/phase09_run_matrix_audit.json")
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(execution["training_run_count"], 720)
        self.assertEqual(execution["duplicate_run_identifiers"], 0)
        self.assertEqual(execution["run_counts_by_protocol"]["RETRAIN_WITHOUT_MODALITY"], 300)
        self.assertEqual(execution["run_counts_by_protocol"]["LEAVE_ONE_SUBJECT_OUT"], 420)
        self.assertEqual(execution["run_counts_by_model"]["hdc_classification"], 300)
        self.assertEqual(execution["run_counts_by_model"]["hdc_regression"], 300)
        self.assertEqual(execution["run_counts_by_model"]["traditional_classification"], 60)
        self.assertEqual(execution["run_counts_by_model"]["traditional_regression"], 60)
        self.assertTrue(all(item["status"] == "COMPLETE_AUDITED" for item in execution["training_runs"]))

    def test_statistics_and_guardrails(self) -> None:
        statistics = self.load("configs/phase09_statistical_rules.json")
        guardrails = self.load("audits/phase09_generalization_guardrail_audit.json")
        self.assertEqual(statistics["statistical_unit"], "subject_id")
        self.assertEqual(statistics["missing_modality"]["bootstrap"]["resamples"], 2000)
        self.assertEqual(guardrails["status"], "PASS")
        self.assertEqual(guardrails["not_feasible"]["unseen_scenario"], "NOT_FEASIBLE_DUE_TO_METADATA")


if __name__ == "__main__":
    unittest.main()
