"""Static tests for the Phase 08 frozen contract. No modeling is imported or run."""

from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT.parent
P3 = EXPERIMENTS / "phase_03_multimodal_dataset_labeling"
EXPECTED_HASHES = {
    "primary_without_performance.csv": "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44",
    "auxiliary_with_performance.csv": "72977a2119e30e37996fb9f0e3404988c4977fb7d2b33992f87bf54bfe5decba",
    "performance_only.csv": "d602282ae41153886d1306494515f2e41a5e7e89a2cec5c192d44b9ca87a07a4",
    "fold_assignments.csv": "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f",
}


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class Phase08ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load("configs/phase08_frozen_contract.json")
        cls.matrix = load("configs/phase08_model_matrix.json")
        cls.execution = load("configs/phase08_execution_manifest.json")
        cls.provenance = load("manifests/phase08_flight_feature_provenance_manifest.json")
        cls.provenance_audit = load("audits/phase08_flight_feature_provenance_audit.json")
        cls.handoff = load("manifests/phase08_to_phase09_generalization_handoff.json")

    def test_data_and_fold_checksums(self) -> None:
        data_dir = P3 / "data"
        for name, expected in EXPECTED_HASHES.items():
            self.assertEqual(sha256(data_dir / name), expected)

    def test_run_key_target_fold_alignment(self) -> None:
        data_dir = P3 / "data"
        names = ["primary_without_performance.csv", "auxiliary_with_performance.csv", "performance_only.csv", "fold_assignments.csv"]
        tables = {name: csv_rows(data_dir / name) for name in names}
        keys = [{row["run_key"] for row in rows} for rows in tables.values()]
        self.assertTrue(all(key_set == keys[0] for key_set in keys[1:]))
        fields = ["subject_id", "target_class", "target_score", "outer_fold"]
        reference = {row["run_key"]: tuple(row[field] for field in fields) for row in tables[names[0]]}
        for name in names[1:]:
            observed = {row["run_key"]: tuple(row[field] for field in fields) for row in tables[name]}
            self.assertEqual(observed, reference)

    def test_fusion_and_performance_counts(self) -> None:
        fusion = load("manifests/phase08_fusion_feature_manifest.json")["combinations"]
        self.assertEqual([fusion[name]["feature_count"] for name in ("physiological_plus_eye", "physiological_plus_eye_plus_head", "physiological_plus_eye_plus_head_plus_flight", "full_multimodal_without_performance")], [649, 808, 1134, 1176])
        self.assertEqual(load("manifests/phase08_performance_feature_manifest.json")["feature_count"], 59)

    def test_flight_provenance_complete_and_exclusive(self) -> None:
        records = self.provenance["features"]
        allowed = {"BEHAVIORAL_RESPONSE", "TASK_SETTING_OR_SCENARIO", "AMBIGUOUS"}
        self.assertEqual(len(records), 326)
        self.assertEqual(len({record["feature_name"] for record in records}), 326)
        self.assertTrue(all(record["semantic_category"] in allowed for record in records))
        required = {"feature_name", "base_variable", "aggregation_statistic", "source_stream", "source_field", "provenance_evidence", "semantic_category", "rationale", "confidence", "ambiguous_reason"}
        self.assertTrue(all(required.issubset(record) for record in records))
        self.assertEqual(sum(self.provenance["category_counts"].values()), 326)
        self.assertEqual(self.provenance_audit["status"], "PASS")

    def test_provenance_sensitivity_subsets(self) -> None:
        records = self.provenance["features"]
        behavioral = {record["feature_name"] for record in records if record["semantic_category"] == "BEHAVIORAL_RESPONSE"}
        task = {record["feature_name"] for record in records if record["semantic_category"] == "TASK_SETTING_OR_SCENARIO"}
        ambiguous = {record["feature_name"] for record in records if record["semantic_category"] == "AMBIGUOUS"}
        self.assertFalse(behavioral & task)
        self.assertFalse(ambiguous & (behavioral | task))
        conditions = self.matrix["flight_sensitivity_conditions"]
        self.assertEqual(conditions["FLIGHT_BEHAVIORAL_ONLY"]["feature_count"], len(behavioral))
        self.assertEqual(conditions["FLIGHT_TASK_SETTING_ONLY"]["feature_count"], len(task))
        self.assertIn(conditions["FLIGHT_BEHAVIORAL_ONLY"]["status"], {"AUTHORIZED_UNIQUE_SENSITIVITY_CONDITION", "REFERENCE_ALIAS_NO_DUPLICATE_TRAINING", "NOT_FEASIBLE_EMPTY_PROVENANCE_GROUP"})
        self.assertIn(conditions["FLIGHT_TASK_SETTING_ONLY"]["status"], {"AUTHORIZED_UNIQUE_SENSITIVITY_CONDITION", "REFERENCE_ALIAS_NO_DUPLICATE_TRAINING", "NOT_FEASIBLE_EMPTY_PROVENANCE_GROUP"})

    def test_hdc_frozen_parameters(self) -> None:
        hdc = self.matrix["HDC"]
        self.assertEqual((hdc["classification"]["dimension"], hdc["classification"]["levels"], hdc["classification"]["seeds"], hdc["classification"]["requested_feature_k"]), (5000, 51, [42, 43, 44, 45, 46], 50))
        self.assertEqual(len(hdc["classification"]["fold_specific_structures"]), 5)
        self.assertEqual((hdc["regression"]["dimension"], hdc["regression"]["levels"], hdc["regression"]["seeds"], hdc["regression"]["ridge_alpha"]), (10000, 51, [42, 43, 44, 45, 46], 0.01))

    def test_traditional_frozen_interfaces(self) -> None:
        traditional = self.matrix["traditional"]
        self.assertEqual(traditional["classification"]["model"], "Gradient Boosting")
        self.assertEqual(traditional["regression"]["model"], "Gradient Boosting Regressor")
        self.assertEqual(traditional["classification"]["random_state"], 42)
        self.assertEqual(traditional["regression"]["random_state"], 42)
        self.assertEqual(len(traditional["classification"]["fold_specific_parameters"]), 5)
        self.assertEqual(len(traditional["regression"]["fold_specific_parameters"]), 5)
        self.assertFalse(traditional["classification"]["parameter_search_authorized"])
        self.assertFalse(traditional["regression"]["parameter_search_authorized"])

    def test_dynamic_run_count_and_unique_identifiers(self) -> None:
        counts = self.matrix["run_counts"]
        expected = 300 + counts["traditional_flight_full_required_runs"] + 60 * counts["unique_nonempty_nonaliased_flight_sensitivity_conditions"]
        self.assertEqual(counts["expected_total_model_runs"], expected)
        self.assertEqual(len(self.execution["run_records"]), expected)
        run_ids = [record["run_id"] for record in self.execution["run_records"]]
        self.assertEqual(len(run_ids), len(set(run_ids)))
        self.assertEqual(self.execution["duplicate_run_identifiers"], 0)

    def test_oof_comparison_and_statistics_frozen(self) -> None:
        self.assertEqual(self.contract["oof_aggregation"]["status"], "FROZEN")
        self.assertEqual(self.contract["comparison_families"]["status"], "FROZEN_SEPARATE_FAMILIES")
        statistics = load("configs/phase08_statistical_analysis_contract.json")
        self.assertEqual((statistics["status"], statistics["statistical_unit"], statistics["n"]), ("FROZEN", "subject_id", 35))
        self.assertEqual(statistics["bootstrap"], {"repetitions": 2000, "seed": 42, "ci": "percentile 95%", "resampling": "paired subjects"})

    def test_shortcut_rules_and_phase09_handoff(self) -> None:
        shortcut = load("configs/phase08_shortcut_evidence_contract.json")
        self.assertEqual(shortcut["status"], "FROZEN")
        self.assertIsNone(shortcut["automatic_leakage_threshold"])
        self.assertFalse(shortcut["high_performance_alone_equals_leakage"])
        self.assertTrue((ROOT / "manifests/phase08_to_phase09_generalization_handoff.json").is_file())
        self.assertFalse(self.handoff["phase09_directory_created"])
        self.assertFalse(self.handoff["holdout_executed"])

    def test_contract_freeze_had_no_training_and_execution_did_not_modify_upstream(self) -> None:
        audit = load("audits/phase08_contract_freeze_audit.json")
        artifact = load("audits/phase08_contract_artifact_audit.json")
        self.assertEqual(audit["training_artifacts_added"], 0)
        self.assertFalse(audit["training_executed"])
        self.assertFalse(audit["outer_test_predictions_generated"])
        self.assertEqual(artifact["upstream_files_modified"], 0)
        execution = load("configs/phase08_execution_manifest.json")
        result_files = sum(item.is_file() for item in (ROOT / "results").rglob("*"))
        if execution["status"] == "CONTRACT_FROZEN_NOT_TRAINED":
            self.assertEqual(result_files, 0)
        elif execution["status"] == "EXECUTION_COMPLETE_PENDING_OOF_CONSOLIDATION":
            self.assertEqual(execution["status"], "EXECUTION_COMPLETE_PENDING_OOF_CONSOLIDATION")
            self.assertEqual(execution["completed_runs"], 370)
            self.assertEqual(result_files, 1110)
        else:
            self.assertEqual(execution["status"], "FROZEN")
            self.assertEqual(execution["completed_runs"], 370)
            self.assertEqual(execution["canonical_oof_rows"], 10894)
            self.assertGreaterEqual(result_files, 1110)


if __name__ == "__main__":
    unittest.main(verbosity=2)
