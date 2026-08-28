"""Unit and read-only integration tests for the Phase 08 final freeze."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import freeze_phase08 as freeze


class Phase08FreezeTests(unittest.TestCase):
    def test_preflight_passes_without_writes(self) -> None:
        before = freeze.FREEZE_FILE.exists()
        result = freeze.preflight()
        if before:
            self.assertTrue(all(value for name, value in result["checks"].items() if name != "freeze_not_already_present"), result)
            self.assertFalse(result["checks"]["freeze_not_already_present"])
        else:
            self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(freeze.FREEZE_FILE.exists(), before)

    def test_exact_counts_and_coverage(self) -> None:
        result = freeze.preflight()
        self.assertEqual(result["actual"]["model_runs"], 370)
        self.assertEqual(result["actual"]["raw_prediction_rows"], 31006)
        self.assertEqual(result["actual"]["canonical_oof_rows"], 10894)
        self.assertTrue(result["checks"]["canonical_run_key_coverage"])

    def test_checksums_and_upstream_integrity(self) -> None:
        checks = freeze.preflight()["checks"]
        for name in ("primary_checksum", "with_performance_checksum", "performance_only_checksum", "frozen_fold_checksum", "upstream_freeze_integrity"):
            self.assertTrue(checks[name], name)

    def test_manifest_has_no_duplicate_paths_or_self_hash(self) -> None:
        pre = freeze.preflight()
        manifest = freeze.build_manifest("2000-01-01T00:00:00+00:00", pre)
        self.assertFalse(manifest["self_hash_included"])
        self.assertEqual(manifest["duplicate_artifacts"], [])
        self.assertEqual(manifest["missing_artifacts"], [])
        self.assertEqual(manifest["hash_mismatches"], [])

    def test_manifest_builder_is_reproducible(self) -> None:
        pre = freeze.preflight(); stamp = "2000-01-01T00:00:00+00:00"
        self.assertEqual(freeze.build_manifest(stamp, pre), freeze.build_manifest(stamp, pre))

    def test_freeze_payload_has_guardrails(self) -> None:
        payload = freeze.freeze_payload("2000-01-01T00:00:00+00:00", "a" * 64)
        self.assertEqual(payload["status"], "FROZEN")
        self.assertFalse(payload["phase09_executed"])
        self.assertIn("do not prove", payload["generalization_guardrail"])


if __name__ == "__main__":
    unittest.main()
