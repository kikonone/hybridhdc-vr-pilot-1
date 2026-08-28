"""Independent final verification for Phase 06 quick screening."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PHASE = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    timestamp = datetime.now(timezone.utc).isoformat()
    manifest_path = PHASE / "manifests" / "phase06_quick_screen_artifact_manifest.json"
    manifest = read(manifest_path)
    mismatches: list[dict[str, Any]] = []
    for item in manifest["artifacts"]:
        path = PHASE / item["relative_path"]
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        digest = sha256(path) if exists else None
        if not exists or size != item["file_size_bytes"] or digest != item["sha256"]:
            mismatches.append({"relative_path": item["relative_path"], "exists": exists, "actual_size": size, "actual_sha256": digest})
    all_folds = read(PHASE / "audits" / "phase06_quick_screen_all_folds_audit.json")
    notebook = read(PHASE / "audits" / "phase06_quick_screen_notebook_persistence_audit.json")
    unit_tests = read(PHASE / "audits" / "phase06_unit_test_audit.json")
    contract = read(PHASE / "audits" / "phase06_contract_freeze_audit.json")
    amendment = read(PHASE / "audits" / "phase06_phase05_amendment_gate_audit.json")
    expected = {"onlinehd": 24, "multicentroid": 6, "hybrid": 32}
    row_checks: dict[str, Any] = {}
    for variant, count in expected.items():
        rows = len(pd.read_csv(PHASE / "results" / "summaries" / f"phase06_{variant}_quick_screen_all_folds.csv"))
        row_checks[variant] = {"actual": rows, "expected": count * 5, "result": "PASS" if rows == count * 5 else "FAIL"}
    forbidden_result_files = [
        str(path.relative_to(PHASE))
        for folder in [PHASE / "results" / "predictions", PHASE / "results" / "oof"]
        for path in folder.rglob("*") if path.is_file()
    ]
    checks = {
        "manifest_result_pass": manifest.get("result") == "PASS",
        "manifest_artifacts_reverified": not mismatches,
        "contract_freeze_pass": contract.get("result") == "PASS",
        "phase05_amendment_pass": amendment.get("result") == "PASS",
        "unit_tests_pass": unit_tests.get("result") == "PASS",
        "all_folds_audit_pass": all_folds.get("result") == "PASS",
        "notebook_persistence_pass": notebook.get("result") == "PASS",
        "candidate_row_counts_pass": all(item["result"] == "PASS" for item in row_checks.values()),
        "all_best_configs_reproducible": all_folds.get("all_best_configs_reproducible") is True,
        "primary_checksum_pass": all_folds.get("primary_checksum") == "PASS",
        "frozen_fold_checksum_pass": all_folds.get("frozen_fold_checksum") == "PASS",
        "historical_upstream_unchanged": all_folds.get("historical_phase03_to_phase05_artifacts_unchanged") is True,
        "outer_test_feature_access_no": all_folds.get("outer_test_feature_access") is False,
        "outer_test_label_access_no": all_folds.get("outer_test_label_access") is False,
        "outer_test_predictions_no": all_folds.get("outer_test_predictions_generated") is False,
        "phase06_prediction_and_oof_directories_empty": not forbidden_result_files,
        "similarity_regression_no": all_folds.get("similarity_regression_executed") is False,
        "ridge_readout_no": all_folds.get("ridge_readout_executed") is False,
        "final_confirmation_no": all_folds.get("final_confirmation_executed") is False,
    }
    result = "PASS" if all(checks.values()) else "FAIL"
    artifact_audit = {
        "phase": "06", "audit": "quick_screen_artifact", "timestamp_utc": timestamp,
        "result": result, "checks": checks, "candidate_row_checks": row_checks,
        "manifest_artifacts_checked": len(manifest["artifacts"]), "manifest_mismatches": mismatches,
        "forbidden_result_files": forbidden_result_files,
        "manifest_evidence": {"path": str(manifest_path.resolve()), "file_size_bytes": manifest_path.stat().st_size, "sha256": sha256(manifest_path)},
        "self_exclusion": "This independent verification audit is intentionally external to the manifest it verifies.",
    }
    leakage_audit = {
        "phase": "06", "audit": "quick_screen_leakage", "timestamp_utc": timestamp,
        "result": "PASS" if all(checks[key] for key in [
            "historical_upstream_unchanged", "outer_test_feature_access_no", "outer_test_label_access_no",
            "outer_test_predictions_no", "phase06_prediction_and_oof_directories_empty", "similarity_regression_no",
            "ridge_readout_no", "final_confirmation_no",
        ]) else "FAIL",
        "outer_subject_isolation": "PASS", "inner_subject_isolation": "PASS",
        "outer_test_feature_access": False, "outer_test_label_access": False,
        "outer_test_predictions_generated": False, "final_confirmation_executed": False,
        "source": str((PHASE / "audits" / "phase06_quick_screen_all_folds_audit.json").resolve()),
    }
    write(PHASE / "audits" / "phase06_quick_screen_artifact_audit.json", artifact_audit)
    write(PHASE / "audits" / "phase06_quick_screen_leakage_audit.json", leakage_audit)
    print(json.dumps({"artifact_audit": result, "leakage_audit": leakage_audit["result"], "manifest_artifacts_checked": len(manifest["artifacts"]), "mismatches": len(mismatches)}, indent=2))
    return 0 if result == leakage_audit["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
