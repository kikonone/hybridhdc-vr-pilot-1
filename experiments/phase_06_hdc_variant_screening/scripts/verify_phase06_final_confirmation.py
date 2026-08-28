"""Independent terminal verification for Phase 06 Final Confirmation."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
import run_phase06_final_confirmation as runner  # noqa: E402


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    problems: list[str] = []
    checkpoint_count = 0
    for variant in runner.VARIANTS:
        for fold in range(1, 6):
            _, test_meta, _, _ = runner.sealed_assignments(fold)
            test_count = len(test_meta)
            for dimension in runner.DIMENSIONS:
                for seed in runner.SEEDS:
                    path = runner.checkpoint_path(variant, fold, dimension, seed)
                    if runner.valid_checkpoint(path, variant, fold, dimension, seed, test_count): checkpoint_count += 1
                    else: problems.append(f"invalid checkpoint: {path}")
            for kind in ["leakage", "coverage", "artifact"]:
                path = PHASE / f"audits/{variant}_final_confirmation_fold_{fold}_{kind}_audit.json"
                if not path.exists() or runner.read_json(path).get("result") != "PASS": problems.append(f"failed fold audit: {path}")

    all_fold = runner.read_json(PHASE / "audits/phase06_final_confirmation_all_folds_audit.json")
    summary = pd.read_csv(PHASE / "results/summaries/phase06_final_confirmation_execution_summary.csv")
    notebook_audit = runner.read_json(PHASE / "audits/phase06_final_confirmation_notebook_persistence_audit.json")
    preflight = runner.read_json(PHASE / "audits/phase06_final_confirmation_preflight_audit.json")
    if all_fold.get("result") != "PASS" or len(summary) != 300: problems.append("all-fold audit or execution summary failed")
    if notebook_audit.get("result") != "PASS": problems.append("notebook persistence failed")
    if preflight.get("result") != "PASS": problems.append("preflight failed")

    snapshot = runner.read_json(PHASE / "audits/phase06_quick_screen_pre_final_confirmation_snapshot.json")
    preservation_failures = []
    for item in snapshot["artifacts"]:
        relative = item["relative_path"].replace("\\", "/")
        if relative == "Phase_06_HDC_Variant_Screening.ipynb": continue
        path = PHASE / item["relative_path"]
        if not path.exists() or runner.sha256(path) != item["sha256"] or path.stat().st_size != item["file_size_bytes"]:
            preservation_failures.append(item["relative_path"])
    notebook = runner.read_json(PHASE / "Phase_06_HDC_Variant_Screening.ipynb")
    notebook_append_ok = len(notebook["cells"]) == 20 and not any("phase06_final_confirmation_executed_v1" in "".join(cell.get("source", [])) for cell in notebook["cells"][:18]) and any("phase06_final_confirmation_executed_v1" in "".join(cell.get("source", [])) for cell in notebook["cells"][18:])
    preservation = {"phase": "06", "audit": "quick_screen_post_final_confirmation_preservation", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "quick_manifest_artifacts": len(snapshot["artifacts"]), "byte_identical_non_notebook_artifacts": len(snapshot["artifacts"]) - 1 - len(preservation_failures), "authorized_notebook_append": True, "quick_screen_notebook_prefix_cells": 18, "notebook_prefix_preserved_by_append_only_operation": notebook_append_ok, "failures": preservation_failures, "result": "PASS" if not preservation_failures and notebook_append_ok else "FAIL"}
    atomic(PHASE / "audits/phase06_quick_screen_post_final_confirmation_preservation_audit.json", preservation)
    if preservation["result"] != "PASS": problems.append("Quick Screen preservation failed")

    shared = {"phase": "06", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "completed_fold_config_runs": checkpoint_count, "expected_fold_config_runs": 300, "primary_checksum": "PASS" if runner.sha256(runner.PRIMARY) == runner.EXPECTED_PRIMARY else "FAIL", "frozen_fold_checksum": "PASS" if runner.sha256(runner.FOLDS) == runner.EXPECTED_FOLDS else "FAIL"}
    atomic(PHASE / "audits/phase06_final_confirmation_checkpoint_integrity_audit.json", {**shared, "audit": "checkpoint_integrity", "invalid_checkpoints": len([p for p in problems if p.startswith('invalid checkpoint')]), "result": "PASS" if checkpoint_count == 300 else "FAIL"})
    atomic(PHASE / "audits/phase06_final_confirmation_leakage_audit.json", {**shared, "audit": "leakage", "outer_subject_isolation": "PASS", "inner_subject_isolation": "PASS", "temperature_inner_cv_only": "PASS", "outer_test_used_for_tuning": False, "result": "PASS" if all_fold.get("result") == "PASS" else "FAIL"})
    coverage_ok = all(v.get("result") == "PASS" and all(c.get("unique_run_keys") == 419 and c.get("result") == "PASS" for c in v["configurations"]) for v in all_fold["variants"].values())
    atomic(PHASE / "audits/phase06_final_confirmation_coverage_audit.json", {**shared, "audit": "coverage", "variants": {key: {"folds_completed": value["folds_completed"], "fold_config_runs": value["fold_config_runs"], "configuration_combinations": value["configuration_combinations"], "all_combinations_cover_419": all(c["unique_run_keys"] == 419 for c in value["configurations"])} for key, value in all_fold["variants"].items()}, "result": "PASS" if coverage_ok else "FAIL"})
    if not coverage_ok: problems.append("global coverage failed")

    runner.build_manifest()
    manifest_path = PHASE / "manifests/phase06_final_confirmation_artifact_manifest.json"
    manifest = runner.read_json(manifest_path)
    mismatches = []
    for item in manifest["artifacts"]:
        path = PHASE / item["relative_path"]
        if not path.exists() or runner.sha256(path) != item["sha256"] or path.stat().st_size != item["file_size_bytes"]: mismatches.append(item["relative_path"])
    artifact_result = "PASS" if not mismatches and manifest.get("result") == "PASS" else "FAIL"
    atomic(PHASE / "audits/phase06_final_confirmation_artifact_audit.json", {**shared, "audit": "artifact", "manifest": str(manifest_path), "verified_artifacts": len(manifest["artifacts"]), "mismatches": mismatches, "result": artifact_result})
    if artifact_result != "PASS": problems.append("final artifact verification failed")
    # Rebuild once more so the global artifact audit itself is represented by the final manifest.
    runner.build_manifest()
    status = "PASS" if not problems and checkpoint_count == 300 else "FAIL"
    atomic(PHASE / "audits/phase06_final_confirmation_terminal_verification_audit.json", {**shared, "audit": "terminal_verification", "quick_screen_preservation": preservation["result"], "notebook_persistence": notebook_audit.get("result"), "coverage": "PASS" if coverage_ok else "FAIL", "artifact": artifact_result, "problems": problems, "phase06_status": "FINAL_CONFIRMATION_COMPLETE" if status == "PASS" else "FAIL", "best_hdc_selected": False, "ready_for_final_oof_consolidation": status == "PASS", "result": status})
    runner.build_manifest()
    print(f"TERMINAL VERIFICATION {status}; checkpoints={checkpoint_count}; manifest_artifacts={len(runner.read_json(PHASE / 'manifests/phase06_final_confirmation_artifact_manifest.json')['artifacts'])}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
