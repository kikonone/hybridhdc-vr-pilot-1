"""Independent read-only verifier for the completed Phase 09 final freeze."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import nbformat
import pandas as pd

from freeze_phase09 import (
    EXPECTED, EXPECTED_FOLDS, EXPECTED_PRIMARY, FINAL_AUDITS, FINAL_MANIFEST,
    FREEZE_FILE, MARKER, NOTEBOOK, ROOT, EXPERIMENTS, PRIMARY, FOLDS,
    MANIFEST_DEPENDENT_AUDITS, iter_manifest_artifacts, notebook_has_errors,
    protected_baseline, read_json, sha256, verify_hash_map,
)


def verify_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    return sorted(row["path"] for row in rows if not (ROOT / row["path"]).is_file() or sha256(ROOT / row["path"]) != row["sha256"])


def verify_freeze() -> dict[str, Any]:
    missing = [str(path.relative_to(ROOT)) for path in [FINAL_MANIFEST, FREEZE_FILE] if not path.is_file()]
    missing += [f"audits/{name}" for name in FINAL_AUDITS if not (ROOT / "audits" / name).is_file()]
    if missing:
        return {"status": "FAIL", "checks": {"required_freeze_artifacts_present": False}, "missing_artifacts": sorted(missing)}
    manifest = read_json(FINAL_MANIFEST)
    freeze = read_json(FREEZE_FILE)
    execution = read_json(ROOT / "configs/phase09_execution_manifest.json")
    final_audits = {name: read_json(ROOT / "audits" / name).get("status") for name in FINAL_AUDITS}
    artifact_mismatches = verify_rows(iter_manifest_artifacts(manifest))
    protected_mismatches = verify_hash_map(protected_baseline())
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    marker_count = sum(MARKER in cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    notebook_text = "\n".join(
        str(output.get("text", output.get("data", {}).get("text/plain", "")))
        for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", [])
    )
    index = pd.read_csv(ROOT / "results/oof/phase09_canonical_oof_index.csv")
    phase10_paths = [str(path.resolve()) for path in EXPERIMENTS.glob("phase_10*")]
    checks = {
        "freeze_status_frozen": freeze.get("status") == "FROZEN",
        "execution_manifest_frozen": execution.get("status") == "FROZEN" and execution.get("phase09_frozen") is True,
        "run_records_complete_720": len(execution.get("training_runs", [])) == EXPECTED["runs"] and all(row.get("status") == "COMPLETE_AUDITED" for row in execution["training_runs"]),
        "raw_prediction_rows_30168": freeze.get("raw_prediction_rows") == execution.get("raw_prediction_rows") == EXPECTED["raw_rows"],
        "canonical_oof_rows_10056": freeze.get("canonical_oof_rows") == len(index) == EXPECTED["canonical"],
        "missing_modality_rows_8380": freeze.get("missing_modality_canonical_rows") == EXPECTED["missing"],
        "loso_rows_1676": freeze.get("loso_canonical_rows") == EXPECTED["loso"],
        "primary_checksum": sha256(PRIMARY) == EXPECTED_PRIMARY,
        "fold_checksum": sha256(FOLDS) == EXPECTED_FOLDS,
        "manifest_hash_matches_freeze": freeze.get("final_manifest", {}).get("sha256") == sha256(FINAL_MANIFEST),
        "manifest_self_hash_excluded": manifest.get("self_hash_included") is False,
        "manifest_artifact_hashes_match": not artifact_mismatches,
        "protected_artifacts_unchanged": not protected_mismatches,
        "missing_artifacts_zero": not manifest.get("missing_artifacts"),
        "duplicate_artifacts_zero": not manifest.get("duplicate_artifacts"),
        "manifest_hash_mismatches_zero": not manifest.get("hash_mismatches"),
        "unexpected_mutable_artifacts_zero": not manifest.get("unexpected_mutable_artifacts"),
        "all_six_final_audits_pass": all(value == "PASS" for value in final_audits.values()),
        "notebook_marker_once": marker_count == 1,
        "notebook_no_error_outputs": not notebook_has_errors(),
        "notebook_freeze_summary_persisted": all(value in notebook_text for value in ["720/720", "30168", "10056", "35/35", "FROZEN"]),
        "claim_boundaries_preserved": freeze.get("generalization_boundaries") == {
            "SUBJECT_GENERALIZATION": "EVALUATED_VIA_35_SUBJECT_LOSO",
            "MISSING_MODALITY_ROBUSTNESS": "EVALUATED_VIA_RETRAIN_WITHOUT_MODALITY",
            "UNSEEN_SESSION": "NOT_FEASIBLE_DUE_TO_METADATA", "UNSEEN_SCENARIO": "NOT_FEASIBLE_DUE_TO_METADATA",
            "TASK_TEMPLATE": "NOT_FEASIBLE_DUE_TO_METADATA", "ROUTE_CONFIGURATION": "NOT_FEASIBLE_DUE_TO_METADATA",
            "FLIGHT_GENERALIZABLE_BEHAVIOR_CLAIM": "INCONCLUSIVE_DUE_TO_METADATA",
        },
        "model_retraining_during_freeze_no": freeze.get("model_retraining_during_freeze") is False,
        "predictions_regenerated_no": freeze.get("predictions_regenerated_during_freeze") is False,
        "phase10_not_executed": freeze.get("phase10_executed") is False and execution.get("phase10_executed") is False,
        "ready_for_phase10": freeze.get("ready_to_proceed_to_phase10") is True and execution.get("ready_to_proceed_to_phase10") is True,
        "dependent_audit_exclusions_explicit": set(manifest.get("self_referential_audit_hash_exclusions", [])) == MANIFEST_DEPENDENT_AUDITS,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "artifact_hash_mismatches": artifact_mismatches, "protected_hash_mismatches": protected_mismatches,
        "final_audit_statuses": final_audits, "phase10_paths": phase10_paths,
        "model_retraining_executed": False, "predictions_regenerated": False,
        "upstream_files_modified": read_json(ROOT / "audits/phase09_upstream_freeze_integrity_final_audit.json").get("upstream_files_modified"),
        "phase10_executed": False, "ready_to_proceed_to_phase10": all(checks.values()),
    }


if __name__ == "__main__":
    result = verify_freeze()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
