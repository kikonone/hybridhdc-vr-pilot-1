"""Independent, read-only verification of the final Phase 08 freeze."""

from __future__ import annotations

import json
from pathlib import Path

from freeze_phase08 import FINAL_MANIFEST, FREEZE_FILE, ROOT, EXPECTED, notebook_has_errors, read_json, sha256


def verify() -> dict:
    required = [
        FINAL_MANIFEST, FREEZE_FILE,
        ROOT / "audits/phase08_freeze_audit.json",
        ROOT / "audits/phase08_final_manifest_audit.json",
        ROOT / "audits/phase08_upstream_freeze_integrity_final_audit.json",
        ROOT / "audits/phase08_notebook_freeze_persistence_audit.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        return {"status": "FAIL", "missing": missing, "ready_to_proceed_to_phase09": False}
    manifest = read_json(FINAL_MANIFEST)
    freeze = read_json(FREEZE_FILE)
    mismatches = []
    sections = [manifest["raw_predictions"]["artifacts"], manifest["fold_metrics"]["artifacts"], manifest["checkpoints"]["artifacts"], manifest["canonical_oof"]["artifacts"], manifest["summaries"], manifest["reports"], manifest["figures"], manifest["audits"]]
    for rows in sections:
        for row in rows:
            path = ROOT / row["path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                mismatches.append(row["path"])
    for key in ("notebook", "phase09_handoff"):
        row = manifest[key]; path = ROOT / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            mismatches.append(row["path"])
    audits = {path.name: read_json(path).get("status") for path in required[2:]}
    execution = read_json(ROOT / "configs/phase08_execution_manifest.json")
    handoff = read_json(ROOT / "configs/phase09_generalization_handoff.json")
    checks = {
        "manifest_status_frozen": manifest.get("status") == "FROZEN",
        "freeze_status_frozen": freeze.get("status") == "FROZEN",
        "execution_status_frozen": execution.get("status") == "FROZEN",
        "counts_exact": freeze.get("model_runs") == EXPECTED["runs"] and freeze.get("raw_prediction_rows") == EXPECTED["raw_rows"] and freeze.get("canonical_oof_rows") == EXPECTED["canonical_rows"],
        "manifest_hash_matches_freeze": freeze.get("final_manifest", {}).get("sha256") == sha256(FINAL_MANIFEST),
        "manifest_artifact_hashes_match": not mismatches,
        "all_freeze_audits_pass": all(value == "PASS" for value in audits.values()),
        "notebook_no_errors": not notebook_has_errors(),
        "notebook_hash_matches": manifest["notebook"]["sha256"] == sha256(ROOT / manifest["notebook"]["path"]),
        "model_retraining_not_executed": freeze.get("model_retraining_during_freeze") is False,
        "predictions_not_regenerated": freeze.get("predictions_regenerated") is False,
        "outer_test_not_used_for_tuning": freeze.get("outer_test_used_for_tuning") is False,
        "phase09_not_executed": freeze.get("phase09_executed") is False and handoff.get("phase09_executed") is False,
        "ready_to_proceed_to_phase09": freeze.get("ready_to_proceed_to_phase09") is True,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "hash_mismatches": mismatches, "audit_statuses": audits, "manifest_sha256": sha256(FINAL_MANIFEST), "ready_to_proceed_to_phase09": all(checks.values())}


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
