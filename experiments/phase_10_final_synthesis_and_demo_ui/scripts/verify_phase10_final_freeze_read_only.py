from __future__ import annotations

import json
from pathlib import Path

from freeze_phase10_core_contract import compare_states, phase00_09_state
from initialize_phase10 import BASE, load_json, sha256


def main() -> None:
    manifest_path = BASE / "manifests/phase10_final_manifest.json"
    manifest = load_json(manifest_path)
    hash_audit = load_json(BASE / "audits/phase10_final_manifest_hash_audit.json")
    freeze = load_json(BASE / "configs/phase10_freeze.json")
    status = load_json(BASE / "configs/phase10_final_status.json")
    freeze_audit = load_json(BASE / "audits/phase10_final_freeze_audit.json")
    upstream_audit = load_json(BASE / "audits/phase10_upstream_freeze_integrity_audit.json")
    readiness = load_json(BASE / "audits/phase10_final_submission_readiness_audit.json")
    notebook_audit = load_json(BASE / "audits/phase10_final_freeze_notebook_persistence_audit.json")
    failures = []
    for item in manifest["artifacts"]:
        path = BASE / item["relative_path"]
        actual = sha256(path) if path.exists() else "MISSING"
        if actual != item["sha256"] or path.stat().st_size != item["file_size"]:
            failures.append({"path": str(path), "expected_sha256": item["sha256"], "actual_sha256": actual})
    manifest_hash_pass = sha256(manifest_path) == hash_audit["manifest_sha256"] and hash_audit["status"] == "PASS"
    baseline = load_json(BASE / "logs/phase10_final_freeze_phase00_09_baseline.json")
    upstream_comparison = compare_states(baseline, phase00_09_state("phase10_final_freeze_read_only_verify"))
    ui_files = [path for path in BASE.rglob("*") if path.is_file() and (path.name == "app.py" or path.suffix.lower() in {".html", ".css", ".js"})]
    expected = {
        "manifest_status": manifest.get("status") == "FROZEN",
        "manifest_hash": manifest_hash_pass,
        "payload_hashes": not failures,
        "freeze_status": freeze.get("status") == "FROZEN",
        "final_status": status.get("phase10_core_status") == "FROZEN",
        "pipeline_complete": status.get("phase00_10_scientific_pipeline_status") == "COMPLETE",
        "project_complete": status.get("core_experiment_project_complete") is True,
        "scientific_readiness": status.get("scientific_readiness") == "PASS",
        "historical_fail_retained": status.get("historical_frozen_immutability_audit") == "FAIL",
        "metadata_differences_6": status.get("nonscientific_metadata_differences_retained") == 6,
        "final_freeze_audit": freeze_audit.get("status") == "PASS",
        "upstream_integrity": upstream_audit.get("status") == "PASS",
        "readiness": readiness.get("status") == "PASS",
        "notebook_persistence": notebook_audit.get("status") == "PASS" and notebook_audit.get("error_outputs") == 0,
        "phase00_09_unchanged": upstream_comparison["modified_count"] == 0,
        "ui_absent": not ui_files,
        "ui_deferred": status.get("ui_status") == "DEFERRED_BY_USER_NOT_EXECUTED",
        "onlinehd_optional": status.get("onlinehd_replay_status") == "OPTIONAL_NOT_EXECUTED",
        "no_training_prediction_statistics": not status.get("model_training_executed") and not status.get("predictions_generated") and not status.get("statistics_recomputed"),
    }
    overall = all(expected.values())
    print(json.dumps({"status": "PASS" if overall else "FAIL", "checks": expected, "final_manifest_artifacts": manifest["artifact_count"], "manifest_sha256": hash_audit["manifest_sha256"], "payload_failures": failures, "phase00_09_files_modified": upstream_comparison["modified_count"], "ui_files": [str(path) for path in ui_files]}, indent=2))
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
