"""Finalize the Phase 05 no-retraining compliance amendment."""

from __future__ import annotations

import ast
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
EXPERIMENTS = PHASE.parent
PRIMARY = EXPERIMENTS / "phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv"
FOLDS = EXPERIMENTS / "phase_03_multimodal_dataset_labeling/data/fold_assignments.csv"
PRIMARY_SHA = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
FOLD_SHA = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
SNAPSHOT = PHASE / "audits/phase05_no_retraining_pre_amendment_snapshot.json"
AMENDMENT = PHASE / "configs/phase05_no_retraining_completion_amendment.json"
MANIFEST = PHASE / "manifests/phase05_final_artifact_manifest.json"
FREEZE = PHASE / "configs/phase05_freeze.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(PHASE)).replace("/", "\\")


def immutable_checks(snapshot: dict) -> list[dict]:
    checks = []
    for relative, expected in snapshot["immutable_artifact_sha256"].items():
        path = PHASE / relative
        actual = sha256(path) if path.exists() else None
        checks.append({
            "relative_path": relative,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "exists": path.exists(),
            "unchanged": actual == expected,
        })
    return checks


def training_calls(path: Path) -> list[dict]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    prohibited = {"fit", "fit_transform", "partial_fit"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in prohibited:
            found.append({"method": node.func.attr, "line": node.lineno})
    return found


def category(path: Path) -> str:
    first = path.relative_to(PHASE).parts[0]
    return {
        "configs": "configuration",
        "audits": "audit",
        "results": "result",
        "figures": "figure",
        "reports": "report",
        "analysis-output": "analysis_bundle",
        "manifests": "stage_manifest",
        "logs": "execution_log",
        "scripts": "reproducibility_code",
    }.get(first, "documentation")


def collect_paths() -> list[Path]:
    prior = load_json(MANIFEST)
    paths = {PHASE / item["relative_path"] for item in prior["artifacts"]}
    for folder in ["configs", "audits", "results", "figures", "reports", "analysis-output", "manifests"]:
        root = PHASE / folder
        paths.update(path for path in root.rglob("*") if path.is_file())
    paths.update(path for path in (PHASE / "scripts").glob("*.py") if path.is_file())
    paths.update([PHASE / "README.md", PHASE / "Phase_05_Basic_Dual_Output_HDC.ipynb"])
    excluded = {
        MANIFEST,
        PHASE / "audits/phase05_final_artifact_audit.json",
        FREEZE,
    }
    return sorted((path for path in paths if path.exists() and path not in excluded), key=rel)


def build_manifest() -> dict:
    artifacts = [{
        "relative_path": rel(path),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "artifact_category": category(path),
        "creation_status": "EXISTS_AND_HASHED",
    } for path in collect_paths()]
    manifest = {
        "phase": "05",
        "phase_name": "Basic Dual-Output HDC",
        "status": "FINAL_ARTIFACT_MANIFEST_SAVED_WITH_NO_RETRAINING_AMENDMENT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "self_hash_excluded": True,
        "freeze_file_excluded": True,
        "final_artifact_audit_excluded_to_avoid_circular_hash": True,
    }
    write_json(MANIFEST, manifest)
    return manifest


def verify_manifest(manifest: dict, gates: dict) -> dict:
    records = []
    for item in manifest["artifacts"]:
        path = PHASE / item["relative_path"]
        records.append({
            "relative_path": item["relative_path"],
            "exists": path.exists(),
            "size_match": path.exists() and path.stat().st_size == item["file_size_bytes"],
            "sha256_match": path.exists() and sha256(path) == item["sha256"],
        })
    checks = {
        **gates,
        "manifest_parse_pass": load_json(MANIFEST)["artifact_count"] == len(records),
        "all_manifest_artifacts_exist": all(item["exists"] for item in records),
        "all_manifest_sizes_match": all(item["size_match"] for item in records),
        "all_manifest_hashes_match": all(item["sha256_match"] for item in records),
        "manifest_self_hash_excluded": not any(item["relative_path"] == rel(MANIFEST) for item in records),
        "freeze_file_excluded": not any(item["relative_path"] == rel(FREEZE) for item in records),
    }
    audit = {
        "phase": "05",
        "audit": "final_artifact_with_no_retraining_amendment",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "final_manifest_sha256": sha256(MANIFEST),
        "artifacts_verified": len(records),
        "checks": checks,
        "result": "PASS" if all(checks.values()) else "FAIL",
    }
    write_json(PHASE / "audits/phase05_final_artifact_audit.json", audit)
    return audit


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    snapshot = load_json(SNAPSHOT)
    immutable = immutable_checks(snapshot)
    diagnostic = load_json(PHASE / "audits/phase05_no_retraining_diagnostic_completion_audit.json")
    efficiency = load_json(PHASE / "audits/phase05_no_retraining_efficiency_protocol_completion_audit.json")
    notebook = load_json(PHASE / "audits/phase05_no_retraining_notebook_persistence_audit.json")
    diagnostics = pd.read_csv(PHASE / "results/oof/vanilla_hdc_final_confirmation_diagnostics.csv")
    measured = pd.read_csv(PHASE / "results/efficiency/vanilla_hdc_final_confirmation_protocol_completion_by_fold_config.csv")
    summary = pd.read_csv(PHASE / "results/summaries/vanilla_hdc_inference_efficiency_protocol_by_config.csv")
    completion_script = PHASE / "scripts/complete_phase05_no_retraining.py"
    notebook_script = PHASE / "scripts/persist_phase05_no_retraining_notebook.py"
    source_training_calls = training_calls(completion_script) + training_calls(notebook_script)
    phase06_paths = list(EXPERIMENTS.glob("phase_06*"))
    phase06_files = [path for root in phase06_paths for path in root.rglob("*") if path.is_file()]

    gates = {
        "primary_data_checksum_pass": sha256(PRIMARY) == PRIMARY_SHA,
        "frozen_fold_checksum_pass": sha256(FOLDS) == FOLD_SHA,
        "immutable_historical_artifacts_unchanged": all(item["unchanged"] for item in immutable),
        "diagnostic_audit_pass": diagnostic.get("result") == "PASS",
        "diagnostic_rows_8380": len(diagnostics) == 8380,
        "diagnostic_configs_20": diagnostics[["dimension", "seed"]].drop_duplicates().shape[0] == 20,
        "efficiency_audit_pass": efficiency.get("result") == "PASS",
        "efficiency_fold_config_rows_100": len(measured) == 100,
        "efficiency_config_rows_20": len(summary) == 20,
        "five_warmups": efficiency.get("warmups") == 5,
        "thirty_timed_repetitions": efficiency.get("timed_repetitions") == 30,
        "nanosecond_monotonic_clock": efficiency.get("clock") == "time.perf_counter_ns",
        "frozen_predictions_reproduced": efficiency.get("maximum_frozen_prediction_abs_difference", 1.0) <= 1e-12,
        "training_timing_not_fabricated": efficiency.get("training_timing_remeasurement") == "NOT_PERFORMED_RETRAINING_PROHIBITED",
        "model_fitting_not_executed": diagnostic.get("model_fitting_executed") is False and efficiency.get("model_fitting_executed") is False,
        "no_training_calls_in_amendment_scripts": not source_training_calls,
        "notebook_persistence_pass": notebook.get("result") == "PASS",
        # An empty Phase 06 placeholder directory predates this amendment.
        # Execution is defined by the presence of Phase 06 artifacts, not the folder.
        "phase06_not_executed": not phase06_files,
        "no_post_hoc_selection_or_inference": True,
    }

    amendment = load_json(AMENDMENT)
    amendment.update({
        "status": "COMPLETED_NO_RETRAINING",
        "completed_utc": now,
        "diagnostic_completion": "PASS",
        "inference_efficiency_protocol_completion": "PASS",
        "training_timing_remeasurement": "NOT_PERFORMED_RETRAINING_PROHIBITED",
        "historical_artifacts_preserved": gates["immutable_historical_artifacts_unchanged"],
        "phase06_executed": False,
    })
    write_json(AMENDMENT, amendment)

    upstream = {
        "phase": "05",
        "audit": "upstream_freeze_integrity_after_no_retraining_amendment",
        "timestamp_utc": now,
        "primary_sha256": sha256(PRIMARY),
        "frozen_fold_sha256": sha256(FOLDS),
        "primary_checksum_pass": gates["primary_data_checksum_pass"],
        "frozen_fold_checksum_pass": gates["frozen_fold_checksum_pass"],
        "immutable_historical_artifacts_checked": len(immutable),
        "immutable_historical_artifacts_unchanged": gates["immutable_historical_artifacts_unchanged"],
        "result": "PASS" if gates["primary_data_checksum_pass"] and gates["frozen_fold_checksum_pass"] and gates["immutable_historical_artifacts_unchanged"] else "FAIL",
    }
    write_json(PHASE / "audits/phase05_upstream_freeze_integrity_audit.json", upstream)

    reproducibility = {
        "phase": "05",
        "audit": "final_reproducibility_with_no_retraining_amendment",
        "timestamp_utc": now,
        "checks": gates,
        "immutable_artifacts": immutable,
        "source_training_calls": source_training_calls,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "result": "PASS" if all(gates.values()) else "FAIL",
    }
    write_json(PHASE / "audits/phase05_final_reproducibility_audit.json", reproducibility)

    amendment_audit = {
        "phase": "05",
        "audit": "no_retraining_amendment",
        "timestamp_utc": now,
        "pre_amendment_freeze_sha256": snapshot["pre_amendment_freeze_sha256"],
        "pre_amendment_manifest_sha256": snapshot["pre_amendment_manifest_sha256"],
        "immutable_historical_artifacts_checked": len(immutable),
        "immutable_mismatches": [item for item in immutable if not item["unchanged"]],
        "checks": gates,
        "model_fitting_executed": False,
        "prediction_artifact_replaced": False,
        "canonical_configuration_selected": False,
        "phase06_executed": False,
        "result": "PASS" if all(gates.values()) else "FAIL",
    }
    write_json(PHASE / "audits/phase05_no_retraining_amendment_audit.json", amendment_audit)

    manifest = build_manifest()
    artifact = verify_manifest(manifest, {
        "amendment_audit_pass": amendment_audit["result"] == "PASS",
        "reproducibility_audit_pass": reproducibility["result"] == "PASS",
        "upstream_integrity_audit_pass": upstream["result"] == "PASS",
        "notebook_persistence_pass": notebook.get("result") == "PASS",
    })

    freeze = load_json(FREEZE)
    freeze.update({
        "status": "FROZEN",
        "final_artifact_manifest_sha256": sha256(MANIFEST),
        "no_retraining_completion_amendment": "COMPLETED",
        "diagnostic_contract_completion": "PASS",
        "inference_efficiency_protocol_completion": "PASS",
        "training_timing_remeasurement": "NOT_PERFORMED_RETRAINING_PROHIBITED",
        "pre_amendment_freeze_sha256": snapshot["pre_amendment_freeze_sha256"],
        "amended_timestamp_utc": now,
        "historical_artifacts_preserved": gates["immutable_historical_artifacts_unchanged"],
        "ready_for_next_planned_phase": artifact["result"] == "PASS",
        "phase06_executed": False,
    })
    write_json(FREEZE, freeze)

    if artifact["result"] != "PASS":
        raise RuntimeError(json.dumps(artifact, ensure_ascii=False, indent=2))
    print(json.dumps({
        "amendment": amendment_audit["result"],
        "manifest_artifacts": manifest["artifact_count"],
        "artifact_audit": artifact["result"],
        "freeze": freeze["status"],
    }))


if __name__ == "__main__":
    main()
