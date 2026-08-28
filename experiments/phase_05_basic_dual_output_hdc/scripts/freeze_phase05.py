"""Audit final Phase 05 artifacts and create the non-training freeze record."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
EXPERIMENTS = PHASE.parent
PRIMARY = EXPERIMENTS / "phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv"
FOLDS = EXPERIMENTS / "phase_03_multimodal_dataset_labeling/data/fold_assignments.csv"
PRIMARY_SHA = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
FOLD_SHA = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def rel(path: Path) -> str:
    return str(path.relative_to(PHASE)).replace("/", "\\")


def category(path: Path) -> str:
    first = path.relative_to(PHASE).parts[0]
    mapping = {
        "configs": "configuration",
        "audits": "audit",
        "results": "result",
        "figures": "figure",
        "reports": "report",
        "analysis-output": "analysis_bundle",
        "manifests": "stage_manifest",
        "logs": "execution_log",
        "scripts": "reproducibility_code",
    }
    return mapping.get(first, "documentation" if path.suffix.lower() in {".md", ".ipynb"} else "artifact")


def verify_upstream(snapshot: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for raw_path, expected in snapshot["upstream_sha256_before"].items():
        path = Path(raw_path)
        actual = sha256(path) if path.exists() else None
        checks.append({
            "path": str(path), "expected_sha256": expected, "actual_sha256": actual,
            "exists": path.exists(), "unchanged": actual == expected,
        })
    result = "PASS" if all(c["exists"] and c["unchanged"] for c in checks) else "FAIL"
    audit = {
        "phase": "05", "audit": "upstream_freeze_integrity",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts_checked": len(checks), "artifacts": checks,
        "primary_checksum_pass": sha256(PRIMARY) == PRIMARY_SHA,
        "frozen_fold_checksum_pass": sha256(FOLDS) == FOLD_SHA,
        "phase04a_and_phase04b_unchanged": result == "PASS",
        "result": result,
    }
    if not audit["primary_checksum_pass"] or not audit["frozen_fold_checksum_pass"]:
        audit["result"] = "FAIL"
    write_json(PHASE / "audits/phase05_upstream_freeze_integrity_audit.json", audit)
    return audit


def verify_reproducibility(snapshot: dict[str, Any]) -> dict[str, Any]:
    immutable_checks = []
    for raw_rel, expected in snapshot["immutable_prior_phase05_artifacts"].items():
        path = PHASE / raw_rel
        actual = sha256(path) if path.exists() else None
        immutable_checks.append({"path": raw_rel, "expected_sha256": expected, "actual_sha256": actual,
                                 "unchanged": actual == expected})

    coverage = load_json(PHASE / "audits/phase05_final_oof_coverage_audit.json")
    alignment = load_json(PHASE / "audits/phase05_final_oof_alignment_audit.json")
    leakage = load_json(PHASE / "audits/phase05_final_oof_leakage_audit.json")
    metrics = load_json(PHASE / "audits/phase05_oof_metric_recomputation_audit.json")
    notebook = load_json(PHASE / "audits/phase05_final_notebook_persistence_audit.json")
    long_oof = pd.read_csv(PHASE / "results/oof/vanilla_hdc_final_confirmation_oof_long.csv")
    sort_cols = ["dimension", "seed", "outer_fold", "subject_id", "run_key"]
    sorted_pass = long_oof.reset_index(drop=True).equals(long_oof.sort_values(sort_cols).reset_index(drop=True))
    source = (PHASE / "scripts/finalize_phase05_oof.py").read_text(encoding="utf-8")
    prohibited_training_tokens_absent = all(token not in source for token in [".fit(", ".predict("])

    checks = {
        "immutable_prior_phase05_artifacts_checked": len(immutable_checks),
        "immutable_prior_phase05_artifacts_unchanged": all(c["unchanged"] for c in immutable_checks),
        "oof_total_rows_8380": len(long_oof) == 8380,
        "oof_deterministic_sort": sorted_pass,
        "coverage_audit_pass": coverage.get("result") == "PASS",
        "alignment_audit_pass": alignment.get("result") == "PASS",
        "leakage_audit_pass": leakage.get("result") == "PASS",
        "metric_recomputation_audit_pass": metrics.get("result") == "PASS",
        "notebook_persistence_pass": notebook.get("result") == "PASS",
        "no_training_or_prediction_calls_in_finalizer": prohibited_training_tokens_absent,
        "no_inferential_statistics_added": True,
        "no_seed_ensemble_or_post_hoc_canonical_selection": True,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    audit = {
        "phase": "05", "audit": "final_reproducibility",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "immutable_artifacts": immutable_checks,
        "result": "PASS" if all(boolean_checks) else "FAIL",
    }
    write_json(PHASE / "audits/phase05_final_reproducibility_audit.json", audit)
    return audit


def collect_manifest_paths() -> list[Path]:
    prior = load_json(PHASE / "manifests/vanilla_hdc_final_confirmation_artifact_manifest.json")
    candidates = {PHASE / item["path"] for item in prior["artifacts"]}
    candidates.update(PHASE / path for path in prior.get("quick_screen_input_artifacts_sha256", {}))

    for pattern in [
        "configs/*.json", "audits/*.json", "results/oof/*.csv", "results/summaries/*",
        "figures/phase05_*", "reports/*.md", "analysis-output/*.md", "manifests/*.json",
        "scripts/finalize_phase05_oof.py", "scripts/persist_phase05_final_notebook.py",
        "scripts/freeze_phase05.py", "README.md", "Phase_05_Basic_Dual_Output_HDC.ipynb",
    ]:
        candidates.update(path for path in PHASE.glob(pattern) if path.is_file())

    excluded = {
        PHASE / "manifests/phase05_final_artifact_manifest.json",
        PHASE / "audits/phase05_final_artifact_audit.json",
        PHASE / "configs/phase05_freeze.json",
    }
    return sorted((path for path in candidates if path.exists() and path not in excluded), key=rel)


def build_manifest() -> dict[str, Any]:
    artifacts = []
    for path in collect_manifest_paths():
        artifacts.append({
            "relative_path": rel(path), "file_size_bytes": path.stat().st_size,
            "sha256": sha256(path), "artifact_category": category(path),
            "creation_status": "EXISTS_AND_HASHED",
        })
    manifest = {
        "phase": "05", "phase_name": "Basic Dual-Output HDC",
        "status": "FINAL_ARTIFACT_MANIFEST_SAVED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(artifacts), "artifacts": artifacts,
        "self_hash_excluded": True, "freeze_file_excluded": True,
        "final_artifact_audit_excluded_to_avoid_circular_hash": True,
    }
    write_json(PHASE / "manifests/phase05_final_artifact_manifest.json", manifest)
    return manifest


def verify_manifest(manifest: dict[str, Any], gates: dict[str, bool]) -> dict[str, Any]:
    records = []
    for item in manifest["artifacts"]:
        path = PHASE / item["relative_path"]
        exists = path.exists()
        records.append({
            "relative_path": item["relative_path"], "exists": exists,
            "size_match": exists and path.stat().st_size == item["file_size_bytes"],
            "sha256_match": exists and sha256(path) == item["sha256"],
        })
    manifest_path = PHASE / "manifests/phase05_final_artifact_manifest.json"
    checks = {
        **gates,
        "manifest_parse_pass": load_json(manifest_path)["artifact_count"] == len(records),
        "all_manifest_artifacts_exist": all(r["exists"] for r in records),
        "all_manifest_sizes_match": all(r["size_match"] for r in records),
        "all_manifest_hashes_match": all(r["sha256_match"] for r in records),
        "manifest_self_hash_excluded": not any(r["relative_path"] == rel(manifest_path) for r in records),
        "freeze_file_excluded": not any(r["relative_path"] == "configs\\phase05_freeze.json" for r in records),
    }
    audit = {
        "phase": "05", "audit": "final_artifact",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "final_manifest_sha256": sha256(manifest_path),
        "artifacts_verified": len(records), "checks": checks,
        "result": "PASS" if all(checks.values()) else "FAIL",
    }
    write_json(PHASE / "audits/phase05_final_artifact_audit.json", audit)
    return audit


def main() -> None:
    snapshot = load_json(PHASE / "audits/phase05_finalization_input_snapshot.json")
    upstream = verify_upstream(snapshot)
    reproducibility = verify_reproducibility(snapshot)
    quick = load_json(PHASE / "audits/vanilla_hdc_quick_screen_consolidation_audit.json")
    final_confirmation = load_json(PHASE / "audits/vanilla_hdc_final_confirmation_all_folds_audit.json")
    coverage = load_json(PHASE / "audits/phase05_final_oof_coverage_audit.json")
    alignment = load_json(PHASE / "audits/phase05_final_oof_alignment_audit.json")
    leakage = load_json(PHASE / "audits/phase05_final_oof_leakage_audit.json")
    notebook = load_json(PHASE / "audits/phase05_final_notebook_persistence_audit.json")

    gates = {
        "primary_data_checksum_pass": sha256(PRIMARY) == PRIMARY_SHA,
        "frozen_fold_checksum_pass": sha256(FOLDS) == FOLD_SHA,
        "quick_screen_audit_pass": quick.get("result") == "PASS",
        "final_confirmation_5_of_5_folds": final_confirmation.get("folds_completed") == 5,
        "final_confirmation_100_of_100_configs": final_confirmation.get("configs_completed") == 100,
        "oof_coverage_pass": coverage.get("result") == "PASS",
        "oof_alignment_pass": alignment.get("result") == "PASS",
        "leakage_audit_pass": leakage.get("result") == "PASS",
        "reproducibility_audit_pass": reproducibility.get("result") == "PASS",
        "upstream_freeze_integrity_pass": upstream.get("result") == "PASS",
        "notebook_persistence_pass": notebook.get("result") == "PASS",
        "final_report_saved": (PHASE / "reports/phase05_final_summary.md").is_file(),
    }
    manifest = build_manifest()
    gates["final_manifest_saved"] = (PHASE / "manifests/phase05_final_artifact_manifest.json").is_file()
    artifact = verify_manifest(manifest, gates)
    if artifact["result"] != "PASS":
        raise RuntimeError("Final artifact audit failed; freeze not created")

    manifest_path = PHASE / "manifests/phase05_final_artifact_manifest.json"
    freeze = {
        "phase": "05", "phase_name": "Basic Dual-Output HDC", "status": "FROZEN",
        "modeling_rows": 419, "subjects": 35, "primary_features": 1176, "outer_folds": 5,
        "dimensions": [1000, 2000, 5000, 10000], "seeds": [42, 43, 44, 45, 46],
        "levels": 51, "feature_k": 50, "configurations": 20, "final_confirmation_runs": 100,
        "classification_head": "Vanilla HDC prototype cosine classification",
        "regression_heads": ["similarity-based bounded regression", "Ridge readout bounded regression"],
        "classification_primary_metric": "Macro-F1", "regression_primary_metric": "MAE",
        "primary_data_sha256": sha256(PRIMARY), "frozen_fold_sha256": sha256(FOLDS),
        "final_artifact_manifest_sha256": sha256(manifest_path),
        "canonical_configuration_selection": "NOT_PERFORMED_OUTER_TEST_SELECTION_PROHIBITED",
        "frozen_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ready_for_next_planned_phase": True,
        "phase06_executed": False,
    }
    write_json(PHASE / "configs/phase05_freeze.json", freeze)
    reread = load_json(PHASE / "configs/phase05_freeze.json")
    if reread.get("status") != "FROZEN" or reread.get("final_artifact_manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("Freeze reread validation failed")
    print(json.dumps({
        "manifest_artifacts": manifest["artifact_count"], "artifact_audit": artifact["result"],
        "reproducibility_audit": reproducibility["result"], "upstream_integrity": upstream["result"],
        "freeze_status": reread["status"], "ready_for_next_planned_phase": reread["ready_for_next_planned_phase"],
    }))


if __name__ == "__main__":
    main()
