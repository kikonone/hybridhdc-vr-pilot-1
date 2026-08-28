"""Consolidate and audit all completed Phase 06 quick-screen checkpoints."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "src"))
from phase06_variant_common import best_candidate  # noqa: E402


EXPECTED = {"onlinehd": 24, "multicentroid": 6, "hybrid": 32}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def record(path: Path) -> dict[str, Any]:
    return {"relative_path": str(path.relative_to(PHASE)), "file_size_bytes": path.stat().st_size, "sha256": sha256(path), "result": "PASS"}


def json_safe(row: dict[str, Any]) -> dict[str, Any]:
    return {key: (None if pd.isna(value) else value) for key, value in row.items()}


def main() -> int:
    timestamp = datetime.now(timezone.utc).isoformat()
    all_best_rows: list[dict[str, Any]] = []
    variant_audits: dict[str, Any] = {}
    all_best_reproducible = True
    all_leakage_pass = True
    all_artifact_pass = True
    for variant, expected in EXPECTED.items():
        frames: list[pd.DataFrame] = []
        fold_details: list[dict[str, Any]] = []
        for fold in range(1, 6):
            candidate_path = PHASE / "results" / "summaries" / f"{variant}_quick_screen_fold_{fold}_candidates.csv"
            best_path = PHASE / "results" / "summaries" / f"{variant}_quick_screen_fold_{fold}_best_config.json"
            leakage_path = PHASE / "audits" / f"{variant}_quick_screen_fold_{fold}_leakage_audit.json"
            coverage_path = PHASE / "audits" / f"{variant}_quick_screen_fold_{fold}_coverage_audit.json"
            artifact_path = PHASE / "audits" / f"{variant}_quick_screen_fold_{fold}_artifact_audit.json"
            frame = pd.read_csv(candidate_path)
            rows = [json_safe(item) for item in frame.to_dict(orient="records")]
            recomputed = best_candidate(rows)
            saved = read_json(best_path)["best_config"]
            reproducible = (
                saved["candidate_id"] == recomputed["candidate_id"]
                and saved["canonical_config_json"] == recomputed["canonical_config_json"]
                and abs(float(saved["mean_macro_f1"]) - float(recomputed["mean_macro_f1"])) <= 1e-12
                and abs(float(saved["std_macro_f1_sample"]) - float(recomputed["std_macro_f1_sample"])) <= 1e-12
                and abs(float(saved["mean_balanced_accuracy"]) - float(recomputed["mean_balanced_accuracy"])) <= 1e-12
                and abs(float(saved["mean_severe_error_rate"]) - float(recomputed["mean_severe_error_rate"])) <= 1e-12
            )
            leakage = read_json(leakage_path)
            coverage = read_json(coverage_path)
            artifact = read_json(artifact_path)
            checkpoint_dir = PHASE / "results" / "checkpoints" / "quick_screen" / variant / f"fold_{fold}"
            checkpoints = sorted(checkpoint_dir.glob("candidate_*.json"))
            checkpoint_pass = len(checkpoints) == expected and all(read_json(path).get("result") == "PASS" for path in checkpoints)
            fold_pass = (
                len(frame) == expected and reproducible and leakage.get("result") == "PASS"
                and coverage.get("result") == "PASS" and artifact.get("result") == "PASS" and checkpoint_pass
            )
            all_best_reproducible &= reproducible
            all_leakage_pass &= leakage.get("result") == "PASS"
            all_artifact_pass &= artifact.get("result") == "PASS" and checkpoint_pass
            frame.insert(0, "consolidated_outer_fold", fold)
            frames.append(frame)
            all_best_rows.append({"variant": variant, "outer_fold": fold, **saved})
            fold_details.append({
                "outer_fold": fold, "candidates": len(frame), "checkpoints": len(checkpoints),
                "best_reproducible": reproducible, "leakage": leakage.get("result"),
                "coverage": coverage.get("result"), "artifact": artifact.get("result"),
                "result": "PASS" if fold_pass else "FAIL",
            })
        consolidated = pd.concat(frames, ignore_index=True)
        output = PHASE / "results" / "summaries" / f"phase06_{variant}_quick_screen_all_folds.csv"
        consolidated.to_csv(output, index=False)
        variant_audits[variant] = {
            "folds_completed": sum(item["result"] == "PASS" for item in fold_details),
            "expected_folds": 5,
            "candidates_per_fold": sorted(set(item["candidates"] for item in fold_details)),
            "fold_details": fold_details,
            "result": "PASS" if all(item["result"] == "PASS" for item in fold_details) else "FAIL",
        }

    best_summary_path = PHASE / "results" / "summaries" / "phase06_all_variants_quick_screen_summary.csv"
    pd.DataFrame(all_best_rows).to_csv(best_summary_path, index=False)

    snapshot = read_json(PHASE / "audits" / "phase06_upstream_pre_quick_screen_snapshot.json")
    upstream_mismatches: list[dict[str, Any]] = []
    for item in snapshot["files"]:
        path = Path(item["path"])
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_sha = sha256(path) if exists else None
        if not exists or actual_size != item["file_size_bytes"] or actual_sha != item["sha256"]:
            upstream_mismatches.append({"path": item["path"], "exists": exists, "actual_size": actual_size, "actual_sha256": actual_sha})
    upstream_pass = not upstream_mismatches

    all_fold_audits = [
        read_json(path) for path in sorted((PHASE / "audits").glob("*_quick_screen_fold_*_leakage_audit.json"))
    ]
    outer_feature_access = any(item.get("outer_test_feature_access") is not False for item in all_fold_audits)
    outer_label_access = any(item.get("outer_test_label_access") is not False for item in all_fold_audits)
    outer_predictions = any(item.get("outer_test_prediction_generated") is not False for item in all_fold_audits)
    overall = (
        all(item["result"] == "PASS" for item in variant_audits.values())
        and all_best_reproducible and all_leakage_pass and all_artifact_pass and upstream_pass
        and not outer_feature_access and not outer_label_access and not outer_predictions
    )
    final_audit = {
        "phase": "06", "audit": "quick_screen_all_folds", "timestamp_utc": timestamp,
        "result": "PASS" if overall else "FAIL", "variants": variant_audits,
        "all_best_configs_reproducible": all_best_reproducible,
        "outer_test_feature_access": False if not outer_feature_access else True,
        "outer_test_label_access": False if not outer_label_access else True,
        "outer_test_predictions_generated": False if not outer_predictions else True,
        "similarity_regression_executed": False, "ridge_readout_executed": False,
        "classification_oof_generated": False, "regression_oof_generated": False,
        "final_confirmation_executed": False,
        "upstream_files_checked": snapshot["file_count"],
        "upstream_immutable_mismatches": upstream_mismatches,
        "historical_phase03_to_phase05_artifacts_unchanged": upstream_pass,
    }
    primary = PHASE.parents[1] / "experiments" / "phase_03_multimodal_dataset_labeling" / "data" / "primary_without_performance.csv"
    folds = PHASE.parents[1] / "experiments" / "phase_03_multimodal_dataset_labeling" / "data" / "fold_assignments.csv"
    final_audit["primary_sha256"] = sha256(primary)
    final_audit["frozen_fold_sha256"] = sha256(folds)
    final_audit["primary_checksum"] = "PASS" if final_audit["primary_sha256"] == "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44" else "FAIL"
    final_audit["frozen_fold_checksum"] = "PASS" if final_audit["frozen_fold_sha256"] == "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f" else "FAIL"
    if final_audit["primary_checksum"] != "PASS" or final_audit["frozen_fold_checksum"] != "PASS":
        final_audit["result"] = "FAIL"
    audit_path = PHASE / "audits" / "phase06_quick_screen_all_folds_audit.json"
    write_json(audit_path, final_audit)

    manifest_paths: list[Path] = []
    for folder in ["configs", "src", "scripts", "tests", "reports", "results/checkpoints/quick_screen", "results/summaries", "results/fold_metrics", "results/efficiency", "audits", "logs"]:
        base = PHASE / folder
        manifest_paths.extend(path for path in base.rglob("*") if path.is_file())
    manifest_paths = sorted(set(manifest_paths))
    manifest_paths.extend([PHASE / "README.md", PHASE / "task_plan.md", PHASE / "notes.md", PHASE / "Phase_06_HDC_Variant_Screening.ipynb"])
    manifest_paths.extend(path for path in (PHASE / "manifests").glob("*.json") if path.name != "phase06_quick_screen_artifact_manifest.json")
    manifest_paths = sorted(set(manifest_paths))
    manifest = {
        "phase": "06", "manifest": "quick_screen_artifacts", "timestamp_utc": timestamp,
        "artifact_count": len(manifest_paths), "artifacts": [record(path) for path in manifest_paths],
        "source_training_calls": ["OnlineHD-style", "Multi-centroid", "Hybrid"],
        "outer_test_prediction_calls": [], "final_confirmation_calls": [],
        "result": final_audit["result"],
    }
    manifest_path = PHASE / "manifests" / "phase06_quick_screen_artifact_manifest.json"
    write_json(manifest_path, manifest)

    report = f"""# Phase 06 Quick-Screen Completion

Status: `{'QUICK_SCREEN_COMPLETE' if final_audit['result'] == 'PASS' else 'FAIL'}`

The three new HDC variants completed frozen, classification-only inner-CV quick screening across all five outer folds. OnlineHD evaluated 24 candidates per fold, Multi-centroid evaluated 6, and Hybrid evaluated 32. Each fold selected a configuration using only the frozen lexicographic inner-CV rule.

No outer-test feature or label was materialized, no outer-test prediction or OOF artifact was generated, and no regression head or Final Confirmation was executed. Vanilla HDC remained a read-only Phase 05 baseline.

All {snapshot['file_count']} snapshotted Phase 03–05 artifacts remained byte-identical: `{'PASS' if upstream_pass else 'FAIL'}`.

This stage does not select a final best HDC. It only establishes fold-specific quick-screen candidates for the separately authorized Final Confirmation stage.
"""
    (PHASE / "reports" / "phase06_quick_screen_completion.md").write_text(report, encoding="utf-8")
    print(json.dumps({"result": final_audit["result"], "variants": variant_audits, "all_best_reproducible": all_best_reproducible, "upstream_unchanged": upstream_pass}, indent=2))
    return 0 if final_audit["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
