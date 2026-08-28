"""Fail-closed final freeze for the completed Phase 08 analysis.

Dry-run is the default.  ``--write`` is the only mode that creates the final
manifest/freeze records and appends the isolated Notebook freeze summary.
No model, prediction, canonical OOF, metric, statistical, report, or upstream
Phase 03--07 artifact is written by this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT.parent
PROJECT = EXPERIMENTS.parent
FINAL_MANIFEST = ROOT / "manifests/phase08_final_manifest.json"
FREEZE_FILE = ROOT / "configs/phase08_freeze.json"
NOTEBOOK = ROOT / "Phase_08_Fusion_and_Shortcut_Analysis.ipynb"
NOTEBOOK_MARKER = "## Phase 08 Final Freeze Summary"

EXPECTED = {"runs": 370, "raw_rows": 31006, "canonical_rows": 10894, "outer_folds": 5, "subjects": 35}
CORE_CONFIGS = [
    "configs/phase08_experiment_contract.json",
    "configs/phase08_execution_manifest.json",
    "configs/phase08_model_matrix.json",
    "configs/phase08_fusion_conditions.json",
    "configs/phase08_shortcut_conditions.json",
]
REQUIRED_REPORTS = [
    "reports/phase08_final_analysis.md",
    "reports/phase08_shortcut_and_generalization_report.md",
    "reports/phase08_statistical_appendix.md",
    "reports/phase08_generalization_limitations.md",
    "reports/analysis-output/analysis-report.md",
    "reports/analysis-output/stats-appendix.md",
    "reports/analysis-output/figure-catalog.md",
]
FIGURE_STEMS = [
    "phase08_classification_condition_comparison",
    "phase08_regression_condition_comparison",
    "phase08_fusion_increment_effects",
    "phase08_shortcut_sensitivity",
    "phase08_subject_level_effects",
]
CRITICAL_AUDITS = [
    "phase08_input_and_fold_audit.json",
    "phase08_execution_coverage_audit.json",
    "phase08_checkpoint_integrity_audit.json",
    "phase08_oof_coverage_audit.json",
    "phase08_oof_alignment_audit.json",
    "phase08_oof_leakage_audit.json",
    "phase08_upstream_reference_integrity_audit.json",
    "phase08_metric_recalculation_audit.json",
    "phase08_statistical_unit_audit.json",
    "phase08_multiple_comparison_audit.json",
    "phase08_final_analysis_artifact_audit.json",
    "phase08_final_analysis_reproducibility_audit.json",
    "phase08_final_notebook_persistence_audit.json",
]
GENERATED_AUDITS = {
    "phase08_freeze_audit.json",
    "phase08_final_manifest_audit.json",
    "phase08_upstream_freeze_integrity_final_audit.json",
    "phase08_notebook_freeze_persistence_audit.json",
}
UPSTREAM = {
    "phase04a": (EXPERIMENTS / "phase_04a_traditional_classification_baselines/configs/phase04a_freeze.json", "34ea8100d9406f9701750a441aa6537323c28bcdb194cb3fd3645c4f7de4a2e1"),
    "phase04b": (EXPERIMENTS / "phase_04b_traditional_regression_baselines/configs/phase04b_freeze.json", "e2c88b1139a50aab6d47b6477c7bceff74f8443095f9d039ea9af84b715ee790"),
    "phase06": (EXPERIMENTS / "phase_06_hdc_variant_screening/configs/phase06_freeze.json", "cd94acdf0bcb463b5f48a7c84110a23059899b6b5bf156131694827d7d7bde66"),
    "phase07": (EXPERIMENTS / "phase_07_unimodal_contribution/configs/phase07_freeze.json", "8569b48a8210f0ca1316d5a140d292edb892e2a556ea13ee299e9b97699af492"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def artifact(path: Path, base: Path = ROOT) -> dict:
    return {"path": path.relative_to(base).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}


def inventory(paths: Iterable[Path], base: Path = ROOT) -> list[dict]:
    return [artifact(path, base) for path in sorted(paths, key=lambda p: p.as_posix().lower())]


def tree_files(relative: str, suffixes: tuple[str, ...] | None = None) -> list[Path]:
    paths = [p for p in (ROOT / relative).rglob("*") if p.is_file()]
    return paths if suffixes is None else [p for p in paths if p.suffix.lower() in suffixes]


def baseline_hashes() -> dict[str, str]:
    audit = read_json(ROOT / "audits/phase08_final_analysis_artifact_audit.json")
    return {row["path"].replace("\\", "/"): row["sha256"] for row in audit.get("artifacts", [])}


def raw_prediction_baseline() -> dict[str, str]:
    audit = read_json(ROOT / "audits/phase08_oof_leakage_audit.json")
    return {key.replace("\\", "/"): value for key, value in audit.get("raw_prediction_hashes_before", {}).items()}


def notebook_has_errors() -> bool:
    import nbformat

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return any(
        output.get("output_type") == "error"
        for cell in notebook.cells if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )


def preflight() -> dict:
    missing: list[str] = []
    mismatches: list[str] = []
    for rel in CORE_CONFIGS + REQUIRED_REPORTS:
        path = ROOT / rel
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(rel)
    for stem in FIGURE_STEMS:
        for suffix in (".pdf", ".png"):
            rel = f"figures/{stem}{suffix}"
            if not (ROOT / rel).is_file():
                missing.append(rel)

    execution = read_json(ROOT / "configs/phase08_execution_manifest.json")
    records = execution.get("run_records", [])
    run_ids = [row.get("run_id") for row in records]
    prediction_files = tree_files("results/predictions", (".csv",))
    raw_rows = sum(csv_rows(path) for path in prediction_files)
    canonical_files = [
        ROOT / "results/oof/phase08_canonical_classification_oof.csv",
        ROOT / "results/oof/phase08_canonical_regression_oof.csv",
    ]
    canonical_rows = sum(csv_rows(path) for path in canonical_files if path.is_file())
    canonical_index = ROOT / "results/oof/phase08_canonical_oof_index.csv"
    if canonical_index.is_file():
        with canonical_index.open("r", encoding="utf-8-sig", newline="") as handle:
            index_rows = list(csv.DictReader(handle))
    else:
        index_rows = []
    canonical_coverage = (
        len(index_rows) == 26
        and sum(int(row["rows"]) for row in index_rows) == EXPECTED["canonical_rows"]
        and all(int(row["unique_run_keys"]) == 419 for row in index_rows)
    )

    raw_baseline = raw_prediction_baseline()
    raw_current = {path.relative_to(ROOT).as_posix(): sha256(path) for path in prediction_files}
    for rel, expected_hash in raw_baseline.items():
        if raw_current.get(rel) != expected_hash:
            mismatches.append(rel)
    stable_baseline = baseline_hashes()
    for rel, expected_hash in stable_baseline.items():
        path = ROOT / rel
        if not path.is_file() or sha256(path) != expected_hash:
            mismatches.append(rel)

    audit_status = {}
    for name in CRITICAL_AUDITS:
        path = ROOT / "audits" / name
        audit_status[name] = path.is_file() and read_json(path).get("status") == "PASS"
        if not audit_status[name]:
            missing.append(f"audits/{name}" if not path.is_file() else f"audits/{name}:status")
    input_audit = read_json(ROOT / "audits/phase08_input_and_fold_audit.json")
    checksum = input_audit.get("checksum_pass", {})
    subject_isolation = all(not row.get("train_test_subject_overlap") for row in input_audit.get("fold_checks", []))
    metric_audit = read_json(ROOT / "audits/phase08_metric_recalculation_audit.json")
    stats_audit = read_json(ROOT / "audits/phase08_statistical_unit_audit.json")
    handoff_path = ROOT / "configs/phase09_generalization_handoff.json"
    handoff = read_json(handoff_path) if handoff_path.is_file() else {}
    upstream_rows = []
    for name, (path, expected_hash) in UPSTREAM.items():
        actual = sha256(path) if path.is_file() else None
        upstream_rows.append({"phase": name, "path": str(path), "expected_sha256": expected_hash, "actual_sha256": actual, "match": actual == expected_hash})

    checks = {
        "core_configs_present": all((ROOT / rel).is_file() for rel in CORE_CONFIGS),
        "model_runs_370": len(records) == EXPECTED["runs"] and all(row.get("status") == "COMPLETE" for row in records),
        "duplicate_run_identifiers_zero": len(run_ids) == len(set(run_ids)) == EXPECTED["runs"],
        "prediction_files_370": len(prediction_files) == EXPECTED["runs"],
        "raw_prediction_rows_31006": raw_rows == EXPECTED["raw_rows"],
        "raw_prediction_hashes_unchanged": len(raw_current) == len(raw_baseline) == EXPECTED["runs"] and not any(rel.startswith("results/predictions/") for rel in mismatches),
        "canonical_oof_rows_10894": canonical_rows == EXPECTED["canonical_rows"],
        "canonical_run_key_coverage": canonical_coverage,
        "primary_checksum": checksum.get("primary") is True,
        "with_performance_checksum": checksum.get("with_performance") is True,
        "performance_only_checksum": checksum.get("performance_only") is True,
        "frozen_fold_checksum": checksum.get("folds") is True,
        "subject_isolation": subject_isolation,
        "critical_audits_pass": all(audit_status.values()),
        "statistical_unit_subject_35": stats_audit.get("statistical_unit") == "subject_id" and stats_audit.get("n") == EXPECTED["subjects"],
        "holm_correction": read_json(ROOT / "audits/phase08_multiple_comparison_audit.json").get("status") == "PASS",
        "bootstrap_2000": stats_audit.get("bootstrap_repetitions") == 2000 and metric_audit.get("checks", {}).get("bootstrap_complete") is True,
        "stable_analysis_artifact_hashes": not mismatches,
        "phase09_handoff_parseable": handoff.get("phase09_executed") is False and handoff.get("holdout_executed") is False,
        "formal_reports_present": all((ROOT / rel).is_file() for rel in REQUIRED_REPORTS),
        "pdf_png_figures_present": all((ROOT / f"figures/{stem}{suffix}").is_file() for stem in FIGURE_STEMS for suffix in (".pdf", ".png")),
        "notebook_has_no_error_output": NOTEBOOK.is_file() and not notebook_has_errors(),
        "upstream_freeze_integrity": all(row["match"] for row in upstream_rows),
        "phase09_not_executed": handoff.get("phase09_executed") is False and not (EXPERIMENTS / "phase_09_generalization").exists(),
        "freeze_not_already_present": not FREEZE_FILE.exists(),
    }
    return {
        "status": "PASS" if all(checks.values()) and not missing and not mismatches else "FAIL",
        "timestamp_utc": now(),
        "checks": checks,
        "actual": {"model_runs": len(records), "prediction_files": len(prediction_files), "raw_prediction_rows": raw_rows, "canonical_oof_rows": canonical_rows},
        "missing_artifacts": sorted(set(missing)),
        "hash_mismatches": sorted(set(mismatches)),
        "upstream": upstream_rows,
    }


def build_manifest(freeze_time: str, pre: dict) -> dict:
    input_manifest = read_json(ROOT / "manifests/phase08_input_manifest.json")
    input_rows = input_manifest["inputs"]
    raw_files = tree_files("results/predictions", (".csv",))
    fold_metrics = tree_files("results/fold_metrics", (".json", ".csv"))
    checkpoints = tree_files("results/checkpoints", (".json",))
    oof_files = tree_files("results/oof", (".csv", ".json"))
    summary_files = tree_files("results/summaries")
    report_files = tree_files("reports")
    figure_files = tree_files("figures")
    audit_files = [p for p in tree_files("audits", (".json",)) if p.name not in {"phase08_freeze_audit.json", "phase08_final_manifest_audit.json"}]
    all_paths = [p.relative_to(ROOT).as_posix() for group in (raw_files, fold_metrics, checkpoints, oof_files, summary_files, report_files, figure_files, audit_files) for p in group]
    duplicates = sorted(path for path, count in Counter(all_paths).items() if count > 1)
    unexpected = sorted(
        p.relative_to(ROOT).as_posix()
        for rel, allowed in {
            "results/predictions": {".csv"}, "results/fold_metrics": {".json", ".csv"}, "results/checkpoints": {".json"},
            "results/oof": {".csv", ".json"}, "results/summaries": {".csv", ".json"}, "reports": {".md"}, "figures": {".pdf", ".png"}, "audits": {".json"},
        }.items()
        for p in tree_files(rel) if p.suffix.lower() not in allowed
    )
    upstream = {row["phase"]: {"path": row["path"], "sha256": row["actual_sha256"], "integrity": "PASS" if row["match"] else "FAIL"} for row in pre["upstream"]}
    execution = read_json(ROOT / "configs/phase08_execution_manifest.json")
    return {
        "phase": "Phase 08",
        "phase_name": "Fusion and Shortcut Analysis",
        "phase_path": str(ROOT),
        "status": "FROZEN",
        "freeze_time_utc": freeze_time,
        "self_hash_included": False,
        "contracts": {rel: sha256(ROOT / rel) for rel in CORE_CONFIGS},
        "datasets": {name: {"path": row["path"], "sha256": sha256(Path(row["path"])), "checksum": "PASS" if sha256(Path(row["path"])) == row["sha256"] else "FAIL"} for name, row in input_rows.items() if name in {"primary", "with_performance", "performance_only", "folds"}},
        "upstream_freezes": upstream,
        "run_coverage": {"expected": EXPECTED["runs"], "actual": len(execution.get("run_records", [])), "complete": sum(row.get("status") == "COMPLETE" for row in execution.get("run_records", [])), "duplicate_run_identifiers": []},
        "raw_predictions": {"expected_rows": EXPECTED["raw_rows"], "actual_rows": pre["actual"]["raw_prediction_rows"], "file_count": len(raw_files), "artifacts": inventory(raw_files)},
        "fold_metrics": {"file_count": len(fold_metrics), "artifacts": inventory(fold_metrics)},
        "checkpoints": {"file_count": len(checkpoints), "artifacts": inventory(checkpoints)},
        "canonical_oof": {"expected_rows": EXPECTED["canonical_rows"], "actual_rows": pre["actual"]["canonical_oof_rows"], "run_key_coverage": "PASS", "artifacts": inventory(oof_files)},
        "summaries": inventory(summary_files),
        "reports": inventory(report_files),
        "figures": inventory(figure_files),
        "audits": inventory(audit_files),
        "notebook": artifact(NOTEBOOK),
        "phase09_handoff": artifact(ROOT / "configs/phase09_generalization_handoff.json"),
        "missing_artifacts": pre["missing_artifacts"],
        "duplicate_artifacts": duplicates,
        "hash_mismatches": pre["hash_mismatches"],
        "unexpected_mutable_artifacts": unexpected,
        "model_retraining_during_freeze": False,
        "predictions_regenerated": False,
        "outer_test_used_for_tuning": False,
        "phase09_executed": False,
    }


def freeze_payload(freeze_time: str, manifest_hash: str) -> dict:
    return {
        "phase": "Phase 08", "status": "FROZEN", "freeze_time_utc": freeze_time,
        "model_runs": EXPECTED["runs"], "raw_prediction_rows": EXPECTED["raw_rows"], "canonical_oof_rows": EXPECTED["canonical_rows"],
        "outer_folds": EXPECTED["outer_folds"], "subjects": EXPECTED["subjects"], "outer_test_used_for_tuning": False,
        "model_retraining_during_freeze": False, "predictions_regenerated": False, "phase09_executed": False,
        "completed_analyses": {"fusion_comparison": True, "flight_behavioral_sensitivity": True, "performance_shortcut_analysis": True, "statistical_analysis": True, "generalization_limitations_saved": True},
        "holdout_feasibility": {"unseen_session": "NOT_FEASIBLE_DUE_TO_METADATA", "unseen_scenario": "NOT_FEASIBLE_DUE_TO_METADATA", "task_template": "NOT_FEASIBLE_DUE_TO_METADATA", "FLIGHT_TASK_SETTING_ONLY": "NOT_FEASIBLE_EMPTY_PROVENANCE_GROUP"},
        "phase09_handoff_saved": True,
        "final_manifest": {"path": "manifests/phase08_final_manifest.json", "sha256": manifest_hash},
        "upstream_freeze_integrity": "PASS", "notebook_persistence": "PASS", "ready_to_proceed_to_phase09": True,
        "scope_statement": "Phase 08 results are frozen only for the registered fusion, shortcut, and sensitivity analyses.",
        "generalization_guardrail": "Current results do not prove cross-session, cross-scenario, or cross-task-template generalization.",
    }


def append_notebook_summary() -> dict:
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    kept = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and NOTEBOOK_MARKER in cell.source:
            break
        kept.append(cell)
    markdown = nbformat.v4.new_markdown_cell(
        NOTEBOOK_MARKER + "\n\nThis isolated, executed summary records the final Phase 08 freeze. Historical cells and outputs above are preserved. No model training, prediction generation, or Phase 09 execution occurs here."
    )
    code = nbformat.v4.new_code_cell(
        "from pathlib import Path\nimport json\nphase08_root = Path.cwd().resolve()\n"
        "freeze = json.loads((phase08_root/'configs/phase08_freeze.json').read_text(encoding='utf-8'))\n"
        "manifest = json.loads((phase08_root/'manifests/phase08_final_manifest.json').read_text(encoding='utf-8'))\n"
        "{'runs':'370/370','raw_prediction_rows':31006,'canonical_oof_rows':10894,'final_audits':'PASS',"
        "'manifest_saved':True,'freeze_file_saved':True,'phase08_status':freeze['status'],"
        "'phase09_executed':freeze['phase09_executed'],'ready_for_phase09':freeze['ready_to_proceed_to_phase09']}"
    )
    temp = nbformat.v4.new_notebook(cells=[code], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    executed = NotebookClient(temp, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    final = nbformat.v4.new_notebook(cells=kept + [markdown] + executed.cells, metadata=notebook.metadata)
    nbformat.write(final, NOTEBOOK)
    outputs = executed.cells[0].get("outputs", [])
    errors = [output for output in outputs if output.get("output_type") == "error"]
    checks = {"summary_present": True, "historical_cells_preserved": True, "added_code_cells": 1, "added_cell_has_output": bool(outputs), "error_outputs_zero": not errors, "model_retraining_not_executed": True, "predictions_not_regenerated": True, "phase09_not_executed": True}
    payload = {"status": "PASS" if all(value is True or value == 1 for value in checks.values()) else "FAIL", "timestamp_utc": now(), "checks": checks, "preserved_prefix_cells": len(kept), "notebook_sha256": sha256(NOTEBOOK)}
    atomic_json(ROOT / "audits/phase08_notebook_freeze_persistence_audit.json", payload)
    return payload


def write_upstream_audit(pre: dict) -> dict:
    checks = {row["phase"]: row["match"] for row in pre["upstream"]}
    payload = {"status": "PASS" if all(checks.values()) else "FAIL", "timestamp_utc": now(), "checks": checks, "interfaces": pre["upstream"], "phase03_to_07_files_modified": False}
    atomic_json(ROOT / "audits/phase08_upstream_freeze_integrity_final_audit.json", payload)
    return payload


def manifest_audit(manifest: dict) -> dict:
    mismatches = []
    for section in ("raw_predictions", "fold_metrics", "checkpoints", "canonical_oof"):
        for row in manifest[section]["artifacts"]:
            path = ROOT / row["path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                mismatches.append(row["path"])
    for section in ("summaries", "reports", "figures", "audits"):
        for row in manifest[section]:
            path = ROOT / row["path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                mismatches.append(row["path"])
    for key in ("notebook", "phase09_handoff"):
        row = manifest[key]; path = ROOT / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            mismatches.append(row["path"])
    checks = {"manifest_parseable": True, "self_hash_excluded": manifest.get("self_hash_included") is False, "missing_required_artifacts_zero": not manifest["missing_artifacts"], "duplicate_artifacts_zero": not manifest["duplicate_artifacts"], "hash_mismatches_zero": not mismatches and not manifest["hash_mismatches"], "unexpected_mutable_artifacts_zero": not manifest["unexpected_mutable_artifacts"]}
    payload = {"status": "PASS" if all(checks.values()) else "FAIL", "timestamp_utc": now(), "checks": checks, "manifest_sha256": sha256(FINAL_MANIFEST), "hash_mismatches": mismatches}
    atomic_json(ROOT / "audits/phase08_final_manifest_audit.json", payload)
    return payload


def freeze_audit(pre: dict, manifest_check: dict, notebook_check: dict) -> dict:
    stable = baseline_hashes(); changed = [rel for rel, expected_hash in stable.items() if not (ROOT / rel).is_file() or sha256(ROOT / rel) != expected_hash]
    raw = raw_prediction_baseline(); raw_changed = [rel for rel, expected_hash in raw.items() if not (ROOT / rel).is_file() or sha256(ROOT / rel) != expected_hash]
    freeze = read_json(FREEZE_FILE)
    checks = {
        "all_expected_artifacts_present": not pre["missing_artifacts"], "hash_mismatch_zero": not changed and not raw_changed,
        "missing_required_artifacts_zero": not pre["missing_artifacts"], "duplicate_run_identifiers_zero": pre["checks"]["duplicate_run_identifiers_zero"],
        "raw_predictions_unchanged": not raw_changed, "canonical_oof_unchanged": not [p for p in changed if p.startswith("results/oof/")],
        "metrics_unchanged": not [p for p in changed if p.startswith("results/summaries/")],
        "statistical_results_unchanged": not [p for p in changed if "pairwise_statistics" in p or "bootstrap_confidence" in p],
        "reports_unchanged": not [p for p in changed if p.startswith("reports/")], "phase03_to_07_files_unmodified": pre["checks"]["upstream_freeze_integrity"],
        "phase09_not_executed": freeze.get("phase09_executed") is False, "manifest_hash_verified": manifest_check.get("status") == "PASS",
        "notebook_persistence": notebook_check.get("status") == "PASS", "freeze_status_reproducible": freeze.get("status") == "FROZEN" and freeze.get("final_manifest", {}).get("sha256") == sha256(FINAL_MANIFEST),
    }
    payload = {"status": "PASS" if all(checks.values()) else "FAIL", "timestamp_utc": now(), "checks": checks, "changed_immutable_artifacts": changed, "changed_raw_predictions": raw_changed, "model_retraining_executed": False, "predictions_regenerated": False, "phase09_executed": False}
    atomic_json(ROOT / "audits/phase08_freeze_audit.json", payload)
    return payload


def write_freeze() -> dict:
    pre = preflight()
    if pre["status"] != "PASS":
        raise RuntimeError("Freeze preflight failed: " + json.dumps({"missing": pre["missing_artifacts"], "mismatches": pre["hash_mismatches"], "checks": {k: v for k, v in pre["checks"].items() if not v}}, ensure_ascii=False))
    freeze_time = now()
    execution_path = ROOT / "configs/phase08_execution_manifest.json"
    original_execution = execution_path.read_bytes()
    original_notebook = NOTEBOOK.read_bytes()
    created = [FINAL_MANIFEST, FREEZE_FILE] + [ROOT / "audits" / name for name in GENERATED_AUDITS]
    try:
        execution = read_json(execution_path)
        execution.update({"status": "FROZEN", "ready_for_phase08_freeze": False, "phase08_frozen": True, "ready_to_proceed_to_phase09": True, "phase09_executed": False, "freeze_time_utc": freeze_time, "last_updated_utc": freeze_time})
        atomic_json(execution_path, execution)
        provisional = build_manifest(freeze_time, pre)
        atomic_json(FINAL_MANIFEST, provisional)
        atomic_json(FREEZE_FILE, freeze_payload(freeze_time, sha256(FINAL_MANIFEST)))
        notebook_check = append_notebook_summary()
        if notebook_check["status"] != "PASS":
            raise RuntimeError("Notebook freeze persistence failed")
        upstream_check = write_upstream_audit(pre)
        if upstream_check["status"] != "PASS":
            raise RuntimeError("Upstream freeze integrity failed")
        final_manifest = build_manifest(freeze_time, pre)
        atomic_json(FINAL_MANIFEST, final_manifest)
        atomic_json(FREEZE_FILE, freeze_payload(freeze_time, sha256(FINAL_MANIFEST)))
        manifest_check = manifest_audit(final_manifest)
        freeze_check = freeze_audit(pre, manifest_check, notebook_check)
        if manifest_check["status"] != "PASS" or freeze_check["status"] != "PASS":
            raise RuntimeError("Final manifest or freeze audit failed")
        return {"status": "PASS", "preflight": pre, "manifest_audit": manifest_check, "freeze_audit": freeze_check, "notebook_audit": notebook_check, "manifest_sha256": sha256(FINAL_MANIFEST)}
    except Exception:
        execution_path.write_bytes(original_execution)
        NOTEBOOK.write_bytes(original_notebook)
        for path in created:
            if path.exists():
                path.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Create the final manifest/freeze artifacts after a passing preflight")
    args = parser.parse_args()
    result = write_freeze() if args.write else preflight()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
