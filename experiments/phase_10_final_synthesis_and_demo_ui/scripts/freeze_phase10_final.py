from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from freeze_phase10_core_contract import compare_states, phase00_09_state
from initialize_phase10 import BASE, EXPERIMENTS, load_json, sha256


ROOT = BASE.parents[1]
NOW = datetime.now(timezone.utc).isoformat()
FREEZE_STATUS = "FROZEN"
PIPELINE_STATUS = "COMPLETE"
NOTEBOOK = BASE / "Phase_10_Final_Synthesis_and_Demo_UI.ipynb"
NOTEBOOK_MARKER = "CORE_FINAL_FREEZE_V1"
KNOWN_PHASE09_REFERENCE_PATHS = {
    str((EXPERIMENTS / "phase_09_robustness_and_generalization/configs/phase09_freeze.json").resolve()),
    str((EXPERIMENTS / "phase_09_robustness_and_generalization/manifests/phase09_final_manifest.json").resolve()),
}


def save_json(relative: str, payload: Any) -> None:
    path = BASE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(relative: str, text: str) -> None:
    path = BASE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def count_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def cell_sha256(cell: dict[str, Any]) -> str:
    payload = json.dumps(cell, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_csv_sources(path: Path, path_field: str, hash_field: str) -> tuple[int, list[dict[str, str]]]:
    failures = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        source = Path(row[path_field])
        actual = sha256(source) if source.exists() else "MISSING"
        if actual != row[hash_field]:
            failures.append({"path": str(source), "expected": row[hash_field], "actual": actual})
    return len(rows), failures


def validate_core_contracts() -> dict[str, Any]:
    freeze_path = BASE / "configs/phase10_core_contract_freeze.json"
    freeze = load_json(freeze_path)
    if freeze.get("status") != "CORE_CONTRACT_FROZEN_NOT_SYNTHESIZED" or not freeze.get("ready_for_phase10_final_synthesis"):
        raise RuntimeError("Phase 10 Core Contract Freeze is not valid")
    failures = []
    for record in freeze["contracts"]:
        path = Path(record["path"])
        actual = sha256(path) if path.exists() else "MISSING"
        if actual != record["sha256"]:
            failures.append({"path": str(path), "expected": record["sha256"], "actual": actual})
    if failures:
        raise RuntimeError(f"Core Contract hash failure: {failures}")
    audit = load_json(BASE / "audits/phase10_core_contract_freeze_audit.json")
    if audit.get("status") != "PASS" or audit.get("scientific_source_conflicts") != 0:
        raise RuntimeError("Core Contract audit is not PASS")
    return {"freeze": str(freeze_path.resolve()), "freeze_sha256": sha256(freeze_path), "contracts_verified": len(freeze["contracts"]), "failures": [], "status": "PASS"}


def validate_final_synthesis() -> dict[str, Any]:
    status = load_json(BASE / "configs/phase10_final_synthesis_status.json")
    artifact = load_json(BASE / "audits/phase10_final_synthesis_artifact_audit.json")
    notebook = load_json(BASE / "audits/phase10_final_synthesis_notebook_persistence_audit.json")
    numerical = load_json(BASE / "audits/phase10_cross_phase_numerical_consistency_audit.json")
    expected_counts = {"prediction_sources": 1406, "statistical_artifacts": 35, "paper_tables": 14, "paper_figures": 61, "rq_rows": 6}
    if status.get("status") != "FINAL_SYNTHESIS_COMPLETE_PENDING_PHASE10_FREEZE" or not status.get("ready_for_phase10_final_freeze"):
        raise RuntimeError("Final Synthesis is not ready for freeze")
    if artifact.get("status") != "PASS" or artifact.get("counts") != expected_counts:
        raise RuntimeError("Final Synthesis artifact gate failed")
    if numerical.get("status") != "PASS" or numerical.get("scientific_source_conflicts") != 0 or numerical.get("unresolved_numerical_differences") != 0:
        raise RuntimeError("Cross-phase numerical gate failed")
    if notebook.get("status") != "PASS" or notebook.get("error_outputs") != 0:
        raise RuntimeError("Final Synthesis notebook gate failed")
    required_audits = [
        "phase10_final_prediction_library_audit.json", "phase10_final_statistics_bundle_audit.json",
        "phase10_paper_table_audit.json", "phase10_paper_figure_audit.json",
        "phase10_rq_evidence_audit.json", "phase10_reproducibility_package_audit.json",
    ]
    for name in required_audits:
        if load_json(BASE / "audits" / name).get("status") != "PASS":
            raise RuntimeError(f"Required synthesis audit failed: {name}")
    if not (BASE / "reports/phase10_final_synthesis_report.md").exists():
        raise RuntimeError("Final Synthesis report missing")
    return {"status_file": str((BASE / "configs/phase10_final_synthesis_status.json").resolve()), "artifact_audit": str((BASE / "audits/phase10_final_synthesis_artifact_audit.json").resolve()), "counts": expected_counts, "status": "PASS"}


def validate_scientific_sources() -> dict[str, Any]:
    pred_count, pred_failures = verify_csv_sources(BASE / "results/final_prediction_library/final_prediction_library_index.csv", "source_path", "source_sha256")
    stat_count, stat_failures = verify_csv_sources(BASE / "results/final_statistics_bundle/final_statistics_index.csv", "source_path", "source_sha256")
    figure_count, figure_failures = verify_csv_sources(BASE / "reports/paper_figures/paper_figure_registry.csv", "source_path", "source_sha256")
    table_map = load_json(BASE / "reports/paper_tables/paper_table_source_map.json")
    table_failures = []
    table_source_count = 0
    for table in table_map["tables"]:
        for item in table["sources"]:
            table_source_count += 1
            source = Path(item["source_path"])
            actual = sha256(source) if source.exists() else "MISSING"
            if actual != item["source_sha256"]:
                table_failures.append({"path": str(source), "expected": item["source_sha256"], "actual": actual})
            if item["exact_copy_path"]:
                copy = Path(item["exact_copy_path"])
                copy_hash = sha256(copy) if copy.exists() else "MISSING"
                if copy_hash != item["source_sha256"]:
                    table_failures.append({"path": str(copy), "expected": item["source_sha256"], "actual": copy_hash})
    checksum = load_json(BASE / "reproducibility/checksum_verification.json")
    failures = pred_failures + stat_failures + figure_failures + table_failures + checksum.get("failures", [])
    if failures or (pred_count, stat_count, len(table_map["tables"]), figure_count) != (1406, 35, 14, 61):
        raise RuntimeError(f"Scientific source verification failed: {failures[:5]}")
    return {"prediction_sources_verified": pred_count, "statistical_artifacts_verified": stat_count, "paper_tables_verified": len(table_map["tables"]), "paper_table_sources_verified": table_source_count, "paper_figures_verified": figure_count, "checksum_registry_verified": checksum["verified_count"], "scientific_source_conflicts": 0, "unresolved_numerical_differences": 0, "status": "PASS"}


def baseline_map(baseline: dict[str, Any]) -> dict[str, str]:
    return {record["path"]: record["sha256"] for record in baseline["artifacts"]}


def validate_upstream_interfaces(baseline: dict[str, Any]) -> dict[str, Any]:
    upstream = load_json(BASE / "manifests/phase10_upstream_freeze_manifest.json")
    baseline_hashes = baseline_map(baseline)
    interfaces = []
    reference_differences = []
    for phase in ("00", "01", "02"):
        phase_dir = next(path for path in EXPERIMENTS.iterdir() if path.name.startswith(f"phase_{phase}"))
        phase_files = [path for path in baseline_hashes if Path(path).is_relative_to(phase_dir.resolve())]
        interfaces.append({"phase": phase, "interface_type": "LEGACY_PHASE_NO_FORMAL_FREEZE_INTERFACE", "integrity_basis": "FULL_PHASE_FILE_HASH_BASELINE", "files_covered": len(phase_files), "status": "PASS"})
    for record in upstream["freeze_interfaces"] + upstream["final_manifests"]:
        path = Path(record["path"])
        actual = sha256(path) if path.exists() else "MISSING"
        stable = baseline_hashes.get(str(path.resolve())) == actual
        reference_match = actual == record["sha256"]
        entry = {"phase": record["phase"], "interface_type": record.get("interface_type", "FREEZE_INTERFACE"), "path": str(path.resolve()), "recorded_sha256": record["sha256"], "actual_sha256": actual, "matches_pre_freeze_baseline": stable, "initialization_reference_match": reference_match, "status": "PASS" if stable and (reference_match or str(path.resolve()) in KNOWN_PHASE09_REFERENCE_PATHS) else "FAIL"}
        interfaces.append(entry)
        if not reference_match:
            reference_differences.append(entry)
    actual_difference_paths = {item["path"] for item in reference_differences}
    if actual_difference_paths != KNOWN_PHASE09_REFERENCE_PATHS or any(item["status"] != "PASS" for item in interfaces):
        raise RuntimeError(f"Unexpected upstream interface difference: {reference_differences}")
    phase09_freeze = load_json(EXPERIMENTS / "phase_09_robustness_and_generalization/configs/phase09_freeze.json")
    phase09_manifest = EXPERIMENTS / "phase_09_robustness_and_generalization/manifests/phase09_final_manifest.json"
    if phase09_freeze["final_manifest"]["sha256"] != sha256(phase09_manifest):
        raise RuntimeError("Current Phase 09 direct freeze chain is inconsistent")
    primary = Path(upstream["primary_data"]["path"])
    folds = Path(upstream["frozen_folds"]["path"])
    if sha256(primary) != upstream["primary_data"]["sha256"] or sha256(folds) != upstream["frozen_folds"]["sha256"]:
        raise RuntimeError("Primary data or frozen fold checksum failed")
    return {"interfaces": interfaces, "formal_interfaces_verified": len(upstream["freeze_interfaces"]) + len(upstream["final_manifests"]), "legacy_phases_covered": 3, "known_phase09_initialization_reference_differences": reference_differences, "primary_data_checksum": "PASS", "frozen_fold_checksum": "PASS", "status": "PASS"}


def validate_historical_caveat() -> dict[str, Any]:
    evidence = load_json(ROOT / "audits/pre_submission_repair/phase06_evidence_chain.json")
    historical = load_json(ROOT / "audits/pre_submission_repair/frozen_artifact_immutability_audit.json")
    scientific = load_json(ROOT / "audits/pre_submission_repair/final_scientific_immutability_audit.json")
    candidate = next(item for item in evidence["candidate_manifests"] if item["source_path"].endswith("phase06_final_artifact_manifest.json") and item["phase06_freeze_reference_consistency"] == "PASS")
    checks = {
        "phase06_original_manifest_hash_verified": evidence["original_final_manifest_hash_verified"] is True,
        "nonscientific_metadata_differences_retained_6": len(candidate["artifact_hash_mismatches"]) == 6,
        "historical_frozen_immutability_fail_retained": historical["status"] == "FAIL",
        "scientific_artifact_changes_zero": historical["scientific_artifact_hash_changes"] == 0,
        "scientific_consistency_pass": scientific["phase00_09_scientific_consistency"] == "PASS",
        "predictions_unmodified": scientific["predictions_modified"] is False,
        "canonical_oof_unmodified": scientific["canonical_oof_modified"] is False,
        "statistics_unmodified": scientific["statistics_modified"] is False,
        "frozen_model_configs_unmodified": scientific["frozen_model_configs_modified"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Historical caveat truth gate failed: {checks}")
    return {"definition": "historical engineering/provenance caveat", "nonscientific_metadata_differences_retained": 6, "historical_changed_nonscientific_files": historical["production_frozen_artifact_hash_changes"], "historical_frozen_immutability_audit": "FAIL", "scientific_artifact_changes": 0, "scientific_consistency": "PASS", "predictions_modified": False, "canonical_oof_modified": False, "statistics_modified": False, "frozen_model_configs_modified": False, "checks": checks, "status": "PASS"}


def validate_ui_absent() -> dict[str, Any]:
    forbidden_names = {"app.py"}
    forbidden_suffixes = {".html", ".css", ".js"}
    forbidden_dirs = {"pages", "assets", "best_hdc_demo_ui"}
    files = [path for path in BASE.rglob("*") if path.is_file() and (path.name in forbidden_names or path.suffix.lower() in forbidden_suffixes)]
    dirs = [path for path in BASE.rglob("*") if path.is_dir() and path.name.lower() in forbidden_dirs]
    if files or dirs:
        raise RuntimeError(f"UI artifact unexpectedly present: {files + dirs}")
    return {"ui_status": "DEFERRED_BY_USER_NOT_EXECUTED", "ui_files_created": False, "ui_server_started": False, "ui_required_for_core_experiment_completion": False, "ui_may_be_developed_after_core_freeze": True, "forbidden_ui_artifacts": [], "status": "PASS"}


def append_and_execute_freeze_notebook(preflight: dict[str, Any]) -> dict[str, Any]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8-sig"))
    if any(cell.get("metadata", {}).get("phase10_stage") == NOTEBOOK_MARKER for cell in notebook["cells"]):
        raise RuntimeError("Core Final Freeze notebook marker already exists; frozen notebook must not be re-executed")
    original_count = len(notebook["cells"])
    original_hashes = [cell_sha256(cell) for cell in notebook["cells"]]
    metadata = {"phase10_stage": NOTEBOOK_MARKER}
    code = """import json\nfrom pathlib import Path\nphase10 = Path.cwd()\nartifact = json.loads((phase10/'audits/phase10_final_synthesis_artifact_audit.json').read_text(encoding='utf-8-sig'))\nstatus = json.loads((phase10/'configs/phase10_final_synthesis_status.json').read_text(encoding='utf-8-sig'))\nprint({'phase10_final_synthesis':artifact['status'],'counts':artifact['counts'],'scientific_source_conflicts':artifact['scientific_source_conflicts'],'unresolved_numerical_differences':artifact['unresolved_numerical_differences'],'phase00_09_files_modified':artifact['phase00_09_files_modified'],'freeze_authorized':status['ready_for_phase10_final_freeze'],'ui':'DEFERRED_BY_USER_NOT_EXECUTED','onlinehd_replay':'OPTIONAL_NOT_EXECUTED'})"""
    notebook["cells"].extend([
        {"cell_type": "markdown", "metadata": metadata, "source": ["## Phase 10 Core Final Freeze\n", "\n", "This cell records the verified read-only freeze preflight. It does not train, predict, recompute statistics, build UI, or execute OnlineHD replay.\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": metadata, "outputs": [], "source": [code]},
    ])
    execution_count = max((cell.get("execution_count") or 0 for cell in notebook["cells"] if cell.get("cell_type") == "code"), default=0) + 1
    cell = notebook["cells"][-1]
    stdout = io.StringIO()
    outputs = []
    failed = False
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            exec(compile(code, "<phase10-core-final-freeze>", "exec"), {"__name__": "__phase10_core_final_freeze__"})
    except Exception as exc:
        failed = True
        outputs.append({"output_type": "error", "ename": type(exc).__name__, "evalue": str(exc), "traceback": traceback.format_exc().splitlines()})
    if stdout.getvalue():
        outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": stdout.getvalue().splitlines(True)})
    cell["execution_count"] = execution_count
    cell["outputs"] = outputs
    notebook.setdefault("metadata", {})["phase10_core_final_freeze_execution"] = {"marker": NOTEBOOK_MARKER, "code_cells_executed": 1, "failed": failed, "historical_cells_reexecuted": False, "ui_code_executed": False}
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    reparsed = json.loads(NOTEBOOK.read_text(encoding="utf-8-sig"))
    original_preserved = [cell_sha256(item) for item in reparsed["cells"][:original_count]] == original_hashes
    freeze_cells = [item for item in reparsed["cells"] if item.get("metadata", {}).get("phase10_stage") == NOTEBOOK_MARKER]
    errors = [output for item in freeze_cells for output in item.get("outputs", []) if output.get("output_type") == "error"]
    passed = original_preserved and len(freeze_cells) == 2 and not errors and not failed
    audit = {"audit": "phase10_final_freeze_notebook_persistence_audit", "notebook_path": str(NOTEBOOK.resolve()), "notebook_sha256": sha256(NOTEBOOK), "original_cell_count": original_count, "original_cells_preserved_exactly": original_preserved, "historical_cells_reexecuted": False, "freeze_summary_cells": len(freeze_cells), "freeze_code_cells_executed": 1, "error_outputs": len(errors), "ui_code_added_or_executed": False, "parseable": True, "status": "PASS" if passed else "FAIL"}
    if not passed:
        raise RuntimeError(f"Final Freeze notebook persistence failed: {audit}")
    save_json("audits/phase10_final_freeze_notebook_persistence_audit.json", audit)
    return audit


def artifact_type(path: Path) -> str:
    relative = path.relative_to(BASE).as_posix()
    if path == NOTEBOOK: return "EXECUTED_NOTEBOOK"
    if relative.startswith("configs/"): return "CONFIG_OR_CONTRACT"
    if relative.startswith("manifests/"): return "MANIFEST_OR_SOURCE_SELECTION"
    if relative.startswith("audits/"): return "AUDIT"
    if relative.startswith("reproducibility/"): return "REPRODUCIBILITY_ARTIFACT"
    if relative.startswith("reports/paper_tables/"): return "PAPER_TABLE_OR_REGISTRY"
    if relative.startswith("reports/paper_figures/"): return "PAPER_FIGURE_REGISTRY"
    if relative.startswith("reports/"): return "SYNTHESIS_OR_FREEZE_REPORT"
    if relative.startswith("results/final_prediction_library/"): return "FINAL_PREDICTION_LIBRARY_INDEX"
    if relative.startswith("results/final_statistics_bundle/"): return "FINAL_STATISTICS_BUNDLE_INDEX"
    if relative.startswith("results/summaries/"): return "FINAL_SYNTHESIS_RESULT"
    if relative.startswith("scripts/"): return "REPRODUCTION_OR_VERIFICATION_SCRIPT"
    return "CORE_PROVENANCE_ARTIFACT"


def scientific_class(path: Path) -> str:
    relative = path.relative_to(BASE).as_posix()
    if relative.startswith(("results/", "reports/paper_", "reports/phase10_rq", "reports/phase10_scientific", "reports/phase10_final_synthesis", "rq_evidence_conclusion_matrix/")):
        return "SCIENTIFIC_DERIVED_OR_FROZEN_SOURCE_INDEX"
    if path == NOTEBOOK:
        return "SCIENTIFIC_DERIVED_EXECUTED_NOTEBOOK"
    return "NONSCIENTIFIC_ENGINEERING_OR_PROVENANCE"


def source_phase(path: Path) -> str:
    match = re.search(r"phase[_-]?(\d\d[a-z]?)", path.name.lower())
    return match.group(1).upper() if match else "10"


def formal_payload_paths() -> list[Path]:
    paths: set[Path] = {NOTEBOOK, BASE / "README.md"}
    for directory in ("configs", "manifests", "audits", "results/final_prediction_library", "results/final_statistics_bundle", "results/summaries", "reports/paper_tables", "reports/paper_figures", "reproducibility", "reproducibility_package", "rq_evidence_conclusion_matrix", "cross_phase_consistency_audit", "final_prediction_library", "final_statistics_bundle", "scripts"):
        root = BASE / directory
        if root.exists():
            paths.update(path for path in root.rglob("*") if path.is_file())
    paths.update(path for path in (BASE / "reports").glob("phase10_*") if path.is_file())
    paths.update(path for path in (BASE / "reports").glob("2026-08-22--phase10-*") if path.is_file())
    excluded_names = {"phase10_final_manifest.json", "phase10_final_manifest_hash_audit.json"}
    return sorted(path for path in paths if path.name not in excluded_names and "__pycache__" not in path.parts and path.suffix.lower() != ".pyc")


def manifest_record(index: int, path: Path) -> dict[str, Any]:
    relative = path.relative_to(BASE).as_posix()
    science = scientific_class(path)
    thesis = science.startswith("SCIENTIFIC") or relative.startswith(("reports/", "reproducibility/")) or path == NOTEBOOK
    ui = relative.startswith("results/final_prediction_library/") or relative in {"configs/phase10_best_dual_task_hdc_interface.json", "configs/phase10_final_status.json", "reports/paper_figures/paper_figure_registry.csv"}
    return {"artifact_id": f"P10-FINAL-{index:04d}", "artifact_type": artifact_type(path), "relative_path": relative, "file_size": path.stat().st_size, "sha256": sha256(path), "source_phase": source_phase(path), "scientific_or_nonscientific": science, "source_of_truth_level": "PHASE10_FROZEN_PAYLOAD" if path.parent != BASE / "scripts" else "REPRODUCTION_ENTRYPOINT", "frozen_status": "FROZEN", "required_for_reproduction": relative.startswith(("configs/", "manifests/", "audits/", "reproducibility/", "scripts/")) or path == NOTEBOOK, "required_for_thesis": thesis, "required_for_ui": ui, "generated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()}


def make_freeze_outputs(preflight: dict[str, Any], notebook_audit: dict[str, Any], upstream: dict[str, Any], caveat: dict[str, Any], ui: dict[str, Any], baseline: dict[str, Any]) -> int:
    common = {"frozen_at_utc": NOW, "historical_caveat_definition": "historical engineering/provenance caveat", "nonscientific_metadata_differences_retained": 6, "scientific_artifact_changes": 0, "historical_frozen_immutability_audit": "FAIL", "primary_data_checksum": "PASS", "frozen_fold_checksum": "PASS", "scientific_consistency": "PASS", "predictions_modified": False, "canonical_oof_modified": False, "statistics_modified": False, "frozen_model_configs_modified": False, "scientific_source_conflicts": 0, "unresolved_numerical_differences": 0, "model_training_executed": False, "predictions_generated": False, "statistics_recomputed": False, "phase00_09_files_modified": 0, "ui_status": "DEFERRED_BY_USER_NOT_EXECUTED", "ui_files_created": False, "ui_server_started": False, "ui_required_for_core_experiment_completion": False, "ui_may_be_developed_after_core_freeze": True, "onlinehd_replay_status": "OPTIONAL_NOT_EXECUTED", "onlinehd_replay_required_for_thesis_core_claims": False}
    freeze_payload = {"phase": "10", "freeze": "phase10_core_final_freeze", "status": "FROZEN", "scope": ["Phase10 core contracts and source-of-truth rules", "Final Prediction Library", "Final Statistics Bundle", "paper tables and figure registries", "RQ evidence matrix", "reproducibility package", "cross-phase audits", "Final Synthesis reports", "executed Phase10 notebook", "formal Phase10 audits"], "final_manifest": "manifests/phase10_final_manifest.json", "manifest_self_hash_seal": "audits/phase10_final_manifest_hash_audit.json", "manifest_seal_design": "The payload manifest excludes itself and its external SHA-256 seal to avoid recursive self-reference.", **common}
    status_payload = {"phase": "10", "phase10_core_status": "FROZEN", "phase00_10_scientific_pipeline_status": "COMPLETE", "core_experiment_project_complete": True, "ready_for_thesis_writing_and_final_packaging": True, "ready_for_optional_local_ui_development": True, "scientific_readiness": "PASS", "engineering_provenance_caveat": "PRESENT", **common}
    save_json("configs/phase10_freeze.json", freeze_payload)
    save_json("configs/phase10_final_status.json", status_payload)
    write_text("reports/phase10_final_freeze_summary.md", f"""# Phase 10 Core Final Freeze Summary

Phase 10 core status is **FROZEN** and the Phase 00–10 scientific pipeline is **COMPLETE**. The frozen payload contains the verified Final Prediction Library (1,406 source records), Final Statistics Bundle (35 artifacts), 14 paper-table candidates, 61 frozen paper-figure references, six RQ evidence rows, the reproducibility package, cross-phase audits, final reports, and the executed Phase 10 notebook.

No model training, prediction generation, tuning, model reselection, statistical recomputation, OnlineHD replay, UI creation, or local server execution occurred. Phase 00–09 file changes during final freeze are zero. Primary data and frozen-fold checksums pass; scientific source conflicts and unresolved numerical differences are zero.

## Historical engineering/provenance caveat

The Phase 06 original manifest SHA-256 remains verified. Six non-scientific metadata differences remain retained. The historical frozen-artifact immutability audit remains **FAIL** for historical non-scientific files; it is not rewritten as PASS. Scientific artifact changes are zero, scientific consistency is PASS, and predictions, canonical OOF, statistics, and frozen model configurations remain unmodified.

Two stale Phase 09 hashes in the earlier Phase 10 initialization reference are preserved alongside the stable current direct Phase 09 freeze chain. The current freeze embeds the current final-manifest hash; this is a non-scientific initialization-reference alignment caveat and does not invalidate scientific results.

## UI and optional replay

UI remains `DEFERRED_BY_USER_NOT_EXECUTED`; no UI file or server exists, and UI is not required for core completion. A later UI may be developed only as an independent read-only display layer. OnlineHD replay remains `OPTIONAL_NOT_EXECUTED` and is not required for thesis core claims.

## Manifest seal

`manifests/phase10_final_manifest.json` freezes the payload. Its own SHA-256 is recorded externally in `audits/phase10_final_manifest_hash_audit.json` to avoid self-referential hashing. After this seal, frozen payload files must not be modified.
""")
    readiness = {"audit": "phase10_final_submission_readiness_audit", "phase10_final_synthesis": "PASS", "core_contract_check": "PASS", "prediction_sources_verified": 1406, "statistical_artifacts_indexed": 35, "paper_tables_verified": 14, "paper_figures_verified": 61, "cross_phase_numerical_consistency": "PASS", "notebook_persistence": notebook_audit["status"], "scientific_readiness": "PASS", "engineering_provenance_caveat": "PRESENT", "historical_frozen_immutability_audit": "FAIL", "ready_for_thesis_writing_and_final_packaging": True, "ready_for_optional_local_ui_development": True, "status": "PASS"}
    freeze_audit = {"audit": "phase10_final_freeze_audit", "preflight": preflight, "upstream_freeze_integrity": upstream["status"], "notebook_persistence": notebook_audit["status"], "historical_caveat": caveat, "ui": ui, "manifest_hash_seal_required": True, "model_training_executed": False, "predictions_generated": False, "statistics_recomputed": False, "phase00_09_files_modified": 0, "status": "PASS"}
    save_json("audits/phase10_final_submission_readiness_audit.json", readiness)
    save_json("audits/phase10_final_freeze_audit.json", freeze_audit)
    paths = formal_payload_paths()
    readiness["planned_final_manifest_artifacts"] = len(paths)
    freeze_audit["planned_final_manifest_artifacts"] = len(paths)
    save_json("audits/phase10_final_submission_readiness_audit.json", readiness)
    save_json("audits/phase10_final_freeze_audit.json", freeze_audit)
    paths = formal_payload_paths()
    records = [manifest_record(index, path) for index, path in enumerate(paths, 1)]
    manifest = {"manifest": "phase10_final_manifest", "phase": "10", "status": "FROZEN", "generated_at": NOW, "artifact_count": len(records), "artifacts": records, "excluded_from_payload": ["manifests/phase10_final_manifest.json (self; sealed externally)", "audits/phase10_final_manifest_hash_audit.json (external self-hash seal)", "logs/**", "tests/**", "**/__pycache__/**", "*.pyc", "task_plan.md", "notes.md", "UI files (none exist)", "historical backups or caches"], "manifest_self_hash_location": "audits/phase10_final_manifest_hash_audit.json", "scientific_source_conflicts": 0, "unresolved_numerical_differences": 0, "phase00_09_files_modified": 0}
    save_json("manifests/phase10_final_manifest.json", manifest)
    manifest_path = BASE / "manifests/phase10_final_manifest.json"
    manifest_hash = sha256(manifest_path)
    save_json("audits/phase10_final_manifest_hash_audit.json", {"audit": "phase10_final_manifest_hash_audit", "manifest_path": str(manifest_path.resolve()), "manifest_sha256": manifest_hash, "manifest_file_size": manifest_path.stat().st_size, "manifest_artifacts": len(records), "algorithm": "SHA-256", "self_reference_policy": "External seal audit; the manifest cannot contain its own final hash without recursive invalidation.", "hash_verified_after_write": sha256(manifest_path) == manifest_hash, "status": "PASS"})
    return len(records)


def main() -> None:
    existing_status = BASE / "configs/phase10_final_status.json"
    if existing_status.exists() and load_json(existing_status).get("phase10_core_status") == "FROZEN":
        raise RuntimeError("Phase 10 is already frozen; the frozen payload cannot be regenerated")
    baseline = phase00_09_state("phase10_core_final_freeze_before")
    save_json("logs/phase10_final_freeze_phase00_09_baseline.json", baseline)
    core = validate_core_contracts()
    synthesis = validate_final_synthesis()
    sources = validate_scientific_sources()
    upstream = validate_upstream_interfaces(baseline)
    caveat = validate_historical_caveat()
    ui = validate_ui_absent()
    preflight = {"core_contract": core, "final_synthesis": synthesis, "scientific_sources": sources, "upstream_interfaces": upstream["status"], "historical_caveat": caveat["status"], "ui": ui["status"], "onlinehd_replay_status": "OPTIONAL_NOT_EXECUTED", "status": "PASS"}
    notebook_audit = append_and_execute_freeze_notebook(preflight)
    after_notebook = phase00_09_state("phase10_core_final_freeze_after_notebook")
    comparison = compare_states(baseline, after_notebook)
    if comparison["modified_count"]:
        raise RuntimeError(f"Phase 00-09 changed during final freeze: {comparison}")
    upstream_audit = {"audit": "phase10_upstream_freeze_integrity_audit", "phase00_09_baseline_files": baseline["file_count"], "phase00_09_comparison": comparison, "interfaces": upstream["interfaces"], "formal_interfaces_verified": upstream["formal_interfaces_verified"], "legacy_phases_covered": upstream["legacy_phases_covered"], "known_phase09_initialization_reference_differences": upstream["known_phase09_initialization_reference_differences"], "primary_data_checksum": upstream["primary_data_checksum"], "frozen_fold_checksum": upstream["frozen_fold_checksum"], "scientific_artifact_changes": 0, "status": "PASS"}
    save_json("audits/phase10_upstream_freeze_integrity_audit.json", upstream_audit)
    artifact_count = make_freeze_outputs(preflight, notebook_audit, upstream, caveat, ui, baseline)
    manifest = load_json(BASE / "manifests/phase10_final_manifest.json")
    failures = []
    for item in manifest["artifacts"]:
        path = BASE / item["relative_path"]
        actual = sha256(path) if path.exists() else "MISSING"
        if actual != item["sha256"]:
            failures.append({"path": str(path), "expected": item["sha256"], "actual": actual})
    final_comparison = compare_states(baseline, phase00_09_state("phase10_core_final_freeze_after_seal"))
    if failures or final_comparison["modified_count"]:
        raise RuntimeError(f"Final seal verification failed: payload={failures[:5]} upstream={final_comparison}")
    print(json.dumps({"status": "PASS", "final_manifest_artifacts": artifact_count, "manifest_sha256": sha256(BASE / "manifests/phase10_final_manifest.json"), "phase00_09_files_modified": 0, "notebook_persistence": "PASS", "phase10_core_status": "FROZEN"}, indent=2))


if __name__ == "__main__":
    main()
