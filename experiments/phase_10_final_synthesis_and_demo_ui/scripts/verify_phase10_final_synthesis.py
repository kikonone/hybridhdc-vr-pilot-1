from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from freeze_phase10_core_contract import compare_states, phase00_09_state
from initialize_phase10 import BASE, load_json, sha256


STATUS = "FINAL_SYNTHESIS_COMPLETE_PENDING_PHASE10_FREEZE"
MARKER = "CORE_FINAL_SYNTHESIS_V1"


def save_json(relative: str, payload: Any) -> None:
    path = BASE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def cell_sha256(cell: dict[str, Any]) -> str:
    payload = json.dumps(cell, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_index_sources(path: Path, path_field: str, hash_field: str) -> tuple[int, list[dict[str, str]]]:
    failures = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        source = Path(row[path_field])
        actual = sha256(source) if source.exists() else "MISSING"
        if actual != row[hash_field]:
            failures.append({"path": str(source), "expected": row[hash_field], "actual": actual})
    return len(rows), failures


def main() -> None:
    required = [
        "results/final_prediction_library/final_prediction_library_index.csv",
        "results/final_prediction_library/final_prediction_library_manifest.json",
        "results/final_statistics_bundle/final_statistics_index.csv",
        "results/final_statistics_bundle/final_effect_size_index.csv",
        "results/final_statistics_bundle/final_confidence_interval_index.csv",
        "results/final_statistics_bundle/final_statistical_test_index.csv",
        "results/final_statistics_bundle/final_statistics_manifest.json",
        "reports/paper_tables/paper_table_registry.csv", "reports/paper_tables/paper_table_source_map.json",
        "reports/paper_figures/paper_figure_registry.csv", "reports/paper_figures/paper_figure_source_map.json",
        "reports/phase10_rq_experiment_evidence_conclusion_matrix.md",
        "results/summaries/phase10_rq_experiment_evidence_conclusion_matrix.csv",
        "reproducibility/README.md", "reproducibility/environment_summary.json", "reproducibility/execution_order.md",
        "reproducibility/frozen_artifact_registry.csv", "reproducibility/checksum_verification.json",
        "reproducibility/notebook_registry.csv", "reproducibility/script_registry.csv",
        "reproducibility/reproduction_scope_and_limits.md",
        "audits/phase10_cross_phase_numerical_consistency_audit.json",
        "reports/phase10_final_synthesis_report.md", "reports/phase10_scientific_claims_and_limitations.md",
        "reports/phase10_thesis_artifact_inventory.md", "results/summaries/phase10_final_key_results.csv",
        "results/summaries/phase10_final_key_findings.json",
    ]
    missing = [name for name in required if not (BASE / name).exists()]
    prediction_count, prediction_failures = verify_index_sources(BASE / required[0], "source_path", "source_sha256")
    statistic_count, statistic_failures = verify_index_sources(BASE / required[2], "source_path", "source_sha256")
    figure_count, figure_failures = verify_index_sources(BASE / "reports/paper_figures/paper_figure_registry.csv", "source_path", "source_sha256")
    table_map = load_json(BASE / "reports/paper_tables/paper_table_source_map.json")
    table_failures = []
    for table in table_map["tables"]:
        for item in table["sources"]:
            source = Path(item["source_path"])
            actual = sha256(source) if source.exists() else "MISSING"
            if actual != item["source_sha256"]:
                table_failures.append({"path": str(source), "expected": item["source_sha256"], "actual": actual})
            if item["exact_copy_path"] and sha256(Path(item["exact_copy_path"])) != item["source_sha256"]:
                table_failures.append({"path": item["exact_copy_path"], "expected": item["source_sha256"], "actual": sha256(Path(item["exact_copy_path"]))})
    checksum = load_json(BASE / "reproducibility/checksum_verification.json")
    numerical = load_json(BASE / "audits/phase10_cross_phase_numerical_consistency_audit.json")
    generation = load_json(BASE / "audits/phase10_final_synthesis_generation_audit.json")
    notebook_path = BASE / "Phase_10_Final_Synthesis_and_Demo_UI.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    snapshot = load_json(BASE / "logs/phase10_notebook_pre_final_synthesis_snapshot.json")
    original_cells = notebook["cells"][:snapshot["original_cell_count"]]
    original_preserved = [cell_sha256(cell) for cell in original_cells] == snapshot["original_cell_hashes"]
    new_cells = [cell for cell in notebook["cells"] if cell.get("metadata", {}).get("phase10_stage") == MARKER]
    new_code = [cell for cell in new_cells if cell.get("cell_type") == "code"]
    notebook_errors = [output for cell in new_code for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    notebook_pass = original_preserved and len(new_code) == 10 and all(cell.get("execution_count") is not None for cell in new_code) and not notebook_errors
    save_json("audits/phase10_final_synthesis_notebook_persistence_audit.json", {
        "audit": "phase10_final_synthesis_notebook_persistence_audit", "notebook_path": str(notebook_path.resolve()),
        "notebook_sha256": sha256(notebook_path), "original_cell_count": snapshot["original_cell_count"],
        "original_cells_preserved_exactly": original_preserved, "final_synthesis_cells": len(new_cells),
        "final_synthesis_code_cells": len(new_code), "executed_final_synthesis_code_cells": sum(cell.get("execution_count") is not None for cell in new_code),
        "error_outputs": len(notebook_errors), "historical_cells_reexecuted": False, "parseable": True,
        "status": "PASS" if notebook_pass else "FAIL",
    })
    baseline = load_json(BASE / "logs/phase10_final_synthesis_phase00_09_baseline.json")
    upstream_comparison = compare_states(baseline, phase00_09_state("phase10_final_synthesis_final_verification"))
    caveat = load_json(BASE.parents[1] / "audits/pre_submission_repair/frozen_artifact_immutability_audit.json")
    scientific = load_json(BASE.parents[1] / "audits/pre_submission_repair/final_scientific_immutability_audit.json")
    counts = {
        "prediction_sources": prediction_count, "statistical_artifacts": statistic_count,
        "paper_tables": count_csv(BASE / "reports/paper_tables/paper_table_registry.csv"),
        "paper_figures": figure_count, "rq_rows": count_csv(BASE / "results/summaries/phase10_rq_experiment_evidence_conclusion_matrix.csv"),
    }
    failures = {
        "missing_required_outputs": missing, "prediction_source_failures": prediction_failures,
        "statistical_source_failures": statistic_failures, "table_source_failures": table_failures,
        "figure_source_failures": figure_failures, "reproducibility_checksum_failures": checksum["failures"],
        "upstream_changes": upstream_comparison,
    }
    overall = (
        not missing and not any((prediction_failures, statistic_failures, table_failures, figure_failures, checksum["failures"]))
        and upstream_comparison["modified_count"] == 0 and numerical["status"] == "PASS"
        and generation["status"] == "PASS" and notebook_pass and counts == {"prediction_sources": 1406, "statistical_artifacts": 35, "paper_tables": 14, "paper_figures": 61, "rq_rows": 6}
        and caveat["status"] == "FAIL" and caveat["scientific_consistency"] == "PASS"
        and scientific["predictions_modified"] is False and scientific["canonical_oof_modified"] is False
        and scientific["statistics_modified"] is False and scientific["frozen_model_configs_modified"] is False
    )
    artifacts = [
        {"path": str((BASE / name).resolve()), "sha256": sha256(BASE / name), "bytes": (BASE / name).stat().st_size}
        for name in required
    ]
    save_json("audits/phase10_final_synthesis_artifact_audit.json", {
        "audit": "phase10_final_synthesis_artifact_audit", "required_artifacts_verified": len(artifacts),
        "artifacts": artifacts, "counts": counts, "failures": failures,
        "scientific_source_conflicts": numerical["scientific_source_conflicts"],
        "unresolved_numerical_differences": numerical["unresolved_numerical_differences"],
        "nonscientific_metadata_differences_retained": 6,
        "historical_changed_nonscientific_files": caveat["production_frozen_artifact_hash_changes"],
        "historical_frozen_immutability_audit": caveat["status"],
        "scientific_consistency": caveat["scientific_consistency"],
        "predictions_modified": scientific["predictions_modified"], "canonical_oof_modified": scientific["canonical_oof_modified"],
        "statistics_modified": scientific["statistics_modified"], "frozen_model_configurations_modified": scientific["frozen_model_configs_modified"],
        "model_training_executed": False, "predictions_generated": False, "statistics_recomputed": False,
        "phase00_09_files_modified": upstream_comparison["modified_count"], "ui_files_created": False,
        "ui_status": "DEFERRED_BY_USER_NOT_EXECUTED", "onlinehd_replay_status": "OPTIONAL_NOT_EXECUTED",
        "phase10_status": STATUS, "phase10_final_frozen": False, "status": "PASS" if overall else "FAIL",
    })
    save_json("configs/phase10_final_synthesis_status.json", {
        "phase": "10", "status": STATUS, "final_synthesis_complete": bool(overall), "phase10_final_frozen": False,
        "ready_for_phase10_final_freeze": bool(overall), "ui_status": "DEFERRED_BY_USER_NOT_EXECUTED",
        "onlinehd_replay_status": "OPTIONAL_NOT_EXECUTED",
    })
    print(json.dumps({"status": "PASS" if overall else "FAIL", "counts": counts, "phase00_09_modified": upstream_comparison["modified_count"], "notebook": "PASS" if notebook_pass else "FAIL", "ready_for_phase10_final_freeze": bool(overall)}, indent=2))
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
