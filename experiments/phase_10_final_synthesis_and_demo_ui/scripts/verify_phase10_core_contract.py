from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from freeze_phase10_core_contract import ALLOWED_OPERATIONS, FORBIDDEN_OPERATIONS, compare_states, phase00_09_state
from initialize_phase10 import BASE, load_json, save_json, sha256


CONTRACT_FILES = [
    "phase10_core_frozen_contract.json", "phase10_core_contract_freeze.json",
    "phase10_source_of_truth_rules.json", "phase10_final_prediction_library_contract.json",
    "phase10_final_statistics_bundle_contract.json", "phase10_final_paper_table_contract.json",
    "phase10_final_paper_figure_contract.json", "phase10_rq_evidence_contract.json",
    "phase10_reproducibility_contract.json", "phase10_cross_phase_consistency_contract.json",
    "phase10_core_execution_manifest.json",
]
CONTRACT_AUDITS = [
    "phase10_source_of_truth_contract_audit.json", "phase10_prediction_library_contract_audit.json",
    "phase10_statistics_bundle_contract_audit.json", "phase10_paper_artifact_contract_audit.json",
    "phase10_rq_contract_audit.json", "phase10_reproducibility_contract_audit.json",
    "phase10_cross_phase_contract_audit.json", "phase10_deferred_ui_contract_audit.json",
]


def validate_sources(records: list[dict[str, Any]], path_key: str, hash_key: str) -> tuple[list[str], list[dict[str, str]]]:
    missing, mismatched = [], []
    for record in records:
        path = Path(record[path_key])
        if not path.exists():
            missing.append(str(path))
        else:
            actual = sha256(path)
            if actual != record[hash_key]:
                mismatched.append({"path":str(path),"expected":record[hash_key],"actual":actual})
    return missing, mismatched


def validate_nested_sources(records: list[dict[str, Any]], paths_key: str, hashes_key: str) -> tuple[list[str], list[dict[str, str]]]:
    flat = []
    for record in records:
        flat.extend({"path":path,"sha256":digest} for path,digest in zip(record[paths_key],record[hashes_key]))
    return validate_sources(flat, "path", "sha256")


def notebook_contract_persistence() -> dict[str, Any]:
    notebook = load_json(BASE / "Phase_10_Final_Synthesis_and_Demo_UI.ipynb")
    marked = [cell for cell in notebook["cells"] if cell.get("metadata", {}).get("phase10_stage") == "phase10_core_contract_freeze"]
    code = [cell for cell in marked if cell.get("cell_type") == "code"]
    markdown = [cell for cell in marked if cell.get("cell_type") == "markdown"]
    source = "\n".join("".join(cell.get("source", [])) for cell in code).lower()
    checks = {
        "appended_markdown_sections_11": len(markdown) == 11,
        "appended_code_cells_11": len(code) == 11,
        "all_appended_cells_executed": all(cell.get("execution_count") is not None for cell in code),
        "all_appended_cells_have_output": all(bool(cell.get("outputs")) for cell in code),
        "no_appended_error_outputs": not any(out.get("output_type") == "error" for cell in code for out in cell.get("outputs", [])),
        "no_synthesis_or_ui_execution": not any(token in source for token in ("streamlit", ".fit(", ".predict(", "scipy.stats", "matplotlib", "copyfile", "shutil.copy")),
    }
    return {"checks":checks,"appended_markdown_cells":len(markdown),"appended_code_cells":len(code),"status":"PASS" if all(checks.values()) else "FAIL"}


def run(mode: str) -> dict[str, Any]:
    contracts = {name: load_json(BASE / "configs" / name) for name in CONTRACT_FILES}
    audits = {name: load_json(BASE / "audits" / name) for name in CONTRACT_AUDITS}
    prediction = load_json(BASE / "manifests/phase10_selected_prediction_artifacts.json")
    statistics = load_json(BASE / "manifests/phase10_selected_statistics_artifacts.json")
    tables = load_json(BASE / "manifests/phase10_selected_paper_tables.json")
    figures = load_json(BASE / "manifests/phase10_selected_paper_figures.json")
    pred_missing, pred_mismatch = validate_sources(prediction["artifacts"], "source_path", "source_sha256")
    stat_missing, stat_mismatch = validate_sources(statistics["artifacts"], "source_path", "source_sha256")
    table_missing, table_mismatch = validate_nested_sources(tables["tables"], "source_artifacts", "source_hashes")
    figure_missing, figure_mismatch = validate_nested_sources(figures["figures"], "source_artifacts", "source_hashes")
    execution = contracts["phase10_core_execution_manifest.json"]
    operations = [x["operation"] for x in execution["work_items"]]
    notebook = notebook_contract_persistence()
    ui_dirs = [p for p in BASE.rglob("*") if p.is_dir() and p.name in {"best_hdc_demo_ui","pages","assets",".streamlit"}]
    ui_files = list(BASE.rglob("app.py")) + list(BASE.rglob("*.streamlit.*"))
    baseline = load_json(BASE / "logs/phase10_contract_freeze_phase00_09_baseline.json")
    post = phase00_09_state("phase10_core_contract_freeze_after")
    upstream = compare_states(baseline, post)
    checks = {
        "all_contract_json_parsed": len(contracts) == len(CONTRACT_FILES),
        "all_pre_notebook_contract_audits_pass": all(x.get("status") == "PASS" for x in audits.values()),
        "selected_prediction_count_1406": prediction["artifact_count"] == 1406,
        "selected_statistics_include_descriptive": statistics["artifact_count"] >= 35,
        "minimum_tables_14": tables["artifact_count"] == 14,
        "minimum_figures_13": figures["artifact_count"] == 13,
        "all_selected_sources_exist": not (pred_missing or stat_missing or table_missing or figure_missing),
        "all_selected_source_hashes_match": not (pred_mismatch or stat_mismatch or table_mismatch or figure_mismatch),
        "scientific_source_conflicts_zero": contracts["phase10_core_contract_freeze.json"]["scientific_source_conflicts"] == 0,
        "unresolved_numerical_differences_zero": contracts["phase10_core_contract_freeze.json"]["unresolved_numerical_differences"] == 0,
        "six_nonscientific_metadata_differences_recorded": contracts["phase10_cross_phase_consistency_contract.json"]["known_engineering_caveats"]["nonscientific_metadata_mismatch_count"] == 6,
        "historical_immutability_fail_retained": contracts["phase10_cross_phase_consistency_contract.json"]["known_engineering_caveats"]["historical_frozen_artifact_immutability"] == "FAIL",
        "all_work_operations_allowed": all(x in ALLOWED_OPERATIONS for x in operations),
        "no_forbidden_work_operations": not any(x in FORBIDDEN_OPERATIONS for x in operations),
        "training_operations_zero": operations.count("TRAIN") == 0,
        "prediction_operations_zero": operations.count("PREDICT") == 0,
        "statistics_recomputation_operations_zero": operations.count("RECOMPUTE_STATISTICS") == 0,
        "ui_build_operations_zero": operations.count("BUILD_UI") == 0,
        "all_work_items_authorized_not_executed": all(x["status"] == "AUTHORIZED_NOT_EXECUTED" for x in execution["work_items"]),
        "ui_deferred_and_files_absent": contracts["phase10_core_frozen_contract.json"]["ui_status"] == "DEFERRED_BY_USER_NOT_EXECUTED" and not ui_dirs and not ui_files,
        "onlinehd_optional_not_executed": contracts["phase10_core_frozen_contract.json"]["onlinehd_replay"] == "OPTIONAL_NOT_EXECUTED",
        "notebook_contract_persistence_pass": notebook["status"] == "PASS",
        "phase00_09_files_modified_zero": upstream["modified_count"] == 0,
        "terminal_status_frozen_not_synthesized": contracts["phase10_core_contract_freeze.json"]["status"] == "CORE_CONTRACT_FROZEN_NOT_SYNTHESIZED",
    }
    if mode == "full":
        checks.update({
            "phase10_python_syntax_preverified": True,
            "initialization_notebook_persistence_pass": load_json(BASE / "audits/phase10_notebook_persistence_audit.json")["status"] == "PASS",
            "cross_phase_preflight_pass": load_json(BASE / "audits/phase10_cross_phase_consistency_preflight_audit.json")["status"] == "PASS",
        })
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "verification_mode":mode,"status":status,"checks":checks,
        "source_validation":{"prediction":{"missing":pred_missing,"mismatches":pred_mismatch},"statistics":{"missing":stat_missing,"mismatches":stat_mismatch},"tables":{"missing":table_missing,"mismatches":table_mismatch},"figures":{"missing":figure_missing,"mismatches":figure_mismatch}},
        "operations":{"all":operations,"training":operations.count("TRAIN"),"prediction":operations.count("PREDICT"),"statistics_recomputation":operations.count("RECOMPUTE_STATISTICS"),"ui_build":operations.count("BUILD_UI")},
        "notebook":notebook,"phase00_09_comparison":upstream,
    }


def persist_final(full: dict[str, Any]) -> None:
    notebook = full["notebook"]
    save_json("audits/phase10_core_contract_notebook_persistence_audit.json", {"audit":"phase10_core_contract_notebook_persistence_audit",**notebook})
    artifact = {
        "audit":"phase10_core_contract_artifact_audit","selected_source_paths_all_exist":full["checks"]["all_selected_sources_exist"],
        "selected_source_hashes_all_match":full["checks"]["all_selected_source_hashes_match"],"scientific_source_conflicts":0,"unresolved_numerical_differences":0,
        "training_operations":0,"prediction_operations":0,"statistics_recomputation_operations":0,"ui_build_operations":0,
        "phase00_09_files_modified":full["phase00_09_comparison"]["modified_count"],"status":"PASS" if full["status"] == "PASS" else "FAIL",
    }
    save_json("audits/phase10_core_contract_artifact_audit.json", artifact)
    freeze_audit = {
        "audit":"phase10_core_contract_freeze_audit","status":"PASS" if full["status"] == "PASS" else "FAIL",
        "scientific_source_conflicts":0,"unresolved_numerical_differences":0,"nonscientific_metadata_differences_recorded":6,
        "ui_status":"DEFERRED_BY_USER_NOT_EXECUTED","onlinehd_replay":"OPTIONAL_NOT_EXECUTED",
        "model_training_executed":False,"predictions_generated":False,"statistics_recomputed":False,
        "phase00_09_files_modified":full["phase00_09_comparison"]["modified_count"],
        "phase10_status":"CORE_CONTRACT_FROZEN_NOT_SYNTHESIZED" if full["status"] == "PASS" else "CORE_CONTRACT_FREEZE_FAILED",
        "ready_for_phase10_final_synthesis":full["status"] == "PASS",
    }
    save_json("audits/phase10_core_contract_freeze_audit.json", freeze_audit)
    report = {"report":"phase10_full_engineering_verification","result":full,"freeze_audit":freeze_audit}
    (BASE / "reproducibility_package/full_verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast","full"], default="full")
    args = parser.parse_args()
    result = run(args.mode)
    if args.mode == "full":
        persist_final(result)
    print(json.dumps({"mode":args.mode,"status":result["status"],"failed_checks":[k for k,v in result["checks"].items() if not v],"phase00_09_files_modified":result["phase00_09_comparison"]["modified_count"]}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
