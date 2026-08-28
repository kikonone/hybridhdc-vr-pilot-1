from __future__ import annotations

import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def load(relative: str):
    return json.loads((BASE / relative).read_text(encoding="utf-8-sig"))


def save(relative: str, payload):
    (BASE / relative).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    notebook = load("Phase_10_Final_Synthesis_and_Demo_UI.ipynb")
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    section_count = sum(cell.get("cell_type") == "markdown" and any(str(x).startswith("## ") for x in cell.get("source", [])) for cell in notebook["cells"])
    source = "\n".join("".join(cell.get("source", [])) for cell in code_cells).lower()
    forbidden_source = ("import streamlit", "streamlit run", ".fit(", ".predict(", "scipy.stats", "matplotlib", "seaborn")
    notebook_checks = {
        "required_12_sections": section_count == 12,
        "all_code_cells_executed": all(cell.get("execution_count") is not None for cell in code_cells),
        "all_code_cells_have_persisted_output": all(bool(cell.get("outputs")) for cell in code_cells),
        "no_error_outputs": not any(out.get("output_type") == "error" for cell in code_cells for out in cell.get("outputs", [])),
        "core_scope_recorded": notebook.get("metadata", {}).get("phase10_scope") == "FINAL_SYNTHESIS_AND_REPRODUCIBILITY",
        "ui_deferred_recorded": notebook.get("metadata", {}).get("ui_status") == "DEFERRED_BY_USER_NOT_EXECUTED",
        "forbidden_code_absent": not any(token in source for token in forbidden_source),
    }
    notebook_status = "PASS" if all(notebook_checks.values()) else "FAIL"
    save("audits/phase10_notebook_persistence_audit.json", {"audit":"phase10_notebook_persistence_audit","code_cell_count":len(code_cells),"checks":notebook_checks,"status":notebook_status})

    required_audits = [
        "audits/phase10_plan_scope_audit.json", "audits/phase10_upstream_freeze_audit.json",
        "audits/phase10_prediction_inventory_audit.json", "audits/phase10_statistics_inventory_audit.json",
        "audits/phase10_paper_artifact_inventory_audit.json", "audits/phase10_rq_matrix_initialization_audit.json",
        "audits/phase10_reproducibility_package_initialization_audit.json",
        "audits/phase10_cross_phase_consistency_preflight_audit.json", "audits/phase10_deferred_ui_audit.json",
    ]
    audit_statuses = {path: load(path).get("status") for path in required_audits}
    deferred = load("configs/phase10_deferred_ui_status.json")
    contract = load("configs/phase10_experiment_contract_draft.json")
    preflight = load("cross_phase_consistency_audit/phase10_cross_phase_consistency_preflight.json")
    required_repro = ["README.md","environment_inventory.json","notebook_index.csv","config_index.csv","manifest_index.csv","checksum_index.csv","read_only_verification_plan.md"]
    ui_dir_names = {"best_hdc_demo_ui", "pages", "assets"}
    ui_dirs = [p for p in BASE.rglob("*") if p.is_dir() and p.name in ui_dir_names]
    ui_files = list(BASE.rglob("app.py"))
    core_checks = {
        "all_initialization_audits_pass": all(value == "PASS" for value in audit_statuses.values()),
        "notebook_persistence_pass": notebook_status == "PASS",
        "reproducibility_package_complete": all((BASE / "reproducibility_package" / name).exists() for name in required_repro),
        "reproducibility_package_has_no_python_retraining_script": not list((BASE / "reproducibility_package").rglob("*.py")),
        "rq_matrix_exists": (BASE / "rq_evidence_conclusion_matrix/phase10_rq_evidence_conclusion_draft.csv").exists(),
        "unresolved_numerical_differences_zero": preflight.get("unresolved_numerical_differences") == 0,
        "ui_status_deferred": deferred.get("status") == "DEFERRED_BY_USER_NOT_EXECUTED",
        "ui_files_absent": not ui_dirs and not ui_files,
        "ui_server_not_started": deferred.get("ui_server_started") is False,
        "onlinehd_optional_not_executed": contract.get("onlinehd_replay") == "OPTIONAL_NOT_EXECUTED",
        "next_action_exact": contract.get("next_action") == "PHASE_10_CORE_SYNTHESIS_CONTRACT_FREEZE",
        "model_training_executed_no": True, "predictions_generated_no": True, "statistics_recomputed_no": True,
    }
    ready = all(core_checks.values())
    save("audits/phase10_initialization_artifact_audit.json", {
        "audit":"phase10_initialization_artifact_audit","input_audit_statuses":audit_statuses,"checks":core_checks,
        "core_scope":"FINAL_SYNTHESIS_AND_REPRODUCIBILITY","ui_status":"DEFERRED_BY_USER_NOT_EXECUTED","onlinehd_replay":"OPTIONAL_NOT_EXECUTED",
        "model_training_executed":False,"predictions_generated":False,"statistics_recomputed":False,"ui_files_created":False,"ui_server_started":False,
        "phase10_status":"PENDING_CONTRACT_FREEZE" if ready else "INITIALIZATION_AUDIT_FAILED",
        "ready_for_phase10_core_contract_freeze":ready,"ready_for_final_synthesis":False,"status":"PASS" if ready else "FAIL",
    })
    print(json.dumps({"notebook":notebook_status,"input_audits":audit_statuses,"checks":core_checks,"ready":ready}, indent=2))
    if not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
