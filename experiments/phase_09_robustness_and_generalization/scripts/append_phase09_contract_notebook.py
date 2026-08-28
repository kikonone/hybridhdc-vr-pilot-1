"""Append and execute Contract Freeze cells while preserving initialization history cells."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient


PHASE09_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PHASE09_ROOT / "Phase_09_Robustness_and_Generalization.ipynb"
MARKER = "# Phase 09 Contract Freeze"


def cell_fingerprint(cell: nbformat.NotebookNode) -> str:
    return hashlib.sha256(json.dumps(cell, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def contract_cells() -> list[nbformat.NotebookNode]:
    metadata = {"tags": ["phase09-contract-freeze"]}
    return [
        nbformat.v4.new_markdown_cell(
            f"{MARKER}\n\nThis appended section freezes the Phase 09 evidence, model, missing-modality, LOSO, aggregation, and statistical contracts. It performs no training or prediction generation.",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 13. Contract Freeze summary", metadata=metadata),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json, subprocess, sys\n"
            "PHASE09_ROOT = Path.cwd().resolve()\n"
            "sys.path.insert(0, str(PHASE09_ROOT / 'scripts'))\n"
            "from freeze_phase09_contract import run_freeze\n"
            "contract_summary = run_freeze()\ncontract_summary",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 14. Frozen selected-model interfaces", metadata=metadata),
        nbformat.v4.new_code_cell(
            "frozen = json.loads((PHASE09_ROOT / 'configs/phase09_frozen_contract.json').read_text(encoding='utf-8'))\n"
            "frozen['selected_model_interfaces']",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 15. Missing-modality conditions and protocol separation", metadata=metadata),
        nbformat.v4.new_code_cell(
            "missing_contract = json.loads((PHASE09_ROOT / 'configs/phase09_missing_modality_contract.json').read_text(encoding='utf-8'))\n"
            "{'primary_protocol': missing_contract['primary_protocol'], 'conditions': missing_contract['conditions'], 'new_training_runs': missing_contract['new_training_runs'], 'full_reference': missing_contract['full_primary_reference_policy'], 'sudden_test_time_missingness': missing_contract['sudden_test_time_missingness'], 'protocol_separation': missing_contract['protocol_separation']} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 16. Deterministic LOSO assignments", metadata=metadata),
        nbformat.v4.new_code_cell(
            "loso_audit = json.loads((PHASE09_ROOT / 'audits/phase09_loso_assignment_audit.json').read_text(encoding='utf-8'))\n"
            "{'status': loso_audit['status'], 'splits': loso_audit['splits'], 'assignment_rows': loso_audit['assignment_rows'], 'duplicate_run_keys': loso_audit['duplicate_run_keys'], 'checks': loso_audit['checks']} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 17. LOSO configuration mapping and leakage guard", metadata=metadata),
        nbformat.v4.new_code_cell(
            "mapping = json.loads((PHASE09_ROOT / 'configs/phase09_loso_config_mapping.json').read_text(encoding='utf-8'))\n"
            "mapping_audit = json.loads((PHASE09_ROOT / 'audits/phase09_config_mapping_leakage_audit.json').read_text(encoding='utf-8'))\n"
            "{'mapping_rule': mapping['mapping_rule'], 'mapped_subjects': len(mapping['mappings']), 'first_mapping': mapping['mappings'][0], 'last_mapping': mapping['mappings'][-1], 'leakage_audit': mapping_audit} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 18. Dynamically enumerated 720-run execution matrix", metadata=metadata),
        nbformat.v4.new_code_cell(
            "execution = json.loads((PHASE09_ROOT / 'configs/phase09_execution_manifest.json').read_text(encoding='utf-8'))\n"
            "run_audit = json.loads((PHASE09_ROOT / 'audits/phase09_run_matrix_audit.json').read_text(encoding='utf-8'))\n"
            "{'training_run_count': execution['training_run_count'], 'duplicate_run_identifiers': execution['duplicate_run_identifiers'], 'by_protocol': execution['run_counts_by_protocol'], 'by_model': execution['run_counts_by_model'], 'first_run': execution['training_runs'][0], 'last_run': execution['training_runs'][-1], 'audit_status': run_audit['status'], 'all_statuses': sorted({item['status'] for item in execution['training_runs']})} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 19. Read-only checkpoint portability conclusion", metadata=metadata),
        nbformat.v4.new_code_cell(
            "portability = json.loads((PHASE09_ROOT / 'audits/phase09_checkpoint_portability_audit.json').read_text(encoding='utf-8'))\n"
            "{'audit_status': portability['status'], 'protocol_status': portability['protocol_status'], 'interfaces': {key: {'portable': value['portable'], 'checks': value['checks'], 'reproduction_attempted': value['reproduction_attempted']} for key, value in portability['interfaces'].items()}, 'training_executed': portability['training_executed'], 'predictions_generated': portability['predictions_generated']} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 20. Generalization boundaries and statistical unit", metadata=metadata),
        nbformat.v4.new_code_cell(
            "guardrails = json.loads((PHASE09_ROOT / 'audits/phase09_generalization_guardrail_audit.json').read_text(encoding='utf-8'))\n"
            "statistics = json.loads((PHASE09_ROOT / 'configs/phase09_statistical_rules.json').read_text(encoding='utf-8'))\n"
            "{'guardrail_status': guardrails['status'], 'allowed': guardrails['allowed'], 'not_feasible': guardrails['not_feasible'], 'forbidden_scenario_proxies': guardrails['forbidden_scenario_proxies'], 'statistical_unit': statistics['statistical_unit'], 'formal_statistics_executed': statistics['formal_statistics_executed']} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 21. Static Contract Freeze tests and audits", metadata=metadata),
        nbformat.v4.new_code_cell(
            "test_process = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(PHASE09_ROOT / 'tests'), '-p', 'test_phase09_contract.py', '-v'], cwd=PHASE09_ROOT, capture_output=True, text=True, check=False)\n"
            "assert test_process.returncode == 0, test_process.stdout + test_process.stderr\n"
            "audit_names = ['phase09_contract_freeze_audit.json', 'phase09_run_matrix_audit.json', 'phase09_loso_assignment_audit.json', 'phase09_config_mapping_leakage_audit.json', 'phase09_missing_modality_contract_audit.json', 'phase09_checkpoint_portability_audit.json', 'phase09_generalization_guardrail_audit.json']\n"
            "audit_statuses = {name: json.loads((PHASE09_ROOT / 'audits' / name).read_text(encoding='utf-8'))['status'] for name in audit_names}\n"
            "{'static_tests': 'PASS', 'audit_statuses': audit_statuses, 'details': (test_process.stdout + test_process.stderr)[-3000:], 'model_training_executed': False, 'predictions_generated': False} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 22. Final Contract Freeze persistence state", metadata=metadata),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json\nPHASE09_ROOT = Path.cwd().resolve()\n"
            "freeze_config = json.loads((PHASE09_ROOT / 'configs/phase09_contract_freeze.json').read_text(encoding='utf-8'))\n"
            "artifact_path = PHASE09_ROOT / 'audits/phase09_contract_artifact_audit.json'\n"
            "artifact = json.loads(artifact_path.read_text(encoding='utf-8')) if artifact_path.exists() else {'status': 'PENDING_EXTERNAL_PERSISTENCE_AUDIT'}\n"
            "{'phase09_status': freeze_config['status'], 'contract_artifact_audit': artifact['status'], 'authorized_training_runs': freeze_config['authorized_training_runs'], 'model_training_executed': freeze_config['training_executed'], 'predictions_generated': freeze_config['predictions_generated'], 'ready_for_execution': freeze_config['ready_for_execution']} ",
            metadata=metadata,
        ),
    ]


def execute_cells(notebook: nbformat.NotebookNode, indices: list[int]) -> None:
    client = NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(PHASE09_ROOT)}})
    with client.setup_kernel():
        for index in indices:
            if notebook.cells[index].cell_type == "code":
                client.execute_cell(notebook.cells[index], index)


def persistence_audit(notebook: nbformat.NotebookNode, original_fingerprints: list[str], appended_start: int) -> dict[str, Any]:
    original_unchanged = [cell_fingerprint(cell) for cell in notebook.cells[:appended_start]] == original_fingerprints
    appended_code = [cell for cell in notebook.cells[appended_start:] if cell.cell_type == "code"]
    errors = [output for cell in appended_code for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    required = ["Contract Freeze summary", "Frozen selected-model", "Missing-modality conditions", "LOSO assignments", "LOSO configuration mapping", "720-run execution matrix", "checkpoint portability", "Generalization boundaries", "Static Contract Freeze tests"]
    appended_text = "\n".join(str(cell.get("source", "")) for cell in notebook.cells[appended_start:])
    status = "PASS" if (
        original_unchanged and appended_code and all(cell.get("execution_count") is not None for cell in appended_code)
        and all(cell.get("outputs") for cell in appended_code) and not errors
        and all(value.lower() in appended_text.lower() for value in required)
    ) else "FAIL"
    audit = {
        "phase": "09", "audit": "contract_notebook_persistence", "status": status,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "path": str(NOTEBOOK_PATH.resolve()),
        "bytes": NOTEBOOK_PATH.stat().st_size, "sha256": hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest(),
        "initialization_history_cells": appended_start, "initialization_history_unchanged": original_unchanged,
        "appended_code_cells": len(appended_code), "executed_appended_code_cells": sum(cell.get("execution_count") is not None for cell in appended_code),
        "appended_code_cells_with_outputs": sum(bool(cell.get("outputs")) for cell in appended_code), "error_outputs": len(errors),
        "required_sections_present": all(value.lower() in appended_text.lower() for value in required),
        "model_training_executed": False, "predictions_generated": False,
    }
    write_json(PHASE09_ROOT / "audits" / "phase09_contract_notebook_persistence_audit.json", audit)
    return audit


def append_execute_and_finalize() -> dict[str, Any]:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    marker_index = next((index for index, cell in enumerate(notebook.cells) if cell.cell_type == "markdown" and str(cell.source).startswith(MARKER)), len(notebook.cells))
    notebook.cells = notebook.cells[:marker_index]
    original_fingerprints = [cell_fingerprint(cell) for cell in notebook.cells]
    appended_start = len(notebook.cells)
    notebook.cells.extend(contract_cells())
    appended_code_indices = [index for index in range(appended_start, len(notebook.cells)) if notebook.cells[index].cell_type == "code"]
    execute_cells(notebook, appended_code_indices)
    nbformat.write(notebook, NOTEBOOK_PATH)
    persistence = persistence_audit(notebook, original_fingerprints, appended_start)
    from freeze_phase09_contract import finalize_contract_artifacts
    artifact = finalize_contract_artifacts()

    final_code_index = appended_code_indices[-1]
    execute_cells(notebook, [final_code_index])
    nbformat.write(notebook, NOTEBOOK_PATH)
    persistence = persistence_audit(notebook, original_fingerprints, appended_start)
    artifact = finalize_contract_artifacts()
    if persistence["status"] != "PASS" or artifact["status"] != "PASS":
        raise RuntimeError({"persistence": persistence, "artifact": artifact})
    return {
        "notebook_persistence": persistence["status"], "initialization_history_unchanged": persistence["initialization_history_unchanged"],
        "contract_artifact_audit": artifact["status"], "phase09_status": artifact["phase09_status"],
        "ready_for_execution": artifact["ready_for_execution"], "model_training_executed": False, "predictions_generated": False,
    }


if __name__ == "__main__":
    print(json.dumps(append_execute_and_finalize(), ensure_ascii=False, indent=2))
