"""Build, execute, persist, and audit the Phase 08 initialization/contract notebook."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient


PHASE08_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PHASE08_ROOT / "Phase_08_Fusion_and_Shortcut_Analysis.ipynb"


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Phase 08 — Fusion and Shortcut Analysis\n\n"
            "## tl;dr\n\n"
            "This executed diagnostic notebook initializes and audits Phase 08 only. "
            "It performs no training, prediction, OOF generation, or fusion execution."
        ),
        nbformat.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "All data, feature membership, folds, and upstream model interfaces are read-only frozen inputs. "
            "Feature groups come from explicit manifests, never column-name prefix inference.\n\n"
            "### Key Assumptions\n\n"
            "- Primary without-performance remains the only main evidence.\n"
            "- With-performance and performance-only are auxiliary shortcut-risk conditions.\n"
            "- Initialization begins at `PENDING_CONTRACT_FREEZE`; the final section freezes the contract without modeling."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json, os, platform, sys\n"
            "PHASE08_ROOT = Path.cwd().resolve()\n"
            "sys.path.insert(0, str(PHASE08_ROOT / 'scripts'))\n"
            "environment = {\n"
            "    'python_version': sys.version,\n"
            "    'python_executable': sys.executable,\n"
            "    'platform': platform.platform(),\n"
            "    'working_directory': os.getcwd(),\n"
            "}\n"
            "environment"
        ),
        nbformat.v4.new_markdown_cell("## Data\n\n### 1. Run the read-only initialization audit"),
        nbformat.v4.new_code_cell(
            "from initialize_phase08 import DATA_PATHS, run_initialization\n"
            "if (PHASE08_ROOT / 'configs' / 'phase08_frozen_contract.json').exists():\n"
            "    summary = json.loads((PHASE08_ROOT / 'audits' / 'phase08_initialization_summary.json').read_text(encoding='utf-8'))\n"
            "else:\n"
            "    summary = run_initialization()\n"
            "summary"
        ),
        nbformat.v4.new_markdown_cell("### 2. Verify datasets, checksums, sizes, targets, and folds"),
        nbformat.v4.new_code_cell(
            "input_fold_audit = json.loads((PHASE08_ROOT / 'audits' / 'phase08_input_and_fold_audit.json').read_text(encoding='utf-8'))\n"
            "{\n"
            "    'status': input_fold_audit['status'],\n"
            "    'dataset_exists': {name: path.exists() for name, path in DATA_PATHS.items() if name != 'folds'},\n"
            "    'sha256': input_fold_audit['actual_hashes'],\n"
            "    'checksum_pass': input_fold_audit['checksum_pass'],\n"
            "    'dataset_summaries': input_fold_audit['dataset_summaries'],\n"
            "    'target_class_values': input_fold_audit['target_class_values'],\n"
            "    'target_score_values': input_fold_audit['target_score_values'],\n"
            "    'outer_folds': input_fold_audit['outer_folds'],\n"
            "    'fold_checks': input_fold_audit['fold_checks'],\n"
            "}"
        ),
        nbformat.v4.new_markdown_cell("## Results\n\n### 3. Verify cross-dataset alignment and feature-set relations"),
        nbformat.v4.new_code_cell(
            "alignment = json.loads((PHASE08_ROOT / 'audits' / 'phase08_dataset_alignment_audit.json').read_text(encoding='utf-8'))\n"
            "alignment"
        ),
        nbformat.v4.new_markdown_cell("### 4. Verify manifest-derived fusion feature counts and exclusions"),
        nbformat.v4.new_code_cell(
            "fusion = json.loads((PHASE08_ROOT / 'audits' / 'phase08_fusion_mapping_audit.json').read_text(encoding='utf-8'))\n"
            "fusion"
        ),
        nbformat.v4.new_markdown_cell("### 5. Persist the 59-feature performance shortcut-risk inventory"),
        nbformat.v4.new_code_cell(
            "performance_risk = json.loads((PHASE08_ROOT / 'audits' / 'phase08_performance_feature_risk_inventory.json').read_text(encoding='utf-8'))\n"
            "{\n"
            "    'status': performance_risk['status'],\n"
            "    'feature_count': performance_risk['feature_count'],\n"
            "    'summary': performance_risk['summary'],\n"
            "    'all_feature_names': [item['feature'] for item in performance_risk['features']],\n"
            "    'target_identifier_exclusion_check': all(not item['reserved_field_name_collision'] for item in performance_risk['features']),\n"
            "}"
        ),
        nbformat.v4.new_markdown_cell("### 6. Verify Phase 04A, 04B, 06, and 07 frozen interfaces"),
        nbformat.v4.new_code_cell(
            "upstream = json.loads((PHASE08_ROOT / 'audits' / 'phase08_upstream_freeze_interface_audit.json').read_text(encoding='utf-8'))\n"
            "upstream"
        ),
        nbformat.v4.new_markdown_cell("### 7. Verify initialized artifacts and prohibited operations"),
        nbformat.v4.new_code_cell(
            "artifact_audit = json.loads((PHASE08_ROOT / 'audits' / 'phase08_initialization_artifact_audit.json').read_text(encoding='utf-8'))\n"
            "{\n"
            "    'artifact_audit': artifact_audit,\n"
            "    'training_executed': 'NO',\n"
            "    'outer_test_predictions_generated': 'NO',\n"
            "    'ready_for_contract_freeze_pre_persistence_audit': summary['ready_for_contract_freeze_pre_notebook'],\n"
            "    'ready_for_modeling': 'NO',\n"
            "}"
        ),
        nbformat.v4.new_markdown_cell(
            "## Phase 08 Contract Freeze\n\n"
            "This appended section freezes the execution contract, flight provenance sensitivity design, "
            "statistics, shortcut-evidence wording, and Phase 09 metadata handoff. It does not train models or create predictions."
        ),
        nbformat.v4.new_markdown_cell("### 8. Freeze contract artifacts after revalidating the initialization gate"),
        nbformat.v4.new_code_cell(
            "from freeze_phase08_contract import freeze_contract\n"
            "contract_summary = freeze_contract()\n"
            "contract_summary"
        ),
        nbformat.v4.new_markdown_cell("### 9. Data conditions, fusion matrix, and shortcut conditions"),
        nbformat.v4.new_code_cell(
            "frozen_contract = json.loads((PHASE08_ROOT / 'configs' / 'phase08_frozen_contract.json').read_text(encoding='utf-8'))\n"
            "{\n"
            "    'evidence_roles': frozen_contract['evidence_roles'],\n"
            "    'fusion_conditions': frozen_contract['fusion_conditions'],\n"
            "    'shortcut_conditions': frozen_contract['shortcut_conditions'],\n"
            "}"
        ),
        nbformat.v4.new_markdown_cell("### 10. Flight provenance categories and sensitivity feasibility"),
        nbformat.v4.new_code_cell(
            "flight_manifest = json.loads((PHASE08_ROOT / 'manifests' / 'phase08_flight_feature_provenance_manifest.json').read_text(encoding='utf-8'))\n"
            "model_matrix = json.loads((PHASE08_ROOT / 'configs' / 'phase08_model_matrix.json').read_text(encoding='utf-8'))\n"
            "{\n"
            "    'flight_features_audited': flight_manifest['feature_count'],\n"
            "    'category_counts': flight_manifest['category_counts'],\n"
            "    'flight_sensitivity_conditions': model_matrix['flight_sensitivity_conditions'],\n"
            "}"
        ),
        nbformat.v4.new_markdown_cell("### 11. Frozen HDC/traditional interfaces and dynamic run count"),
        nbformat.v4.new_code_cell(
            "{\n"
            "    'interfaces': model_matrix['interfaces'],\n"
            "    'run_counts': model_matrix['run_counts'],\n"
            "    'duplicate_run_identifiers': json.loads((PHASE08_ROOT / 'configs' / 'phase08_execution_manifest.json').read_text(encoding='utf-8'))['duplicate_run_identifiers'],\n"
            "}"
        ),
        nbformat.v4.new_markdown_cell("### 12. Comparison families, statistics, and shortcut-evidence wording"),
        nbformat.v4.new_code_cell(
            "statistics = json.loads((PHASE08_ROOT / 'configs' / 'phase08_statistical_analysis_contract.json').read_text(encoding='utf-8'))\n"
            "shortcut = json.loads((PHASE08_ROOT / 'configs' / 'phase08_shortcut_evidence_contract.json').read_text(encoding='utf-8'))\n"
            "{\n"
            "    'comparison_families': frozen_contract['comparison_families'],\n"
            "    'oof_aggregation': frozen_contract['oof_aggregation'],\n"
            "    'statistics': statistics,\n"
            "    'shortcut_evidence_wording': shortcut,\n"
            "}"
        ),
        nbformat.v4.new_markdown_cell("### 13. Phase 09 metadata feasibility handoff"),
        nbformat.v4.new_code_cell(
            "handoff = json.loads((PHASE08_ROOT / 'manifests' / 'phase08_to_phase09_generalization_handoff.json').read_text(encoding='utf-8'))\n"
            "{\n"
            "    'holdout_feasibility': {name: value['feasibility'] for name, value in handoff['holdouts'].items()},\n"
            "    'guardrails': handoff['generalization_guardrails'],\n"
            "}"
        ),
        nbformat.v4.new_markdown_cell("### 14. Run static tests without invoking modeling"),
        nbformat.v4.new_code_cell(
            "import subprocess\n"
            "test_process = subprocess.run(\n"
            "    [sys.executable, '-m', 'unittest', 'discover', '-s', str(PHASE08_ROOT / 'tests'), '-p', 'test_phase08_contract.py', '-v'],\n"
            "    cwd=PHASE08_ROOT, capture_output=True, text=True, check=False,\n"
            ")\n"
            "assert test_process.returncode == 0, test_process.stdout + test_process.stderr\n"
            "{'returncode': test_process.returncode, 'result': 'PASS', 'details': (test_process.stdout + test_process.stderr)[-3000:]}"
        ),
        nbformat.v4.new_markdown_cell("### 15. Confirm execution readiness and prohibited-operation state"),
        nbformat.v4.new_code_cell(
            "freeze_audit = json.loads((PHASE08_ROOT / 'audits' / 'phase08_contract_freeze_audit.json').read_text(encoding='utf-8'))\n"
            "{\n"
            "    'contract_freeze': freeze_audit['status'],\n"
            "    'training_executed': 'NO',\n"
            "    'outer_test_predictions_generated': 'NO',\n"
            "    'ready_for_execution_pending_notebook_persistence_audit': freeze_audit['status'] == 'PASS',\n"
            "}"
        ),
        nbformat.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "Phase 08 is contract-frozen and not trained. The persisted contract authorizes a later, separate execution step only after this notebook's external persistence audit passes."
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": sys.version.split()[0]}
    return notebook


def execute_and_audit() -> dict[str, Any]:
    notebook = build_notebook()
    nbformat.write(notebook, NOTEBOOK_PATH)
    client = NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(PHASE08_ROOT)}})
    executed = client.execute()
    nbformat.write(executed, NOTEBOOK_PATH)

    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    error_outputs = [
        output for cell in code_cells for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    output_cells = sum(bool(cell.get("outputs")) for cell in code_cells)
    persistence_pass = (
        NOTEBOOK_PATH.exists()
        and all(cell.get("execution_count") is not None for cell in code_cells)
        and not error_outputs
        and output_cells == len(code_cells)
    )
    audit = {
        "status": "PASS" if persistence_pass else "FAIL",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "notebook_path": str(NOTEBOOK_PATH),
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": sum(cell.get("execution_count") is not None for cell in code_cells),
        "code_cells_with_persisted_outputs": output_cells,
        "error_output_count": len(error_outputs),
        "training_executed": False,
        "outer_test_predictions_generated": False,
        "ready_for_contract_freeze": persistence_pass,
        "ready_for_modeling": False,
    }
    write_json(PHASE08_ROOT / "audits" / "phase08_notebook_persistence_audit.json", audit)

    artifact_path = PHASE08_ROOT / "audits" / "phase08_initialization_artifact_audit.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    required_artifacts = dict(artifact["required_artifacts"])
    required_artifacts["Phase_08_Fusion_and_Shortcut_Analysis.ipynb"] = NOTEBOOK_PATH.is_file()
    required_artifacts["audits/phase08_initialization_artifact_audit.json"] = artifact_path.is_file()
    required_artifacts["audits/phase08_notebook_persistence_audit.json"] = True
    all_artifacts_present = all(required_artifacts.values())
    all_directories_present = all(artifact["required_directories"].values())
    overall_pass = bool(artifact["overall_pre_notebook_pass"] and persistence_pass and all_artifacts_present and all_directories_present)
    artifact.update({
        "status": "PASS" if overall_pass else "FAIL",
        "required_artifacts": required_artifacts,
        "notebook_persistence_pass": persistence_pass,
        "all_required_directories_present": all_directories_present,
        "all_required_artifacts_present": all_artifacts_present,
        "ready_for_contract_freeze": overall_pass,
        "ready_for_modeling": False,
    })
    write_json(artifact_path, artifact)

    summary_path = PHASE08_ROOT / "audits" / "phase08_initialization_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "notebook_persistence_pass": persistence_pass,
        "phase08_status": "PENDING_CONTRACT_FREEZE" if overall_pass else "FAIL",
        "ready_for_contract_freeze": overall_pass,
        "ready_for_modeling": False,
    })
    write_json(summary_path, summary)

    contract_marker_present = any(
        cell.cell_type == "markdown" and "## Phase 08 Contract Freeze" in cell.source
        for cell in executed.cells
    )
    test_process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(PHASE08_ROOT / "tests"), "-p", "test_phase08_contract.py", "-v"],
        cwd=PHASE08_ROOT, capture_output=True, text=True, check=False,
    )
    contract_freeze_audit = json.loads((PHASE08_ROOT / "audits" / "phase08_contract_freeze_audit.json").read_text(encoding="utf-8"))
    contract_persistence_pass = bool(
        persistence_pass and contract_marker_present and test_process.returncode == 0
        and contract_freeze_audit.get("status") == "PASS"
    )
    contract_notebook_audit = {
        "status": "PASS" if contract_persistence_pass else "FAIL",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "notebook_path": str(NOTEBOOK_PATH),
        "contract_freeze_section_present": contract_marker_present,
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": sum(cell.get("execution_count") is not None for cell in code_cells),
        "code_cells_with_persisted_outputs": output_cells,
        "error_output_count": len(error_outputs),
        "static_unit_test_returncode": test_process.returncode,
        "static_unit_tests": "PASS" if test_process.returncode == 0 else "FAIL",
        "training_executed": False,
        "outer_test_predictions_generated": False,
        "ready_for_execution": contract_persistence_pass,
    }
    write_json(PHASE08_ROOT / "audits" / "phase08_contract_notebook_persistence_audit.json", contract_notebook_audit)

    contract_artifact_path = PHASE08_ROOT / "audits" / "phase08_contract_artifact_audit.json"
    contract_artifact = json.loads(contract_artifact_path.read_text(encoding="utf-8"))
    contract_artifact.update({
        "status": "PASS" if contract_persistence_pass else "FAIL",
        "contract_notebook_persistence": contract_notebook_audit["status"],
        "static_unit_tests": contract_notebook_audit["static_unit_tests"],
        "ready_for_execution": contract_persistence_pass,
        "training_executed": False,
        "outer_test_predictions_generated": False,
    })
    write_json(contract_artifact_path, contract_artifact)

    contract_summary_path = PHASE08_ROOT / "audits" / "phase08_contract_freeze_summary.json"
    contract_summary = json.loads(contract_summary_path.read_text(encoding="utf-8"))
    contract_summary.update({
        "static_unit_tests": contract_notebook_audit["static_unit_tests"],
        "contract_artifact_audit": contract_artifact["status"],
        "notebook_persistence": contract_notebook_audit["status"],
        "phase08_status": "CONTRACT_FROZEN_NOT_TRAINED" if contract_persistence_pass else "FAIL",
        "ready_for_phase08_execution": contract_persistence_pass,
        "model_training_executed": False,
        "outer_test_predictions_generated": False,
    })
    write_json(contract_summary_path, contract_summary)
    return contract_notebook_audit


if __name__ == "__main__":
    print(json.dumps(execute_and_audit(), ensure_ascii=False, indent=2))
