"""Build, execute, persist, and externally audit the Phase 09 initialization notebook."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient


PHASE09_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PHASE09_ROOT / "Phase_09_Robustness_and_Generalization.ipynb"


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Phase 09: Robustness and Generalization\n\n"
            "Executed initialization and feasibility audit only. No model training, prediction generation, OOF generation, hyperparameter search, or formal statistical analysis is performed."
        ),
        nbformat.v4.new_markdown_cell(
            "## 1. Stage purpose and evidence boundary\n\n"
            "The main evidence is Phase 03 Primary without performance features. Frozen Phase 03 folds and Phase 04A/04B/06/07/08 interfaces are read-only. "
            "The planned workstreams are missing-modality robustness, selected-model LOSO, and subject-level stability."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json, os, platform, subprocess, sys\n"
            "PHASE09_ROOT = Path.cwd().resolve()\n"
            "assert PHASE09_ROOT.name == 'phase_09_robustness_and_generalization'\n"
            "sys.path.insert(0, str(PHASE09_ROOT / 'scripts'))\n"
            "from initialize_phase09 import run_initialization, REQUIRED_ARTIFACTS, REQUIRED_DIRECTORIES\n"
            "environment = {'python': sys.version, 'platform': platform.platform(), 'working_directory': os.getcwd()}\n"
            "environment"
        ),
        nbformat.v4.new_markdown_cell("## 2. Run read-only initialization and upstream freeze validation"),
        nbformat.v4.new_code_cell(
            "summary = run_initialization()\n"
            "upstream = json.loads((PHASE09_ROOT / 'audits/phase09_upstream_freeze_audit.json').read_text(encoding='utf-8'))\n"
            "{'initialization': summary, 'phase_freeze_interfaces': upstream['interface_results'], 'actual_freeze_sha256': upstream['actual_freeze_sha256'], 'status': upstream['status']}"
        ),
        nbformat.v4.new_markdown_cell("## 3. Primary data preflight"),
        nbformat.v4.new_code_cell(
            "input_audit = json.loads((PHASE09_ROOT / 'audits/phase09_input_and_fold_audit.json').read_text(encoding='utf-8'))\n"
            "{'actual': input_audit['actual'], 'checks': input_audit['checks'], 'status': input_audit['status']}"
        ),
        nbformat.v4.new_markdown_cell("## 4. Frozen five-fold verification"),
        nbformat.v4.new_code_cell(
            "{'fold_sha256': input_audit['actual']['fold_sha256'], 'checksum_pass': input_audit['checks']['fold_checksum'], 'outer_folds': input_audit['actual']['outer_folds'], 'fold_alignment': input_audit['checks']['fold_alignment'], 'subject_isolation': input_audit['checks']['outer_subject_isolation'], 'fold_details': input_audit['fold_details']}"
        ),
        nbformat.v4.new_markdown_cell("## 5. Five-modality feature counts and coverage"),
        nbformat.v4.new_code_cell(
            "modality = json.loads((PHASE09_ROOT / 'audits/phase09_modality_coverage_audit.json').read_text(encoding='utf-8'))\n"
            "{'counts': modality['modality_counts'], 'union_count': modality['union_count'], 'overlap_count': modality['overlap_count'], 'checks': modality['checks'], 'status': modality['status']}"
        ),
        nbformat.v4.new_markdown_cell("## 6. Missing-modality condition feasibility"),
        nbformat.v4.new_code_cell(
            "missing_plan = json.loads((PHASE09_ROOT / 'configs/phase09_missing_modality_plan.json').read_text(encoding='utf-8'))\n"
            "{'conditions': missing_plan['conditions'], 'protocols_must_remain_separate': missing_plan['protocols_requiring_separate_contracts'], 'hdc_feasibility': missing_plan['hdc_feasibility'], 'traditional_feasibility': missing_plan['traditional_feasibility'], 'training_executed': missing_plan['training_executed'], 'predictions_generated': missing_plan['predictions_generated']}"
        ),
        nbformat.v4.new_markdown_cell("## 7. Deterministic 35-fold LOSO feasibility"),
        nbformat.v4.new_code_cell(
            "loso = json.loads((PHASE09_ROOT / 'audits/phase09_loso_feasibility_audit.json').read_text(encoding='utf-8'))\n"
            "loso_manifest = json.loads((PHASE09_ROOT / 'manifests/phase09_loso_feasibility_manifest.json').read_text(encoding='utf-8'))\n"
            "{'split_count': loso['split_count'], 'checks': loso['checks'], 'first_split': loso_manifest['splits'][0], 'last_split': loso_manifest['splits'][-1], 'status': loso['status']}"
        ),
        nbformat.v4.new_markdown_cell("## 8. Subject run counts and target coverage"),
        nbformat.v4.new_code_cell(
            "{'subject_run_counts': loso['subject_run_counts'], 'subject_target_coverage': loso['subject_target_coverage'], 'empty_training_sets': loso['empty_training_sets'], 'empty_test_sets': loso['empty_test_sets']}"
        ),
        nbformat.v4.new_markdown_cell("## 9. Generalization-scope limitations"),
        nbformat.v4.new_code_cell(
            "scope = json.loads((PHASE09_ROOT / 'configs/phase09_generalization_scope.json').read_text(encoding='utf-8'))\n"
            "scope_audit = json.loads((PHASE09_ROOT / 'audits/phase09_generalization_scope_audit.json').read_text(encoding='utf-8'))\n"
            "{'subject_generalization': scope['subject_generalization'], 'missing_modality_robustness': scope['missing_modality_robustness'], 'unseen_session_generalization': scope['unseen_session_generalization'], 'unseen_scenario_generalization': scope['unseen_scenario_generalization'], 'task_template_generalization': scope['task_template_generalization'], 'route_configuration_generalization': scope['route_configuration_generalization'], 'flight_claim': scope['flight_generalizable_behavior_claim'], 'checks': scope_audit['checks']}"
        ),
        nbformat.v4.new_markdown_cell("## 10. Selected frozen model interfaces"),
        nbformat.v4.new_code_cell(
            "interfaces = json.loads((PHASE09_ROOT / 'configs/phase09_selected_model_interfaces.json').read_text(encoding='utf-8'))\n"
            "{'traditional_classification': {'model': interfaces['traditional_classification']['model'], 'interface_pass': interfaces['traditional_classification']['interface_pass'], 'fold_specific_parameters': interfaces['traditional_classification']['fold_specific_parameters']}, 'traditional_regression': {'model': interfaces['traditional_regression']['model'], 'interface_pass': interfaces['traditional_regression']['interface_pass'], 'fold_specific_parameters': interfaces['traditional_regression']['fold_specific_parameters']}, 'hdc_classification': {'model': interfaces['hdc_classification']['selected_variant_name'], 'dimension': interfaces['hdc_classification']['selected_fixed_dimension'], 'interface_pass': interfaces['hdc_classification']['interface_pass']}, 'hdc_regression': {'model': interfaces['hdc_regression']['selected_regression_head'], 'dimension': interfaces['hdc_regression']['selected_fixed_dimension'], 'interface_pass': interfaces['hdc_regression']['interface_pass']}, 'policy': interfaces['policy']}"
        ),
        nbformat.v4.new_markdown_cell("## 11. Initialization artifact audit and static tests"),
        nbformat.v4.new_code_cell(
            "pre_persistence_missing_dirs = [p for p in REQUIRED_DIRECTORIES if not (PHASE09_ROOT / p).is_dir()]\n"
            "deferred_self_audits = {'audits/phase09_notebook_persistence_audit.json', 'audits/phase09_initialization_artifact_audit.json'}\n"
            "pre_persistence_missing_artifacts = [p for p in REQUIRED_ARTIFACTS if p not in deferred_self_audits and not (PHASE09_ROOT / p).is_file()]\n"
            "test_process = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(PHASE09_ROOT / 'tests'), '-p', 'test_phase09_initialization.py', '-v'], cwd=PHASE09_ROOT, capture_output=True, text=True, check=False)\n"
            "assert test_process.returncode == 0, test_process.stdout + test_process.stderr\n"
            "{'pre_persistence_directory_audit': 'PASS' if not pre_persistence_missing_dirs else 'FAIL', 'pre_persistence_artifact_audit': 'PASS' if not pre_persistence_missing_artifacts else 'FAIL', 'missing_directories': pre_persistence_missing_dirs, 'missing_artifacts': pre_persistence_missing_artifacts, 'static_tests': 'PASS', 'test_details': (test_process.stdout + test_process.stderr)[-2500:], 'external_notebook_persistence_audit': 'RUN_AFTER_NOTEBOOK_SAVE'}"
        ),
        nbformat.v4.new_markdown_cell("## 12. Contract Freeze entry gate"),
        nbformat.v4.new_code_cell(
            "contract = json.loads((PHASE09_ROOT / 'configs/phase09_experiment_contract_draft.json').read_text(encoding='utf-8'))\n"
            "statistical_plan = json.loads((PHASE09_ROOT / 'configs/phase09_statistical_plan.json').read_text(encoding='utf-8'))\n"
            "{'contract_status': contract['status'], 'next_action': contract['next_action'], 'subject_level_stability_plan': statistical_plan['planned_analyses'], 'training_authorized': contract['training_authorized'], 'prediction_generation_authorized': contract['prediction_generation_authorized'], 'formal_statistics_authorized': contract['formal_statistics_authorized'], 'ready_for_modeling': False}"
        ),
        nbformat.v4.new_markdown_cell(
            "## Phase Validation Summary\n\n"
            "**VERIFIED:** Primary input/folds, upstream frozen interfaces, five-modality partition, missing-modality computational feasibility, 35 deterministic LOSO splits, subject target coverage, and metadata-based scope limits.\n\n"
            "**NOT VERIFIED:** Performance after modality loss or under LOSO; no such experiment was run.\n\n"
            "**WARNINGS:** LOSO supports only subject-generalization claims. Flight generalizable behavior remains `INCONCLUSIVE_DUE_TO_METADATA`.\n\n"
            "**KEY RESULTS:** Initialization only; no model metrics or predictions.\n\n"
            "**OUTPUT FILES:** Draft configs, manifests, audits, this executed notebook, README, and notes.\n\n"
            "**NEXT PHASE REQUIREMENTS:** Separate Phase 09 Contract Freeze before any modeling."
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": sys.version.split()[0]}
    return notebook


def execute_and_audit() -> dict[str, Any]:
    notebook = build_notebook()
    nbformat.write(notebook, NOTEBOOK_PATH)
    client = NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(PHASE09_ROOT)}})
    executed = client.execute()
    nbformat.write(executed, NOTEBOOK_PATH)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    errors = [output for cell in code_cells for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    required_headings = ["upstream freeze", "Primary data", "Frozen five-fold", "Five-modality", "Missing-modality", "35-fold LOSO", "Subject run counts", "Generalization-scope", "Selected frozen model", "Initialization artifact", "Contract Freeze"]
    notebook_text = "\n".join(str(cell.get("source", "")) for cell in executed.cells)
    status = "PASS" if (
        NOTEBOOK_PATH.exists() and all(cell.get("execution_count") is not None for cell in code_cells)
        and all(cell.get("outputs") for cell in code_cells) and not errors
        and all(heading.lower() in notebook_text.lower() for heading in required_headings)
    ) else "FAIL"
    persistence = {
        "phase": "09", "audit": "notebook_persistence", "status": status,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "path": str(NOTEBOOK_PATH.resolve()),
        "bytes": NOTEBOOK_PATH.stat().st_size, "sha256": __import__('hashlib').sha256(NOTEBOOK_PATH.read_bytes()).hexdigest(),
        "code_cells": len(code_cells), "executed_code_cells": sum(cell.get("execution_count") is not None for cell in code_cells),
        "code_cells_with_outputs": sum(bool(cell.get("outputs")) for cell in code_cells), "error_outputs": len(errors),
        "required_sections_present": all(heading.lower() in notebook_text.lower() for heading in required_headings),
        "model_training_executed": False, "loso_predictions_generated": False, "missing_modality_predictions_generated": False,
    }
    write_json(PHASE09_ROOT / "audits" / "phase09_notebook_persistence_audit.json", persistence)
    from initialize_phase09 import finalize_artifact_audit
    artifact = finalize_artifact_audit()
    if status != "PASS" or artifact["status"] != "PASS":
        raise RuntimeError({"persistence": persistence, "artifact": artifact})
    return {"notebook_persistence": status, "artifact_audit": artifact["status"], "phase09_status": artifact["phase09_status"]}


if __name__ == "__main__":
    print(json.dumps(execute_and_audit(), ensure_ascii=False, indent=2))
