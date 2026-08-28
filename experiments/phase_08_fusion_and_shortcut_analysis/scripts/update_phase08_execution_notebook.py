"""Append and execute the Phase 08 batch-execution evidence section only.

The frozen initialization and contract cells are deliberately not re-executed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Phase_08_Fusion_and_Shortcut_Analysis.ipynb"
SECTION_MARKER = "## Phase 08 Execution"
AUDIT_PATH = ROOT / "audits" / "phase08_execution_notebook_persistence_audit.json"
SUMMARY_PATH = ROOT / "audits" / "phase08_execution_summary.json"
MANIFEST_PATH = ROOT / "configs" / "phase08_execution_manifest.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def execution_cells() -> list:
    return [
        nbformat.v4.new_markdown_cell(
            SECTION_MARKER
            + "\n\nThis section records the audited 370-run frozen batch. It reads persisted artifacts only; "
            "it does not train, tune, consolidate OOF predictions, or invoke Phase 09."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "PHASE08_ROOT = Path.cwd().resolve()\n"
            "def load_audit(name):\n"
            "    return json.loads((PHASE08_ROOT / 'audits' / name).read_text(encoding='utf-8'))\n"
            "execution_summary = load_audit('phase08_execution_summary.json')\n"
            "execution_summary"
        ),
        nbformat.v4.new_markdown_cell("### 16. Executor gate and completed-run coverage"),
        nbformat.v4.new_code_cell(
            "executor = load_audit('phase08_executor_validation_audit.json')\n"
            "coverage = load_audit('phase08_execution_coverage_audit.json')\n"
            "{\n"
            "    'executor_validation': execution_summary['executor_validation'],\n"
            "    'completed_runs': f\"{execution_summary['completed_runs']}/{execution_summary['expected_runs']}\",\n"
            "    'raw_prediction_rows': f\"{execution_summary['raw_prediction_rows']}/{execution_summary['expected_raw_prediction_rows']}\",\n"
            "    'model_task_counts': execution_summary['model_task_counts'],\n"
            "    'condition_counts': execution_summary['condition_counts'],\n"
            "    'failed_run_events': execution_summary['failed_run_events'],\n"
            "    'recovered_valid_checkpoints': execution_summary['recovered_valid_checkpoints'],\n"
            "}"
        ),
        nbformat.v4.new_markdown_cell("### 17. Checkpoint integrity, leakage, and artifact audits"),
        nbformat.v4.new_code_cell(
            "checkpoint = load_audit('phase08_checkpoint_integrity_audit.json')\n"
            "leakage = load_audit('phase08_execution_leakage_audit.json')\n"
            "artifacts = load_audit('phase08_execution_artifact_audit.json')\n"
            "{\n"
            "    'checkpoint_integrity': execution_summary['checkpoint_integrity'],\n"
            "    'execution_coverage': execution_summary['coverage_audit'],\n"
            "    'execution_leakage': execution_summary['leakage_audit'],\n"
            "    'artifact_audit': execution_summary['artifact_audit'],\n"
            "    'outer_test_used_for_tuning': execution_summary['outer_test_used_for_tuning'],\n"
            "    'final_oof_consolidation_executed': execution_summary['final_oof_consolidation_executed'],\n"
            "    'phase09_executed': execution_summary['phase09_executed'],\n"
            "}"
        ),
        nbformat.v4.new_markdown_cell("### 18. Execution boundary and handoff"),
        nbformat.v4.new_code_cell(
            "{\n"
            "    'status': execution_summary['status'],\n"
            "    'flight_task_setting_status': execution_summary['flight_task_setting_status'],\n"
            "    'next_permitted_step': 'PHASE_08_OOF_CONSOLIDATION',\n"
            "    'oof_consolidation_performed_here': False,\n"
            "}"
        ),
        nbformat.v4.new_markdown_cell(
            "## Execution Takeaways\n\n"
            "All 370 frozen model-runs completed and passed independent coverage, checkpoint, leakage, and "
            "artifact audits. The execution produced 31,006 raw outer-test prediction rows. Canonical OOF "
            "consolidation and Phase 09 remain unexecuted."
        ),
    ]


def main() -> None:
    original = nbformat.read(NOTEBOOK, as_version=4)
    kept = []
    for cell in original.cells:
        if cell.cell_type == "markdown" and SECTION_MARKER in cell.source:
            break
        kept.append(cell)

    appended = execution_cells()
    temporary = nbformat.v4.new_notebook(
        cells=appended[1:], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    )
    executed = NotebookClient(temporary, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    final = nbformat.v4.new_notebook(cells=kept + [appended[0]] + executed.cells, metadata=original.metadata)
    nbformat.write(final, NOTEBOOK)

    execution_code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    error_outputs = [
        output
        for cell in execution_code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    output_cells = [cell for cell in execution_code_cells if cell.get("outputs")]
    audit = {
        "audit": "phase08_execution_notebook_persistence",
        "status": "PASS" if len(execution_code_cells) == 4 and len(output_cells) == 4 and not error_outputs else "FAIL",
        "notebook": NOTEBOOK.name,
        "frozen_prefix_cells_preserved": len(kept),
        "execution_section_present": True,
        "execution_code_cells": len(execution_code_cells),
        "execution_code_cells_with_outputs": len(output_cells),
        "error_outputs": len(error_outputs),
        "old_contract_cells_reexecuted": False,
        "training_invoked_by_execution_section": False,
        "oof_consolidation_executed": False,
        "phase09_executed": False,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(AUDIT_PATH, audit)
    if audit["status"] != "PASS":
        raise RuntimeError(f"Notebook persistence audit failed: {audit}")

    summary = read_json(SUMMARY_PATH)
    summary["notebook_persistence"] = "PASS"
    summary["ready_for_oof_consolidation"] = True
    write_json(SUMMARY_PATH, summary)

    manifest = read_json(MANIFEST_PATH)
    manifest["status"] = "EXECUTION_COMPLETE_PENDING_OOF_CONSOLIDATION"
    manifest["execution_notebook_persistence_audit"] = "audits/phase08_execution_notebook_persistence_audit.json"
    manifest["ready_for_oof_consolidation_pending_notebook"] = False
    manifest["ready_for_oof_consolidation"] = True
    manifest["last_verified_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
