"""Append executed Phase 09 training evidence without changing prior notebook cells."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient


PHASE09_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PHASE09_ROOT / "Phase_09_Robustness_and_Generalization.ipynb"
MARKER = "# Phase 09 Execution Evidence"


def cell_fingerprint(cell: nbformat.NotebookNode) -> str:
    return hashlib.sha256(
        json.dumps(cell, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def execution_cells() -> list[nbformat.NotebookNode]:
    metadata = {"tags": ["phase09-execution-evidence"]}
    setup = (
        "from pathlib import Path\nimport json, subprocess, sys\n"
        "PHASE09_ROOT = Path.cwd().resolve()\n"
        "sys.path.insert(0, str(PHASE09_ROOT / 'scripts'))\n"
    )
    return [
        nbformat.v4.new_markdown_cell(
            f"{MARKER}\n\nThis appended section records the authorized 720-run execution and raw prediction evidence. It performs no OOF consolidation, seed aggregation, formal statistics, or Phase 10 work.",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 23. Executor static validation", metadata=metadata),
        nbformat.v4.new_code_cell(
            setup
            + "validation = json.loads((PHASE09_ROOT / 'audits/phase09_executor_validation_audit.json').read_text(encoding='utf-8'))\n"
            + "{'status': validation['status'], 'checks': validation['checks'], 'authorized_run_count': validation['dry_run_unique_runs'], 'training_executed': validation['training_executed']} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 24. Exact 720-run dry-run summary", metadata=metadata),
        nbformat.v4.new_code_cell(
            "dry_run = validation\n"
            "{'status': dry_run['status'], 'authorized_run_count': dry_run['dry_run_unique_runs'], 'expected_raw_prediction_rows': dry_run['expected_raw_prediction_rows'], 'exact_manifest_match': dry_run['checks']['exact_frozen_manifest_match'], 'training_executed': dry_run['training_executed']} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 25. Missing-modality execution coverage", metadata=metadata),
        nbformat.v4.new_code_cell(
            "coverage = json.loads((PHASE09_ROOT / 'audits/phase09_execution_coverage_audit.json').read_text(encoding='utf-8'))\n"
            "{'status': coverage['status'], 'completed_by_protocol': coverage['run_counts_by_protocol'], 'completed_conditions': coverage['missing_modality_conditions_completed'], 'checks': coverage['checks']} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 26. LOSO execution coverage", metadata=metadata),
        nbformat.v4.new_code_cell(
            "{'status': coverage['status'], 'completed_subject_count': len(coverage['loso_subjects_completed']), 'completed_subjects': coverage['loso_subjects_completed'], 'loso_runs': coverage['run_counts_by_protocol']['LEAVE_ONE_SUBJECT_OUT']} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 27. Checkpoint and recovery integrity", metadata=metadata),
        nbformat.v4.new_code_cell(
            "checkpoint = json.loads((PHASE09_ROOT / 'audits/phase09_checkpoint_integrity_audit.json').read_text(encoding='utf-8'))\n"
            "{'status': checkpoint['status'], 'completed_checkpoints': checkpoint['completed_checkpoints'], 'expected_checkpoints': checkpoint['expected_checkpoints'], 'invalid_runs': checkpoint['invalid_runs'], 'load_or_hash_failures': checkpoint['load_or_hash_failures']} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 28. Raw prediction row evidence", metadata=metadata),
        nbformat.v4.new_code_cell(
            "{'status': coverage['status'], 'raw_prediction_rows': coverage['raw_prediction_rows'], 'expected_raw_prediction_rows': coverage['expected_raw_prediction_rows'], 'duplicate_run_id_run_key_pairs': coverage['duplicate_run_id_run_key_pairs'], 'completed_by_model': coverage['run_counts_by_model']} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 29. Leakage and feature-exclusion audits", metadata=metadata),
        nbformat.v4.new_code_cell(
            "leakage = json.loads((PHASE09_ROOT / 'audits/phase09_execution_leakage_audit.json').read_text(encoding='utf-8'))\n"
            "features = json.loads((PHASE09_ROOT / 'audits/phase09_feature_exclusion_audit.json').read_text(encoding='utf-8'))\n"
            "{'leakage_status': leakage['status'], 'feature_exclusion_status': features['status'], 'leakage_failures': leakage['leakage_failures'], 'feature_failures': features['feature_failures']} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 30. Execution artifact audit", metadata=metadata),
        nbformat.v4.new_code_cell(
            "artifacts = json.loads((PHASE09_ROOT / 'audits/phase09_execution_artifact_audit.json').read_text(encoding='utf-8'))\n"
            "{'status': artifacts['status'], 'artifact_count': artifacts['artifact_count'], 'expected_artifact_count': artifacts['expected_artifact_count'], 'missing_or_invalid_runs': artifacts['missing_or_invalid_runs'], 'upstream_files_modified': artifacts['upstream_files_modified'], 'frozen_contract_files_modified': artifacts['frozen_contract_files_modified']} ",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 31. Execution limitations and stopping boundary", metadata=metadata),
        nbformat.v4.new_code_cell(
            "limitations = {'full_primary_reference_retrained': False, 'sudden_test_time_missingness': 'NOT_FEASIBLE_DUE_TO_CHECKPOINT_INTERFACE', 'oof_consolidation_executed': False, 'seed_aggregation_executed': False, 'formal_statistics_executed': False, 'phase10_started': False, 'phase09_final_freeze_executed': False}\nlimitations",
            metadata=metadata,
        ),
        nbformat.v4.new_markdown_cell("## 32. Final execution persistence state", metadata=metadata),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json\nPHASE09_ROOT = Path.cwd().resolve()\n"
            "manifest = json.loads((PHASE09_ROOT / 'configs/phase09_execution_manifest.json').read_text(encoding='utf-8'))\n"
            "persistence_path = PHASE09_ROOT / 'audits/phase09_execution_notebook_persistence_audit.json'\n"
            "persistence = json.loads(persistence_path.read_text(encoding='utf-8')) if persistence_path.exists() else {'status': 'PENDING_EXTERNAL_PERSISTENCE_AUDIT'}\n"
            "{'phase09_execution_status': manifest['status'], 'completed_training_runs': manifest['completed_training_runs'], 'raw_prediction_rows': manifest['raw_prediction_rows'], 'notebook_persistence': persistence['status'], 'ready_for_oof_consolidation': manifest.get('ready_for_oof_consolidation', False)} ",
            metadata=metadata,
        ),
    ]


def execute_cells(notebook: nbformat.NotebookNode, indices: list[int]) -> None:
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(PHASE09_ROOT)}},
    )
    with client.setup_kernel():
        for index in indices:
            if notebook.cells[index].cell_type == "code":
                client.execute_cell(notebook.cells[index], index)


def persistence_audit(
    notebook: nbformat.NotebookNode,
    original_fingerprints: list[str],
    appended_start: int,
) -> dict[str, Any]:
    history_unchanged = (
        [cell_fingerprint(cell) for cell in notebook.cells[:appended_start]]
        == original_fingerprints
    )
    appended = notebook.cells[appended_start:]
    code_cells = [cell for cell in appended if cell.cell_type == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    appended_text = "\n".join(str(cell.get("source", "")) for cell in appended)
    required = [
        "Executor static validation", "720-run dry-run", "Missing-modality execution",
        "LOSO execution", "Checkpoint and recovery", "Raw prediction row",
        "Leakage and feature-exclusion", "Execution artifact", "Execution limitations",
    ]
    status = "PASS" if (
        history_unchanged
        and code_cells
        and all(cell.get("execution_count") is not None for cell in code_cells)
        and all(cell.get("outputs") for cell in code_cells)
        and not errors
        and all(item.lower() in appended_text.lower() for item in required)
    ) else "FAIL"
    audit = {
        "phase": "09",
        "audit": "execution_notebook_persistence",
        "status": status,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(NOTEBOOK_PATH.resolve()),
        "bytes": NOTEBOOK_PATH.stat().st_size,
        "sha256": hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest(),
        "prior_cells": appended_start,
        "prior_history_unchanged": history_unchanged,
        "appended_code_cells": len(code_cells),
        "executed_appended_code_cells": sum(cell.get("execution_count") is not None for cell in code_cells),
        "appended_code_cells_with_outputs": sum(bool(cell.get("outputs")) for cell in code_cells),
        "error_outputs": len(errors),
        "required_sections_present": all(item.lower() in appended_text.lower() for item in required),
    }
    write_json(PHASE09_ROOT / "audits" / "phase09_execution_notebook_persistence_audit.json", audit)
    return audit


def append_execute_and_finalize() -> dict[str, Any]:
    scripts_path = str(PHASE09_ROOT / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    from verify_phase09_execution import finalize_execution_status, verify_execution

    verification = verify_execution()
    if not verification["ready_pending_notebook_persistence"]:
        raise RuntimeError(verification)

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    marker_index = next(
        (
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "markdown" and str(cell.source).startswith(MARKER)
        ),
        len(notebook.cells),
    )
    notebook.cells = notebook.cells[:marker_index]
    original_fingerprints = [cell_fingerprint(cell) for cell in notebook.cells]
    appended_start = len(notebook.cells)
    notebook.cells.extend(execution_cells())
    code_indices = [
        index
        for index in range(appended_start, len(notebook.cells))
        if notebook.cells[index].cell_type == "code"
    ]
    execute_cells(notebook, code_indices)
    nbformat.write(notebook, NOTEBOOK_PATH)
    persistence = persistence_audit(notebook, original_fingerprints, appended_start)
    finalization = finalize_execution_status()

    execute_cells(notebook, [code_indices[-1]])
    nbformat.write(notebook, NOTEBOOK_PATH)
    persistence = persistence_audit(notebook, original_fingerprints, appended_start)
    finalization = finalize_execution_status()
    if persistence["status"] != "PASS" or not finalization["ready_for_oof_consolidation"]:
        raise RuntimeError({"persistence": persistence, "finalization": finalization})
    return {
        "notebook_persistence": persistence["status"],
        "prior_history_unchanged": persistence["prior_history_unchanged"],
        "phase09_execution_status": finalization["status"],
        "ready_for_oof_consolidation": finalization["ready_for_oof_consolidation"],
    }


if __name__ == "__main__":
    print(json.dumps(append_execute_and_finalize(), ensure_ascii=False, indent=2))
