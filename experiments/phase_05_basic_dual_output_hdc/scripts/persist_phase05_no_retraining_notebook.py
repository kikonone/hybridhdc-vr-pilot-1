"""Append and execute the Phase 05 no-retraining compliance amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


PHASE = Path(__file__).resolve().parents[1]
NOTEBOOK = PHASE / "Phase_05_Basic_Dual_Output_HDC.ipynb"
SNAPSHOT = PHASE / "audits/phase05_no_retraining_pre_amendment_snapshot.json"
AUDIT = PHASE / "audits/phase05_no_retraining_notebook_persistence_audit.json"
STANDARD_AUDIT = PHASE / "audits/phase05_final_notebook_persistence_audit.json"
MARKER = "No-Retraining Compliance Completion"
STATUS_MARKER = "No-Retraining Amendment Final Status"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell_digest(cell: dict) -> str:
    payload = json.dumps(cell, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def execute_cells(nb: nbformat.NotebookNode, indices: list[int]) -> None:
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(PHASE)}},
        allow_errors=False,
    )
    with client.setup_kernel():
        for index in indices:
            client.execute_cell(nb.cells[index], index)


def write_notebook(nb: nbformat.NotebookNode) -> None:
    temp = NOTEBOOK.with_suffix(".ipynb.tmp")
    nbformat.write(nb, temp)
    temp.replace(NOTEBOOK)


def evidence_cells() -> list[nbformat.NotebookNode]:
    heading = nbformat.v4.new_markdown_cell(
        "## No-Retraining Compliance Completion\n\n"
        "This amendment derives diagnostics from the unchanged frozen OOF table and benchmarks "
        "read-only inference from saved fitted artifacts. It does not refit a model, replace a "
        "prediction, select a canonical configuration, or start Phase 06."
    )
    diagnostics = nbformat.v4.new_code_cell(
        """from pathlib import Path
import json
import pandas as pd
from IPython.display import display

phase = Path.cwd()
def load_amendment_json(rel):
    return json.loads((phase / rel).read_text(encoding='utf-8'))

diagnostic_audit = load_amendment_json('audits/phase05_no_retraining_diagnostic_completion_audit.json')
class_diag = pd.read_csv(phase / 'results/summaries/vanilla_hdc_classification_diagnostics_by_config.csv')
similarity_diag = pd.read_csv(phase / 'results/summaries/vanilla_hdc_similarity_regression_diagnostics_by_config.csv')
ridge_diag = pd.read_csv(phase / 'results/summaries/vanilla_hdc_ridge_regression_diagnostics_by_config.csv')
cross_task = pd.read_csv(phase / 'results/summaries/vanilla_hdc_cross_task_consistency_by_config.csv')
display({'diagnostic_audit': diagnostic_audit})
for label, frame in [('classification', class_diag), ('similarity', similarity_diag),
                     ('ridge', ridge_diag), ('cross_task', cross_task)]:
    print(f'{label}: rows={len(frame)}, columns={len(frame.columns)}')
    display(frame)
assert diagnostic_audit.get('result') == 'PASS'
assert diagnostic_audit.get('model_fitting_executed') is False
"""
    )
    efficiency = nbformat.v4.new_code_cell(
        """efficiency_audit = load_amendment_json('audits/phase05_no_retraining_efficiency_protocol_completion_audit.json')
efficiency_by_config = pd.read_csv(phase / 'results/summaries/vanilla_hdc_inference_efficiency_protocol_by_config.csv')
efficiency_by_dimension = pd.read_csv(phase / 'results/summaries/vanilla_hdc_inference_efficiency_protocol_seed_aggregate_by_dimension.csv')
display({'efficiency_audit': efficiency_audit})
display(efficiency_by_config)
display(efficiency_by_dimension)
assert efficiency_audit.get('result') == 'PASS'
assert efficiency_audit.get('warmups') == 5
assert efficiency_audit.get('timed_repetitions') == 30
assert efficiency_audit.get('clock') == 'time.perf_counter_ns'
assert efficiency_audit.get('training_timing_remeasurement') == 'NOT_PERFORMED_RETRAINING_PROHIBITED'
print('NO-RETRAINING COMPLETION EVIDENCE: PASS')
"""
    )
    return [heading, diagnostics, efficiency]


def status_cells() -> list[nbformat.NotebookNode]:
    heading = nbformat.v4.new_markdown_cell(
        "### No-Retraining Amendment Final Status\n\n"
        "The saved amendment, audits, manifest, and freeze record are read below."
    )
    status = nbformat.v4.new_code_cell(
        """from pathlib import Path
import json
from IPython.display import display

phase = Path.cwd()
def load_final_json(rel):
    return json.loads((phase / rel).read_text(encoding='utf-8'))

amendment = load_final_json('configs/phase05_no_retraining_completion_amendment.json')
amendment_audit = load_final_json('audits/phase05_no_retraining_amendment_audit.json')
artifact_audit = load_final_json('audits/phase05_final_artifact_audit.json')
freeze = load_final_json('configs/phase05_freeze.json')
display({'amendment': amendment, 'amendment_audit': amendment_audit,
         'artifact_audit': artifact_audit, 'freeze': freeze})
assert amendment.get('status') == 'COMPLETED_NO_RETRAINING'
assert amendment_audit.get('result') == 'PASS'
assert artifact_audit.get('result') == 'PASS'
assert freeze.get('status') == 'FROZEN'
assert freeze.get('phase06_executed') is False
print('PHASE 05 NO-RETRAINING AMENDMENT: COMPLETE; PHASE 05 STATUS: FROZEN')
"""
    )
    return [heading, status]


def audit_notebook(nb: nbformat.NotebookNode) -> dict:
    snapshot = load_json(SNAPSHOT)
    baseline_count = snapshot["pre_amendment_notebook_cell_count"]
    expected = snapshot["pre_amendment_notebook_cell_sha256"]
    # The snapshot hashes the raw JSON representation. nbformat joins source
    # line arrays in memory, so hash the saved raw cells to compare like-for-like.
    raw_notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    current = [cell_digest(cell) for cell in raw_notebook["cells"][:baseline_count]]
    appended = nb.cells[baseline_count:]
    code_cells = [cell for cell in appended if cell.cell_type == "code"]
    checks = {
        "baseline_cell_count": baseline_count,
        "current_cell_count": len(nb.cells),
        "all_baseline_cells_and_outputs_preserved": current == expected,
        "amendment_heading_present_once": sum(MARKER in str(c.get("source", "")) for c in nb.cells) == 1,
        "status_heading_present_at_most_once": sum(STATUS_MARKER in str(c.get("source", "")) for c in nb.cells) <= 1,
        "appended_code_cells_executed": bool(code_cells) and all(c.get("execution_count") is not None for c in code_cells),
        "appended_outputs_saved": bool(code_cells) and all(len(c.get("outputs", [])) > 0 for c in code_cells),
        "appended_cells_error_free": all(
            not any(output.get("output_type") == "error" for output in c.get("outputs", []))
            for c in code_cells
        ),
    }
    result = "PASS" if all(value for value in checks.values() if isinstance(value, bool)) else "FAIL"
    return {
        "phase": "05",
        "audit": "no_retraining_notebook_persistence",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "notebook": NOTEBOOK.name,
        "notebook_sha256": sha256(NOTEBOOK),
        "checks": checks,
        "model_fitting_executed": False,
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["evidence", "status"])
    args = parser.parse_args()
    nb = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(str(cell.get("source", "")) for cell in nb.cells)
    marker = MARKER if args.mode == "evidence" else STATUS_MARKER
    if marker not in source:
        new_cells = evidence_cells() if args.mode == "evidence" else status_cells()
        start = len(nb.cells)
        nb.cells.extend(new_cells)
        execute_cells(nb, [i for i in range(start, len(nb.cells)) if nb.cells[i].cell_type == "code"])
        write_notebook(nb)
        nb = nbformat.read(NOTEBOOK, as_version=4)
    audit = audit_notebook(nb)
    write_json(AUDIT, audit)
    write_json(STANDARD_AUDIT, audit)
    if audit["result"] != "PASS":
        raise RuntimeError(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({"notebook": "PASS", "mode": args.mode, "cells": len(nb.cells)}))


if __name__ == "__main__":
    main()
