"""Append and execute Phase 05 final evidence cells without rerunning prior cells."""

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
AUDIT = PHASE / "audits/phase05_final_notebook_persistence_audit.json"
BASELINE_CELLS = 29
MARKER = "Final OOF Consolidation and Phase 05 Freeze"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cell_digest(cell: dict) -> str:
    payload = json.dumps(cell, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def execute_new_cells(nb: nbformat.NotebookNode, indices: list[int]) -> None:
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
        "## Final OOF Consolidation and Phase 05 Freeze\n\n"
        "This section consolidates the already-generated Final Confirmation predictions. "
        "It does not fit a model, generate a prediction, tune on outer-test data, or start Phase 06. "
        "All results are descriptive over the preregistered 4 dimensions × 5 seeds matrix."
    )
    audit_code = nbformat.v4.new_code_cell(
        """from pathlib import Path
import hashlib, json
import pandas as pd
from IPython.display import display

phase = Path.cwd()
def load_json(rel):
    return json.loads((phase / rel).read_text(encoding='utf-8'))
def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

coverage = load_json('audits/phase05_final_oof_coverage_audit.json')
alignment = load_json('audits/phase05_final_oof_alignment_audit.json')
leakage = load_json('audits/phase05_final_oof_leakage_audit.json')
metric_audit = load_json('audits/phase05_oof_metric_recomputation_audit.json')
compatibility = load_json('audits/phase05_baseline_compatibility_audit.json')
primary_path = phase.parent / 'phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv'
fold_path = phase.parent / 'phase_03_multimodal_dataset_labeling/data/fold_assignments.csv'
checksum_evidence = {
    'primary_sha256': file_sha256(primary_path),
    'primary_checksum_pass': file_sha256(primary_path) == '0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44',
    'fold_sha256': file_sha256(fold_path),
    'fold_checksum_pass': file_sha256(fold_path) == 'e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f',
}
display({'oof_coverage_audit': coverage, 'oof_alignment_audit': alignment,
         'final_leakage_audit': leakage, 'metric_recomputation_audit': metric_audit,
         'baseline_compatibility_audit': compatibility, 'checksums': checksum_evidence})
"""
    )
    table_code = nbformat.v4.new_code_cell(
        """table_paths = [
    'results/summaries/vanilla_hdc_classification_oof_metrics_by_config.csv',
    'results/summaries/vanilla_hdc_similarity_regression_oof_metrics_by_config.csv',
    'results/summaries/vanilla_hdc_ridge_regression_oof_metrics_by_config.csv',
    'results/summaries/vanilla_hdc_classification_seed_aggregate_by_dimension.csv',
    'results/summaries/vanilla_hdc_similarity_regression_seed_aggregate_by_dimension.csv',
    'results/summaries/vanilla_hdc_ridge_regression_seed_aggregate_by_dimension.csv',
    'results/summaries/vanilla_hdc_efficiency_by_config.csv',
    'results/summaries/vanilla_hdc_efficiency_seed_aggregate_by_dimension.csv',
    'results/summaries/phase05_vs_phase04a_classification_comparison.csv',
    'results/summaries/phase05_vs_phase04b_regression_comparison.csv',
    'results/summaries/phase05_dual_output_final_comparison.csv',
]
for rel in table_paths:
    frame = pd.read_csv(phase / rel)
    print(f'{rel} — rows={len(frame)}, columns={len(frame.columns)}')
    display(frame)
"""
    )
    figure_code = nbformat.v4.new_code_cell(
        """from IPython.display import Image, display
figure_paths = [
    'figures/phase05_classification_macro_f1_vs_dimension.png',
    'figures/phase05_classification_seed_stability.png',
    'figures/phase05_similarity_regression_mae_vs_dimension.png',
    'figures/phase05_ridge_regression_mae_vs_dimension.png',
    'figures/phase05_regression_seed_stability.png',
    'figures/phase05_accuracy_efficiency_tradeoff.png',
    'figures/phase05_classification_vs_traditional_baselines.png',
    'figures/phase05_regression_vs_traditional_baselines.png',
]
for rel in figure_paths:
    print(rel)
    display(Image(filename=str(phase / rel), width=980))
"""
    )
    return [heading, audit_code, table_code, figure_code]


def status_cells() -> list[nbformat.NotebookNode]:
    heading = nbformat.v4.new_markdown_cell(
        "### Final audit and frozen status\n\n"
        "The status below is read from the saved final audits, manifest, and freeze record."
    )
    code = nbformat.v4.new_code_cell(
        """from pathlib import Path
import json
from IPython.display import display

phase = Path.cwd()
def load_json(rel):
    return json.loads((phase / rel).read_text(encoding='utf-8'))

final_audits = {
    'artifact': load_json('audits/phase05_final_artifact_audit.json'),
    'reproducibility': load_json('audits/phase05_final_reproducibility_audit.json'),
    'upstream_freeze_integrity': load_json('audits/phase05_upstream_freeze_integrity_audit.json'),
}
final_manifest = load_json('manifests/phase05_final_artifact_manifest.json')
phase05_freeze = load_json('configs/phase05_freeze.json')
freeze_evidence = {key: phase05_freeze.get(key) for key in [
    'phase', 'phase_name', 'status', 'modeling_rows', 'subjects', 'primary_features',
    'outer_folds', 'dimensions', 'seeds', 'levels', 'feature_k', 'configurations',
    'final_confirmation_runs', 'canonical_configuration_selection',
    'ready_for_next_planned_phase',
]}
display({'final_audits': final_audits,
         'final_manifest_status': final_manifest.get('status'),
         'phase05_freeze': freeze_evidence})
assert all(item.get('result') == 'PASS' for item in final_audits.values())
assert phase05_freeze.get('status') == 'FROZEN'
print('PHASE 05 STATUS: FROZEN; READY TO PROCEED TO NEXT PLANNED PHASE: YES')
"""
    )
    return [heading, code]


def write_audit(nb: nbformat.NotebookNode, baseline: list[str], expected_min_cells: int) -> dict:
    preserved = len(nb.cells) >= BASELINE_CELLS and [cell_digest(c) for c in nb.cells[:BASELINE_CELLS]] == baseline
    appended_code = [c for c in nb.cells[BASELINE_CELLS:] if c.cell_type == "code"]
    outputs_saved = all(c.get("execution_count") is not None and len(c.get("outputs", [])) > 0 for c in appended_code)
    no_errors = all(not any(o.get("output_type") == "error" for o in c.get("outputs", [])) for c in appended_code)
    checks = {
        "baseline_cell_count": BASELINE_CELLS,
        "current_cell_count": len(nb.cells),
        "expected_minimum_cell_count": expected_min_cells,
        "prior_cells_and_outputs_preserved": preserved,
        "appended_code_cells_executed": outputs_saved,
        "appended_outputs_error_free": no_errors,
        "final_section_present": any(MARKER in c.source for c in nb.cells if c.cell_type == "markdown"),
    }
    result = "PASS" if all([
        checks["current_cell_count"] >= expected_min_cells,
        checks["prior_cells_and_outputs_preserved"],
        checks["appended_code_cells_executed"],
        checks["appended_outputs_error_free"],
        checks["final_section_present"],
    ]) else "FAIL"
    audit = {
        "phase": "05",
        "audit": "final_notebook_persistence",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "notebook": NOTEBOOK.name,
        "notebook_sha256": sha256(NOTEBOOK),
        "checks": checks,
        "result": result,
    }
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["evidence", "status"])
    args = parser.parse_args()
    nb = nbformat.read(NOTEBOOK, as_version=4)

    if args.mode == "evidence":
        if len(nb.cells) != BASELINE_CELLS:
            raise RuntimeError(f"Expected untouched {BASELINE_CELLS}-cell notebook, found {len(nb.cells)}")
        baseline = [cell_digest(c) for c in nb.cells]
        new_cells = evidence_cells()
        start = len(nb.cells)
        nb.cells.extend(new_cells)
        execute_new_cells(nb, [start + 1, start + 2, start + 3])
        write_notebook(nb)
        (PHASE / "audits/phase05_notebook_baseline_cell_hashes.json").write_text(
            json.dumps({"baseline_cell_count": BASELINE_CELLS, "cell_sha256": baseline}, indent=2) + "\n",
            encoding="utf-8",
        )
        audit = write_audit(nb, baseline, BASELINE_CELLS + len(new_cells))
    else:
        baseline_record = json.loads((PHASE / "audits/phase05_notebook_baseline_cell_hashes.json").read_text(encoding="utf-8"))
        baseline = baseline_record["cell_sha256"]
        if any("Final audit and frozen status" in c.source for c in nb.cells if c.cell_type == "markdown"):
            raise RuntimeError("Final frozen-status cells already exist")
        new_cells = status_cells()
        start = len(nb.cells)
        nb.cells.extend(new_cells)
        execute_new_cells(nb, [start + 1])
        write_notebook(nb)
        audit = write_audit(nb, baseline, start + len(new_cells))

    print(json.dumps({"mode": args.mode, "cells": len(nb.cells), "audit": audit["result"], "notebook_sha256": audit["notebook_sha256"]}))


if __name__ == "__main__":
    main()
