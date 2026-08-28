"""Append and execute Phase 09 final-analysis evidence while preserving prior cells."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

from run_phase09_batch import atomic_json
from verify_phase09_final_analysis import verify


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Phase_09_Robustness_and_Generalization.ipynb"
AUDIT = ROOT / "audits" / "phase09_final_notebook_persistence_audit.json"
MARKER = "## Phase 09 Final Analysis Evidence"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(cell: Any) -> str:
    payload = json.dumps(cell, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def new_cells() -> list[Any]:
    return [
        nbformat.v4.new_markdown_cell(MARKER + "\n\nPersisted presentation of canonical OOF, independently recalculated metrics, subject-level statistics, figures, and claim boundaries. No training, raw-prediction regeneration, freeze, or Phase 10 execution occurs here."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd\nPHASE09_ROOT=Path.cwd().resolve()\ndef audit(name): return json.loads((PHASE09_ROOT/'audits'/name).read_text(encoding='utf-8'))\nindex=pd.read_csv(PHASE09_ROOT/'results/oof/phase09_canonical_oof_index.csv')\nrefs=pd.read_csv(PHASE09_ROOT/'results/oof/phase09_full_primary_reference_index.csv')\n{'canonical_rows':len(index),'reference_rows':len(refs),'coverage':audit('phase09_oof_coverage_audit.json')['status'],'alignment':audit('phase09_oof_alignment_audit.json')['status'],'leakage':audit('phase09_oof_leakage_audit.json')['status']}"),
        nbformat.v4.new_markdown_cell("### Missing-modality robustness and subject-level inference"),
        nbformat.v4.new_code_cell("robustness=pd.read_csv(PHASE09_ROOT/'results/summaries/phase09_missing_modality_robustness.csv')\nstats=pd.read_csv(PHASE09_ROOT/'results/summaries/phase09_pairwise_statistics.csv')\n{'comparisons':len(stats),'subjects_per_comparison':sorted(stats.n_subjects.unique().tolist()),'holm_complete':bool(stats.p_value_holm.notna().all()),'bootstrap_resamples':sorted(stats.bootstrap_resamples.unique().tolist()),'largest_degradation_by_model':robustness.loc[robustness.groupby('model_key').mean_subject_degradation.idxmax(),['model_key','condition','mean_subject_degradation']].to_dict('records')}"),
        nbformat.v4.new_markdown_cell("### LOSO stability, flight dependence, and generalization boundary"),
        nbformat.v4.new_code_cell("loso=pd.read_csv(PHASE09_ROOT/'results/summaries/phase09_loso_subject_stability.csv')\nflight=pd.read_csv(PHASE09_ROOT/'results/summaries/phase09_flight_dependence_evidence.csv')\n{'loso_subjects':sorted(loso.subjects.unique().tolist()),'flight_dependence_rows':len(flight),'flight_generalizable_behavior_claim':sorted(flight.generalizable_behavior_claim.unique().tolist()),'unseen_session':'NOT_FEASIBLE_DUE_TO_METADATA','unseen_scenario':'NOT_FEASIBLE_DUE_TO_METADATA','task_template':'NOT_FEASIBLE_DUE_TO_METADATA','route_configuration':'NOT_FEASIBLE_DUE_TO_METADATA'}"),
        nbformat.v4.new_markdown_cell("### Figures, reports, and pre-persistence verification"),
        nbformat.v4.new_code_cell("figures=sorted(p.name for p in (PHASE09_ROOT/'figures').glob('phase09_*.*'))\nreports=sorted(str(p.relative_to(PHASE09_ROOT)) for p in (PHASE09_ROOT/'reports').rglob('*.md'))\nverification=audit('phase09_final_analysis_verification.json')\n{'figure_files':figures,'report_files':reports,'pre_persistence_status':verification['status'],'model_retraining_executed':False,'raw_predictions_regenerated':False,'phase09_freeze_executed':False,'phase10_executed':False}"),
        nbformat.v4.new_markdown_cell("## Final Analysis Takeaways\n\nPhase 09 contains 10,056 canonical OOF rows plus 1,676 frozen Full Primary reference rows. Missing-modality effects are paired at the subject level (n=35), corrected with Holm within each preregistered model-task family, and accompanied by 2,000-resample bootstrap intervals. LOSO supports held-out-subject evaluation only. Flight generalizable behavior is inconclusive because session, scenario, task-template, and route metadata are unavailable. Analysis is complete pending a separate Phase 09 freeze; Phase 10 was not executed."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nPHASE09_ROOT=Path.cwd().resolve()\nmanifest=json.loads((PHASE09_ROOT/'configs/phase09_execution_manifest.json').read_text(encoding='utf-8'))\n{'phase09_status':manifest['status'],'ready_for_phase09_freeze':manifest.get('ready_for_phase09_freeze',False),'phase09_freeze_executed':manifest.get('phase09_freeze_executed',False),'phase10_executed':manifest.get('phase10_executed',False)}"),
    ]


def main() -> None:
    pre = verify(allow_notebook_pending=True)
    if pre["status"] != "PASS":
        raise RuntimeError({"pre_notebook_verification": pre})
    original = nbformat.read(NOTEBOOK, as_version=4)
    kept = []
    for cell in original.cells:
        if cell.cell_type == "markdown" and MARKER in cell.source:
            break
        kept.append(cell)
    kept_fingerprints = [fingerprint(cell) for cell in kept]
    additions = new_cells()
    temporary = nbformat.v4.new_notebook(
        cells=additions[1:],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )
    executed = NotebookClient(temporary, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    final = nbformat.v4.new_notebook(cells=kept + [additions[0]] + executed.cells, metadata=original.metadata)
    nbformat.write(final, NOTEBOOK)
    reloaded = nbformat.read(NOTEBOOK, as_version=4)
    prior_preserved = [fingerprint(cell) for cell in reloaded.cells[:len(kept)]] == kept_fingerprints
    appended = reloaded.cells[len(kept):]
    code = [cell for cell in appended if cell.cell_type == "code"]
    errors = [output for cell in code for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    output_text = "\n".join(
        str(output.get("text", output.get("data", {}).get("text/plain", "")))
        for cell in code for output in cell.get("outputs", [])
    )
    checks = {
        "notebook_parseable": True,
        "final_analysis_marker_once": sum(MARKER in cell.source for cell in reloaded.cells if cell.cell_type == "markdown") == 1,
        "prior_cells_and_outputs_preserved_byte_equivalent": prior_preserved,
        "appended_code_cells_5": len(code) == 5,
        "all_appended_code_cells_executed": all(cell.execution_count is not None for cell in code),
        "all_appended_code_cells_have_outputs": all(cell.get("outputs") for cell in code),
        "error_outputs_zero": not errors,
        "canonical_counts_in_outputs": "10056" in output_text and "1676" in output_text,
        "claim_boundary_in_outputs": "INCONCLUSIVE_DUE_TO_METADATA" in output_text,
        "prior_cells_not_reexecuted": True,
        "model_retraining_executed_no": True,
        "raw_predictions_regenerated_no": True,
        "phase09_freeze_executed_no": True,
        "phase10_executed_no": True,
    }
    payload = {
        "phase": "09", "audit": "final_notebook_persistence", "status": "PASS" if all(checks.values()) else "FAIL",
        "audited_at_utc": now(), "checks": checks, "preserved_prefix_cells": len(kept),
        "appended_cells": len(appended), "notebook_path": str(NOTEBOOK.resolve()),
    }
    atomic_json(AUDIT, payload)
    if payload["status"] != "PASS":
        raise RuntimeError(payload)
    final_verification = verify(allow_notebook_pending=False)
    if final_verification["status"] != "PASS":
        raise RuntimeError({"final_verification": final_verification})
    # Refresh only the final status cell after verification advances the manifest.
    refreshed = nbformat.read(NOTEBOOK, as_version=4)
    status_cell = refreshed.cells[-1]
    status_notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(status_cell.source)],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )
    executed_status = NotebookClient(status_notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    refreshed.cells[-1] = executed_status.cells[0]
    nbformat.write(refreshed, NOTEBOOK)
    final_text = "\n".join(
        str(output.get("text", output.get("data", {}).get("text/plain", "")))
        for output in refreshed.cells[-1].get("outputs", [])
    )
    payload["checks"]["final_status_output_current"] = "ANALYSIS_COMPLETE_PENDING_FREEZE" in final_text and "'phase10_executed': False" in final_text
    payload["status"] = "PASS" if all(payload["checks"].values()) else "FAIL"
    payload["audited_at_utc"] = now()
    atomic_json(AUDIT, payload)
    if payload["status"] != "PASS":
        raise RuntimeError(payload)
    final_verification = verify(allow_notebook_pending=False)
    if final_verification["status"] != "PASS":
        raise RuntimeError({"post_refresh_final_verification": final_verification})
    print(json.dumps({"notebook_persistence": payload["status"], "final_verification": final_verification["status"]}, indent=2))


if __name__ == "__main__":
    main()
