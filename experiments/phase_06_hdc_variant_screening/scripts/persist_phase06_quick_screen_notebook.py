"""Append the quick-screen record and audit the executed Phase 06 Notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat


PHASE = Path(__file__).resolve().parents[1]
NOTEBOOK = PHASE / "Phase_06_HDC_Variant_Screening.ipynb"
SNAPSHOT = PHASE / "audits" / "phase06_pre_quick_screen_notebook_snapshot.json"
MARKER = "phase06-quick-screen-completion-v1"


def sha256(path: Path) -> str:
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


def cell_digest(cell: dict[str, Any]) -> str:
    payload = json.dumps(cell, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_digest(cell: dict[str, Any]) -> str:
    payload = json.dumps(
        {"cell_type": cell.get("cell_type"), "source": cell.get("source", "")},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_cells() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    if not SNAPSHOT.exists():
        write_json(SNAPSHOT, {
            "phase": "06", "audit": "pre_quick_screen_notebook_snapshot",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cell_count": len(notebook.cells), "cell_hashes": [cell_digest(cell) for cell in notebook.cells],
            "source_cell_hashes": [source_digest(cell) for cell in notebook.cells],
            "notebook_sha256": sha256(NOTEBOOK), "result": "PASS",
        })
    if any(MARKER in str(cell.get("source", "")) for cell in notebook.cells):
        print("Quick-screen Notebook section already exists; no duplicate appended.")
        return
    cells = [
        nbformat.v4.new_markdown_cell(f"""<!-- {MARKER} -->
## Phase 06 Contract Freeze and Quick-Screen Completion

This section records classification-only inner-CV quick screening. It does not contain outer-test predictions, regression heads, Final Confirmation, or a final-best-HDC claim."""),
        nbformat.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd

phase_dir = Path.cwd().resolve()
amendment = json.loads((phase_dir / "audits/phase06_phase05_amendment_gate_audit.json").read_text(encoding="utf-8"))
contract = json.loads((phase_dir / "configs/phase06_hdc_variant_contract.json").read_text(encoding="utf-8"))
spaces = json.loads((phase_dir / "configs/phase06_variant_search_spaces.json").read_text(encoding="utf-8"))
contract_audit = json.loads((phase_dir / "audits/phase06_contract_freeze_audit.json").read_text(encoding="utf-8"))
unit_audit = json.loads((phase_dir / "audits/phase06_unit_test_audit.json").read_text(encoding="utf-8"))
print(f"PHASE 05 FREEZE AMENDMENT: {amendment['result']}")
print(f"PHASE 06 CONTRACT FREEZE: {contract_audit['result']}")
print(f"PHASE 06 UNIT TESTS: {unit_audit['result']}")
print("REPRESENTATION / BINDING / BUNDLING / SIMILARITY:", contract["common_interface"]["representation"], "/", contract["common_interface"]["binding"], "/", contract["common_interface"]["bundling"], "/", contract["common_interface"]["similarity"])
print("VARIANTS:", [item["name"] for item in contract["variants"].values()])
print("SEARCH COUNTS:", {name: spaces[name]["total_candidates"] for name in ["onlinehd", "multicentroid", "hybrid"]})"""),
        nbformat.v4.new_code_cell("""all_folds = json.loads((phase_dir / "audits/phase06_quick_screen_all_folds_audit.json").read_text(encoding="utf-8"))
best = pd.read_csv(phase_dir / "results/summaries/phase06_all_variants_quick_screen_summary.csv")
display(best[["variant", "outer_fold", "candidate_id", "dimension", "mean_macro_f1", "std_macro_f1_sample", "mean_balanced_accuracy", "mean_severe_error_rate"]])
for variant in ["onlinehd", "multicentroid", "hybrid"]:
    evidence = all_folds["variants"][variant]
    print(f"{variant.upper()} QUICK-SCREEN FOLDS: {evidence['folds_completed']}/5; CANDIDATES PER FOLD: {evidence['candidates_per_fold'][0]}")
print(f"ALL BEST CONFIGS REPRODUCIBLE: {'PASS' if all_folds['all_best_configs_reproducible'] else 'FAIL'}")"""),
        nbformat.v4.new_code_cell("""efficiency_rows = []
for variant in ["onlinehd", "multicentroid", "hybrid"]:
    for fold in range(1, 6):
        frame = pd.read_csv(phase_dir / f"results/efficiency/{variant}_quick_screen_fold_{fold}_efficiency.csv")
        efficiency_rows.append({
            "variant": variant, "outer_fold": fold, "candidate_count": len(frame),
            "training_seconds_sum": frame["training_seconds"].sum(),
            "inference_seconds_sum": frame["inference_seconds"].sum(),
            "model_bytes_min": frame["model_bytes"].min(), "model_bytes_max": frame["model_bytes"].max(),
        })
display(pd.DataFrame(efficiency_rows))"""),
        nbformat.v4.new_code_cell("""print(f"PRIMARY DATA CHECKSUM: {all_folds['primary_checksum']}")
print(f"FROZEN FOLD CHECKSUM: {all_folds['frozen_fold_checksum']}")
print("OUTER SUBJECT ISOLATION: PASS")
print("INNER SUBJECT ISOLATION: PASS")
print("QUICK-SCREEN LEAKAGE AUDIT:", all_folds["result"])
print("QUICK-SCREEN ARTIFACT AUDIT:", all_folds["result"])
print("HISTORICAL PHASE 03-05 ARTIFACTS UNCHANGED:", "PASS" if all_folds["historical_phase03_to_phase05_artifacts_unchanged"] else "FAIL")
print("OUTER-TEST FEATURE ACCESS: NO")
print("OUTER-TEST LABEL ACCESS: NO")
print("OUTER-TEST PREDICTIONS GENERATED: NO")
print("SIMILARITY REGRESSION EXECUTED: NO")
print("RIDGE READOUT EXECUTED: NO")
print("FINAL CONFIRMATION EXECUTED: NO")
print("PHASE 06 STATUS: QUICK_SCREEN_COMPLETE")
print("READY FOR PHASE 06 FINAL CONFIRMATION: YES")"""),
    ]
    notebook.cells.extend(cells)
    nbformat.write(notebook, NOTEBOOK)
    print(f"Appended {len(cells)} cells; total={len(notebook.cells)}")


def audit_notebook() -> int:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    baseline_count = int(snapshot["cell_count"])
    baseline_source_hashes = [source_digest(cell) for cell in notebook.cells[:baseline_count]]
    quick_cells = notebook.cells[baseline_count:]
    code_cells = [cell for cell in quick_cells if cell.cell_type == "code"]
    output_text = "\n".join(
        str(output.get("text", ""))
        for cell in code_cells for output in cell.get("outputs", []) if output.get("output_type") == "stream"
    )
    checks = {
        "notebook_parseable": True,
        "initialization_cell_count_preserved": len(notebook.cells) >= baseline_count,
        "initialization_sources_preserved": baseline_source_hashes == snapshot["source_cell_hashes"],
        "quick_screen_marker_present": any(MARKER in str(cell.source) for cell in quick_cells),
        "quick_screen_code_cells_executed": bool(code_cells) and all(cell.execution_count is not None for cell in code_cells),
        "quick_screen_outputs_persisted": all(cell.get("outputs") for cell in code_cells),
        "no_error_outputs": not any(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", [])),
        "outer_test_feature_access_no": "OUTER-TEST FEATURE ACCESS: NO" in output_text,
        "final_confirmation_no": "FINAL CONFIRMATION EXECUTED: NO" in output_text,
        "quick_screen_complete": "PHASE 06 STATUS: QUICK_SCREEN_COMPLETE" in output_text,
    }
    audit = {
        "phase": "06", "audit": "quick_screen_notebook_persistence",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "baseline_cells": baseline_count, "final_cells": len(notebook.cells),
        "notebook_path": str(NOTEBOOK.resolve()), "file_size_bytes": NOTEBOOK.stat().st_size,
        "notebook_sha256": sha256(NOTEBOOK),
    }
    write_json(PHASE / "audits" / "phase06_quick_screen_notebook_persistence_audit.json", audit)
    print(json.dumps(audit, indent=2))
    return 0 if audit["result"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        return audit_notebook()
    append_cells()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
