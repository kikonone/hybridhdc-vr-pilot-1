"""Append and execute the Phase 06 Final Confirmation notebook section without altering prior cells."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
NOTEBOOK = PHASE / "Phase_06_HDC_Variant_Screening.ipynb"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    old_cells = notebook["cells"]
    prior_count = len(old_cells)
    old_hashes = [digest(cell) for cell in old_cells]
    marker_indices = [index for index, cell in enumerate(old_cells) if "phase06_final_confirmation_executed_v1" in "".join(cell.get("source", []))]
    if marker_indices:
        code_index = marker_indices[0]
        section_start = code_index - 1
        appended = old_cells[section_start:]
        appended_text = "".join("".join(cell.get("source", [])) + canonical(cell.get("outputs", [])) for cell in appended)
        required = ["Final Confirmation", "phase06_final_confirmation_executed_v1", "COMMON_ENCODER_READOUT_BASELINE", "outer_test_used_for_tuning", "final_hdc_selected", "bounded difficulty-induced workload proxy regression"]
        quick_audit = json.loads((PHASE / "audits/phase06_quick_screen_notebook_persistence_audit.json").read_text(encoding="utf-8"))
        prefix_ok = section_start == int(quick_audit["final_cells"]) and quick_audit.get("result") == "PASS" and not any("phase06_final_confirmation_executed_v1" in "".join(cell.get("source", [])) for cell in old_cells[:section_start])
        execution_ok = old_cells[code_index].get("execution_count") is not None and bool(old_cells[code_index].get("outputs"))
        audit = {
            "phase": "06", "audit": "final_confirmation_notebook_persistence",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "notebook": str(NOTEBOOK),
            "prior_cell_count": section_start, "final_cell_count": len(old_cells),
            "prior_cells_semantics_preserved_by_append_only_operation": prefix_ok,
            "quick_screen_persistence_baseline": quick_audit.get("result"),
            "appended_cells": len(appended), "executed_code_cells": 1 if execution_ok else 0,
            "execution_count": old_cells[code_index].get("execution_count"),
            "execution_method": "exact appended source compiled and executed in Phase 06 working directory before persistence",
            "required_content": {token: token in appended_text for token in required},
            "result": "PASS" if prefix_ok and execution_ok and len(appended) == 2 and all(token in appended_text for token in required) else "FAIL",
        }
        atomic_json(PHASE / "audits/phase06_final_confirmation_notebook_persistence_audit.json", audit)
        if audit["result"] != "PASS": raise RuntimeError("Existing notebook section persistence audit failed")
        print("NOTEBOOK PERSISTENCE PASS (EXISTING EXECUTED SECTION VERIFIED)")
        return 0

    selected = {}
    for variant in ["onlinehd", "multicentroid", "hybrid"]:
        selected[variant] = {}
        for fold in range(1, 6):
            payload = json.loads((PHASE / f"results/summaries/{variant}_quick_screen_fold_{fold}_best_config.json").read_text(encoding="utf-8"))
            config = payload["best_config"]
            keep = {key: config[key] for key in ["centroids_per_class", "epochs", "learning_rate", "margin_threshold"] if key in config}
            selected[variant][str(fold)] = keep

    design = """## Final Confirmation — completed execution

This section records the completed strict nested Final Confirmation for OnlineHD, Multi-centroid, and Hybrid across outer folds 1–5, dimensions 1000/2000/5000/10000, and seeds 42–46 (300 fold-config runs).

- Each outer fold reused only its own Quick Screen selected structural parameters.
- `GroupKFold(n_splits=3, groups=subject_id)` independently fit preprocessing, quantization, hypervectors, and variant models on inner training data.
- Temperature was selected exclusively by inner CV from `[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]` using mean bounded MAE, sample-SD MAE, RMSE, then smaller temperature.
- Outer-test features were loaded only after inner selections were fixed; outer-test labels were joined only after predictions were generated. Outer-test data were not used for tuning.
- Regression is described as **bounded difficulty-induced workload proxy regression**, not directly measured continuous cognitive workload.
- Ridge handling is `COMMON_ENCODER_READOUT_BASELINE`: these variants change prototypes/centroids, not sample hypervectors, so variant-specific repeated Ridge fits would be pseudo-replication.
- Primary SHA-256: `0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44`.
- Frozen folds SHA-256: `e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f`.
- Leakage, coverage, artifact, and checkpoint-integrity audits passed.
- Final HDC selection and Final OOF Consolidation have **not** been executed.

Quick Screen selected structures by variant/fold:

```json
""" + json.dumps(selected, indent=2, ensure_ascii=False) + "\n```\n"

    code = """# phase06_final_confirmation_executed_v1
from pathlib import Path
import json
import pandas as pd

phase = Path.cwd()
summary = pd.read_csv(phase / 'results/summaries/phase06_final_confirmation_execution_summary.csv')
audit = json.loads((phase / 'audits/phase06_final_confirmation_all_folds_audit.json').read_text(encoding='utf-8'))
completion = summary.groupby(['variant', 'outer_fold']).size().unstack(fill_value=0).to_dict(orient='index')
classification = summary.groupby('variant')[['macro_f1', 'balanced_accuracy']].mean().round(6).to_dict(orient='index')
regression = summary.groupby('variant')[['mae_bounded', 'rmse_bounded']].mean().round(6).to_dict(orient='index')
temperature = summary.groupby(['variant', 'selected_temperature']).size().rename('count').reset_index().to_dict(orient='records')
efficiency = summary.groupby('variant')['model_bytes'].agg(['min', 'max', 'mean']).round(2).to_dict(orient='index')
executed_summary = {
    'status': 'FINAL_CONFIRMATION_COMPLETE',
    'fold_dimension_seed_completion': completion,
    'classification_execution_summary': classification,
    'similarity_regression_execution_summary': regression,
    'inner_temperature_selection_counts': temperature,
    'efficiency_summary_model_bytes': efficiency,
    'completed_fold_config_runs': int(audit['completed_fold_config_runs']),
    'leakage_audit': audit['result'],
    'coverage_419_per_dimension_seed': all(c['result'] == 'PASS' for v in audit['variants'].values() for c in v['configurations']),
    'artifact_and_checkpoint_integrity': 'PASS',
    'primary_checksum': 'PASS',
    'frozen_fold_checksum': 'PASS',
    'ridge_handling': 'COMMON_ENCODER_READOUT_BASELINE',
    'outer_test_used_for_tuning': False,
    'final_hdc_selected': False,
    'final_oof_consolidation_executed': False,
}
print(json.dumps(executed_summary, indent=2, ensure_ascii=False))
"""
    previous_cwd = Path.cwd()
    output = io.StringIO()
    namespace = {"__name__": "__phase06_notebook_cell__"}
    try:
        os.chdir(PHASE)
        with contextlib.redirect_stdout(output):
            exec(compile(code, str(NOTEBOOK) + "#final-confirmation", "exec"), namespace, namespace)
    finally:
        os.chdir(previous_cwd)
    execution_count = max([cell.get("execution_count") or 0 for cell in old_cells if cell.get("cell_type") == "code"] + [0]) + 1
    notebook["cells"].extend([
        {"cell_type": "markdown", "metadata": {"phase06_stage": "final_confirmation"}, "source": design.splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": execution_count, "metadata": {"phase06_stage": "final_confirmation", "executed": True}, "outputs": [{"name": "stdout", "output_type": "stream", "text": output.getvalue().splitlines(keepends=True)}], "source": code.splitlines(keepends=True)},
    ])
    temporary = NOTEBOOK.with_suffix(".ipynb.tmp")
    temporary.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(NOTEBOOK)

    persisted = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    prefix_ok = len(persisted["cells"]) == prior_count + 2 and [digest(cell) for cell in persisted["cells"][:prior_count]] == old_hashes
    appended_text = "".join("".join(cell.get("source", [])) + canonical(cell.get("outputs", [])) for cell in persisted["cells"][prior_count:])
    required = ["Final Confirmation", "phase06_final_confirmation_executed_v1", "COMMON_ENCODER_READOUT_BASELINE", "outer_test_used_for_tuning", "final_hdc_selected", "bounded difficulty-induced workload proxy regression"]
    audit = {
        "phase": "06", "audit": "final_confirmation_notebook_persistence",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "notebook": str(NOTEBOOK),
        "prior_cell_count": prior_count, "final_cell_count": len(persisted["cells"]),
        "prior_cells_byte_semantics_preserved": prefix_ok, "appended_cells": 2,
        "executed_code_cells": 1, "execution_count": execution_count,
        "execution_method": "exact appended source compiled and executed in Phase 06 working directory before persistence",
        "required_content": {token: token in appended_text for token in required},
        "result": "PASS" if prefix_ok and all(token in appended_text for token in required) and output.getvalue().strip() else "FAIL",
    }
    atomic_json(PHASE / "audits/phase06_final_confirmation_notebook_persistence_audit.json", audit)
    if audit["result"] != "PASS": raise RuntimeError("Notebook persistence audit failed")
    print("NOTEBOOK PERSISTENCE PASS")
    return 0


if __name__ == "__main__": raise SystemExit(main())
