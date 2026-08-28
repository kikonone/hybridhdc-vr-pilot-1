"""Build the Phase 06 initialization Notebook; execution is a separate explicit step."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PHASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PHASE_DIR / "Phase_06_HDC_Variant_Screening.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nb["cells"] = [
    markdown(
        """
# Phase 06 — HDC Variant Screening

This executed Notebook initializes the Phase 06 interface and validates frozen upstream evidence. It does not train, tune, select, or evaluate a model.

Required P1 variants:

1. Vanilla Prototype HDC
2. OnlineHD-style HDC
3. Multi-centroid HDC
4. HDC+OnlineHD Hybrid

Vanilla Prototype HDC is reused read-only from frozen Phase 05. The regression task is **bounded difficulty-induced workload proxy regression**, and project conclusions are limited to **workload-proxy classification and regression**. The target is not directly measured continuous cognitive workload.
"""
    ),
    markdown("## 1. Project root and read-only preflight"),
    code(
        """
from pathlib import Path
import json
import sys

phase_dir = Path.cwd().resolve()
assert phase_dir.name == "phase_06_hdc_variant_screening"
project_root = phase_dir.parents[1]
assert (project_root / "最新完整实验计划_分类回归双任务.md").is_file()
assert (project_root / "CODEX_NOTEBOOK_RULES.md").is_file()
sys.path.insert(0, str(phase_dir / "src"))

from phase06_preflight import run_preflight

summary = run_preflight(phase_dir)
print(f"PROJECT ROOT: {project_root}")
print("UPSTREAM ACCESS MODE: READ ONLY")
print("PRIMARY DATA OR FOLDS COPIED: NO")
"""
    ),
    markdown("## 2. Primary data and frozen-fold evidence"),
    code(
        """
print(f"PRIMARY DATA ROWS: {summary['modeling_rows']}")
print(f"SUBJECTS: {summary['subjects']}")
print(f"PRIMARY FEATURES: {summary['primary_predictive_features']}")
print(f"UNIQUE RUN KEYS: {summary['unique_run_keys']}")
print(f"TARGET_CLASS VALUES: {summary['target_class_values']}")
print(f"TARGET_CLASS MISSING: {summary['target_class_missing']}")
print(f"TARGET_SCORE VALUES: {summary['target_score_values']}")
print(f"TARGET_SCORE MISSING: {summary['target_score_missing']}")
print(f"PRIMARY DATA SHA-256: {summary['primary_sha256']}")
print(f"PRIMARY DATA CHECKSUM: {summary['primary_checksum']}")
print(f"FROZEN FOLD SHA-256: {summary['frozen_fold_sha256']}")
print(f"FROZEN FOLD CHECKSUM: {summary['frozen_fold_checksum']}")
print(f"FOLD ROWS / UNIQUE RUNS: {summary['fold_assignment_rows']} / {summary['fold_unique_run_keys']}")
print(f"OUTER FOLDS: {summary['outer_folds']}")
print(f"OUTER SUBJECT ISOLATION: {summary['outer_subject_isolation']}")
print(f"INNER 3-FOLD GROUPKFOLD FEASIBILITY: {summary['inner_groupkfold_3_feasibility']}")
"""
    ),
    markdown("## 3. Frozen Phase 05 Vanilla HDC interface"),
    code(
        """
print(f"PHASE 05 STATUS: {summary['phase05']['status']}")
print(f"PHASE 05 FREEZE INTERFACE: {summary['phase05']['freeze_interface']}")
print(f"PHASE 05 VANILLA RESULTS AVAILABLE: {summary['phase05']['vanilla_results_available']}")
print("PHASE 05 VANILLA BASELINE ACCESS: READ-ONLY REUSE; RETRAINING PROHIBITED")
print(f"PHASE 05 CANONICAL CONFIGURATION SELECTION: {summary['phase05']['canonical_configuration_selection']}")
print("PHASE 05 OUTER-TEST OBSERVED BEST DIMENSION/SEED USED FOR PHASE 06 SELECTION: NO")
"""
    ),
    markdown("## 4. Frozen Phase 04 comparison interfaces"),
    code(
        """
print(f"PHASE 04A INTERFACE: {summary['phase04']['phase04a']}")
print(f"PHASE 04B INTERFACE: {summary['phase04']['phase04b']}")
print("TRADITIONAL MODEL RESULTS RECOMPUTED: NO")
"""
    ),
    markdown("## 5. Initialization audit readback"),
    code(
        """
audit_paths = {
    "INPUT AND FOLD AUDIT": phase_dir / "audits" / "phase06_input_and_fold_audit.json",
    "PHASE 05 FREEZE INTERFACE AUDIT": phase_dir / "audits" / "phase06_phase05_freeze_interface_audit.json",
    "PHASE 04 BASELINE INTERFACE AUDIT": phase_dir / "audits" / "phase06_phase04_baseline_interface_audit.json",
    "INITIALIZATION ARTIFACT AUDIT": phase_dir / "audits" / "phase06_initialization_artifact_audit.json",
}
for label, path in audit_paths.items():
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"{label}: {payload['result']}")
print(f"INPUT MANIFEST SAVED: {summary['input_manifest_saved']}")
"""
    ),
    markdown(
        """
## Phase Validation Summary

VERIFIED: actual Primary data, frozen folds, outer subject isolation, inner GroupKFold feasibility, frozen Phase 04A/04B interfaces, and frozen Phase 05 Vanilla HDC interface.

NOT VERIFIED: Phase 06 algorithm definitions and model-selection rules remain intentionally deferred to contract freeze.

WARNINGS: no Phase 05 outer-test observed best dimension or seed may be promoted to a canonical Phase 06 configuration.

KEY RESULTS: initialization evidence only; no model result was generated.

OUTPUT FILES: Phase 06 configs, manifest, audits, README, source validator, tests, and this executed Notebook.

NEXT PHASE REQUIREMENTS: complete and freeze the Phase 06 modeling contract before any HDC variant modeling.
"""
    ),
    code(
        """
gates = [
    summary["primary_checksum"] == "PASS",
    summary["frozen_fold_checksum"] == "PASS",
    summary["outer_subject_isolation"] == "PASS",
    summary["inner_groupkfold_3_feasibility"] == "PASS",
    summary["phase05"]["freeze_interface"] == "PASS",
    summary["phase04"]["phase04a"] == "PASS",
    summary["phase04"]["phase04b"] == "PASS",
    summary["initialization_audit"] == "PASS",
]
status = "PENDING_CONTRACT_FREEZE" if all(gates) else "FAIL"
print("HDC VARIANT TRAINING EXECUTED: NO")
print(f"PHASE 06 STATUS: {status}")
print(f"READY FOR PHASE 06 CONTRACT FREEZE: {'YES' if all(gates) else 'NO'}")
"""
    ),
]

nbf.write(nb, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
