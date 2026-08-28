"""Independent final verification for Phase 08 analysis (does not freeze)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from analyze_phase08_conditions import analyze, class_metrics, regression_metrics
from consolidate_phase08_oof import ROOT, EXPECTED_CANONICAL_ROWS, atomic_json, consolidate, now, read_json, sha256


REQUIRED = [
    "results/oof/phase08_canonical_classification_oof.csv", "results/oof/phase08_canonical_regression_oof.csv", "results/oof/phase08_canonical_oof_index.csv", "results/oof/phase08_upstream_reference_index.csv",
    "results/summaries/phase08_classification_metrics.csv", "results/summaries/phase08_regression_metrics.csv", "results/summaries/phase08_seed_stability.csv", "results/summaries/phase08_fusion_condition_comparison.csv", "results/summaries/phase08_fusion_increment_analysis.csv", "results/summaries/phase08_flight_behavioral_sensitivity.csv", "results/summaries/phase08_shortcut_evidence_matrix.csv", "results/summaries/phase08_generalization_evidence_limits.csv", "results/summaries/phase08_pairwise_statistics.csv", "results/summaries/phase08_bootstrap_confidence_intervals.csv", "results/summaries/phase08_final_comparison.csv",
    "reports/phase08_generalization_limitations.md", "reports/phase08_final_analysis.md", "reports/phase08_shortcut_and_generalization_report.md", "reports/phase08_statistical_appendix.md", "reports/analysis-output/analysis-report.md", "reports/analysis-output/stats-appendix.md", "reports/analysis-output/figure-catalog.md", "configs/phase09_generalization_handoff.json",
    "figures/phase08_classification_condition_comparison.pdf", "figures/phase08_classification_condition_comparison.png", "figures/phase08_regression_condition_comparison.pdf", "figures/phase08_regression_condition_comparison.png", "figures/phase08_fusion_increment_effects.pdf", "figures/phase08_fusion_increment_effects.png", "figures/phase08_shortcut_sensitivity.pdf", "figures/phase08_shortcut_sensitivity.png", "figures/phase08_subject_level_effects.pdf", "figures/phase08_subject_level_effects.png",
]


def metric_reconciliation() -> dict:
    cls = pd.read_csv(ROOT / "results/oof/phase08_canonical_classification_oof.csv")
    reg = pd.read_csv(ROOT / "results/oof/phase08_canonical_regression_oof.csv")
    cm = pd.read_csv(ROOT / "results/summaries/phase08_classification_metrics.csv")
    rm = pd.read_csv(ROOT / "results/summaries/phase08_regression_metrics.csv")
    mismatches = []
    for (condition, model), x in cls.groupby(["condition", "model_family"]):
        expected = class_metrics(x); row = cm[(cm.condition == condition) & (cm.model_family == model) & (cm.source_status == "NEW_PHASE08_RUN")]
        if len(row) != 1 or not np.isclose(row.macro_f1.iat[0], expected["macro_f1"], atol=1e-12): mismatches.append(f"classification:{condition}:{model}")
    for (condition, model), x in reg.groupby(["condition", "model_family"]):
        expected = regression_metrics(x); row = rm[(rm.condition == condition) & (rm.model_family == model) & (rm.source_status == "NEW_PHASE08_RUN")]
        if len(row) != 1 or not np.isclose(row.bounded_mae.iat[0], expected["bounded_mae"], atol=1e-12): mismatches.append(f"regression:{condition}:{model}")
    return {"status": "PASS" if not mismatches else "FAIL", "mismatches": mismatches, "independent_recalculation": True}


def verify(allow_notebook_pending: bool = False) -> dict:
    consolidation = consolidate(write=False); analysis = analyze(write=False); metrics = metric_reconciliation()
    existing = [p for p in REQUIRED if (ROOT / p).is_file() and (ROOT / p).stat().st_size > 0]
    artifact_rows = [{"path": p, "sha256": sha256(ROOT / p), "bytes": (ROOT / p).stat().st_size} for p in existing]
    notebook_audit_path = ROOT / "audits/phase08_final_notebook_persistence_audit.json"
    notebook_pass = notebook_audit_path.exists() and read_json(notebook_audit_path).get("status") == "PASS"
    audit_names = ["phase08_oof_coverage_audit.json", "phase08_oof_alignment_audit.json", "phase08_oof_leakage_audit.json", "phase08_upstream_reference_integrity_audit.json", "phase08_metric_recalculation_audit.json", "phase08_statistical_unit_audit.json", "phase08_multiple_comparison_audit.json"]
    audits_pass = all(read_json(ROOT / "audits" / x)["status"] == "PASS" for x in audit_names)
    cls = pd.read_csv(ROOT / "results/oof/phase08_canonical_classification_oof.csv"); reg = pd.read_csv(ROOT / "results/oof/phase08_canonical_regression_oof.csv")
    checks = {
        "consolidation_dry_run": consolidation["status"] == "PASS", "analysis_dry_run": analysis["status"] == "PASS",
        "canonical_rows_10894": len(cls) + len(reg) == EXPECTED_CANONICAL_ROWS, "required_artifacts_present": len(existing) == len(REQUIRED),
        "component_audits_pass": audits_pass, "metric_reconciliation": metrics["status"] == "PASS",
        "phase08_freeze_absent": not (ROOT / "configs/phase08_freeze.json").exists(), "phase09_not_executed": read_json(ROOT / "configs/phase09_generalization_handoff.json").get("phase09_executed") is False,
        "notebook_persistence": notebook_pass or allow_notebook_pending,
    }
    overall = all(checks.values())
    artifact_audit = {"status": "PASS" if len(existing) == len(REQUIRED) else "FAIL", "timestamp_utc": now(), "checks": {"required_count": len(existing) == len(REQUIRED), "all_nonempty": all(x["bytes"] > 0 for x in artifact_rows), "pdf_png_pairs_5": sum(x["path"].endswith(".pdf") for x in artifact_rows) == 5 and sum(x["path"].endswith(".png") for x in artifact_rows) == 5}, "artifact_count": len(artifact_rows), "artifacts": artifact_rows}
    reproducibility = {"status": "PASS" if consolidation["status"] == analysis["status"] == metrics["status"] == "PASS" else "FAIL", "timestamp_utc": now(), "read_only_consolidation_dry_run": consolidation, "read_only_analysis_dry_run": analysis, "metric_reconciliation": metrics, "model_retraining_executed": False, "outer_test_used_for_tuning": False, "phase09_executed": False}
    atomic_json(ROOT / "audits/phase08_final_analysis_artifact_audit.json", artifact_audit)
    atomic_json(ROOT / "audits/phase08_final_analysis_reproducibility_audit.json", reproducibility)
    result = {"status": "PASS" if overall else "FAIL", "timestamp_utc": now(), "checks": checks, "ready_for_phase08_freeze": overall and notebook_pass, "model_retraining_executed": False, "outer_test_used_for_tuning": False, "phase09_executed": False}
    atomic_json(ROOT / "audits/phase08_final_analysis_verification.json", result)
    manifest = read_json(ROOT / "configs/phase08_execution_manifest.json")
    if overall and notebook_pass:
        manifest["status"] = "ANALYSIS_COMPLETE_PENDING_FREEZE"; manifest["ready_for_phase08_freeze"] = True
    else:
        manifest["status"] = "ANALYSIS_COMPLETE_PENDING_NOTEBOOK" if allow_notebook_pending and overall else "ANALYSIS_VERIFICATION_FAILED"; manifest["ready_for_phase08_freeze"] = False
    manifest["analysis_completed"] = overall and notebook_pass; manifest["phase09_executed"] = False; manifest["last_updated_utc"] = now()
    atomic_json(ROOT / "configs/phase08_execution_manifest.json", manifest)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--allow-notebook-pending", action="store_true"); a = p.parse_args()
    result = verify(a.allow_notebook_pending); print(json.dumps(result, indent=2)); raise SystemExit(0 if result["status"] == "PASS" else 1)
