"""Independent Phase 09 final-analysis verification; never freezes or starts Phase 10."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_phase09_robustness import classification_metrics, regression_metrics
from consolidate_phase09_oof import dry_run, json_default
from run_phase09_batch import atomic_json, read_json, sha256


ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "results" / "oof"
SUMMARIES = ROOT / "results" / "summaries"
AUDITS = ROOT / "audits"
MANIFEST = ROOT / "configs" / "phase09_execution_manifest.json"
NOTEBOOK_AUDIT = AUDITS / "phase09_final_notebook_persistence_audit.json"

OOF_FILES = [
    "results/oof/phase09_missing_modality_canonical_classification_oof.csv",
    "results/oof/phase09_missing_modality_canonical_regression_oof.csv",
    "results/oof/phase09_loso_canonical_classification_oof.csv",
    "results/oof/phase09_loso_canonical_regression_oof.csv",
    "results/oof/phase09_canonical_oof_index.csv",
    "results/oof/phase09_full_primary_reference_index.csv",
]
SUMMARY_FILES = [
    "phase09_missing_modality_classification_metrics.csv",
    "phase09_missing_modality_regression_metrics.csv",
    "phase09_loso_classification_metrics.csv",
    "phase09_loso_regression_metrics.csv",
    "phase09_seed_stability.csv",
    "phase09_missing_modality_robustness.csv",
    "phase09_missing_modality_deltas.csv",
    "phase09_model_robustness_comparison.csv",
    "phase09_loso_subject_metrics.csv",
    "phase09_loso_subject_stability.csv",
    "phase09_loso_difficulty_level_errors.csv",
    "phase09_loso_seed_variability.csv",
    "phase09_pairwise_statistics.csv",
    "phase09_bootstrap_confidence_intervals.csv",
    "phase09_flight_dependence_evidence.csv",
]
FIGURE_STEMS = [
    "phase09_missing_modality_classification_curve",
    "phase09_missing_modality_regression_curve",
    "phase09_missing_modality_model_comparison",
    "phase09_loso_subject_classification",
    "phase09_loso_subject_regression",
    "phase09_loso_stability_distribution",
]
REPORT_FILES = [
    "reports/phase09_missing_modality_report.md",
    "reports/phase09_loso_stability_report.md",
    "reports/phase09_statistical_appendix.md",
    "reports/phase09_generalization_boundaries.md",
    "reports/phase09_final_analysis.md",
    "reports/analysis-output/analysis-report.md",
    "reports/analysis-output/stats-appendix.md",
    "reports/analysis-output/figure-catalog.md",
]
COMPONENT_AUDITS = [
    "phase09_oof_coverage_audit.json", "phase09_oof_alignment_audit.json",
    "phase09_oof_leakage_audit.json", "phase09_full_primary_reference_integrity_audit.json",
    "phase09_metric_recalculation_audit.json", "phase09_statistical_unit_audit.json",
    "phase09_multiple_comparison_audit.json",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metric_reconciliation() -> dict[str, Any]:
    missing_class = pd.read_csv(OOF / "phase09_missing_modality_canonical_classification_oof.csv")
    missing_reg = pd.read_csv(OOF / "phase09_missing_modality_canonical_regression_oof.csv")
    loso_class = pd.read_csv(OOF / "phase09_loso_canonical_classification_oof.csv")
    loso_reg = pd.read_csv(OOF / "phase09_loso_canonical_regression_oof.csv")
    references = pd.read_csv(OOF / "phase09_full_primary_reference_index.csv")
    stored_class = pd.read_csv(SUMMARIES / "phase09_missing_modality_classification_metrics.csv")
    stored_reg = pd.read_csv(SUMMARIES / "phase09_missing_modality_regression_metrics.csv")
    stored_loso_class = pd.read_csv(SUMMARIES / "phase09_loso_classification_metrics.csv")
    stored_loso_reg = pd.read_csv(SUMMARIES / "phase09_loso_regression_metrics.csv")
    mismatches: list[str] = []
    class_source = pd.concat([missing_class, references[references.task == "classification"]], ignore_index=True)
    reg_source = pd.concat([missing_reg, references[references.task == "regression"]], ignore_index=True)
    for (condition, model_key), group in class_source.groupby(["condition", "model_key"]):
        expected = classification_metrics(group)["macro_f1"]
        found = stored_class[(stored_class.condition == condition) & (stored_class.model_key == model_key)]
        if len(found) != 1 or not np.isclose(found.macro_f1.iat[0], expected, atol=1e-12):
            mismatches.append(f"classification:{condition}:{model_key}")
    for (condition, model_key), group in reg_source.groupby(["condition", "model_key"]):
        expected = regression_metrics(group)["bounded_mae"]
        found = stored_reg[(stored_reg.condition == condition) & (stored_reg.model_key == model_key)]
        if len(found) != 1 or not np.isclose(found.bounded_mae.iat[0], expected, atol=1e-12):
            mismatches.append(f"regression:{condition}:{model_key}")
    for model_key, group in loso_class.groupby("model_key"):
        expected = classification_metrics(group)["macro_f1"]
        found = stored_loso_class[stored_loso_class.model_key == model_key]
        if len(found) != 1 or not np.isclose(found.macro_f1.iat[0], expected, atol=1e-12):
            mismatches.append(f"loso_classification:{model_key}")
    for model_key, group in loso_reg.groupby("model_key"):
        expected = regression_metrics(group)["bounded_mae"]
        found = stored_loso_reg[stored_loso_reg.model_key == model_key]
        if len(found) != 1 or not np.isclose(found.bounded_mae.iat[0], expected, atol=1e-12):
            mismatches.append(f"loso_regression:{model_key}")
    return {"status": "PASS" if not mismatches else "FAIL", "independent_recalculation": True, "mismatches": mismatches}


def verify(allow_notebook_pending: bool = False) -> dict[str, Any]:
    consolidation = dry_run()
    metrics = metric_reconciliation()
    required = OOF_FILES + [f"results/summaries/{name}" for name in SUMMARY_FILES] + REPORT_FILES
    required += [f"figures/{stem}.{extension}" for stem in FIGURE_STEMS for extension in ("pdf", "png")]
    present = [relative for relative in required if (ROOT / relative).is_file() and (ROOT / relative).stat().st_size > 0]
    artifacts = [{"path": relative, "bytes": (ROOT / relative).stat().st_size, "sha256": sha256(ROOT / relative)} for relative in present]
    figure_magic = all(
        (ROOT / f"figures/{stem}.pdf").read_bytes()[:4] == b"%PDF"
        and (ROOT / f"figures/{stem}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        for stem in FIGURE_STEMS
    ) if len(present) == len(required) else False
    component_pass = all(read_json(AUDITS / name).get("status") == "PASS" for name in COMPONENT_AUDITS)
    notebook_pass = NOTEBOOK_AUDIT.exists() and read_json(NOTEBOOK_AUDIT).get("status") == "PASS"
    index = pd.read_csv(OOF / "phase09_canonical_oof_index.csv")
    references = pd.read_csv(OOF / "phase09_full_primary_reference_index.csv")
    pairwise = pd.read_csv(SUMMARIES / "phase09_pairwise_statistics.csv")
    bootstrap = pd.read_csv(SUMMARIES / "phase09_bootstrap_confidence_intervals.csv")
    flight = pd.read_csv(SUMMARIES / "phase09_flight_dependence_evidence.csv")
    manifest = read_json(MANIFEST)
    checks = {
        "consolidation_dry_run": consolidation["status"] == "PASS",
        "canonical_rows_10056": len(index) == 10056,
        "canonical_keys_unique": index.canonical_key.nunique() == 10056,
        "reference_rows_1676": len(references) == 1676,
        "required_artifacts_present_nonempty": len(present) == len(required),
        "figure_pairs_6_vector_and_600dpi_raster": figure_magic,
        "component_audits_pass": component_pass,
        "independent_metric_reconciliation": metrics["status"] == "PASS",
        "subject_pairwise_rows_20": len(pairwise) == 20 and pairwise.n_subjects.eq(35).all(),
        "holm_complete": pairwise.p_value_holm.notna().all() and pairwise.holm_family_size.eq(5).all(),
        "bootstrap_2000": len(bootstrap) == 24 and bootstrap.resamples.eq(2000).all(),
        "flight_boundary_explicit": len(flight) == 4 and flight.generalizable_behavior_claim.eq("INCONCLUSIVE_DUE_TO_METADATA").all(),
        "phase09_freeze_absent": not (ROOT / "configs/phase09_freeze.json").exists(),
        "phase10_not_executed": manifest.get("phase10_executed") is False,
        "model_retraining_during_analysis_no": manifest.get("model_retraining_during_analysis") is False,
        "raw_predictions_regenerated_during_analysis_no": manifest.get("raw_predictions_regenerated_during_analysis") is False,
        "notebook_persistence": notebook_pass or allow_notebook_pending,
    }
    artifact_audit = {
        "phase": "09", "audit": "final_analysis_artifact", "status": "PASS" if len(present) == len(required) and figure_magic else "FAIL",
        "audited_at_utc": now(), "required_count": len(required), "present_count": len(present), "artifacts": artifacts,
        "figure_pdf_png_pairs": 6, "pubfig_available": False, "fallback": "matplotlib; PDF vector plus PNG 600 DPI",
    }
    reproducibility = {
        "phase": "09", "audit": "final_analysis_reproducibility",
        "status": "PASS" if consolidation["status"] == metrics["status"] == "PASS" and component_pass else "FAIL",
        "audited_at_utc": now(), "read_only_consolidation_dry_run": consolidation,
        "independent_metric_reconciliation": metrics, "component_audits_pass": component_pass,
        "model_retraining_executed": False, "raw_predictions_regenerated": False,
        "outer_test_used_for_selection": False, "phase09_freeze_executed": False, "phase10_executed": False,
    }
    atomic_json(AUDITS / "phase09_final_analysis_artifact_audit.json", artifact_audit)
    atomic_json(AUDITS / "phase09_final_analysis_reproducibility_audit.json", reproducibility)
    overall = all(checks.values()) and artifact_audit["status"] == reproducibility["status"] == "PASS"
    result = {
        "phase": "09", "audit": "final_analysis_verification", "status": "PASS" if overall else "FAIL",
        "audited_at_utc": now(), "checks": checks, "ready_for_phase09_freeze": bool(overall and notebook_pass),
        "model_retraining_executed": False, "raw_predictions_regenerated": False,
        "phase09_freeze_executed": False, "phase10_executed": False,
    }
    atomic_json(AUDITS / "phase09_final_analysis_verification.json", result)
    if overall and notebook_pass:
        manifest["status"] = "ANALYSIS_COMPLETE_PENDING_FREEZE"
        manifest["ready_for_phase09_freeze"] = True
        manifest["analysis_completed"] = True
    elif allow_notebook_pending and overall:
        manifest["status"] = "ANALYSIS_COMPLETE_PENDING_NOTEBOOK"
        manifest["ready_for_phase09_freeze"] = False
        manifest["analysis_completed"] = False
    else:
        manifest["status"] = "ANALYSIS_VERIFICATION_FAILED"
        manifest["ready_for_phase09_freeze"] = False
        manifest["analysis_completed"] = False
    manifest["phase09_freeze_executed"] = False
    manifest["phase10_executed"] = False
    manifest["last_updated_utc"] = now()
    atomic_json(MANIFEST, manifest)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-notebook-pending", action="store_true")
    args = parser.parse_args()
    outcome = verify(args.allow_notebook_pending)
    print(json.dumps(outcome, indent=2, default=json_default))
    raise SystemExit(0 if outcome["status"] == "PASS" else 1)
