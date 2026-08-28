"""Fail-closed Phase 09 final freeze. Dry-run is the default; --write freezes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

from run_phase09_batch import output_paths, read_json, reusable_run, sha256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT.parent
PRIMARY = EXPERIMENTS / "phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv"
FOLDS = EXPERIMENTS / "phase_03_multimodal_dataset_labeling/data/fold_assignments.csv"
EXECUTION = ROOT / "configs/phase09_execution_manifest.json"
FINAL_MANIFEST = ROOT / "manifests/phase09_final_manifest.json"
FREEZE_FILE = ROOT / "configs/phase09_freeze.json"
NOTEBOOK = ROOT / "Phase_09_Robustness_and_Generalization.ipynb"
MARKER = "## Phase 09 Final Freeze Summary"
EXPECTED_PRIMARY = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLDS = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
EXPECTED = {"runs": 720, "raw_rows": 30168, "canonical": 10056, "missing": 8380, "loso": 1676, "subjects": 35}

CONTRACTS = [
    "configs/phase09_frozen_contract.json", "configs/phase09_contract_freeze.json",
    "configs/phase09_execution_manifest.json", "configs/phase09_missing_modality_contract.json",
    "configs/phase09_loso_contract.json", "configs/phase09_loso_config_mapping.json",
    "configs/phase09_oof_aggregation_rules.json", "configs/phase09_statistical_rules.json",
    "manifests/phase09_loso_assignments.csv",
]
REQUIRED_REPORTS = [
    "reports/phase09_missing_modality_report.md", "reports/phase09_loso_stability_report.md",
    "reports/phase09_statistical_appendix.md", "reports/phase09_generalization_boundaries.md",
    "reports/phase09_final_analysis.md", "reports/analysis-output/analysis-report.md",
    "reports/analysis-output/stats-appendix.md", "reports/analysis-output/figure-catalog.md",
]
FIGURE_STEMS = [
    "phase09_missing_modality_classification_curve", "phase09_missing_modality_regression_curve",
    "phase09_missing_modality_model_comparison", "phase09_loso_subject_classification",
    "phase09_loso_subject_regression", "phase09_loso_stability_distribution",
]
CRITICAL_AUDITS = [
    "phase09_checkpoint_integrity_audit.json", "phase09_execution_coverage_audit.json",
    "phase09_execution_leakage_audit.json", "phase09_feature_exclusion_audit.json",
    "phase09_execution_artifact_audit.json", "phase09_config_mapping_leakage_audit.json",
    "phase09_oof_coverage_audit.json", "phase09_oof_alignment_audit.json",
    "phase09_oof_leakage_audit.json", "phase09_full_primary_reference_integrity_audit.json",
    "phase09_metric_recalculation_audit.json", "phase09_statistical_unit_audit.json",
    "phase09_multiple_comparison_audit.json", "phase09_final_analysis_artifact_audit.json",
    "phase09_final_analysis_reproducibility_audit.json", "phase09_final_notebook_persistence_audit.json",
    "phase09_final_analysis_verification.json",
]
FINAL_AUDITS = {
    "phase09_freeze_audit.json", "phase09_final_manifest_audit.json",
    "phase09_final_hash_integrity_audit.json", "phase09_upstream_freeze_integrity_final_audit.json",
    "phase09_notebook_freeze_persistence_audit.json", "phase09_generalization_claim_audit.json",
}
MANIFEST_DEPENDENT_AUDITS = {
    "phase09_freeze_audit.json", "phase09_final_manifest_audit.json", "phase09_final_hash_integrity_audit.json",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def tree(relative: str) -> list[Path]:
    return sorted((path for path in (ROOT / relative).rglob("*") if path.is_file()), key=lambda path: path.as_posix().lower())


def artifact(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "absolute_path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def inventory(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [artifact(path) for path in sorted(paths, key=lambda value: value.as_posix().lower())]


def notebook_has_errors(path: Path = NOTEBOOK) -> bool:
    notebook = nbformat.read(path, as_version=4)
    return any(output.get("output_type") == "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.get("outputs", []))


def protected_baseline() -> dict[str, str]:
    execution = read_json(ROOT / "audits/phase09_execution_artifact_audit.json")
    analysis = read_json(ROOT / "audits/phase09_final_analysis_artifact_audit.json")
    rows = execution.get("artifacts", []) + analysis.get("artifacts", [])
    return {row["path"].replace("\\", "/"): row["sha256"] for row in rows}


def verify_hash_map(expected: dict[str, str]) -> list[str]:
    return sorted(relative for relative, digest in expected.items() if not (ROOT / relative).is_file() or sha256(ROOT / relative) != digest)


def upstream_inventory() -> list[dict[str, Any]]:
    source = read_json(ROOT / "manifests/phase09_upstream_freeze_manifest.json")
    rows = []
    for row in source["sources"]:
        path = Path(row["path"])
        actual = sha256(path) if path.is_file() else None
        rows.append({**row, "actual_sha256": actual, "match": actual == row["sha256"]})
    return rows


def preflight(require_freeze_absent: bool = True) -> dict[str, Any]:
    missing: list[str] = []
    for relative in CONTRACTS + REQUIRED_REPORTS:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)
    for stem in FIGURE_STEMS:
        for extension in ("pdf", "png"):
            relative = f"figures/{stem}.{extension}"
            if not (ROOT / relative).is_file():
                missing.append(relative)
    audits = {name: (ROOT / "audits" / name).is_file() and read_json(ROOT / "audits" / name).get("status") == "PASS" for name in CRITICAL_AUDITS}
    for name, passed in audits.items():
        if not passed:
            missing.append(f"audits/{name}:missing_or_not_pass")

    manifest = read_json(EXECUTION)
    records = manifest.get("training_runs", [])
    run_ids = [row.get("run_identifier") for row in records]
    reusable = [row["run_identifier"] for row in records if reusable_run(row, output_paths(row))]
    prediction_paths = [output_paths(row)["prediction"] for row in records]
    raw_rows = sum(csv_rows(path) for path in prediction_paths if path.is_file())
    output_path_rows = [str(path.resolve()).lower() for row in records for name, path in output_paths(row).items() if name != "checkpoint_dir"]
    missing_rows = sum(csv_rows(ROOT / relative) for relative in [
        "results/oof/phase09_missing_modality_canonical_classification_oof.csv",
        "results/oof/phase09_missing_modality_canonical_regression_oof.csv",
    ])
    loso_rows = sum(csv_rows(ROOT / relative) for relative in [
        "results/oof/phase09_loso_canonical_classification_oof.csv",
        "results/oof/phase09_loso_canonical_regression_oof.csv",
    ])
    canonical_rows = csv_rows(ROOT / "results/oof/phase09_canonical_oof_index.csv")
    canonical = pd.read_csv(ROOT / "results/oof/phase09_canonical_oof_index.csv")
    bootstrap = pd.read_csv(ROOT / "results/summaries/phase09_bootstrap_confidence_intervals.csv")
    pairwise = pd.read_csv(ROOT / "results/summaries/phase09_pairwise_statistics.csv")
    protected_mismatches = verify_hash_map(protected_baseline())
    upstream = upstream_inventory()
    phase10_paths = [str(path.resolve()) for path in EXPERIMENTS.glob("phase_10*")]
    frozen = read_json(ROOT / "configs/phase09_frozen_contract.json")
    contract_freeze = read_json(ROOT / "configs/phase09_contract_freeze.json")
    coverage = read_json(ROOT / "audits/phase09_oof_coverage_audit.json")
    leakage = read_json(ROOT / "audits/phase09_execution_leakage_audit.json")
    feature = read_json(ROOT / "audits/phase09_feature_exclusion_audit.json")
    checks = {
        "analysis_status_pending_freeze": manifest.get("status") == "ANALYSIS_COMPLETE_PENDING_FREEZE",
        "phase09_contract_frozen": frozen.get("status") == contract_freeze.get("status") == "CONTRACT_FROZEN_NOT_TRAINED",
        "authorized_runs_720": len(records) == manifest.get("training_run_count") == EXPECTED["runs"],
        "completed_runs_720_from_artifacts": len(reusable) == EXPECTED["runs"] and manifest.get("completed_training_runs") == EXPECTED["runs"],
        "run_identifiers_unique": len(set(run_ids)) == len(run_ids) == EXPECTED["runs"],
        "output_paths_unique": len(output_path_rows) == len(set(output_path_rows)) == EXPECTED["runs"] * 5,
        "raw_prediction_rows_30168": raw_rows == EXPECTED["raw_rows"],
        "canonical_oof_rows_10056": canonical_rows == EXPECTED["canonical"],
        "missing_modality_rows_8380": missing_rows == EXPECTED["missing"],
        "loso_rows_1676": loso_rows == EXPECTED["loso"],
        "canonical_keys_unique": canonical.canonical_key.nunique() == len(canonical) == EXPECTED["canonical"],
        "hdc_five_seed_coverage": coverage.get("five_seed_coverage") == "PASS",
        "full_primary_references": read_json(ROOT / "audits/phase09_full_primary_reference_integrity_audit.json").get("status") == "PASS",
        "canonical_run_key_coverage": coverage.get("checks", {}).get("run_key_coverage") is True,
        "canonical_alignment": read_json(ROOT / "audits/phase09_oof_alignment_audit.json").get("status") == "PASS",
        "subject_isolation": leakage.get("status") == "PASS" and leakage.get("leakage_failures") == [],
        "loso_config_mapping_leakage": read_json(ROOT / "audits/phase09_config_mapping_leakage_audit.json").get("status") == "PASS",
        "performance_features_included_no": feature.get("status") == "PASS" and feature.get("performance_features_included") is False,
        "metric_recalculation": read_json(ROOT / "audits/phase09_metric_recalculation_audit.json").get("status") == "PASS",
        "statistical_unit": read_json(ROOT / "audits/phase09_statistical_unit_audit.json").get("status") == "PASS",
        "holm_correction": read_json(ROOT / "audits/phase09_multiple_comparison_audit.json").get("status") == "PASS" and pairwise.p_value_holm.notna().all(),
        "bootstrap_2000": len(bootstrap) == 24 and bootstrap.resamples.eq(2000).all(),
        "critical_audits_pass": all(audits.values()),
        "protected_artifact_hashes_unchanged": not protected_mismatches,
        "figures_present_6_pairs": all((ROOT / f"figures/{stem}.{extension}").is_file() for stem in FIGURE_STEMS for extension in ("pdf", "png")),
        "reports_present": all((ROOT / relative).is_file() for relative in REQUIRED_REPORTS),
        "notebook_no_error_outputs": NOTEBOOK.is_file() and not notebook_has_errors(),
        "primary_checksum": sha256(PRIMARY) == EXPECTED_PRIMARY,
        "fold_checksum": sha256(FOLDS) == EXPECTED_FOLDS,
        "upstream_integrity": all(row["match"] for row in upstream),
        "phase10_not_executed": manifest.get("phase10_executed") is False and not phase10_paths,
        "freeze_absent_or_verification_mode": (not FREEZE_FILE.exists()) if require_freeze_absent else True,
    }
    return {
        "status": "PASS" if all(checks.values()) and not missing else "FAIL", "checked_at_utc": now(),
        "checks": checks, "actual": {"authorized_runs": len(records), "completed_runs": len(reusable), "raw_prediction_rows": raw_rows, "canonical_oof_rows": canonical_rows, "missing_modality_rows": missing_rows, "loso_rows": loso_rows},
        "missing_artifacts": sorted(set(missing)), "hash_mismatches": protected_mismatches,
        "duplicate_artifact_paths": sorted(path for path, count in Counter(output_path_rows).items() if count > 1),
        "upstream": upstream, "phase10_paths": phase10_paths,
        "run_status_metadata_repair_needed": sum(row.get("status") != "COMPLETE_AUDITED" for row in records),
    }


def normalize_execution_manifest(freeze_time: str) -> None:
    manifest = read_json(EXECUTION)
    for record in manifest["training_runs"]:
        paths = output_paths(record)
        if not reusable_run(record, paths):
            raise RuntimeError(f"Cannot normalize invalid run: {record['run_identifier']}")
        record.update({
            "status": "COMPLETE_AUDITED",
            "actual_checkpoint_path": paths["checkpoint"].relative_to(ROOT).as_posix(),
            "actual_model_path": paths["model"].relative_to(ROOT).as_posix(),
            "actual_audit_path": paths["audit"].relative_to(ROOT).as_posix(),
            "actual_prediction_path": paths["prediction"].relative_to(ROOT).as_posix(),
            "actual_metrics_path": paths["metrics"].relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256(paths["checkpoint"]), "prediction_sha256": sha256(paths["prediction"]),
        })
    manifest.update({
        "status": "FROZEN", "completed_training_runs": EXPECTED["runs"], "raw_prediction_rows": EXPECTED["raw_rows"],
        "canonical_oof_rows": EXPECTED["canonical"], "phase09_frozen": True, "ready_for_phase09_freeze": False,
        "ready_to_proceed_to_phase10": True, "phase10_executed": False, "freeze_time_utc": freeze_time,
        "model_retraining_during_freeze": False, "predictions_regenerated_during_freeze": False, "last_updated_utc": freeze_time,
    })
    atomic_json(EXECUTION, manifest)


def audit_files_for_manifest() -> list[Path]:
    return [path for path in tree("audits") if path.suffix.lower() == ".json" and path.name not in MANIFEST_DEPENDENT_AUDITS]


def build_manifest(freeze_time: str, pre: dict[str, Any]) -> dict[str, Any]:
    groups = {
        "checkpoints": tree("results/checkpoints"), "predictions": tree("results/predictions"),
        "fold_metrics": tree("results/fold_metrics"), "oof": tree("results/oof"),
        "summaries": tree("results/summaries"), "audits": audit_files_for_manifest(),
        "figures": tree("figures"), "reports": tree("reports"),
    }
    all_relative = [path.relative_to(ROOT).as_posix() for paths in groups.values() for path in paths]
    duplicates = sorted(path for path, count in Counter(all_relative).items() if count > 1)
    allowed = {
        "checkpoints": {".json", ".npz", ".joblib"}, "predictions": {".csv"}, "fold_metrics": {".json"},
        "oof": {".csv", ".json"}, "summaries": {".csv", ".json"}, "audits": {".json"},
        "figures": {".pdf", ".png"}, "reports": {".md"},
    }
    unexpected = sorted(path.relative_to(ROOT).as_posix() for name, paths in groups.items() for path in paths if path.suffix.lower() not in allowed[name])
    upstream = {row["role"]: {"path": row["path"], "bytes": row["bytes"], "sha256": row["actual_sha256"], "integrity": "PASS" if row["match"] else "FAIL"} for row in pre["upstream"]}
    execution = read_json(EXECUTION)
    run_status = Counter(row.get("status") for row in execution["training_runs"])
    return {
        "phase": "Phase 09", "phase_name": "Robustness and Generalization", "phase_path": str(ROOT.resolve()),
        "status": "FROZEN", "freeze_timestamp_utc": freeze_time, "self_hash_included": False,
        "self_referential_audit_hash_exclusions": sorted(MANIFEST_DEPENDENT_AUDITS),
        "datasets": {"primary": {"path": str(PRIMARY.resolve()), "bytes": PRIMARY.stat().st_size, "sha256": sha256(PRIMARY)}, "frozen_folds": {"path": str(FOLDS.resolve()), "bytes": FOLDS.stat().st_size, "sha256": sha256(FOLDS)}},
        "upstream_frozen_interfaces": upstream,
        "contracts": {relative: artifact(ROOT / relative) for relative in CONTRACTS},
        "coverage": {
            "authorized_runs": EXPECTED["runs"], "completed_runs": EXPECTED["runs"], "run_status_counts": dict(run_status),
            "raw_prediction_rows": EXPECTED["raw_rows"], "canonical_oof_rows": EXPECTED["canonical"],
            "missing_modality_canonical_rows": EXPECTED["missing"], "loso_canonical_rows": EXPECTED["loso"],
            "subjects": EXPECTED["subjects"], "missing_modality_conditions": 5, "loso_splits": 35,
        },
        "checkpoint_artifacts": {"count": len(groups["checkpoints"]), "artifacts": inventory(groups["checkpoints"])},
        "prediction_artifacts": {"count": len(groups["predictions"]), "rows": pre["actual"]["raw_prediction_rows"], "artifacts": inventory(groups["predictions"])},
        "fold_metric_artifacts": {"count": len(groups["fold_metrics"]), "artifacts": inventory(groups["fold_metrics"])},
        "oof_artifacts": {"count": len(groups["oof"]), "rows": EXPECTED["canonical"], "artifacts": inventory(groups["oof"])},
        "summary_artifacts": inventory(groups["summaries"]), "audit_artifacts": inventory(groups["audits"]),
        "figure_artifacts": inventory(groups["figures"]), "report_artifacts": inventory(groups["reports"]),
        "notebook": artifact(NOTEBOOK), "missing_artifacts": pre["missing_artifacts"], "duplicate_artifacts": duplicates,
        "hash_mismatches": pre["hash_mismatches"], "unexpected_mutable_artifacts": unexpected,
        "outer_test_used_for_tuning": False, "model_retraining_during_freeze": False,
        "predictions_regenerated_during_freeze": False, "phase10_executed": False,
    }


def freeze_payload(freeze_time: str, manifest_hash: str) -> dict[str, Any]:
    return {
        "phase": "Phase 09", "phase_name": "Robustness and Generalization", "status": "FROZEN",
        "freeze_timestamp_utc": freeze_time, "subjects": 35, "primary_rows": 419, "authorized_runs": 720,
        "completed_runs": 720, "raw_prediction_rows": 30168, "canonical_oof_rows": 10056,
        "missing_modality_canonical_rows": 8380, "loso_canonical_rows": 1676,
        "missing_modality_conditions": 5, "loso_splits": 35, "outer_test_used_for_tuning": False,
        "model_retraining_during_freeze": False, "predictions_regenerated_during_freeze": False, "phase10_executed": False,
        "analyses": {"missing_modality": "COMPLETE", "loso_stability": "COMPLETE", "subject_level_statistics": "COMPLETE"},
        "FULL_PRIMARY_REFERENCE": "REUSED_NOT_RETRAINED", "TEST_TIME_MISSINGNESS": "NOT_FEASIBLE_DUE_TO_CHECKPOINT_INTERFACE",
        "performance_features_included": False,
        "generalization_boundaries": {
            "SUBJECT_GENERALIZATION": "EVALUATED_VIA_35_SUBJECT_LOSO",
            "MISSING_MODALITY_ROBUSTNESS": "EVALUATED_VIA_RETRAIN_WITHOUT_MODALITY",
            "UNSEEN_SESSION": "NOT_FEASIBLE_DUE_TO_METADATA", "UNSEEN_SCENARIO": "NOT_FEASIBLE_DUE_TO_METADATA",
            "TASK_TEMPLATE": "NOT_FEASIBLE_DUE_TO_METADATA", "ROUTE_CONFIGURATION": "NOT_FEASIBLE_DUE_TO_METADATA",
            "FLIGHT_GENERALIZABLE_BEHAVIOR_CLAIM": "INCONCLUSIVE_DUE_TO_METADATA",
        },
        "final_manifest": {"path": "manifests/phase09_final_manifest.json", "sha256": manifest_hash},
        "notebook_persistence": "PASS", "ready_to_proceed_to_phase10": True,
    }


def write_upstream_audit(pre: dict[str, Any]) -> dict[str, Any]:
    checks = {row["role"]: row["match"] for row in pre["upstream"]}
    payload = {"phase": "09", "audit": "upstream_freeze_integrity_final", "status": "PASS" if all(checks.values()) else "FAIL", "audited_at_utc": now(), "checks": checks, "interfaces": pre["upstream"], "upstream_files_modified": sum(not value for value in checks.values()), "phase03_to_08_files_modified": False}
    atomic_json(ROOT / "audits/phase09_upstream_freeze_integrity_final_audit.json", payload)
    return payload


def write_claim_audit() -> dict[str, Any]:
    freeze = read_json(FREEZE_FILE)
    expected = freeze_payload(freeze["freeze_timestamp_utc"], freeze["final_manifest"]["sha256"])["generalization_boundaries"]
    checks = {key: freeze.get("generalization_boundaries", {}).get(key) == value for key, value in expected.items()}
    payload = {"phase": "09", "audit": "generalization_claim", "status": "PASS" if all(checks.values()) else "FAIL", "audited_at_utc": now(), "checks": checks, "boundaries": expected, "causal_claim_added": False, "equivalence_claim_added": False, "unseen_scenario_claim_added": False}
    atomic_json(ROOT / "audits/phase09_generalization_claim_audit.json", payload)
    return payload


def append_notebook_summary() -> dict[str, Any]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    kept = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown" and MARKER in cell.source:
            break
        kept.append(cell)
    before = [hashlib.sha256(json.dumps(cell, sort_keys=True, default=str).encode()).hexdigest() for cell in kept]
    markdown = nbformat.v4.new_markdown_cell(MARKER + "\n\nExecuted final-freeze evidence. All historical cells and outputs above are preserved; no model training, prediction regeneration, or Phase 10 execution occurs here.")
    code = nbformat.v4.new_code_cell(
        "from pathlib import Path\nimport json\nroot=Path.cwd().resolve()\n"
        "freeze=json.loads((root/'configs/phase09_freeze.json').read_text(encoding='utf-8'))\n"
        "manifest=json.loads((root/'manifests/phase09_final_manifest.json').read_text(encoding='utf-8'))\n"
        "{'runs':'720/720','raw_prediction_rows':30168,'canonical_oof_rows':10056,'loso_subjects':'35/35',"
        "'missing_modality_analysis':'COMPLETE','subject_level_statistics':'COMPLETE','final_audits':'PASS',"
        "'final_manifest_saved':True,'phase09_freeze_saved':True,'phase09_status':freeze['status'],"
        "'phase10_executed':freeze['phase10_executed'],'ready_for_phase10':freeze['ready_to_proceed_to_phase10']}"
    )
    temp = nbformat.v4.new_notebook(cells=[code], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    executed = NotebookClient(temp, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    final = nbformat.v4.new_notebook(cells=kept + [markdown] + executed.cells, metadata=notebook.metadata)
    nbformat.write(final, NOTEBOOK)
    reloaded = nbformat.read(NOTEBOOK, as_version=4)
    after = [hashlib.sha256(json.dumps(cell, sort_keys=True, default=str).encode()).hexdigest() for cell in reloaded.cells[:len(kept)]]
    outputs = reloaded.cells[-1].get("outputs", [])
    text = "\n".join(str(output.get("text", output.get("data", {}).get("text/plain", ""))) for output in outputs)
    checks = {
        "marker_once": sum(MARKER in cell.source for cell in reloaded.cells if cell.cell_type == "markdown") == 1,
        "historical_cells_and_outputs_preserved": before == after, "summary_code_executed": reloaded.cells[-1].execution_count is not None,
        "summary_output_persisted": bool(outputs), "notebook_error_outputs_zero": not notebook_has_errors(),
        "required_counts_present": all(value in text for value in ["720/720", "30168", "10056", "35/35"]),
        "frozen_status_present": "FROZEN" in text, "phase10_executed_no": "'phase10_executed': False" in text,
        "ready_for_phase10_yes": "'ready_for_phase10': True" in text,
    }
    payload = {"phase": "09", "audit": "notebook_freeze_persistence", "status": "PASS" if all(checks.values()) else "FAIL", "audited_at_utc": now(), "checks": checks, "preserved_prefix_cells": len(kept), "notebook_path": str(NOTEBOOK.resolve()), "notebook_sha256": sha256(NOTEBOOK), "model_retraining_executed": False, "predictions_regenerated": False, "phase10_executed": False}
    atomic_json(ROOT / "audits/phase09_notebook_freeze_persistence_audit.json", payload)
    return payload


def iter_manifest_artifacts(manifest: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for key in ["checkpoint_artifacts", "prediction_artifacts", "fold_metric_artifacts", "oof_artifacts"]:
        yield from manifest[key]["artifacts"]
    for key in ["summary_artifacts", "audit_artifacts", "figure_artifacts", "report_artifacts"]:
        yield from manifest[key]
    yield manifest["notebook"]
    yield from manifest["contracts"].values()


def write_manifest_and_hash_audits(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    mismatches = [row["path"] for row in iter_manifest_artifacts(manifest) if not (ROOT / row["path"]).is_file() or sha256(ROOT / row["path"]) != row["sha256"]]
    checks = {"manifest_parseable": True, "self_hash_excluded": manifest.get("self_hash_included") is False, "missing_required_artifacts_zero": not manifest["missing_artifacts"], "duplicate_artifact_paths_zero": not manifest["duplicate_artifacts"], "hash_mismatches_zero": not mismatches and not manifest["hash_mismatches"], "unexpected_mutable_artifacts_zero": not manifest["unexpected_mutable_artifacts"]}
    manifest_audit = {"phase": "09", "audit": "final_manifest", "status": "PASS" if all(checks.values()) else "FAIL", "audited_at_utc": now(), "checks": checks, "final_manifest_path": str(FINAL_MANIFEST.resolve()), "final_manifest_sha256": sha256(FINAL_MANIFEST), "hash_mismatches": mismatches, "excluded_self_referential_audits": sorted(MANIFEST_DEPENDENT_AUDITS)}
    atomic_json(ROOT / "audits/phase09_final_manifest_audit.json", manifest_audit)
    freeze = read_json(FREEZE_FILE)
    hash_checks = {"manifest_hash_matches_freeze": freeze["final_manifest"]["sha256"] == sha256(FINAL_MANIFEST), "all_manifest_artifact_hashes_match": not mismatches, "protected_baseline_hashes_match": not verify_hash_map(protected_baseline()), "manifest_self_hash_absent": manifest.get("self_hash_included") is False}
    hash_audit = {"phase": "09", "audit": "final_hash_integrity", "status": "PASS" if all(hash_checks.values()) else "FAIL", "audited_at_utc": now(), "checks": hash_checks, "final_manifest_sha256": sha256(FINAL_MANIFEST), "hash_mismatches": mismatches, "protected_hash_mismatches": verify_hash_map(protected_baseline())}
    atomic_json(ROOT / "audits/phase09_final_hash_integrity_audit.json", hash_audit)
    return manifest_audit, hash_audit


def write_freeze_audit(pre: dict[str, Any], manifest_audit: dict[str, Any], hash_audit: dict[str, Any], notebook_audit: dict[str, Any], upstream_audit: dict[str, Any], claim_audit: dict[str, Any]) -> dict[str, Any]:
    changed = verify_hash_map(protected_baseline())
    freeze = read_json(FREEZE_FILE)
    execution = read_json(EXECUTION)
    categories = {
        "raw_predictions": [path for path in changed if path.startswith("results/predictions/")],
        "checkpoints": [path for path in changed if path.startswith("results/checkpoints/")],
        "canonical_oof": [path for path in changed if path.startswith("results/oof/")],
        "metrics": [path for path in changed if path.startswith("results/summaries/") and "pairwise_statistics" not in path and "bootstrap_confidence" not in path],
        "statistics": [path for path in changed if "pairwise_statistics" in path or "bootstrap_confidence" in path],
        "reports": [path for path in changed if path.startswith("reports/")],
        "figures": [path for path in changed if path.startswith("figures/")],
    }
    checks = {
        "preflight_pass": pre["status"] == "PASS", "missing_required_artifacts_zero": not pre["missing_artifacts"],
        "duplicate_artifact_paths_zero": not pre["duplicate_artifact_paths"], "hash_mismatches_zero": not changed,
        "raw_predictions_changed_zero": not categories["raw_predictions"], "checkpoints_changed_zero": not categories["checkpoints"],
        "canonical_oof_changed_zero": not categories["canonical_oof"], "metrics_changed_zero": not categories["metrics"],
        "statistics_changed_zero": not categories["statistics"], "reports_unexpectedly_changed_zero": not categories["reports"],
        "figures_unexpectedly_changed_zero": not categories["figures"], "upstream_files_modified_zero": upstream_audit.get("upstream_files_modified") == 0,
        "manifest_audit_pass": manifest_audit["status"] == "PASS", "hash_audit_pass": hash_audit["status"] == "PASS",
        "notebook_persistence_pass": notebook_audit["status"] == "PASS", "generalization_claim_audit_pass": claim_audit["status"] == "PASS",
        "execution_manifest_frozen": execution.get("status") == "FROZEN" and all(row.get("status") == "COMPLETE_AUDITED" for row in execution["training_runs"]),
        "freeze_file_frozen": freeze.get("status") == "FROZEN", "model_training_executed_no": freeze.get("model_retraining_during_freeze") is False,
        "predictions_regenerated_no": freeze.get("predictions_regenerated_during_freeze") is False, "phase10_executed_no": freeze.get("phase10_executed") is False,
    }
    payload = {"phase": "09", "audit": "freeze", "status": "PASS" if all(checks.values()) else "FAIL", "audited_at_utc": now(), "checks": checks, "changed_protected_artifacts": changed, "changed_by_category": categories, "model_retraining_executed": False, "predictions_regenerated": False, "upstream_files_modified": upstream_audit.get("upstream_files_modified"), "phase10_executed": False, "ready_to_proceed_to_phase10": all(checks.values())}
    atomic_json(ROOT / "audits/phase09_freeze_audit.json", payload)
    return payload


def write_freeze() -> dict[str, Any]:
    pre = preflight(require_freeze_absent=True)
    if pre["status"] != "PASS":
        raise RuntimeError({"freeze_preflight": pre})
    freeze_time = now()
    original_execution = EXECUTION.read_bytes()
    original_notebook = NOTEBOOK.read_bytes()
    generated = [FINAL_MANIFEST, FREEZE_FILE] + [ROOT / "audits" / name for name in FINAL_AUDITS]
    existed = {path: path.exists() for path in generated}
    original_generated = {path: path.read_bytes() for path in generated if path.exists()}
    try:
        normalize_execution_manifest(freeze_time)
        provisional = build_manifest(freeze_time, pre)
        atomic_json(FINAL_MANIFEST, provisional)
        atomic_json(FREEZE_FILE, freeze_payload(freeze_time, sha256(FINAL_MANIFEST)))
        upstream_audit = write_upstream_audit(pre)
        claim_audit = write_claim_audit()
        notebook_audit = append_notebook_summary()
        if any(audit["status"] != "PASS" for audit in [upstream_audit, claim_audit, notebook_audit]):
            raise RuntimeError("A pre-final freeze audit failed")
        final_manifest = build_manifest(freeze_time, pre)
        atomic_json(FINAL_MANIFEST, final_manifest)
        atomic_json(FREEZE_FILE, freeze_payload(freeze_time, sha256(FINAL_MANIFEST)))
        manifest_audit, hash_audit = write_manifest_and_hash_audits(final_manifest)
        freeze_audit = write_freeze_audit(pre, manifest_audit, hash_audit, notebook_audit, upstream_audit, claim_audit)
        if any(audit["status"] != "PASS" for audit in [upstream_audit, claim_audit, notebook_audit, manifest_audit, hash_audit, freeze_audit]):
            raise RuntimeError("One or more final freeze audits failed")
        return {"status": "FROZEN", "preflight": pre["status"], "final_manifest_sha256": sha256(FINAL_MANIFEST), "freeze_audit": freeze_audit["status"], "notebook_audit": notebook_audit["status"], "phase10_executed": False}
    except Exception:
        EXECUTION.write_bytes(original_execution)
        NOTEBOOK.write_bytes(original_notebook)
        for path in generated:
            if existed[path]:
                path.write_bytes(original_generated[path])
            elif path.exists():
                path.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Create the final manifest, freeze, notebook summary, and audits")
    args = parser.parse_args()
    result = write_freeze() if args.write else preflight(require_freeze_absent=True)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=json_default))
    return 0 if result["status"] in {"PASS", "FROZEN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
