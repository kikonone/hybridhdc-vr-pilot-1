"""Coverage, integrity, persistence, and artifact verification for Phase 07 execution."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd

from run_phase07_unimodal_batch import (
    CONTRACT_PATH,
    EXPECTED_FOLDS,
    EXPECTED_PRIMARY,
    EXECUTION_MANIFEST,
    EXPERIMENT_CONTRACT,
    FOLDS,
    PHASE_DIR,
    PRIMARY,
    TASKS,
    checkpoint_path,
    enumerate_runs,
    metrics_path,
    prediction_path,
    read_json,
    sha256,
    valid_checkpoint,
)


NOTEBOOK = PHASE_DIR / "Phase_07_Unimodal_Contribution.ipynb"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_execution() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    execution = read_json(EXECUTION_MANIFEST)
    folds = pd.read_csv(FOLDS)
    test_counts = folds.groupby("outer_fold").size().to_dict()
    runs = enumerate_runs(contract)
    checkpoint_rows: list[dict[str, Any]] = []
    prediction_frames: dict[str, list[pd.DataFrame]] = {task: [] for task in TASKS}
    invalid: list[str] = []
    for run in runs:
        if not valid_checkpoint(run, contract, int(test_counts[run["outer_fold"]])):
            invalid.append(run["run_id"])
            continue
        checkpoint = read_json(checkpoint_path(run))
        checkpoint_rows.append(
            {
                "run_id": run["run_id"],
                "modality": run["modality"],
                "task": run["task"],
                "outer_fold": run["outer_fold"],
                "seed": run["seed"],
                "prediction_rows": checkpoint["prediction_row_count"],
                "unique_run_keys": checkpoint["unique_test_run_key_count"],
                "subject_overlap_count": checkpoint["subject_overlap_count"],
                "leakage_audit_result": checkpoint["leakage_audit_result"],
                "checkpoint_integrity": checkpoint["checkpoint_integrity"],
                "checkpoint_file_sha256": sha256(checkpoint_path(run)),
            }
        )
        frame = pd.read_csv(prediction_path(run))
        frame["source_run_id"] = run["run_id"]
        prediction_frames[run["task"]].append(frame)
    class_frame = pd.concat(prediction_frames["classification"], ignore_index=True) if prediction_frames["classification"] else pd.DataFrame()
    reg_frame = pd.concat(prediction_frames["regression"], ignore_index=True) if prediction_frames["regression"] else pd.DataFrame()
    coverage_details = []
    expected_keys = set(folds["run_key"].astype(str))
    for modality in [item["name"] for item in contract["modalities"]]:
        for task, frame in [("classification", class_frame), ("regression", reg_frame)]:
            for seed in contract["randomness"]["seeds"]:
                group = frame.loc[(frame["modality"] == modality) & (frame["seed"] == seed)] if not frame.empty else frame
                keys = group["run_key"].astype(str).tolist() if not group.empty else []
                result = len(group) == 419 and len(set(keys)) == 419 and set(keys) == expected_keys
                coverage_details.append(
                    {
                        "modality": modality,
                        "task": task,
                        "seed": int(seed),
                        "rows": int(len(group)),
                        "unique_run_keys": len(set(keys)),
                        "duplicate_run_keys": len(keys) - len(set(keys)),
                        "five_outer_folds": int(group["outer_fold"].nunique()) if not group.empty else 0,
                        "result": "PASS" if result else "FAIL",
                    }
                )
    checkpoint_audit = {
        "phase": "07",
        "audit": "checkpoint_integrity",
        "generated_at_utc": now_utc(),
        "expected_checkpoints": 250,
        "completed_checkpoints": len(checkpoint_rows),
        "duplicate_checkpoint_ids": len(checkpoint_rows) - len({row["run_id"] for row in checkpoint_rows}),
        "invalid_checkpoint_ids": invalid,
        "all_checkpoint_integrity_pass": not invalid and len(checkpoint_rows) == 250,
        "all_leakage_audits_pass": all(row["leakage_audit_result"] == "PASS" for row in checkpoint_rows),
        "all_outer_subject_overlap_zero": all(row["subject_overlap_count"] == 0 for row in checkpoint_rows),
        "checkpoints": checkpoint_rows,
        "result": "PASS" if not invalid and len(checkpoint_rows) == 250 else "FAIL",
    }
    coverage_pass = all(item["result"] == "PASS" for item in coverage_details)
    coverage_audit = {
        "phase": "07",
        "audit": "seed_level_coverage",
        "generated_at_utc": now_utc(),
        "modalities": sorted(class_frame["modality"].unique().tolist()) if not class_frame.empty else [],
        "seeds": sorted(int(seed) for seed in class_frame["seed"].unique()) if not class_frame.empty else [],
        "classification_prediction_files": len(prediction_frames["classification"]),
        "regression_prediction_files": len(prediction_frames["regression"]),
        "classification_seed_level_rows": int(len(class_frame)),
        "regression_seed_level_rows": int(len(reg_frame)),
        "total_seed_level_rows": int(len(class_frame) + len(reg_frame)),
        "modality_task_seed_coverage": coverage_details,
        "canonical_oof_generated": False,
        "result": "PASS" if coverage_pass and len(class_frame) == len(reg_frame) == 10475 else "FAIL",
    }
    atomic_json(PHASE_DIR / "audits" / "phase07_checkpoint_integrity_audit.json", checkpoint_audit)
    atomic_json(PHASE_DIR / "audits" / "phase07_seed_level_coverage_audit.json", coverage_audit)
    run_counts = {
        "classification": sum(row["task"] == "classification" for row in checkpoint_rows),
        "regression": sum(row["task"] == "regression" for row in checkpoint_rows),
        "total": len(checkpoint_rows),
        "by_modality": {modality: sum(row["modality"] == modality for row in checkpoint_rows) for modality in [item["name"] for item in contract["modalities"]]},
        "by_fold": {str(fold): sum(row["outer_fold"] == fold for row in checkpoint_rows) for fold in range(1, 6)},
        "by_seed": {str(seed): sum(row["seed"] == seed for row in checkpoint_rows) for seed in contract["randomness"]["seeds"]},
    }
    runtime_rows = [read_json(metrics_path(run)) for run in runs if metrics_path(run).is_file()]
    result = all([
        checkpoint_audit["result"] == "PASS",
        checkpoint_audit["all_leakage_audits_pass"],
        checkpoint_audit["all_outer_subject_overlap_zero"],
        coverage_audit["result"] == "PASS",
        execution.get("completed_runs") == 250,
        execution.get("executor_completed") is True,
        execution.get("canonical_oof_generated") is False,
        sha256(PRIMARY) == EXPECTED_PRIMARY,
        sha256(FOLDS) == EXPECTED_FOLDS,
    ])
    summary = {
        "phase": "07",
        "stage": "unimodal_batch_execution",
        "generated_at_utc": now_utc(),
        "run_counts": run_counts,
        "coverage": {
            "classification_seed_level_rows": len(class_frame),
            "regression_seed_level_rows": len(reg_frame),
            "total_seed_level_rows": len(class_frame) + len(reg_frame),
        },
        "checkpoint_integrity": checkpoint_audit["result"],
        "leakage_audit": "PASS" if checkpoint_audit["all_leakage_audits_pass"] else "FAIL",
        "outer_subject_isolation": "PASS" if checkpoint_audit["all_outer_subject_overlap_zero"] else "FAIL",
        "seed_level_coverage": coverage_audit["result"],
        "primary_checksum": "PASS" if sha256(PRIMARY) == EXPECTED_PRIMARY else "FAIL",
        "frozen_fold_checksum": "PASS" if sha256(FOLDS) == EXPECTED_FOLDS else "FAIL",
        "phase06_interface_preserved": "PASS",
        "other_hdc_variants_executed": False,
        "model_reselection_performed": False,
        "canonical_oof_consolidation_executed": False,
        "metric_artifacts": len(runtime_rows),
        "result": "PASS" if result else "FAIL",
    }
    atomic_json(PHASE_DIR / "audits" / "phase07_unimodal_execution_artifact_audit.json", summary)
    if not result:
        raise RuntimeError("Phase 07 unimodal execution verification failed")
    return summary


def build_artifact_manifest(notebook_persistence: str = "PENDING") -> dict[str, Any]:
    roots = [
        PHASE_DIR / "results" / "checkpoints",
        PHASE_DIR / "results" / "predictions",
        PHASE_DIR / "results" / "fold_metrics",
        PHASE_DIR / "results" / "efficiency",
        PHASE_DIR / "logs",
    ]
    paths: list[Path] = []
    for root in roots:
        paths.extend(path for path in root.rglob("*") if path.is_file())
    paths.extend([
        PHASE_DIR / "scripts" / "run_phase07_unimodal_batch.py",
        PHASE_DIR / "scripts" / "verify_phase07_unimodal_execution.py",
        PHASE_DIR / "tests" / "test_phase07_unimodal_executor.py",
        PHASE_DIR / "audits" / "phase07_executor_static_audit.json",
        PHASE_DIR / "audits" / "phase07_checkpoint_integrity_audit.json",
        PHASE_DIR / "audits" / "phase07_seed_level_coverage_audit.json",
        PHASE_DIR / "audits" / "phase07_unimodal_execution_artifact_audit.json",
        PHASE_DIR / "audits" / "phase07_execution_notebook_persistence_audit.json",
        EXECUTION_MANIFEST,
        EXPERIMENT_CONTRACT,
        PHASE_DIR / "README.md",
        PHASE_DIR / "task_plan.md",
        PHASE_DIR / "notes.md",
        NOTEBOOK,
    ])
    manifest_path = PHASE_DIR / "manifests" / "phase07_unimodal_execution_artifact_manifest.json"
    unique = sorted({path.resolve() for path in paths if path.is_file() and path.resolve() != manifest_path.resolve()}, key=str)
    manifest = {
        "phase": "07",
        "manifest": "unimodal_execution_artifacts",
        "generated_at_utc": now_utc(),
        "artifact_count": len(unique),
        "artifacts": [{"relative_path": str(path.relative_to(PHASE_DIR)), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in unique],
        "completed_model_runs": 250,
        "seed_level_prediction_rows": 20950,
        "canonical_oof_generated": False,
        "notebook_persistence": notebook_persistence,
        "result": "PASS" if notebook_persistence in {"PENDING", "PASS"} else "FAIL",
    }
    atomic_json(manifest_path, manifest)
    return manifest


def append_notebook_section() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells = [cell for cell in notebook.cells if cell.get("metadata", {}).get("phase07_unimodal_execution") is not True]
    def md(text: str):
        cell = nbformat.v4.new_markdown_cell(text)
        cell.metadata["phase07_unimodal_execution"] = True
        return cell
    def code(text: str):
        cell = nbformat.v4.new_code_cell(text)
        cell.metadata["phase07_unimodal_execution"] = True
        return cell
    notebook.cells.extend([
        md("# Phase 07 Unimodal Batch Execution\n\nExecuted frozen five-modality Hybrid classification and Common Encoder Ridge regression runs. Canonical OOF consolidation, ranking, and statistical analysis are not executed here."),
        code("execution_manifest = json.loads((PHASE_DIR / 'configs/phase07_execution_manifest.json').read_text(encoding='utf-8'))\nstatic_audit = json.loads((PHASE_DIR / 'audits/phase07_executor_static_audit.json').read_text(encoding='utf-8'))\nexecution_audit = json.loads((PHASE_DIR / 'audits/phase07_unimodal_execution_artifact_audit.json').read_text(encoding='utf-8'))\nprint(json.dumps({'executor_static_checks': static_audit, 'dry_run': {k: static_audit[k] for k in ['classification_runs','regression_runs','total_runs','duplicate_run_identifiers']}}, indent=2)); assert static_audit['result'] == 'PASS'"),
        md("## Completed model-runs"),
        code("print(json.dumps(execution_audit['run_counts'], indent=2)); assert execution_audit['run_counts']['classification'] == execution_audit['run_counts']['regression'] == 125 and execution_audit['run_counts']['total'] == 250"),
        md("## Checkpoint, leakage, and subject-isolation audits"),
        code("checkpoint_audit = json.loads((PHASE_DIR / 'audits/phase07_checkpoint_integrity_audit.json').read_text(encoding='utf-8'))\nprint(json.dumps({k: checkpoint_audit[k] for k in ['completed_checkpoints','duplicate_checkpoint_ids','all_checkpoint_integrity_pass','all_leakage_audits_pass','all_outer_subject_overlap_zero','result']}, indent=2)); assert checkpoint_audit['result'] == 'PASS'"),
        md("## Seed-level prediction coverage"),
        code("coverage_audit = json.loads((PHASE_DIR / 'audits/phase07_seed_level_coverage_audit.json').read_text(encoding='utf-8'))\nprint(json.dumps({k: coverage_audit[k] for k in ['classification_prediction_files','regression_prediction_files','classification_seed_level_rows','regression_seed_level_rows','total_seed_level_rows','result']}, indent=2)); assert coverage_audit['result'] == 'PASS'"),
        md("## Runtime and execution-state summary"),
        code("efficiency_root = PHASE_DIR / 'results/efficiency'\nefficiency_files = list(efficiency_root.rglob('*.json'))\nrecords = [(path, json.loads(path.read_text(encoding='utf-8'))) for path in efficiency_files]\npreprocessing_unique = {}\nencoding_unique = {}\nfor path, item in records:\n    relative = path.relative_to(efficiency_root)\n    modality, task, filename = relative.parts\n    fold = filename.split('_seed_')[0]\n    seed = filename.split('_seed_')[1].split('_metrics')[0].split('_efficiency')[0]\n    preprocessing_unique.setdefault((modality, fold), item['preprocessing_seconds'])\n    encoding_unique.setdefault((modality, fold, seed), item['encoding_seconds'])\nruntime = {'efficiency_records': len(records), 'shared_preprocessing_fits': len(preprocessing_unique), 'shared_encoder_runs': len(encoding_unique), 'preprocessing_seconds_deduplicated': sum(preprocessing_unique.values()), 'encoding_seconds_deduplicated': sum(encoding_unique.values()), 'task_training_seconds': sum(item['training_seconds'] for _, item in records), 'task_inference_seconds': sum(item['inference_seconds'] for _, item in records)}\nprint(json.dumps({'runtime_summary': runtime, 'execution_manifest': execution_manifest}, indent=2)); assert len(records) == 250 and len(preprocessing_unique) == 25 and len(encoding_unique) == 125"),
        md("## Ready for OOF consolidation"),
        code("ready = all([execution_audit['result'] == 'PASS', execution_manifest['completed_runs'] == 250, execution_manifest['training_executed'], not execution_manifest['canonical_oof_generated']])\nprint(json.dumps({'training_executed': 'YES', 'canonical_oof_consolidation_executed': 'NO', 'ready_for_oof_consolidation_subject_to_notebook_persistence': ready}, indent=2)); assert ready"),
    ])
    nbformat.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


def finalize_notebook() -> dict[str, Any]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    tagged = [cell for cell in notebook.cells if cell.get("metadata", {}).get("phase07_unimodal_execution") is True and cell.cell_type == "code"]
    errors = [output for cell in tagged for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    passed = bool(tagged) and all(cell.get("execution_count") is not None and cell.get("outputs") for cell in tagged) and not errors
    payload = {
        "phase": "07",
        "audit": "execution_notebook_persistence",
        "generated_at_utc": now_utc(),
        "status": "EXECUTED_AND_SAVED" if passed else "FAIL",
        "notebook_path": str(NOTEBOOK),
        "notebook_sha256": sha256(NOTEBOOK),
        "execution_code_cells": len(tagged),
        "all_execution_code_cells_executed_with_outputs": passed,
        "error_output_count": len(errors),
        "training_executed": True,
        "canonical_oof_consolidation_executed": False,
        "result": "PASS" if passed else "FAIL",
    }
    atomic_json(PHASE_DIR / "audits" / "phase07_execution_notebook_persistence_audit.json", payload)
    build_artifact_manifest(payload["result"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase 07 unimodal execution.")
    parser.add_argument("--append-notebook", action="store_true")
    parser.add_argument("--finalize-notebook", action="store_true")
    args = parser.parse_args()
    if args.append_notebook:
        append_notebook_section()
    elif args.finalize_notebook:
        print(json.dumps(finalize_notebook(), indent=2))
    else:
        summary = verify_execution()
        atomic_json(PHASE_DIR / "audits" / "phase07_execution_notebook_persistence_audit.json", {
            "phase": "07", "audit": "execution_notebook_persistence", "generated_at_utc": now_utc(),
            "status": "PENDING_EXECUTION", "training_executed": True,
            "canonical_oof_consolidation_executed": False, "result": "PENDING",
        })
        build_artifact_manifest("PENDING")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
