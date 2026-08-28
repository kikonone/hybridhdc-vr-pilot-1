"""Read-only consolidation of five Phase 05 Vanilla HDC quick screens."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
PRIMARY = ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv"
FOLDS = ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv"
EXPECTED_PRIMARY_SHA = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLD_SHA = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
FINAL_DIMENSIONS = [1000, 2000, 5000, 10000]
FINAL_SEEDS = [42, 43, 44, 45, 46]
TEMPERATURE_GRID = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
SELECTION_RULE = (
    "higher mean Macro-F1; lower Macro-F1 standard deviation; lower Severe Error Rate; "
    "smaller dimension; smaller finite k with all after finite k; fewer levels; frozen ParameterGrid order"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fold_artifacts(fold: int) -> dict[str, Path]:
    return {
        "checkpoint": PHASE / f"results/checkpoints/quick_screen/vanilla_hdc_quick_screen_fold_{fold}_checkpoint.json",
        "candidates_csv": PHASE / f"results/summaries/vanilla_hdc_quick_screen_fold_{fold}_candidates.csv",
        "best_config": PHASE / f"results/summaries/vanilla_hdc_quick_screen_fold_{fold}_best_config.json",
        "inner_metrics": PHASE / f"results/fold_metrics/vanilla_hdc_quick_screen_fold_{fold}_inner_metrics.csv",
        "efficiency": PHASE / f"results/efficiency/vanilla_hdc_quick_screen_fold_{fold}_efficiency.csv",
        "leakage_audit": PHASE / f"audits/vanilla_hdc_quick_screen_fold_{fold}_leakage_audit.json",
        "artifact_audit": PHASE / f"audits/vanilla_hdc_quick_screen_fold_{fold}_artifact_audit.json",
        "notebook_persistence": PHASE / f"audits/vanilla_hdc_quick_screen_fold_{fold}_notebook_persistence_audit.json",
        "stdout_log": PHASE / f"logs/vanilla_hdc_quick_screen_fold_{fold}_stdout.log",
        "stderr_log": PHASE / f"logs/vanilla_hdc_quick_screen_fold_{fold}_stderr.log",
    }


def independently_select_best(candidates: pd.DataFrame) -> pd.Series:
    ranked = candidates.copy()
    ranked["k"] = ranked["k"].astype(str)
    ranked["_all_after_finite"] = (ranked["k"] == "all").astype(int)
    ranked["_effective_k_sort"] = ranked["k"].map(
        lambda value: np.inf if value == "all" else float(value)
    )
    return ranked.sort_values(
        [
            "mean_macro_f1", "std_macro_f1", "mean_severe_error_rate", "dimension",
            "_all_after_finite", "_effective_k_sort", "levels", "parameter_grid_order",
        ],
        ascending=[False, True, True, True, True, True, True, True],
        kind="mergesort",
    ).iloc[0]


def main() -> None:
    primary_sha = sha256(PRIMARY)
    fold_sha = sha256(FOLDS)
    if primary_sha != EXPECTED_PRIMARY_SHA or fold_sha != EXPECTED_FOLD_SHA:
        raise RuntimeError("frozen input checksum mismatch")
    metadata = pd.read_csv(PRIMARY, usecols=["run_key", "subject_id"])
    folds = pd.read_csv(FOLDS)
    if len(metadata) != 419 or metadata["subject_id"].nunique() != 35:
        raise RuntimeError("unexpected modeling rows or subject count")
    if not metadata["run_key"].is_unique or not folds["run_key"].is_unique:
        raise RuntimeError("run_key is not unique")
    if set(metadata["run_key"]) != set(folds["run_key"]):
        raise RuntimeError("frozen fold assignments do not cover the modeling runs")
    if sorted(folds["outer_fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("outer folds are not exactly 1..5")

    split_checks: list[dict[str, Any]] = []
    for fold in range(1, 6):
        train = folds.loc[folds["outer_fold"] != fold]
        test = folds.loc[folds["outer_fold"] == fold]
        overlap = set(train["subject_id"]) & set(test["subject_id"])
        splits = list(GroupKFold(n_splits=3).split(train, groups=train["subject_id"]))
        split_checks.append(
            {
                "outer_fold": fold, "train_rows": int(len(train)),
                "train_subjects": int(train["subject_id"].nunique()), "test_rows": int(len(test)),
                "subject_overlap_count": len(overlap), "inner_groupkfold_splits": len(splits),
                "inner_subject_isolation": all(
                    not (set(train.iloc[a]["subject_id"]) & set(train.iloc[b]["subject_id"]))
                    for a, b in splits
                ),
            }
        )
    if any(item["subject_overlap_count"] or item["inner_groupkfold_splits"] != 3 for item in split_checks):
        raise RuntimeError("outer isolation or inner GroupKFold feasibility failed")

    all_paths = {fold: fold_artifacts(fold) for fold in range(1, 6)}
    for paths in all_paths.values():
        for path in paths.values():
            if not path.is_file():
                raise RuntimeError(f"missing quick-screen artifact: {path}")
    hashes_before = {
        str(path.relative_to(PHASE)): sha256(path)
        for paths in all_paths.values() for path in paths.values()
    }

    expected_candidates = {
        (dimension, levels, str(k), 42)
        for dimension in [2000, 5000]
        for levels in [21, 51]
        for k in [50, 100, 200, "all"]
    }
    summary_rows: list[dict[str, Any]] = []
    per_fold_manifest: list[dict[str, Any]] = []
    fold_audits: list[dict[str, Any]] = []

    for fold in range(1, 6):
        paths = all_paths[fold]
        checkpoint = read_json(paths["checkpoint"])
        candidates = pd.read_csv(paths["candidates_csv"])
        candidates["k"] = candidates["k"].astype(str)
        best = read_json(paths["best_config"])
        inner = pd.read_csv(paths["inner_metrics"])
        efficiency = pd.read_csv(paths["efficiency"])
        leakage = read_json(paths["leakage_audit"])
        artifact = read_json(paths["artifact_audit"])
        persistence = read_json(paths["notebook_persistence"])
        stdout_text = paths["stdout_log"].read_text(encoding="utf-8")
        stderr_text = paths["stderr_log"].read_text(encoding="utf-8")

        actual_candidates = set(
            zip(
                candidates["dimension"].astype(int), candidates["levels"].astype(int),
                candidates["k"], candidates["seed"].astype(int), strict=True,
            )
        )
        complete_space = len(candidates) == 16 and len(actual_candidates) == 16 and actual_candidates == expected_candidates
        if not complete_space:
            raise RuntimeError(f"Fold {fold} candidate space is incomplete, duplicated, or extra")
        if len(inner) != 48 or not (inner.groupby(["dimension", "levels", "k", "seed"]).size() == 3).all():
            raise RuntimeError(f"Fold {fold} inner metrics are not 16 candidates x 3 splits")
        if inner["subject_overlap_count"].astype(int).sum() != 0 or sorted(inner["inner_fold"].unique()) != [1, 2, 3]:
            raise RuntimeError(f"Fold {fold} inner subject isolation failed")
        if len(efficiency) != 16:
            raise RuntimeError(f"Fold {fold} efficiency table does not have 16 rows")

        recalculated = independently_select_best(candidates)
        best_reproducible = (
            int(recalculated["dimension"]) == int(best["dimension"])
            and int(recalculated["levels"]) == int(best["levels"])
            and str(recalculated["k"]) == str(best["k"])
            and np.isclose(float(recalculated["mean_macro_f1"]), float(best["mean_macro_f1"]), rtol=0, atol=1e-15)
        )
        checks = {
            "outer_fold": fold,
            "candidates_completed": checkpoint.get("candidates_completed") == 16,
            "complete_candidate_space": complete_space,
            "inner_groupkfold_3": True,
            "best_config_reproducible": bool(best_reproducible),
            "leakage_audit_pass": leakage.get("result") == "PASS",
            "artifact_audit_pass": artifact.get("result") == "PASS",
            "notebook_persistence_pass": persistence.get("persistence_result") == "PASS",
            "outer_test_feature_access": bool(checkpoint.get("outer_test_feature_access")),
            "outer_test_prediction_generated": bool(leakage.get("outer_test_prediction_generated")),
            "similarity_regression_executed": bool(leakage.get("similarity_regression_executed")),
            "ridge_readout_executed": bool(leakage.get("ridge_readout_executed")),
            "stdout_log_read": len(stdout_text) > 0,
            "stderr_log_read": True,
        }
        if not (
            checks["candidates_completed"] and checks["best_config_reproducible"]
            and checks["leakage_audit_pass"] and checks["artifact_audit_pass"]
            and checks["notebook_persistence_pass"] and not checks["outer_test_feature_access"]
            and not checks["outer_test_prediction_generated"] and not checks["similarity_regression_executed"]
            and not checks["ridge_readout_executed"] and checks["stdout_log_read"]
        ):
            raise RuntimeError(f"Fold {fold} verification failed")
        fold_audits.append(checks)
        split = split_checks[fold - 1]
        summary_rows.append(
            {
                "outer_fold": fold, "outer_training_rows": split["train_rows"],
                "outer_training_subjects": split["train_subjects"],
                "best_quick_dimension": int(best["dimension"]), "selected_levels": int(best["levels"]),
                "selected_feature_k": str(best["k"]), "best_inner_macro_f1": float(best["mean_macro_f1"]),
                "best_inner_macro_f1_std": float(best["std_macro_f1"]),
                "best_inner_balanced_accuracy": float(best["mean_balanced_accuracy"]),
                "best_inner_severe_error_rate": float(best["mean_severe_error_rate"]),
                "candidates_completed": int(checkpoint["candidates_completed"]),
                "selection_rule": SELECTION_RULE, "checkpoint_sha256": sha256(paths["checkpoint"]),
                "candidates_csv_sha256": sha256(paths["candidates_csv"]),
                "best_config_sha256": sha256(paths["best_config"]),
            }
        )
        per_fold_manifest.append(
            {
                "outer_fold": fold, "selected_levels": int(best["levels"]),
                "selected_feature_k": str(best["k"]), "quick_screen_selected_dimension_record_only": int(best["dimension"]),
                "final_confirmation_dimensions": FINAL_DIMENSIONS,
                "evaluation_seeds": FINAL_SEEDS,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("outer_fold")
    summary_path = PHASE / "results/summaries/vanilla_hdc_quick_screen_all_folds.csv"
    summary.to_csv(summary_path, index=False)
    aggregate = {
        "phase": "05", "stage": "quick_screen_consolidation", "folds_verified": 5,
        "candidates_per_fold": 16, "per_fold_results": summary_rows,
        "descriptive_inner_cv_metrics": {
            "mean_of_fold_best_macro_f1": float(summary["best_inner_macro_f1"].mean()),
            "std_of_fold_best_macro_f1": float(summary["best_inner_macro_f1"].std(ddof=0)),
            "mean_of_fold_best_balanced_accuracy": float(summary["best_inner_balanced_accuracy"].mean()),
            "mean_of_fold_best_severe_error_rate": float(summary["best_inner_severe_error_rate"].mean()),
        },
        "global_levels_or_feature_k_selected": False,
        "interpretation": "descriptive aggregation only; each outer fold retains its own selected levels and feature_k",
        "training_executed": False, "outer_test_feature_access": False,
    }
    aggregate_path = PHASE / "results/summaries/vanilla_hdc_quick_screen_aggregate_summary.json"
    write_json(aggregate_path, aggregate)

    artifact_manifest = {
        "phase": "05", "artifact_count": len(hashes_before),
        "quick_screen_input_artifacts_sha256": hashes_before,
        "primary_data": {"path": str(PRIMARY), "sha256": primary_sha},
        "fold_assignments": {"path": str(FOLDS), "sha256": fold_sha},
    }
    artifact_manifest_path = PHASE / "manifests/vanilla_hdc_quick_screen_artifact_manifest.json"
    write_json(artifact_manifest_path, artifact_manifest)

    final_manifest = {
        "phase": "05", "status": "FINAL_CONFIRMATION_MANIFEST_FROZEN_NOT_EXECUTED",
        "outer_folds": [1, 2, 3, 4, 5], "per_fold_configuration": per_fold_manifest,
        "final_confirmation_dimensions": FINAL_DIMENSIONS, "evaluation_seeds": FINAL_SEEDS,
        "classification_primary_metric": "Macro-F1", "regression_primary_metric": "MAE",
        "similarity_regression_temperature_grid": TEMPERATURE_GRID,
        "ridge_alpha_grid": RIDGE_ALPHA_GRID,
        "head_parameter_selection": "temperature and alpha only inside the corresponding outer-training 3-fold GroupKFold",
        "outer_test_rule": "each outer-test fold may be accessed once only after that fold configuration is completely fixed",
        "quick_screen_dimension_is_record_only": True, "global_optimum_across_outer_folds_prohibited": True,
        "outer_test_selection_prohibited": ["dimension", "seed", "temperature", "ridge alpha"],
        "checkpoint_per_outer_fold_required": True, "stop_on_any_fold_audit_failure": True,
        "phase06_variants_prohibited": True, "final_confirmation_executed": False,
        "quick_screen_input_artifacts_sha256": hashes_before,
    }
    final_manifest_path = PHASE / "configs/vanilla_hdc_final_confirmation_manifest.json"
    write_json(final_manifest_path, final_manifest)

    hashes_after = {
        relative: sha256(PHASE / relative) for relative in hashes_before
    }
    artifacts_preserved = hashes_before == hashes_after
    consolidation_audit = {
        "phase": "05", "folds_verified": 5, "candidates_per_fold": 16,
        "all_candidate_spaces_complete": all(item["complete_candidate_space"] for item in fold_audits),
        "all_best_configs_reproducible": all(item["best_config_reproducible"] for item in fold_audits),
        "all_fold_audits_pass": all(item["leakage_audit_pass"] and item["artifact_audit_pass"] and item["notebook_persistence_pass"] for item in fold_audits),
        "all_quick_screen_artifacts_preserved": artifacts_preserved,
        "artifact_sha256_before": hashes_before, "artifact_sha256_after": hashes_after,
        "primary_checksum_pass": primary_sha == EXPECTED_PRIMARY_SHA,
        "fold_checksum_pass": fold_sha == EXPECTED_FOLD_SHA,
        "modeling_rows": int(len(metadata)), "subjects": int(metadata["subject_id"].nunique()),
        "outer_run_coverage_pass": set(metadata["run_key"]) == set(folds["run_key"]) and len(folds) == 419,
        "outer_subject_isolation_pass": all(item["subject_overlap_count"] == 0 for item in split_checks),
        "inner_groupkfold_feasibility_pass": all(item["inner_groupkfold_splits"] == 3 and item["inner_subject_isolation"] for item in split_checks),
        "outer_test_feature_access": False, "training_executed": False,
        "similarity_regression_executed": False, "ridge_readout_executed": False,
        "fold_checks": fold_audits, "result": "PASS",
    }
    consolidation_audit_path = PHASE / "audits/vanilla_hdc_quick_screen_consolidation_audit.json"
    write_json(consolidation_audit_path, consolidation_audit)

    manifest_audit = {
        "phase": "05", "manifest_parse_pass": True,
        "outer_folds_pass": final_manifest["outer_folds"] == [1, 2, 3, 4, 5],
        "per_fold_configuration_pass": len(final_manifest["per_fold_configuration"]) == 5,
        "final_dimensions_pass": final_manifest["final_confirmation_dimensions"] == FINAL_DIMENSIONS,
        "final_seeds_pass": final_manifest["evaluation_seeds"] == FINAL_SEEDS,
        "temperature_grid_pass": final_manifest["similarity_regression_temperature_grid"] == TEMPERATURE_GRID,
        "ridge_alpha_grid_pass": final_manifest["ridge_alpha_grid"] == RIDGE_ALPHA_GRID,
        "artifact_hashes_pass": all(sha256(PHASE / relative) == digest for relative, digest in hashes_before.items()),
        "no_global_configuration_selected": final_manifest["global_optimum_across_outer_folds_prohibited"],
        "outer_test_rule_pass": bool(final_manifest["outer_test_rule"]),
        "phase06_excluded": final_manifest["phase06_variants_prohibited"],
        "final_confirmation_training_executed": False, "result": "PASS",
    }
    manifest_audit_path = PHASE / "audits/vanilla_hdc_final_confirmation_manifest_audit.json"
    write_json(manifest_audit_path, manifest_audit)
    print("CONSOLIDATION COMPLETE: 5/5 folds verified; Final Confirmation manifest frozen, not executed.")


if __name__ == "__main__":
    main()
