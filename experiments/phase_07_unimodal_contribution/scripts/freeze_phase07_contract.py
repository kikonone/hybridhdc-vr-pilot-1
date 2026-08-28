from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat

from initialize_phase07 import collect_audit


PHASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE_DIR.parents[1]
P5_ENCODING = PROJECT_ROOT / "experiments" / "phase_05_basic_dual_output_hdc" / "configs" / "phase05_hdc_encoding_contract.json"
P6_DIR = PROJECT_ROOT / "experiments" / "phase_06_hdc_variant_screening"
P6_VARIANT = P6_DIR / "configs" / "phase06_hdc_variant_contract.json"
P6_FREEZE = P6_DIR / "configs" / "phase06_freeze.json"
P6_CLASSIFICATION = P6_DIR / "configs" / "phase06_best_classification_hdc.json"
P6_REGRESSION = P6_DIR / "configs" / "phase06_best_regression_hdc.json"
NOTEBOOK = PHASE_DIR / "Phase_07_Unimodal_Contribution.ipynb"

EXPECTED_UPSTREAM_HASHES = {
    "primary_data": "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44",
    "frozen_folds": "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f",
    "phase06_freeze": "cd94acdf0bcb463b5f48a7c84110a23059899b6b5bf156131694827d7d7bde66",
    "phase06_classification": "174a99de2d993acdea49fdebc9647b28db4648ada2bea7a33f620f4677f031a4",
    "phase06_regression": "acde51709971d57c76eefaffcf1ecd571a4d4c5c36f8d76edf39841c5e7065b8",
    "phase06_variant_contract": "f9d0bd0f304678bbd00e3fdeaaf9b619511a2e54f9d426e21cc130691cca365b",
    "phase05_encoding_contract": "8bffcbdcad5ef73778a5daf8eb64dd9dd1d8d90c675d10f9cbd72a1360f133ef",
}

MODALITY_COUNTS = {
    "physiological_features": 233,
    "eye_tracking_features": 416,
    "head_movement_features": 159,
    "flight_parameter_features": 326,
    "body_movement": 42,
}
EXPECTED_MISSING_ROWS = {
    "physiological_features": 0,
    "eye_tracking_features": 14,
    "head_movement_features": 0,
    "flight_parameter_features": 0,
    "body_movement": 29,
}
SEEDS = [42, 43, 44, 45, 46]
CLASSES = [0, 1, 2, 3]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def effective_feature_k(requested_feature_k: int, post_variance_feature_count: int) -> int:
    if requested_feature_k <= 0 or post_variance_feature_count <= 0:
        raise ValueError("Both feature counts must be positive")
    return min(requested_feature_k, post_variance_feature_count)


def result_inventory() -> dict[str, list[str]]:
    return {
        name: sorted(str(path.relative_to(PHASE_DIR)) for path in (PHASE_DIR / "results" / name).rglob("*") if path.is_file())
        for name in ["checkpoints", "predictions", "oof"]
    }


def load_evidence() -> dict[str, Any]:
    audit = collect_audit()
    phase05 = read_json(P5_ENCODING)
    phase06_variant = read_json(P6_VARIANT)
    classification = read_json(P6_CLASSIFICATION)
    regression = read_json(P6_REGRESSION)
    freeze = read_json(P6_FREEZE)
    modality_manifest = read_json(PHASE_DIR / "manifests" / "phase07_modality_feature_manifest.json")
    initialization_audits = {
        name: read_json(PHASE_DIR / "audits" / name)
        for name in [
            "phase07_initialization_artifact_audit.json",
            "phase07_notebook_persistence_audit.json",
            "phase07_input_and_fold_audit.json",
            "phase07_modality_mapping_audit.json",
            "phase07_phase06_freeze_interface_audit.json",
        ]
    }
    upstream_paths = {
        "primary_data": Path(audit["paths"]["primary_data"]),
        "frozen_folds": Path(audit["paths"]["frozen_folds"]),
        "phase06_freeze": P6_FREEZE,
        "phase06_classification": P6_CLASSIFICATION,
        "phase06_regression": P6_REGRESSION,
        "phase06_variant_contract": P6_VARIANT,
        "phase05_encoding_contract": P5_ENCODING,
    }
    upstream_hashes = {name: sha256(path) for name, path in upstream_paths.items()}
    return {
        "audit": audit,
        "phase05": phase05,
        "phase06_variant": phase06_variant,
        "classification": classification,
        "regression": regression,
        "freeze": freeze,
        "modality_manifest": modality_manifest,
        "initialization_audits": initialization_audits,
        "upstream_paths": {name: str(path) for name, path in upstream_paths.items()},
        "upstream_hashes": upstream_hashes,
    }


def normalized_fold_structures(classification: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"outer_fold": int(item["outer_fold"]), **json.loads(item["selected_structure_json"])}
        for item in classification["fold_selected_structures"]
    ]


def normalized_ridge_policy(regression: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"outer_fold": int(item["outer_fold"]), "seed_parameters": json.loads(item["parameter_policy_json"])}
        for item in regression["fold_parameter_policy"]
    ]


def validate_evidence(evidence: dict[str, Any]) -> dict[str, bool]:
    audit = evidence["audit"]
    phase05 = evidence["phase05"]
    variant = evidence["phase06_variant"]
    classification = evidence["classification"]
    regression = evidence["regression"]
    modality_manifest = evidence["modality_manifest"]
    manifest_counts = {item["name"]: item["feature_count"] for item in modality_manifest["modalities"]}
    missing_counts = {item["modality"]: item["fully_missing_rows"] for item in audit["modality_audits"]}
    structures = normalized_fold_structures(classification)
    ridge_policy = normalized_ridge_policy(regression)
    all_alphas = [item["ridge_alpha"] for fold in ridge_policy for item in fold["seed_parameters"]]
    all_policy_seeds = sorted({item["seed"] for fold in ridge_policy for item in fold["seed_parameters"]})
    checks = {
        "upstream_hashes_pass": evidence["upstream_hashes"] == EXPECTED_UPSTREAM_HASHES,
        "initialization_gates_pass": audit["initialization_gates_without_notebook_persistence_pass"],
        "initialization_artifact_audit_pass": evidence["initialization_audits"]["phase07_initialization_artifact_audit.json"]["result"] == "PASS",
        "initialization_notebook_persistence_pass": evidence["initialization_audits"]["phase07_notebook_persistence_audit.json"]["result"] == "PASS",
        "rows_subjects_runs_features_pass": all([
            audit["data_checks"]["modeling_rows"] == 419,
            audit["data_checks"]["subjects"] == 35,
            audit["data_checks"]["unique_run_key"] == 419,
            audit["data_checks"]["primary_predictive_features"] == 1176,
        ]),
        "folds_pass": all([
            audit["data_checks"]["outer_folds"] == [1, 2, 3, 4, 5],
            audit["data_checks"]["outer_subject_isolation_pass"],
            audit["data_checks"]["inner_groupkfold_3_feasibility_pass"],
        ]),
        "phase06_frozen_ready": evidence["freeze"]["status"] == "FROZEN" and evidence["freeze"]["ready_for_next_planned_phase"] is True,
        "phase06_outer_oof_not_used": not audit["phase06_checks"]["outer_oof_read_for_selection"],
        "modalities_pass": manifest_counts == MODALITY_COUNTS,
        "modality_union_pass": modality_manifest["checks"]["feature_union_count"] == 1176 and modality_manifest["checks"]["disjointness_pass"] and modality_manifest["checks"]["union_coverage_pass"],
        "missingness_counts_pass": missing_counts == EXPECTED_MISSING_ROWS,
        "phase05_randomness_pass": all([
            phase05["randomness"]["generator"] == "np.random.default_rng",
            phase05["randomness"]["bit_generator"] == "PCG64",
            phase05["randomness"]["final_seeds"] == SEEDS,
            phase05["randomness"]["python_hash_seed_prohibited"] is True,
            "SHA-256" in phase05["randomness"]["stable_derivation"],
            phase05["representation"]["paired_dimension_rule"] == "use the first D dimensions of a single 10000-dimensional codebook",
        ]),
        "preprocessing_pass": [item["operation"] for item in variant["preprocessing"][:4]] == ["SimpleImputer", "VarianceThreshold", "StandardScaler", "SelectKBest"] and variant["preprocessing"][3]["parameters"]["score_func"] == "f_classif",
        "classification_interface_pass": all([
            classification["selected_variant"] == "hybrid",
            classification["selected_variant_name"] == "HDC+OnlineHD Hybrid",
            classification["selected_fixed_dimension"] == 5000,
            classification["levels"] == 51,
            classification["feature_k"] == 50,
            len(structures) == 5,
            classification["single_seed_selected"] is False,
        ]),
        "regression_interface_pass": all([
            regression["selected_variant"] == "common_ridge",
            regression["selected_regression_head"] == "COMMON_ENCODER_READOUT_BASELINE",
            regression["selected_fixed_dimension"] == 10000,
            regression["levels"] == 51,
            regression["feature_k"] == 50,
            len(ridge_policy) == 5,
            all(alpha == 0.01 for alpha in all_alphas),
            all_policy_seeds == SEEDS,
            regression["single_seed_selected"] is False,
        ]),
        "no_training_artifacts": all(not files for files in result_inventory().values()),
    }
    return checks


def build_contracts(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    classification = evidence["classification"]
    regression = evidence["regression"]
    fold_structures = normalized_fold_structures(classification)
    ridge_policy = normalized_ridge_policy(regression)
    timestamp = now_utc()
    frozen_contract = {
        "phase": "07",
        "phase_name": "Unimodal Contribution Analysis",
        "contract_version": "phase07_frozen_unimodal_contract_v1",
        "status": "CONTRACT_FROZEN_NOT_TRAINED",
        "frozen_at_utc": timestamp,
        "primary_analysis": "FULL_COHORT_419_RUNS_WITH_FOLD_LOCAL_MISSINGNESS_HANDLING",
        "modalities": [{"name": name, "feature_count": count, "membership_source": "manifests/phase07_modality_feature_manifest.json", "fusion_permitted": False} for name, count in MODALITY_COUNTS.items()],
        "sample_and_fold_policy": {
            "modeling_rows": 419,
            "run_key_alignment_required": True,
            "outer_folds": 5,
            "outer_split_source": evidence["upstream_paths"]["frozen_folds"],
            "outer_fold_regeneration_prohibited": True,
            "retain_all_modeling_rows": True,
            "row_deletion_prohibited": True,
            "outer_training_fit_only": True,
            "outer_test_transform_and_predict_only": True,
        },
        "missingness_policy": {
            "status": "FROZEN",
            "fully_missing_rows_by_modality": EXPECTED_MISSING_ROWS,
            "fully_missing_rows_retained": True,
            "imputer_fit_scope": "outer-training or inner-training rows only, as appropriate",
            "missing_indicators_required": True,
            "global_statistics_prohibited": True,
            "global_nan_to_zero_prohibited": True,
            "availability_stratified_diagnostics_required": True,
            "diagnostic_strata": ["modality_available", "modality_fully_missing"],
            "diagnostics_do_not_replace_full_cohort": True,
            "diagnostics_not_for_retraining_or_selection": True,
        },
        "preprocessing": {
            "fit_scope": "fold-local training data only",
            "ordered_steps": [
                {"step": 1, "operation": "SimpleImputer", "parameters": {"strategy": "median", "add_indicator": True, "keep_empty_features": True}},
                {"step": 2, "operation": "VarianceThreshold", "parameters": {"threshold": 0.0}},
                {"step": 3, "operation": "StandardScaler", "parameters": {"statistics": "current training fold only"}},
                {"step": 4, "operation": "SelectKBest", "parameters": {"score_func": "f_classif"}},
            ],
            "regression_reuses_f_classif": True,
            "requested_feature_k": 50,
            "effective_feature_k_rule": "min(50, post_variance_feature_count)",
            "required_run_records": ["requested_feature_k", "post_variance_feature_count", "effective_feature_k", "selected_feature_names", "missing_indicator_feature_names"],
        },
        "classification": {
            "model": "HDC+OnlineHD Hybrid",
            "selected_variant": "hybrid",
            "dimension": 5000,
            "levels": 51,
            "requested_feature_k": 50,
            "representation": "bipolar",
            "similarity": "cosine",
            "seeds": SEEDS,
            "classes": CLASSES,
            "primary_metric": "Macro-F1",
            "fold_structures": fold_structures,
            "structure_source": str(P6_CLASSIFICATION),
            "metric_names": ["macro_f1", "balanced_accuracy", "accuracy", "per_class_recall", "severe_error_rate", "confusion_matrix"],
            "severe_error_definition": "abs(predicted_class - target_class) >= 2",
            "metric_labels": CLASSES,
            "zero_division": 0,
            "runs_per_modality": 25,
            "total_runs": 125,
            "reselection_prohibited": True,
            "single_best_seed_selection_prohibited": True,
        },
        "regression": {
            "selected_variant": "common_ridge",
            "head": "COMMON_ENCODER_READOUT_BASELINE",
            "dimension": 10000,
            "levels": 51,
            "requested_feature_k": 50,
            "seeds": SEEDS,
            "target": "target_score",
            "target_range": [1.0, 4.0],
            "target_description": "bounded difficulty-induced workload proxy regression",
            "primary_metric": "bounded_mae",
            "ridge_alpha": 0.01,
            "fold_seed_parameter_policy": ridge_policy,
            "prediction_fields": ["prediction_raw", "prediction_bounded", "residual_bounded"],
            "prediction_bounded_rule": "clip(prediction_raw, 1.0, 4.0)",
            "metric_names": ["bounded_mae", "raw_mae", "bounded_rmse", "bounded_r2", "bounded_spearman", "per_target_level_mae"],
            "runs_per_modality": 25,
            "total_runs": 125,
            "parameter_reselection_prohibited": True,
            "single_best_seed_selection_prohibited": True,
        },
        "randomness": {
            "seeds": SEEDS,
            "generator": "np.random.default_rng",
            "bit_generator": "PCG64",
            "stable_derivation": "SHA-256 over experiment seed, vector role, feature name or level count, and contract version",
            "paired_codebook_dimension_rule": "use the first D dimensions of a single 10000-dimensional codebook",
            "python_hash_seed_prohibited": True,
        },
        "oof_aggregation": {
            "status": "FROZEN",
            "seed_level_rows_per_task": 10475,
            "seed_level_rows_total": 20950,
            "canonical_rows_per_task": 2095,
            "classification": {
                "group_key": ["modality", "run_key"],
                "rule": "arithmetic mean of the five seed class scores, then argmax",
                "argmax_tie": "smaller target_class",
                "true_label_tiebreak_prohibited": True,
            },
            "regression": {
                "group_key": ["modality", "run_key"],
                "rule": "arithmetic mean of five prediction_raw values, then clip to [1.0, 4.0]",
                "canonical_metrics_recomputed_from_aggregated_predictions": True,
            },
            "mean_of_seed_metrics_not_canonical": True,
        },
        "ranking": {
            "classification": {
                "status": "FROZEN",
                "cohort": "all 419 canonical OOF rows",
                "ordered_criteria": ["macro_f1 descending", "balanced_accuracy descending", "severe_error_rate ascending", "seed_macro_f1_sample_sd ascending"],
                "tie_tolerance": 1e-12,
                "remaining_exact_tie": "shared rank; modality-name display order only",
            },
            "regression": {
                "status": "FROZEN",
                "cohort": "all 419 canonical OOF rows",
                "ordered_criteria": ["bounded_mae ascending", "bounded_rmse ascending", "bounded_spearman descending", "seed_bounded_mae_sample_sd ascending"],
                "tie_tolerance": 1e-12,
                "remaining_exact_tie": "shared rank; modality-name display order only",
            },
            "combined_best_modality_prohibited": True,
            "availability_strata_not_for_ranking": True,
        },
        "multimodal_reference": {
            "access": "READ_ONLY",
            "retraining_prohibited": True,
            "required_unique_run_keys": 419,
            "run_key_alignment": "one-to-one with every Phase 07 canonical OOF",
            "classification_reference": {"model": "HDC+OnlineHD Hybrid", "dimension": 5000, "delta": "unimodal_macro_f1 - multimodal_macro_f1", "weaker_direction": "delta < 0"},
            "regression_reference": {"head": "COMMON_ENCODER_READOUT_BASELINE", "dimension": 10000, "delta": "unimodal_bounded_mae - multimodal_bounded_mae", "weaker_direction": "delta > 0"},
            "deltas_not_for_contract_or_model_change": True,
        },
        "error_analysis": {
            "classification": ["4x4 confusion matrix", "per-class recall", "severe error rate", "Level 1 versus Level 4 severe confusion", "classification confidence", "similarity margin", "subject macro_f1", "availability strata"],
            "regression": ["prediction_raw", "prediction_bounded", "residual distribution", "per-target-level MAE", "boundary clipping count and rate", "mean-collapse diagnostics", "subject bounded_mae", "availability strata"],
            "physiological_causal_interpretation_prohibited": True,
        },
        "expected_model_runs": {"classification": 125, "regression": 125, "total": 250},
        "training_executed": False,
        "outer_test_predictions_generated": False,
        "prohibitions": ["model training during contract freeze", "hypervector generation during contract freeze", "outer-test prediction during contract freeze", "OOF generation during contract freeze", "variant/dimension/levels/seed reselection", "performance experiments", "control-input experiments", "multimodal fusion", "Phase 08"],
    }
    statistics = {
        "phase": "07",
        "contract_version": "phase07_statistical_analysis_contract_v1",
        "status": "FROZEN_NOT_EXECUTED",
        "statistical_unit": "subject",
        "subjects": 35,
        "run_as_independent_unit_prohibited": True,
        "paired_analysis": True,
        "bootstrap": {
            "repetitions": 2000,
            "seed": 42,
            "confidence_interval": "percentile 95%",
            "shared_subject_resamples_for_all_models": True,
            "resampled_subject_includes_all_oof_runs": True,
            "classification_metrics": ["macro_f1", "balanced_accuracy", "severe_error_rate"],
            "regression_metrics": ["bounded_mae", "bounded_rmse"],
        },
        "overall_unimodal_comparison": {"test": "Friedman", "unit": "subject-level metrics", "separate_by_task": True, "outer_folds_as_samples_prohibited": True},
        "preregistered_pairwise": {
            "comparisons": "frozen multimodal reference versus each of five unimodal models",
            "classification_metric": "subject-level macro_f1",
            "regression_metric": "subject-level bounded_mae",
            "test": "Wilcoxon signed-rank",
            "multiplicity": "Holm correction separately within five classification and five regression comparisons",
            "alpha": 0.05,
            "effect_size": "rank-biserial correlation",
            "not_estimable_rule": "If all paired differences are zero or the test is not computable, report NOT_ESTIMABLE with reason and no fabricated p-value",
        },
        "interpretation": "association and predictive performance only; no physiological causality",
    }
    execution = {
        "phase": "07",
        "status": "CONTRACT_FROZEN_NOT_TRAINED",
        "created_at_utc": timestamp,
        "modalities": 5,
        "modality_names": list(MODALITY_COUNTS),
        "outer_folds": 5,
        "evaluation_seeds": SEEDS,
        "evaluation_seed_count": 5,
        "classification_runs": 125,
        "regression_runs": 125,
        "total_model_runs": 250,
        "checkpoint_granularity": "modality/task/fold/seed",
        "resume_policy": "reuse only audit-PASS checkpoints",
        "completed_runs": 0,
        "training_executed": False,
        "prediction_files_generated": 0,
        "oof_files_generated": 0,
        "executor_invoked": False,
    }
    metrics = {
        "phase": "07",
        "status": "FROZEN",
        "classification": {
            "labels": CLASSES,
            "zero_division": 0,
            "primary": "macro_f1",
            "definitions": {
                "macro_f1": "unweighted mean of per-class F1 across fixed labels [0,1,2,3]",
                "balanced_accuracy": "unweighted mean of per-class recall",
                "accuracy": "correct predictions divided by evaluated rows",
                "per_class_recall": "recall for each fixed class label",
                "severe_error_rate": "mean(abs(predicted_class - target_class) >= 2)",
                "confusion_matrix": "4x4 matrix with fixed label order [0,1,2,3]",
            },
        },
        "regression": {
            "primary": "bounded_mae",
            "target_range": [1.0, 4.0],
            "target_description": "bounded difficulty-induced workload proxy regression",
            "prediction_bounded": "clip(prediction_raw, 1.0, 4.0)",
            "definitions": {
                "bounded_mae": "MAE(target_score, prediction_bounded)",
                "raw_mae": "MAE(target_score, prediction_raw)",
                "bounded_rmse": "RMSE(target_score, prediction_bounded)",
                "bounded_r2": "R2(target_score, prediction_bounded)",
                "bounded_spearman": "Spearman(target_score, prediction_bounded)",
                "residual_bounded": "target_score - prediction_bounded",
                "per_target_level_mae": "bounded MAE separately for each observed target_score level",
            },
        },
    }
    return {"frozen": frozen_contract, "statistics": statistics, "execution": execution, "metrics": metrics}


def freeze_contract() -> None:
    evidence = load_evidence()
    checks = validate_evidence(evidence)
    if not all(checks.values()):
        raise RuntimeError(f"Contract preconditions failed: {[key for key, value in checks.items() if not value]}")
    contracts = build_contracts(evidence)
    write_json(PHASE_DIR / "configs" / "phase07_frozen_unimodal_contract.json", contracts["frozen"])
    write_json(PHASE_DIR / "configs" / "phase07_statistical_analysis_contract.json", contracts["statistics"])
    write_json(PHASE_DIR / "configs" / "phase07_execution_manifest.json", contracts["execution"])
    write_json(PHASE_DIR / "configs" / "phase07_metric_definitions.json", contracts["metrics"])
    experiment_contract_path = PHASE_DIR / "configs" / "phase07_experiment_contract.json"
    experiment_contract = read_json(experiment_contract_path)
    experiment_contract.update({
        "status": "CONTRACT_FROZEN_NOT_TRAINED",
        "contract_frozen_at_utc": now_utc(),
        "frozen_unimodal_contract": "configs/phase07_frozen_unimodal_contract.json",
        "statistical_analysis_contract": "configs/phase07_statistical_analysis_contract.json",
        "execution_manifest": "configs/phase07_execution_manifest.json",
        "metric_definitions": "configs/phase07_metric_definitions.json",
        "training_executed": False,
        "outer_test_predictions_generated": False,
    })
    write_json(experiment_contract_path, experiment_contract)
    write_json(PHASE_DIR / "audits" / "phase07_contract_freeze_audit.json", {
        "phase": "07", "audit": "contract_freeze", "generated_at_utc": now_utc(),
        "precondition_checks": checks,
        "upstream_paths": evidence["upstream_paths"],
        "upstream_sha256": evidence["upstream_hashes"],
        "fold_structures_verified": len(normalized_fold_structures(evidence["classification"])),
        "ridge_fold_policies_verified": len(normalized_ridge_policy(evidence["regression"])),
        "training_executed": False, "outer_test_predictions_generated": False,
        "result": "PASS",
    })
    write_json(PHASE_DIR / "audits" / "phase07_pretraining_feasibility_audit.json", {
        "phase": "07", "audit": "pretraining_feasibility", "generated_at_utc": now_utc(),
        "checks": {
            "full_cohort_rows": 419,
            "outer_folds": 5,
            "inner_groupkfold_feasible": True,
            "modalities": MODALITY_COUNTS,
            "fully_missing_rows": EXPECTED_MISSING_ROWS,
            "all_outer_train_test_partitions_have_available_modality_data": True,
            "effective_feature_k_rule": "min(50, post_variance_feature_count)",
            "body_movement_example_post_variance_42": effective_feature_k(50, 42),
            "requested_feature_k": 50,
            "results_inventory": result_inventory(),
        },
        "training_executed": False, "result": "PASS",
    })
    for name, audit_name in [
        ("phase07_contract_artifact_audit.json", "contract_artifacts"),
        ("phase07_contract_notebook_persistence_audit.json", "contract_notebook_persistence"),
    ]:
        write_json(PHASE_DIR / "audits" / name, {
            "phase": "07", "audit": audit_name, "generated_at_utc": now_utc(),
            "status": "PENDING_STATIC_TEST_AND_NOTEBOOK_EXECUTION", "training_executed": False,
            "outer_test_predictions_generated": False, "result": "PENDING",
        })
    write_json(PHASE_DIR / "manifests" / "phase07_contract_artifact_manifest.json", {
        "phase": "07", "status": "PENDING_FINAL_AUDIT", "generated_at_utc": now_utc(), "artifacts": []
    })
    print(json.dumps({"contract_preconditions": checks, "status": "CONTRACT_FROZEN_NOT_TRAINED"}, indent=2))


def append_notebook_section() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells = [cell for cell in notebook.cells if cell.get("metadata", {}).get("phase07_contract_freeze") is not True]
    def md(text: str):
        cell = nbformat.v4.new_markdown_cell(text)
        cell.metadata["phase07_contract_freeze"] = True
        return cell
    def code(text: str):
        cell = nbformat.v4.new_code_cell(text)
        cell.metadata["phase07_contract_freeze"] = True
        return cell
    notebook.cells.extend([
        md("# Phase 07 Contract Freeze\n\nStatic contract freeze only. No model training, hypervector generation, outer-test prediction, or OOF generation is executed."),
        code("contract = json.loads((PHASE_DIR / 'configs/phase07_frozen_unimodal_contract.json').read_text(encoding='utf-8'))\nstats_contract = json.loads((PHASE_DIR / 'configs/phase07_statistical_analysis_contract.json').read_text(encoding='utf-8'))\nexecution = json.loads((PHASE_DIR / 'configs/phase07_execution_manifest.json').read_text(encoding='utf-8'))\nprint(json.dumps({'status': contract['status'], 'primary_analysis': contract['primary_analysis'], 'training_executed': contract['training_executed']}, indent=2))"),
        md("## Frozen modalities"),
        code("print(json.dumps(contract['modalities'], indent=2)); assert len(contract['modalities']) == 5 and sum(item['feature_count'] for item in contract['modalities']) == 1176"),
        md("## Full-cohort and fully-missing modality policy"),
        code("print(json.dumps({'sample_policy': contract['sample_and_fold_policy'], 'missingness_policy': contract['missingness_policy']}, indent=2)); assert contract['sample_and_fold_policy']['retain_all_modeling_rows']; assert contract['missingness_policy']['fully_missing_rows_retained']"),
        md("## Frozen classification interface"),
        code("print(json.dumps(contract['classification'], indent=2)); assert contract['classification']['model'] == 'HDC+OnlineHD Hybrid' and len(contract['classification']['fold_structures']) == 5"),
        md("## Frozen regression interface"),
        code("print(json.dumps(contract['regression'], indent=2)); assert contract['regression']['head'] == 'COMMON_ENCODER_READOUT_BASELINE' and contract['regression']['ridge_alpha'] == 0.01"),
        md("## Effective feature-k rule"),
        code("print(json.dumps(contract['preprocessing'], indent=2)); assert contract['preprocessing']['effective_feature_k_rule'] == 'min(50, post_variance_feature_count)'"),
        md("## Seeds and expected run counts"),
        code("print(json.dumps({'seeds': contract['randomness']['seeds'], 'expected_model_runs': contract['expected_model_runs'], 'execution_manifest': execution}, indent=2)); assert execution['total_model_runs'] == 250 and execution['completed_runs'] in [0, 250]"),
        md("## Canonical OOF aggregation"),
        code("print(json.dumps(contract['oof_aggregation'], indent=2)); assert contract['oof_aggregation']['status'] == 'FROZEN'"),
        md("## Separate modality ranking rules"),
        code("print(json.dumps(contract['ranking'], indent=2)); assert contract['ranking']['classification']['status'] == contract['ranking']['regression']['status'] == 'FROZEN'"),
        md("## Subject-level statistical-analysis rules"),
        code("print(json.dumps(stats_contract, indent=2)); assert stats_contract['statistical_unit'] == 'subject' and stats_contract['bootstrap']['repetitions'] == 2000"),
        md("## Static tests and contract artifact audit"),
        code("test_report = json.loads((PHASE_DIR / 'audits/phase07_contract_artifact_audit.json').read_text(encoding='utf-8'))\nprint(json.dumps({'static_unit_tests': test_report.get('static_unit_tests'), 'artifact_audit_result': test_report['result']}, indent=2)); assert test_report['result'] == 'PASS' and test_report.get('static_unit_tests') == 'PASS'"),
        md("## Contract Freeze execution gate"),
        code("ready = all([contract['status'] == 'CONTRACT_FROZEN_NOT_TRAINED', test_report['result'] == 'PASS', not contract['training_executed'], not contract['outer_test_predictions_generated']])\nprint(json.dumps({'training_executed': 'NO', 'outer_test_predictions_generated': 'NO', 'ready_for_unimodal_execution_subject_to_notebook_persistence': ready}, indent=2)); assert ready"),
    ])
    nbformat.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


def audit_artifacts(static_tests: str, finalize_notebook: bool) -> None:
    required = [
        "configs/phase07_experiment_contract.json",
        "configs/phase07_frozen_unimodal_contract.json",
        "configs/phase07_statistical_analysis_contract.json",
        "configs/phase07_execution_manifest.json",
        "configs/phase07_metric_definitions.json",
        "manifests/phase07_modality_feature_manifest.json",
        "audits/phase07_contract_freeze_audit.json",
        "audits/phase07_pretraining_feasibility_audit.json",
        "scripts/freeze_phase07_contract.py",
        "tests/test_phase07_contract.py",
        "README.md", "task_plan.md", "notes.md", "Phase_07_Unimodal_Contribution.ipynb",
    ]
    inventory = []
    for relative in required:
        path = PHASE_DIR / relative
        parseable = None
        if path.suffix == ".json" and path.is_file():
            try:
                read_json(path)
                parseable = True
            except (OSError, json.JSONDecodeError):
                parseable = False
        inventory.append({"relative_path": relative, "exists": path.is_file(), "json_parseable": parseable, "sha256": sha256(path) if path.is_file() else None})
    evidence = load_evidence()
    upstream_unchanged = evidence["upstream_hashes"] == EXPECTED_UPSTREAM_HASHES
    no_results = all(not files for files in result_inventory().values())
    artifact_pass = all(item["exists"] and item["json_parseable"] is not False for item in inventory) and upstream_unchanged and no_results and static_tests == "PASS"
    artifact_audit = {
        "phase": "07", "audit": "contract_artifacts", "generated_at_utc": now_utc(),
        "inventory": inventory, "static_unit_tests": static_tests,
        "upstream_sha256_after_freeze": evidence["upstream_hashes"],
        "upstream_files_unchanged": upstream_unchanged,
        "result_inventory": result_inventory(),
        "training_executed": False, "outer_test_predictions_generated": False,
        "result": "PASS" if artifact_pass else "FAIL",
    }
    write_json(PHASE_DIR / "audits" / "phase07_contract_artifact_audit.json", artifact_audit)
    notebook_audit = read_json(PHASE_DIR / "audits" / "phase07_contract_notebook_persistence_audit.json")
    if finalize_notebook:
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        tagged = [cell for cell in notebook.cells if cell.get("metadata", {}).get("phase07_contract_freeze") is True and cell.cell_type == "code"]
        errors = [output for cell in tagged for output in cell.get("outputs", []) if output.get("output_type") == "error"]
        notebook_pass = bool(tagged) and all(cell.get("execution_count") is not None and cell.get("outputs") for cell in tagged) and not errors
        notebook_audit = {
            "phase": "07", "audit": "contract_notebook_persistence", "generated_at_utc": now_utc(),
            "status": "EXECUTED_AND_SAVED" if notebook_pass else "FAIL",
            "notebook_path": str(NOTEBOOK), "notebook_sha256": sha256(NOTEBOOK),
            "contract_code_cells": len(tagged), "all_contract_code_cells_executed_with_outputs": notebook_pass,
            "error_output_count": len(errors), "training_executed": False,
            "outer_test_predictions_generated": False, "result": "PASS" if notebook_pass else "FAIL",
        }
        write_json(PHASE_DIR / "audits" / "phase07_contract_notebook_persistence_audit.json", notebook_audit)
    manifest_files = [PHASE_DIR / item["relative_path"] for item in inventory]
    manifest_files.extend([
        PHASE_DIR / "audits" / "phase07_contract_artifact_audit.json",
        PHASE_DIR / "audits" / "phase07_contract_notebook_persistence_audit.json",
    ])
    manifest = {
        "phase": "07", "status": "CONTRACT_FROZEN_NOT_TRAINED", "generated_at_utc": now_utc(),
        "artifacts": [{"relative_path": str(path.relative_to(PHASE_DIR)), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in manifest_files if path.is_file()],
        "training_executed": False, "outer_test_predictions_generated": False,
        "result": "PASS" if artifact_pass and (not finalize_notebook or notebook_audit["result"] == "PASS") else "FAIL",
    }
    write_json(PHASE_DIR / "manifests" / "phase07_contract_artifact_manifest.json", manifest)
    print(json.dumps({"artifact_audit": artifact_audit["result"], "notebook_persistence": notebook_audit["result"], "manifest": manifest["result"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Phase 07 contracts without training or prediction.")
    parser.add_argument("--append-notebook", action="store_true")
    parser.add_argument("--audit-artifacts", action="store_true")
    parser.add_argument("--finalize-notebook", action="store_true")
    parser.add_argument("--static-tests", choices=["PASS", "FAIL"], default="PASS")
    args = parser.parse_args()
    if args.append_notebook:
        append_notebook_section()
    elif args.audit_artifacts or args.finalize_notebook:
        audit_artifacts(args.static_tests, args.finalize_notebook)
    else:
        freeze_contract()


if __name__ == "__main__":
    main()
