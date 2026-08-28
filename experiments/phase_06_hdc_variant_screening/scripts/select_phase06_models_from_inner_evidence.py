"""Deterministically select Phase 06 canonical models from allowlisted inner evidence only."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
PHASE05 = ROOT / "experiments/phase_05_basic_dual_output_hdc"
AMENDMENT = PHASE / "configs/phase06_final_model_selection_rules_amendment_v2.json"
ORIGINAL_RULE = PHASE / "configs/phase06_model_selection_rules.json"
VARIANTS = ["vanilla", "onlinehd", "multicentroid", "hybrid"]
DISPLAY = {"vanilla": "Vanilla Prototype HDC", "onlinehd": "OnlineHD-style HDC", "multicentroid": "Multi-centroid HDC", "hybrid": "HDC+OnlineHD Hybrid", "common_ridge": "COMMON_ENCODER_READOUT_BASELINE"}
CLASS_DIMS = [2000, 5000]
REG_DIMS = [1000, 2000, 5000, 10000]
SEEDS = [42, 43, 44, 45, 46]
FOLDS = [1, 2, 3, 4, 5]

ALLOWLIST = [
    "experiments/phase_06_hdc_variant_screening/configs/phase06_model_selection_rules.json",
    "experiments/phase_06_hdc_variant_screening/configs/phase06_final_model_selection_rules_amendment_v2.json",
    "experiments/phase_05_basic_dual_output_hdc/results/summaries/vanilla_hdc_quick_screen_fold_*_candidates.csv",
    "experiments/phase_05_basic_dual_output_hdc/results/fold_metrics/vanilla_hdc_quick_screen_fold_*_inner_metrics.csv",
    "experiments/phase_05_basic_dual_output_hdc/results/efficiency/vanilla_hdc_quick_screen_fold_*_efficiency.csv",
    "experiments/phase_06_hdc_variant_screening/results/summaries/*_quick_screen_fold_*_candidates.csv",
    "experiments/phase_06_hdc_variant_screening/results/fold_metrics/*_quick_screen_fold_*_inner_metrics.csv",
    "experiments/phase_06_hdc_variant_screening/results/efficiency/*_quick_screen_fold_*_efficiency.csv",
    "experiments/phase_05_basic_dual_output_hdc/results/fold_metrics/vanilla_hdc_final_confirmation_fold_*_inner_selection.csv",
    "experiments/phase_05_basic_dual_output_hdc/results/efficiency/vanilla_hdc_final_confirmation_fold_*_efficiency.csv",
    "experiments/phase_05_basic_dual_output_hdc/results/efficiency/vanilla_hdc_final_confirmation_protocol_completion_by_fold_config.csv",
    "experiments/phase_06_hdc_variant_screening/results/fold_metrics/*_final_confirmation_fold_*_inner_selection.csv",
    "experiments/phase_06_hdc_variant_screening/results/efficiency/*_final_confirmation_fold_*_efficiency.csv"
]
BANNED_PARTS = ["/results/oof/", "/results/predictions/", "_classification_metrics.csv", "_similarity_regression_metrics.csv", "oof_metrics_by_config.csv"]
READ_PATHS: set[str] = set()
FORBIDDEN_ATTEMPTS: list[str] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def assert_allowed(path: Path) -> None:
    relative = project_relative(path)
    lowered = "/" + relative.lower()
    if any(token in lowered for token in BANNED_PARTS) or not any(fnmatch.fnmatch(relative, pattern) for pattern in ALLOWLIST):
        FORBIDDEN_ATTEMPTS.append(relative)
        raise PermissionError(f"Selector input is not allowlisted: {relative}")


def read_csv(path: Path) -> pd.DataFrame:
    assert_allowed(path); READ_PATHS.add(project_relative(path)); return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    assert_allowed(path); READ_PATHS.add(project_relative(path)); return json.loads(path.read_text(encoding="utf-8"))


def stable_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"); temporary.replace(path)


def stable_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n", float_format="%.17g"); temporary.replace(path)


def new_quick_key(row: pd.Series) -> tuple[Any, ...]:
    return (-float(row.mean_macro_f1), float(row.std_macro_f1_sample), -float(row.mean_balanced_accuracy), float(row.mean_severe_error_rate), int(row.dimension), int(row.get("epochs", 0) or 0), int(row.get("centroids_per_class", 0) or 0), float(row.get("learning_rate", 0) or 0), float(row.get("margin_threshold", 0) or 0), str(row.canonical_config_json))


def validate_new_candidate(row: pd.Series, inner: pd.DataFrame) -> tuple[float, int]:
    rows = inner[inner.candidate_id == row.candidate_id]
    if len(rows) != 3 or rows.inner_fold.nunique() != 3: raise RuntimeError(f"Incomplete inner evidence for {row.candidate_id}")
    checks = {"mean_macro_f1": rows.macro_f1.mean(), "mean_balanced_accuracy": rows.balanced_accuracy.mean(), "mean_severe_error_rate": rows.severe_error_rate.mean()}
    for field, value in checks.items():
        if abs(float(row[field]) - float(value)) > 1e-12: raise RuntimeError(f"Candidate/inner mismatch {row.candidate_id} {field}")
    runtime = float((rows.training_seconds + rows.inference_seconds + rows.preprocessing_seconds_shared + rows.encoding_seconds_shared).sum())
    return runtime, int(rows.model_bytes.max())


def validate_vanilla_candidate(row: pd.Series, inner: pd.DataFrame) -> tuple[float, int]:
    rows = inner[(inner.dimension == int(row.dimension)) & (inner.levels == int(row.levels)) & (inner.k.astype(str) == str(row.k)) & (inner.seed == int(row.seed))]
    if len(rows) != 3 or rows.inner_fold.nunique() != 3: raise RuntimeError("Incomplete Vanilla inner evidence")
    if abs(float(row.mean_macro_f1) - float(rows.macro_f1.mean())) > 1e-12: raise RuntimeError("Vanilla candidate/inner mismatch")
    runtime = float((rows.preprocessing_seconds + rows.encoding_seconds + rows.prototype_training_seconds + rows.inference_seconds).sum())
    return runtime, int(rows.model_bytes.max())


def classification_candidates() -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for fold in FOLDS:
            if variant == "vanilla":
                candidates = read_csv(PHASE05 / f"results/summaries/vanilla_hdc_quick_screen_fold_{fold}_candidates.csv")
                inner = read_csv(PHASE05 / f"results/fold_metrics/vanilla_hdc_quick_screen_fold_{fold}_inner_metrics.csv")
                read_csv(PHASE05 / f"results/efficiency/vanilla_hdc_quick_screen_fold_{fold}_efficiency.csv")
                candidates = candidates[(candidates.levels == 51) & (candidates.k.astype(str) == "50") & (candidates.seed == 42)].copy()
            else:
                candidates = read_csv(PHASE / f"results/summaries/{variant}_quick_screen_fold_{fold}_candidates.csv")
                inner = read_csv(PHASE / f"results/fold_metrics/{variant}_quick_screen_fold_{fold}_inner_metrics.csv")
                read_csv(PHASE / f"results/efficiency/{variant}_quick_screen_fold_{fold}_efficiency.csv")
            for dimension in CLASS_DIMS:
                family = candidates[candidates.dimension == dimension].copy()
                if family.empty: raise RuntimeError(f"No classification candidates: {variant} d={dimension} fold={fold}")
                if variant == "vanilla":
                    family = family.sort_values(["mean_macro_f1", "std_macro_f1", "mean_severe_error_rate", "parameter_grid_order"], ascending=[False, True, True, True], kind="mergesort")
                    selected = family.iloc[0]; runtime, model_bytes = validate_vanilla_candidate(selected, inner)
                    structure = {"levels": 51, "feature_k": 50, "seed": 42}
                else:
                    selected = min((row for _, row in family.iterrows()), key=new_quick_key); runtime, model_bytes = validate_new_candidate(selected, inner)
                    structure = {key: selected[key].item() if hasattr(selected[key], "item") else selected[key] for key in ["epochs", "centroids_per_class", "learning_rate", "margin_threshold"] if key in selected and pd.notna(selected[key])}
                fold_rows.append({"variant": variant, "canonical_variant_name": DISPLAY[variant], "dimension": dimension, "outer_fold": fold, "fold_mean_inner_macro_f1": float(selected.mean_macro_f1), "fold_mean_inner_balanced_accuracy": float(selected.mean_balanced_accuracy), "fold_mean_inner_severe_error_rate": float(selected.mean_severe_error_rate), "inner_measured_runtime_seconds": runtime, "model_bytes": model_bytes, "selected_structure_json": json.dumps(structure, sort_keys=True, separators=(",", ":"))})
    family_rows = []
    fold_frame = pd.DataFrame(fold_rows)
    for (variant, dimension), group in fold_frame.groupby(["variant", "dimension"], sort=True):
        family_rows.append({"task": "classification", "variant": variant, "canonical_variant_name": DISPLAY[variant], "dimension": int(dimension), "outer_training_task_count": 5, "inner_fold_count_per_task": 3, "mean_inner_macro_f1_across_outer_tasks": float(group.fold_mean_inner_macro_f1.mean()), "sample_sd_outer_task_mean_macro_f1": float(group.fold_mean_inner_macro_f1.std(ddof=1)), "mean_inner_balanced_accuracy_across_outer_tasks": float(group.fold_mean_inner_balanced_accuracy.mean()), "mean_inner_severe_error_rate_across_outer_tasks": float(group.fold_mean_inner_severe_error_rate.mean()), "mean_inner_measured_runtime_seconds": float(group.inner_measured_runtime_seconds.mean()), "mean_model_bytes": float(group.model_bytes.mean()), "model_bytes_complete": True, "fold_selected_structures_json": json.dumps(group.sort_values("outer_fold")[["outer_fold", "selected_structure_json"]].to_dict(orient="records"), sort_keys=True, separators=(",", ":"))})
    frame = pd.DataFrame(family_rows)
    return frame, {"fold_rows": fold_frame.to_dict(orient="records")}


def phase06_regression_fold_rows(variant: str) -> list[dict[str, Any]]:
    rows = []
    for fold in FOLDS:
        inner = read_csv(PHASE / f"results/fold_metrics/{variant}_final_confirmation_fold_{fold}_inner_selection.csv")
        read_csv(PHASE / f"results/efficiency/{variant}_final_confirmation_fold_{fold}_efficiency.csv")
        for dimension in REG_DIMS:
            seed_rows, parameter_policy = [], []
            for seed in SEEDS:
                group = inner[(inner.dimension == dimension) & (inner.seed == seed)].copy()
                selected_temperature = float(group.selected_temperature.iloc[0]); chosen = group[np.isclose(group.temperature.astype(float), selected_temperature, atol=0, rtol=0)]
                if len(chosen) != 3 or chosen.inner_fold.nunique() != 3: raise RuntimeError(f"Incomplete selected temperature evidence {variant} fold={fold} d={dimension} seed={seed}")
                seed_rows.append({"mae": chosen.bounded_mae.mean(), "rmse": chosen.bounded_rmse.mean(), "runtime": chosen.model_seconds.sum(), "model_bytes": chosen.model_bytes.max()})
                parameter_policy.append({"seed": seed, "temperature": selected_temperature})
            values = pd.DataFrame(seed_rows)
            rows.append({"head_family": f"{variant}_similarity", "variant": variant, "regression_head": "similarity_regression", "dimension": dimension, "outer_fold": fold, "fold_mean_inner_bounded_mae": float(values.mae.mean()), "fold_mean_inner_bounded_rmse": float(values.rmse.mean()), "fold_mean_inner_measured_runtime_seconds": float(values.runtime.mean()), "fold_mean_model_bytes": float(values.model_bytes.mean()), "parameter_policy_json": json.dumps(parameter_policy, sort_keys=True, separators=(",", ":")), "rmse_available": True, "runtime_scope": "mean across seeds of sum of three selected inner-split model_seconds"})
    return rows


def phase05_regression_fold_rows() -> list[dict[str, Any]]:
    rows = []
    protocol = read_csv(PHASE05 / "results/efficiency/vanilla_hdc_final_confirmation_protocol_completion_by_fold_config.csv")
    for fold in FOLDS:
        inner = read_csv(PHASE05 / f"results/fold_metrics/vanilla_hdc_final_confirmation_fold_{fold}_inner_selection.csv")
        efficiency = read_csv(PHASE05 / f"results/efficiency/vanilla_hdc_final_confirmation_fold_{fold}_efficiency.csv")
        selected = inner[inner.selected.astype(str).str.lower() == "true"].copy()
        for head, family, variant, output_head in [("similarity_regression", "vanilla_similarity", "vanilla", "similarity_regression"), ("ridge_regression", "COMMON_ENCODER_READOUT_BASELINE", "common_ridge", "COMMON_ENCODER_READOUT_BASELINE")]:
            for dimension in REG_DIMS:
                group = selected[(selected["head"] == head) & (selected.dimension == dimension)]
                if len(group) != 5 or set(group.seed) != set(SEEDS): raise RuntimeError(f"Incomplete Phase 05 selected inner evidence {head} fold={fold} d={dimension}")
                eff = efficiency[efficiency.dimension == dimension]; fold_protocol = protocol[(protocol.outer_fold == fold) & (protocol.dimension == dimension)]
                runtime = float(eff.inner_selection_seconds.mean())
                bytes_column = "prototype_bytes" if head == "similarity_regression" else "ridge_readout_bytes"
                parameter = [{"seed": int(row.seed), "temperature": float(row.temperature)} if head == "similarity_regression" else {"seed": int(row.seed), "ridge_alpha": float(row.ridge_alpha)} for row in group.itertuples(index=False)]
                rows.append({"head_family": family, "variant": variant, "regression_head": output_head, "dimension": dimension, "outer_fold": fold, "fold_mean_inner_bounded_mae": float(group.mean_inner_mae.mean()), "fold_mean_inner_bounded_rmse": np.nan, "fold_mean_inner_measured_runtime_seconds": runtime, "fold_mean_model_bytes": float(fold_protocol[bytes_column].mean()), "parameter_policy_json": json.dumps(parameter, sort_keys=True, separators=(",", ":")), "rmse_available": False, "runtime_scope": "Phase 05 recorded inner_selection_seconds shared by the two heads"})
    return rows


def regression_candidates() -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_rows = phase05_regression_fold_rows()
    for variant in ["onlinehd", "multicentroid", "hybrid"]: fold_rows.extend(phase06_regression_fold_rows(variant))
    fold_frame = pd.DataFrame(fold_rows); candidates = []
    for (family, dimension), group in fold_frame.groupby(["head_family", "dimension"], sort=True):
        if len(group) != 5: raise RuntimeError(f"Regression fold coverage failure {family} d={dimension}")
        candidates.append({"task": "regression", "head_family": family, "canonical_head_family_name": family, "variant": group.variant.iloc[0], "regression_head": group.regression_head.iloc[0], "dimension": int(dimension), "outer_fold_count": 5, "seed_count_per_fold": 5, "mean_inner_bounded_mae": float(group.fold_mean_inner_bounded_mae.mean()), "sample_sd_outer_fold_mean_mae": float(group.fold_mean_inner_bounded_mae.std(ddof=1)), "mean_inner_bounded_rmse": float(group.fold_mean_inner_bounded_rmse.mean()) if group.rmse_available.all() else np.nan, "rmse_complete": bool(group.rmse_available.all()), "mean_inner_measured_runtime_seconds": float(group.fold_mean_inner_measured_runtime_seconds.mean()), "mean_model_bytes": float(group.fold_mean_model_bytes.mean()), "model_bytes_complete": bool(group.fold_mean_model_bytes.notna().all()), "fold_parameter_policy_json": json.dumps(group.sort_values("outer_fold")[["outer_fold", "parameter_policy_json"]].to_dict(orient="records"), sort_keys=True, separators=(",", ":")), "runtime_scope": group.runtime_scope.iloc[0]})
    return pd.DataFrame(candidates), {"fold_rows": fold_frame.to_dict(orient="records")}


def nondominated(frame: pd.DataFrame, performance: str, maximize: bool) -> pd.Series:
    values = frame[[performance, "mean_inner_measured_runtime_seconds", "mean_model_bytes"]].to_numpy(float); keep = np.ones(len(frame), dtype=bool)
    for i in range(len(frame)):
        better_perf = values[:, 0] >= values[i, 0] if maximize else values[:, 0] <= values[i, 0]
        no_worse = better_perf & (values[:, 1] <= values[i, 1]) & (values[:, 2] <= values[i, 2])
        strict = ((values[:, 0] > values[i, 0]) if maximize else (values[:, 0] < values[i, 0])) | (values[:, 1] < values[i, 1]) | (values[:, 2] < values[i, 2])
        if np.any(no_worse & strict): keep[i] = False
    return pd.Series(keep, index=frame.index)


def rank_and_save(classification: pd.DataFrame, regression: pd.DataFrame, amendment_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    class_bytes_complete = bool(classification.model_bytes_complete.all())
    class_key = lambda row: (-row.mean_inner_macro_f1_across_outer_tasks, row.sample_sd_outer_task_mean_macro_f1, -row.mean_inner_balanced_accuracy_across_outer_tasks, row.mean_inner_severe_error_rate_across_outer_tasks, row.mean_inner_measured_runtime_seconds, row.mean_model_bytes if class_bytes_complete else 0.0, row.dimension, row.canonical_variant_name)
    class_order = sorted((row for row in classification.itertuples(index=False)), key=class_key); class_rank = {f"{row.variant}|{row.dimension}": rank for rank, row in enumerate(class_order, 1)}
    classification["ranking_order"] = classification.apply(lambda row: class_rank[f"{row.variant}|{row.dimension}"], axis=1); classification["selected"] = classification.ranking_order == 1; classification["selection_reason"] = classification.ranking_order.map(lambda value: "SELECTED_BY_AMENDMENT_V2_PRIMARY_AND_TIEBREAK_ORDER" if value == 1 else "LOWER_DETERMINISTIC_RANK")
    reg_bytes_complete = bool(regression.model_bytes_complete.all()); rmse_complete = bool(regression.rmse_complete.all())
    # Phase 05 inner-selection records did not persist RMSE. It is a downstream tie-break and is not invoked because priorities 1-2 uniquely rank the winner.
    primary_order = regression.sort_values(["mean_inner_bounded_mae", "sample_sd_outer_fold_mean_mae"], kind="mergesort")
    winner_primary = primary_order.iloc[0]
    winner_unique_before_rmse = int(((regression.mean_inner_bounded_mae == winner_primary.mean_inner_bounded_mae) & (regression.sample_sd_outer_fold_mean_mae == winner_primary.sample_sd_outer_fold_mean_mae)).sum()) == 1
    if not rmse_complete and not winner_unique_before_rmse:
        raise RuntimeError("Regression winner is not unique before the unavailable RMSE tie-break")
    reg_key = lambda row: (row.mean_inner_bounded_mae, row.sample_sd_outer_fold_mean_mae, row.mean_inner_bounded_rmse if rmse_complete else 0.0, row.mean_inner_measured_runtime_seconds, row.mean_model_bytes if reg_bytes_complete else 0.0, row.dimension, row.canonical_head_family_name)
    reg_order = sorted((row for row in regression.itertuples(index=False)), key=reg_key); reg_rank = {f"{row.head_family}|{row.dimension}": rank for rank, row in enumerate(reg_order, 1)}
    regression["ranking_order"] = regression.apply(lambda row: reg_rank[f"{row.head_family}|{row.dimension}"], axis=1); regression["selected"] = regression.ranking_order == 1; regression["selection_reason"] = regression.ranking_order.map(lambda value: "SELECTED_BY_AMENDMENT_V2_PRIMARY_AND_TIEBREAK_ORDER" if value == 1 else "LOWER_DETERMINISTIC_RANK"); regression["rmse_tiebreak_status"] = "AVAILABLE" if rmse_complete else "NOT_INVOKED_INCOMPLETE_PHASE05_INNER_RECORDS"
    classification["pareto"] = nondominated(classification, "mean_inner_macro_f1_across_outer_tasks", True); regression["pareto"] = nondominated(regression, "mean_inner_bounded_mae", False)
    stable_csv(PHASE / "results/summaries/phase06_inner_only_classification_selection_trace.csv", classification.sort_values("ranking_order"))
    stable_csv(PHASE / "results/summaries/phase06_inner_only_regression_selection_trace.csv", regression.sort_values("ranking_order"))
    pareto = pd.concat([classification.assign(candidate_family=classification.variant + "|" + classification.dimension.astype(str), primary_metric="inner_macro_f1", primary_value=classification.mean_inner_macro_f1_across_outer_tasks)[["task", "candidate_family", "dimension", "primary_metric", "primary_value", "mean_inner_measured_runtime_seconds", "mean_model_bytes", "pareto"]], regression.assign(candidate_family=regression.head_family + "|" + regression.dimension.astype(str), primary_metric="inner_bounded_mae", primary_value=regression.mean_inner_bounded_mae)[["task", "candidate_family", "dimension", "primary_metric", "primary_value", "mean_inner_measured_runtime_seconds", "mean_model_bytes", "pareto"]]], ignore_index=True)
    pareto["model_bytes_used"] = class_bytes_complete and reg_bytes_complete
    pareto["memory_status"] = "MEMORY_USED_AS_LATE_TIEBREAK_COMPLETE_COVERAGE" if class_bytes_complete and reg_bytes_complete else "MEMORY_NOT_USED_IN_SELECTION_INCOMPLETE_COVERAGE"
    stable_csv(PHASE / "results/summaries/phase06_inner_only_model_selection_pareto.csv", pareto.sort_values(["task", "candidate_family", "dimension"], kind="mergesort"))
    best_c = classification[classification.selected].iloc[0]; best_r = regression[regression.selected].iloc[0]
    best_class = {"phase": "06", "selection_rule": "INNER_CV_ONLY_POST_FREEZE_AMENDMENT_V2", "selection_amendment_sha256": amendment_hash, "selection_evidence": "INNER_CV_ONLY", "selected_variant": best_c.variant, "selected_variant_name": best_c.canonical_variant_name, "selected_fixed_dimension": int(best_c.dimension), "levels": 51, "feature_k": 50, "structure_selection_policy": "fold-local inner-CV selection under the frozen Quick Screen grid", "seed_policy": "frozen evaluation seeds; no single seed selected", "single_seed_selected": False, "candidate_family_count": 8, "inner_evidence": {"mean_macro_f1": float(best_c.mean_inner_macro_f1_across_outer_tasks), "sample_sd_outer_task_mean_macro_f1": float(best_c.sample_sd_outer_task_mean_macro_f1), "mean_balanced_accuracy": float(best_c.mean_inner_balanced_accuracy_across_outer_tasks), "mean_severe_error_rate": float(best_c.mean_inner_severe_error_rate_across_outer_tasks), "mean_runtime_seconds": float(best_c.mean_inner_measured_runtime_seconds), "mean_model_bytes": float(best_c.mean_model_bytes), "pareto": bool(best_c.pareto)}, "fold_selected_structures": json.loads(best_c.fold_selected_structures_json)}
    best_reg = {"phase": "06", "selection_rule": "INNER_CV_ONLY_POST_FREEZE_AMENDMENT_V2", "selection_amendment_sha256": amendment_hash, "selection_evidence": "INNER_CV_ONLY", "selected_variant": best_r.variant, "selected_head_family": best_r.head_family, "selected_regression_head": best_r.regression_head, "selected_fixed_dimension": int(best_r.dimension), "levels": 51, "feature_k": 50, "parameter_policy": "fold-local inner-CV temperature or Ridge alpha selection", "seed_policy": "frozen evaluation seeds; no single seed selected", "single_seed_selected": False, "candidate_family_count": 20, "common_ridge_copies": 0, "inner_evidence": {"mean_bounded_mae": float(best_r.mean_inner_bounded_mae), "sample_sd_outer_fold_mean_mae": float(best_r.sample_sd_outer_fold_mean_mae), "mean_bounded_rmse": None if pd.isna(best_r.mean_inner_bounded_rmse) else float(best_r.mean_inner_bounded_rmse), "rmse_tiebreak_status": best_r.rmse_tiebreak_status, "mean_runtime_seconds": float(best_r.mean_inner_measured_runtime_seconds), "mean_model_bytes": float(best_r.mean_model_bytes), "pareto": bool(best_r.pareto)}, "fold_parameter_policy": json.loads(best_r.fold_parameter_policy_json)}
    stable_json(PHASE / "configs/phase06_best_classification_hdc.json", best_class); stable_json(PHASE / "configs/phase06_best_regression_hdc.json", best_reg)
    return best_class, best_reg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--probe-input", type=Path)
    arguments = parser.parse_args()
    if arguments.probe_input is not None: assert_allowed(arguments.probe_input); return 0
    amendment = read_json(AMENDMENT); read_json(ORIGINAL_RULE)
    if amendment.get("status") != "INNER_CV_ONLY_POST_FREEZE_AMENDMENT": raise RuntimeError("Amendment status invalid")
    classification, _ = classification_candidates(); regression, _ = regression_candidates()
    if len(classification) != 8 or len(regression) != 20: raise RuntimeError("Candidate-family count mismatch")
    best_class, best_reg = rank_and_save(classification, regression, sha256(AMENDMENT))
    output_paths = ["configs/phase06_best_classification_hdc.json", "configs/phase06_best_regression_hdc.json", "results/summaries/phase06_inner_only_classification_selection_trace.csv", "results/summaries/phase06_inner_only_regression_selection_trace.csv", "results/summaries/phase06_inner_only_model_selection_pareto.csv"]
    rmse_complete = bool(regression.rmse_complete.all())
    primary_order = regression.sort_values(["mean_inner_bounded_mae", "sample_sd_outer_fold_mean_mae"], kind="mergesort")
    winner_primary = primary_order.iloc[0]
    winner_unique_before_rmse = int(((regression.mean_inner_bounded_mae == winner_primary.mean_inner_bounded_mae) & (regression.sample_sd_outer_fold_mean_mae == winner_primary.sample_sd_outer_fold_mean_mae)).sum()) == 1
    audit = {"phase": "06", "audit": "inner_only_selection_isolation", "selector": "scripts/select_phase06_models_from_inner_evidence.py", "selector_sha256": sha256(Path(__file__)), "input_allowlist": ALLOWLIST, "actual_read_paths": sorted(READ_PATHS), "forbidden_path_attempts": FORBIDDEN_ATTEMPTS, "outer_oof_read_by_selector": False, "outer_prediction_read_by_selector": False, "outer_metric_read_by_selector": False, "training_calls": 0, "prediction_calls": 0, "classification_candidate_families": len(classification), "regression_candidate_families": len(regression), "single_seed_selected": False, "regression_rmse_tiebreak_status": "AVAILABLE" if rmse_complete else "NOT_INVOKED_INCOMPLETE_PHASE05_INNER_RECORDS", "regression_winner_unique_before_rmse_tiebreak": winner_unique_before_rmse, "runtime_comparability_limitation": "Phase 05 persists shared inner-selection time whereas Phase 06 persists selected inner-split model time; runtime is therefore a late deterministic tie-break with explicitly heterogeneous scope.", "outputs": output_paths, "best_classification": {"variant": best_class["selected_variant"], "dimension": best_class["selected_fixed_dimension"]}, "best_regression": {"head_family": best_reg["selected_head_family"], "dimension": best_reg["selected_fixed_dimension"]}, "result": "PASS"}
    stable_json(PHASE / "audits/phase06_inner_only_selection_isolation_audit.json", audit)
    print(f"INNER-ONLY SELECTION PASS: classification={best_class['selected_variant']} d={best_class['selected_fixed_dimension']}; regression={best_reg['selected_head_family']} d={best_reg['selected_fixed_dimension']}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
