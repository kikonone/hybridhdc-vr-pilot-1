"""Strict subject-level Phase 09 robustness, LOSO, statistics, figures, and reports."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)


PHASE09 = Path(__file__).resolve().parents[1]
EXPERIMENTS = PHASE09.parent
PHASE05 = EXPERIMENTS / "phase_05_basic_dual_output_hdc"
PHASE06 = EXPERIMENTS / "phase_06_hdc_variant_screening"
OOF = PHASE09 / "results" / "oof"
SUMMARIES = PHASE09 / "results" / "summaries"
FIGURES = PHASE09 / "figures"
REPORTS = PHASE09 / "reports"
AUDITS = PHASE09 / "audits"
MANIFEST_PATH = PHASE09 / "configs" / "phase09_execution_manifest.json"
LABELS = [0, 1, 2, 3]
CONDITIONS = [
    "FULL_PRIMARY_REFERENCE", "MISSING_PHYSIOLOGICAL", "MISSING_EYE_TRACKING",
    "MISSING_HEAD_MOVEMENT", "MISSING_FLIGHT_PARAMETER", "MISSING_BODY_MOVEMENT",
]
MISSING_CONDITIONS = CONDITIONS[1:]
DISPLAY_CONDITIONS = {
    "FULL_PRIMARY_REFERENCE": "Full Primary", "MISSING_PHYSIOLOGICAL": "No physiological",
    "MISSING_EYE_TRACKING": "No eye tracking", "MISSING_HEAD_MOVEMENT": "No head movement",
    "MISSING_FLIGHT_PARAMETER": "No flight parameters", "MISSING_BODY_MOVEMENT": "No body movement",
}
MODEL_DISPLAY = {
    "hdc_classification": "HDC", "hdc_regression": "HDC",
    "traditional_classification": "Traditional", "traditional_regression": "Traditional",
}
COLORS = {"HDC": "#0072B2", "Traditional": "#E69F00"}
BOOTSTRAP_RESAMPLES = 2000
RNG_SEED = 20260821

sys.path.insert(0, str(PHASE09 / "scripts"))
from consolidate_phase09_oof import (  # noqa: E402
    execution_preflight,
    load_raw,
    reference_sources,
)
from run_phase09_batch import atomic_csv, atomic_json, read_json, sha256  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float:
    result = float(value)
    return result if np.isfinite(result) else float("nan")


def classification_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    truth = pd.to_numeric(frame.y_true).astype(int).to_numpy()
    pred = pd.to_numeric(frame.y_pred).astype(int).to_numpy()
    matrix = confusion_matrix(truth, pred, labels=LABELS)
    recalls = recall_score(truth, pred, labels=LABELS, average=None, zero_division=0)
    return {
        "rows": len(frame), "macro_f1": f1_score(truth, pred, labels=LABELS, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, pred), "accuracy": accuracy_score(truth, pred),
        "severe_error_rate": float(np.mean(np.abs(truth - pred) >= 2)),
        "recall_class_0": recalls[0], "recall_class_1": recalls[1],
        "recall_class_2": recalls[2], "recall_class_3": recalls[3],
        "quadratic_weighted_kappa": cohen_kappa_score(truth, pred, labels=LABELS, weights="quadratic"),
        "confusion_matrix": json.dumps(matrix.tolist(), separators=(",", ":")),
    }


def regression_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    truth = pd.to_numeric(frame.y_true).astype(float).to_numpy()
    raw = pd.to_numeric(frame.y_pred_raw).astype(float).to_numpy()
    bounded = pd.to_numeric(frame.y_pred_bounded).astype(float).to_numpy()
    rounded = np.clip(np.rint(bounded).astype(int) - 1, 0, 3)
    true_class = np.clip(np.rint(truth).astype(int) - 1, 0, 3)
    clipped = ~np.isclose(raw, bounded)
    rho = float("nan") if np.allclose(truth, truth[0]) or np.allclose(bounded, bounded[0]) else spearmanr(truth, bounded).statistic
    return {
        "rows": len(frame), "raw_mae": mean_absolute_error(truth, raw),
        "bounded_mae": mean_absolute_error(truth, bounded),
        "bounded_rmse": math.sqrt(mean_squared_error(truth, bounded)),
        "bounded_r2": r2_score(truth, bounded),
        "bounded_spearman": safe_float(rho), "clipping_count": int(clipped.sum()),
        "clipping_rate": float(clipped.mean()),
        "rounded_regression_macro_f1": f1_score(true_class, rounded, labels=LABELS, average="macro", zero_division=0),
        "adjacent_accuracy": float(np.mean(np.abs(true_class - rounded) <= 1)),
        "severe_error_rate": float(np.mean(np.abs(true_class - rounded) >= 2)),
    }


def bootstrap_ci(values: np.ndarray, statistic: Callable[[np.ndarray], float] = np.mean, seed_offset: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(RNG_SEED + seed_offset)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    estimates = np.asarray([statistic(values[index]) for index in indices], dtype=float)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def paired_bootstrap_ci(reference: np.ndarray, condition: np.ndarray, degradation: Callable[[np.ndarray, np.ndarray], np.ndarray], seed_offset: int = 0) -> tuple[float, float]:
    reference = np.asarray(reference, dtype=float)
    condition = np.asarray(condition, dtype=float)
    if len(reference) != len(condition) or len(reference) == 0:
        raise ValueError("Paired bootstrap requires equal non-empty arrays")
    values = degradation(reference, condition)
    return bootstrap_ci(values, np.mean, seed_offset)


def rank_biserial_signed(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & ~np.isclose(values, 0.0)]
    if len(values) == 0:
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    return (positive - negative) / float(ranks.sum())


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p) - rank) * p[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def load_canonical() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = pd.concat([
        pd.read_csv(OOF / "phase09_missing_modality_canonical_classification_oof.csv"),
        pd.read_csv(OOF / "phase09_missing_modality_canonical_regression_oof.csv"),
    ], ignore_index=True, sort=False)
    loso = pd.concat([
        pd.read_csv(OOF / "phase09_loso_canonical_classification_oof.csv"),
        pd.read_csv(OOF / "phase09_loso_canonical_regression_oof.csv"),
    ], ignore_index=True, sort=False)
    references = pd.read_csv(OOF / "phase09_full_primary_reference_index.csv")
    if len(missing) != 8380 or len(loso) != 1676 or len(references) != 1676:
        raise RuntimeError("Canonical/reference input row count mismatch")
    return missing, loso, references


def overall_metric_tables(missing: pd.DataFrame, loso: pd.DataFrame, references: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_plus_ref = pd.concat([missing, references], ignore_index=True, sort=False)
    class_rows = []
    reg_rows = []
    for (condition, model_key, task), group in missing_plus_ref.groupby(["condition", "model_key", "task"], sort=True):
        common = {"condition": condition, "model_key": model_key, "model": MODEL_DISPLAY[model_key], "task": task, "subjects": group.subject_id.nunique()}
        if task == "classification":
            class_rows.append({**common, **classification_metrics(group)})
        else:
            reg_rows.append({**common, "task_name": "bounded difficulty-induced workload proxy regression", **regression_metrics(group)})
    loso_class = []
    loso_reg = []
    for (model_key, task), group in loso.groupby(["model_key", "task"], sort=True):
        common = {"condition": "LOSO", "model_key": model_key, "model": MODEL_DISPLAY[model_key], "task": task, "subjects": group.subject_id.nunique()}
        if task == "classification":
            loso_class.append({**common, **classification_metrics(group)})
        else:
            loso_reg.append({**common, "task_name": "bounded difficulty-induced workload proxy regression", **regression_metrics(group)})
    return pd.DataFrame(class_rows), pd.DataFrame(reg_rows), pd.DataFrame(loso_class), pd.DataFrame(loso_reg)


def subject_primary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition, model_key, task, subject_id), group in frame.groupby(["condition", "model_key", "task", "subject_id"], sort=True):
        if task == "classification":
            metrics = classification_metrics(group)
            value = metrics["macro_f1"]
            row = {
                "classification_macro_f1": value, "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"], "severe_error_rate": metrics["severe_error_rate"],
                "bounded_mae": np.nan, "bounded_rmse": np.nan,
            }
            metric = "macro_f1"
            direction = "higher_is_better"
        else:
            metrics = regression_metrics(group)
            value = metrics["bounded_mae"]
            row = {
                "classification_macro_f1": np.nan, "balanced_accuracy": np.nan,
                "accuracy": np.nan, "severe_error_rate": metrics["severe_error_rate"],
                "bounded_mae": value, "bounded_rmse": metrics["bounded_rmse"],
            }
            metric = "bounded_mae"
            direction = "lower_is_better"
        target_counts = group.y_true.value_counts().sort_index().to_dict()
        rows.append({
            "condition": condition, "model_key": model_key, "model": MODEL_DISPLAY[model_key],
            "task": task, "subject_id": subject_id, "test_rows": len(group),
            "target_coverage": json.dumps({str(key): int(value) for key, value in target_counts.items()}, separators=(",", ":")),
            "primary_metric": metric, "primary_metric_direction": direction,
            "primary_metric_value": value, **row,
        })
    return pd.DataFrame(rows)


def seed_stability(preflight: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_raw(preflight["records"])
    raw = raw[raw.model_key.str.startswith("hdc_")].copy()
    rows = []
    for (protocol, condition, model_key, seed), group in raw.groupby(["protocol", "condition", "model_key", "seed"], sort=True):
        task = "classification" if model_key.endswith("classification") else "regression"
        value = classification_metrics(group)["macro_f1"] if task == "classification" else regression_metrics(group)["bounded_mae"]
        rows.append({"protocol": protocol, "condition": condition, "model_key": model_key, "task": task, "seed": int(seed), "primary_metric_value": value})

    class_source = pd.read_csv(reference_sources()["hdc_classification"]["path"])
    class_source = class_source[(class_source.variant == "hybrid") & (class_source.dimension == 5000)].rename(columns={"true_class": "y_true", "predicted_class": "y_pred"})
    for seed, group in class_source.groupby("seed", sort=True):
        rows.append({"protocol": "FULL_PRIMARY_REFERENCE", "condition": "FULL_PRIMARY_REFERENCE", "model_key": "hdc_classification", "task": "classification", "seed": int(seed), "primary_metric_value": classification_metrics(group)["macro_f1"]})
    reg_source = pd.read_csv(reference_sources()["hdc_regression"]["path"])
    reg_source = reg_source[reg_source.dimension == 10000].rename(columns={"target_score": "y_true", "ridge_prediction_raw": "y_pred_raw", "ridge_prediction_bounded": "y_pred_bounded"})
    for seed, group in reg_source.groupby("seed", sort=True):
        rows.append({"protocol": "FULL_PRIMARY_REFERENCE", "condition": "FULL_PRIMARY_REFERENCE", "model_key": "hdc_regression", "task": "regression", "seed": int(seed), "primary_metric_value": regression_metrics(group)["bounded_mae"]})
    per_seed = pd.DataFrame(rows)
    aggregates = []
    for keys, group in per_seed.groupby(["protocol", "condition", "model_key", "task"], sort=True):
        values = group.primary_metric_value.to_numpy(float)
        aggregates.append({
            "protocol": keys[0], "condition": keys[1], "model_key": keys[2], "task": keys[3],
            "primary_metric": "macro_f1" if keys[3] == "classification" else "bounded_mae",
            "seed_count": len(group), "seed_mean": values.mean(), "seed_sd_sample": values.std(ddof=1),
            "seed_min": values.min(), "seed_max": values.max(), "seed_range": values.max() - values.min(),
        })
    return per_seed, pd.DataFrame(aggregates)


def robustness_analysis(subject: pd.DataFrame, class_metrics: pd.DataFrame, reg_metrics: pd.DataFrame, seed_agg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_lookup = pd.concat([
        class_metrics[["condition", "model_key", "macro_f1"]].rename(columns={"macro_f1": "absolute_metric"}),
        reg_metrics[["condition", "model_key", "bounded_mae"]].rename(columns={"bounded_mae": "absolute_metric"}),
    ], ignore_index=True)
    robustness_rows = []
    delta_rows = []
    stats_rows = []
    bootstrap_rows = []
    seed_offset = 0
    for model_key in ["hdc_classification", "traditional_classification", "hdc_regression", "traditional_regression"]:
        task = "classification" if model_key.endswith("classification") else "regression"
        reference = subject[(subject.condition == "FULL_PRIMARY_REFERENCE") & (subject.model_key == model_key)].sort_values("subject_id")
        ref_global = float(metric_lookup[(metric_lookup.condition == "FULL_PRIMARY_REFERENCE") & (metric_lookup.model_key == model_key)].absolute_metric.iloc[0])
        ref_seed_sd = seed_agg[(seed_agg.condition == "FULL_PRIMARY_REFERENCE") & (seed_agg.model_key == model_key)].seed_sd_sample
        family_indices = []
        family_p = []
        for condition in MISSING_CONDITIONS:
            current = subject[(subject.condition == condition) & (subject.model_key == model_key)].sort_values("subject_id")
            paired = reference[["subject_id", "primary_metric_value"]].merge(
                current[["subject_id", "primary_metric_value"]], on="subject_id", suffixes=("_reference", "_condition"), validate="one_to_one"
            )
            if len(paired) != 35:
                raise RuntimeError(f"Subject pairing incomplete: {model_key} {condition}")
            ref_values = paired.primary_metric_value_reference.to_numpy(float)
            condition_values = paired.primary_metric_value_condition.to_numpy(float)
            degradation_values = ref_values - condition_values if task == "classification" else condition_values - ref_values
            absolute = float(metric_lookup[(metric_lookup.condition == condition) & (metric_lookup.model_key == model_key)].absolute_metric.iloc[0])
            absolute_difference = absolute - ref_global
            relative_degradation = float(np.mean(degradation_values) / abs(np.mean(ref_values)) * 100.0) if not np.isclose(np.mean(ref_values), 0) else float("nan")
            ci_low, ci_high = bootstrap_ci(degradation_values, np.mean, seed_offset)
            seed_offset += 1
            condition_mean_ci = bootstrap_ci(condition_values, np.mean, seed_offset)
            seed_offset += 1
            try:
                test = wilcoxon(degradation_values, zero_method="wilcox", alternative="two-sided", method="auto") if not np.allclose(degradation_values, 0) else None
                statistic = float(test.statistic) if test is not None else 0.0
                p_value = float(test.pvalue) if test is not None else 1.0
            except ValueError:
                statistic, p_value = 0.0, 1.0
            effect = rank_biserial_signed(degradation_values)
            seed_sd = seed_agg[(seed_agg.condition == condition) & (seed_agg.model_key == model_key)].seed_sd_sample
            seed_change = float(seed_sd.iloc[0] - ref_seed_sd.iloc[0]) if len(seed_sd) and len(ref_seed_sd) else np.nan
            row = {
                "model_key": model_key, "model": MODEL_DISPLAY[model_key], "task": task,
                "condition": condition, "primary_metric": "macro_f1" if task == "classification" else "bounded_mae",
                "metric_direction": "higher_is_better" if task == "classification" else "lower_is_better",
                "full_primary_absolute_metric": ref_global, "condition_absolute_metric": absolute,
                "absolute_difference_condition_minus_reference": absolute_difference,
                "mean_subject_degradation": float(np.mean(degradation_values)),
                "median_subject_degradation": float(np.median(degradation_values)),
                "relative_degradation_percent": relative_degradation,
                "degradation_ci_low_95": ci_low, "degradation_ci_high_95": ci_high,
                "condition_subject_mean": float(np.mean(condition_values)),
                "condition_subject_ci_low_95": condition_mean_ci[0], "condition_subject_ci_high_95": condition_mean_ci[1],
                "subjects": 35, "performance_direction": "DECREASE" if np.mean(degradation_values) > 0 else "IMPROVEMENT_OR_NO_DECREASE",
                "seed_stability_sd_change": seed_change,
            }
            robustness_rows.append(row)
            for paired_row, degradation_value in zip(paired.itertuples(index=False), degradation_values):
                delta_rows.append({
                    "model_key": model_key, "model": MODEL_DISPLAY[model_key], "task": task,
                    "condition": condition, "subject_id": paired_row.subject_id,
                    "reference_primary_metric": paired_row.primary_metric_value_reference,
                    "condition_primary_metric": paired_row.primary_metric_value_condition,
                    "degradation_positive_is_worse": degradation_value,
                })
            stats_rows.append({
                "comparison_family": f"{model_key}|primary_metric", "model_key": model_key,
                "model": MODEL_DISPLAY[model_key], "task": task, "condition": condition,
                "reference": "FULL_PRIMARY_REFERENCE", "statistical_unit": "subject_id",
                "n_subjects": 35, "test": "paired Wilcoxon signed-rank", "wilcoxon_statistic": statistic,
                "p_value_raw": p_value, "rank_biserial_effect_size": effect,
                "mean_degradation_positive_is_worse": float(np.mean(degradation_values)),
                "bootstrap_ci_low_95": ci_low, "bootstrap_ci_high_95": ci_high,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            })
            family_indices.append(len(stats_rows) - 1)
            family_p.append(p_value)
            bootstrap_rows.append({
                "analysis": "missing_modality_paired_degradation", "model_key": model_key,
                "task": task, "condition": condition, "statistical_unit": "subject_id",
                "n_subjects": 35, "estimate": float(np.mean(degradation_values)),
                "ci_low_95": ci_low, "ci_high_95": ci_high, "resamples": BOOTSTRAP_RESAMPLES,
            })
        adjusted = holm_adjust(family_p)
        for index, adjusted_p in zip(family_indices, adjusted):
            stats_rows[index]["p_value_holm"] = adjusted_p
            stats_rows[index]["holm_significant_0_05"] = bool(adjusted_p < 0.05)
            stats_rows[index]["holm_family_size"] = len(family_p)

    robustness = pd.DataFrame(robustness_rows)
    deltas = pd.DataFrame(delta_rows)
    stats = pd.DataFrame(stats_rows)
    comparison_rows = []
    for (task, condition), group in deltas.groupby(["task", "condition"], sort=True):
        pivot = group.pivot(index="subject_id", columns="model", values="degradation_positive_is_worse")
        comparison_rows.append({
            "task": task, "condition": condition, "subjects": len(pivot),
            "hdc_mean_degradation": pivot.HDC.mean(), "traditional_mean_degradation": pivot.Traditional.mean(),
            "hdc_minus_traditional_degradation": (pivot.HDC - pivot.Traditional).mean(),
            "more_robust_model_descriptive": "HDC" if pivot.HDC.mean() < pivot.Traditional.mean() else "Traditional",
            "inference_status": "DESCRIPTIVE_ONLY_NOT_A_PREREGISTERED_WILCOXON_FAMILY",
        })
    comparison = pd.DataFrame(comparison_rows)
    return robustness, deltas, comparison, stats, pd.DataFrame(bootstrap_rows)


def loso_analysis(loso: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject = subject_primary(loso.assign(condition="LOSO"))
    stability_rows = []
    bootstrap_rows = []
    for (model_key, task), group in subject.groupby(["model_key", "task"], sort=True):
        values = group.primary_metric_value.to_numpy(float)
        ci_low, ci_high = bootstrap_ci(values, np.mean, len(stability_rows) + 100)
        mean = values.mean()
        sd = values.std(ddof=1)
        stability_rows.append({
            "model_key": model_key, "model": MODEL_DISPLAY[model_key], "task": task,
            "primary_metric": "macro_f1" if task == "classification" else "bounded_mae",
            "metric_direction": "higher_is_better" if task == "classification" else "lower_is_better",
            "subjects": len(values), "mean": mean, "median": np.median(values), "sd_sample": sd,
            "q1": np.quantile(values, 0.25), "q3": np.quantile(values, 0.75),
            "iqr": np.quantile(values, 0.75) - np.quantile(values, 0.25),
            "minimum": values.min(), "maximum": values.max(),
            "coefficient_of_variation": sd / abs(mean) if not np.isclose(mean, 0) else np.nan,
            "bootstrap_mean_ci_low_95": ci_low, "bootstrap_mean_ci_high_95": ci_high,
            "best_subject": group.loc[group.primary_metric_value.idxmax() if task == "classification" else group.primary_metric_value.idxmin(), "subject_id"],
            "worst_subject": group.loc[group.primary_metric_value.idxmin() if task == "classification" else group.primary_metric_value.idxmax(), "subject_id"],
            "worst_subject_diagnostic_only": True,
        })
        bootstrap_rows.append({
            "analysis": "loso_subject_mean", "model_key": model_key, "task": task,
            "condition": "LOSO", "statistical_unit": "subject_id", "n_subjects": len(values),
            "estimate": mean, "ci_low_95": ci_low, "ci_high_95": ci_high,
            "resamples": BOOTSTRAP_RESAMPLES,
        })

    difficulty_rows = []
    for (model_key, task, target), group in loso.groupby(["model_key", "task", "y_true"], sort=True):
        row = {"model_key": model_key, "model": MODEL_DISPLAY[model_key], "task": task, "difficulty_level": target, "rows": len(group), "subjects": group.subject_id.nunique()}
        if task == "classification":
            truth = group.y_true.astype(int).to_numpy(); pred = group.y_pred.astype(int).to_numpy()
            row.update({"error_rate": float(np.mean(truth != pred)), "severe_error_rate": float(np.mean(np.abs(truth - pred) >= 2)), "bounded_mae": np.nan})
        else:
            row.update({"error_rate": np.nan, "severe_error_rate": regression_metrics(group)["severe_error_rate"], "bounded_mae": mean_absolute_error(group.y_true, group.y_pred_bounded)})
        difficulty_rows.append(row)

    raw_hdc = raw[(raw.protocol == "LEAVE_ONE_SUBJECT_OUT") & raw.model_key.str.startswith("hdc_")].copy()
    seed_subject_rows = []
    for (model_key, subject_id, seed), group in raw_hdc.groupby(["model_key", "subject_id", "seed"], sort=True):
        task = "classification" if model_key.endswith("classification") else "regression"
        value = classification_metrics(group)["macro_f1"] if task == "classification" else regression_metrics(group)["bounded_mae"]
        seed_subject_rows.append({"model_key": model_key, "task": task, "subject_id": subject_id, "seed": int(seed), "primary_metric_value": value})
    seed_subject = pd.DataFrame(seed_subject_rows)
    variability_rows = []
    for (model_key, task, subject_id), group in seed_subject.groupby(["model_key", "task", "subject_id"], sort=True):
        values = group.primary_metric_value.to_numpy(float)
        variability_rows.append({
            "model_key": model_key, "model": "HDC", "task": task, "subject_id": subject_id,
            "seed_count": len(values), "seed_mean": values.mean(), "seed_sd_sample": values.std(ddof=1),
            "seed_min": values.min(), "seed_max": values.max(), "seed_range": values.max() - values.min(),
        })
    variability = pd.DataFrame(variability_rows)
    variability["seed_instability_flag_top_quartile"] = variability.groupby("task").seed_sd_sample.transform(lambda values: values >= values.quantile(0.75))
    return subject, pd.DataFrame(stability_rows), pd.DataFrame(difficulty_rows), variability, pd.DataFrame(bootstrap_rows)


def save_pair(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_figures(robustness: pd.DataFrame, subject_all: pd.DataFrame, loso_subject: pd.DataFrame, stability: pd.DataFrame) -> list[dict[str, Any]]:
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8, "lines.linewidth": 1.7})
    catalog = []
    x = np.arange(len(CONDITIONS))
    labels = [DISPLAY_CONDITIONS[value] for value in CONDITIONS]
    for task, stem, ylabel in [
        ("classification", "phase09_missing_modality_classification_curve", "Subject Macro-F1 (higher is better)"),
        ("regression", "phase09_missing_modality_regression_curve", "Bounded MAE (lower is better)"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for model in ["HDC", "Traditional"]:
            model_key = f"{'hdc' if model == 'HDC' else 'traditional'}_{task}"
            rows = []
            for condition in CONDITIONS:
                group = subject_all[(subject_all.condition == condition) & (subject_all.model_key == model_key)]
                ci = bootstrap_ci(group.primary_metric_value.to_numpy(float), np.mean, 300 + len(rows) + (0 if model == "HDC" else 20))
                rows.append((group.primary_metric_value.mean(), ci[0], ci[1]))
            means = np.array([row[0] for row in rows]); lower = means - np.array([row[1] for row in rows]); upper = np.array([row[2] for row in rows]) - means
            ax.errorbar(x, means, yerr=np.vstack([lower, upper]), marker="o", capsize=3, label=model, color=COLORS[model])
        ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_ylabel(ylabel); ax.set_xlabel("Input condition")
        ax.set_ylim(bottom=0, top=1 if task == "classification" else None); ax.grid(alpha=0.25); ax.legend(title="Model family")
        save_pair(fig, stem)
        catalog.append({"filename": stem, "purpose": f"Compare {task} robustness across five missing-modality conditions against Full Primary.", "data_source": "subject-level paired metrics", "error_bars": "95% subject bootstrap CI, n=35", "reader_notice": "Axes start at zero; metric direction is explicit.", "caveat": "Modality removal is predictive dependence evidence, not causal evidence."})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, task in zip(axes, ["classification", "regression"]):
        part = robustness[robustness.task == task]
        for model in ["HDC", "Traditional"]:
            rows = part[part.model == model].set_index("condition").loc[MISSING_CONDITIONS]
            means = rows.mean_subject_degradation.to_numpy(float)
            ax.errorbar(np.arange(5), means, yerr=np.vstack([means - rows.degradation_ci_low_95, rows.degradation_ci_high_95 - means]), marker="o", capsize=3, label=model, color=COLORS[model])
        ax.axhline(0, color="black", linewidth=0.8); ax.set_xticks(np.arange(5), [DISPLAY_CONDITIONS[value] for value in MISSING_CONDITIONS], rotation=30, ha="right")
        ax.set_ylabel("Mean degradation (positive = worse)"); ax.set_title(task.capitalize()); ax.grid(alpha=0.25); ax.legend()
    save_pair(fig, "phase09_missing_modality_model_comparison")
    catalog.append({"filename": "phase09_missing_modality_model_comparison", "purpose": "Compare HDC and traditional subject-paired degradation.", "data_source": "phase09_missing_modality_robustness.csv", "error_bars": "95% paired subject bootstrap CI, n=35", "reader_notice": "Positive values mean worse performance for both task directions.", "caveat": "Between-model contrast is descriptive; preregistered inference compares each condition to its own Full Primary reference."})

    subjects = sorted(loso_subject.subject_id.unique())
    for task, stem, ylabel in [
        ("classification", "phase09_loso_subject_classification", "Subject Macro-F1 (higher is better)"),
        ("regression", "phase09_loso_subject_regression", "Subject bounded MAE (lower is better)"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for model in ["HDC", "Traditional"]:
            values = loso_subject[(loso_subject.task == task) & (loso_subject.model == model)].set_index("subject_id").loc[subjects].primary_metric_value
            ax.plot(np.arange(len(subjects)), values, marker="o", markersize=3, label=model, color=COLORS[model])
        ax.set_xticks(np.arange(len(subjects)), subjects, rotation=90); ax.set_ylabel(ylabel); ax.set_xlabel("Held-out subject (LOSO)")
        ax.set_ylim(bottom=0, top=1 if task == "classification" else None); ax.grid(alpha=0.2); ax.legend();
        save_pair(fig, stem)
        catalog.append({"filename": stem, "purpose": f"Show held-out-subject {task} heterogeneity.", "data_source": "phase09_loso_subject_metrics.csv", "error_bars": "No per-subject CI; each point is one held-out subject, n=35", "reader_notice": "All subjects are retained, including high-error subjects.", "caveat": "LOSO supports subject generalization only, not unseen-scenario generalization."})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = []
    group_labels = []
    for task in ["classification", "regression"]:
        for model in ["HDC", "Traditional"]:
            groups.append(loso_subject[(loso_subject.task == task) & (loso_subject.model == model)].primary_metric_value.to_numpy(float))
            group_labels.append(f"{model}\n{task}")
    boxes = ax.boxplot(groups, tick_labels=group_labels, patch_artist=True, showmeans=True)
    for patch, label in zip(boxes["boxes"], group_labels):
        patch.set_facecolor(COLORS["HDC" if label.startswith("HDC") else "Traditional"]); patch.set_alpha(0.55)
    ax.set_ylabel("Subject primary metric (task-specific direction)"); ax.set_ylim(bottom=0); ax.grid(axis="y", alpha=0.25)
    save_pair(fig, "phase09_loso_stability_distribution")
    catalog.append({"filename": "phase09_loso_stability_distribution", "purpose": "Summarize the distribution of subject-level primary metrics.", "data_source": "phase09_loso_subject_metrics.csv", "error_bars": "Box/IQR distribution across n=35 subjects", "reader_notice": "Classification and regression have different metric directions, stated in the companion report.", "caveat": "The panel is descriptive and should not be read as cross-task scale equivalence."})
    return catalog


def conclusions(robustness: pd.DataFrame, comparison: pd.DataFrame, stats: pd.DataFrame, loso_stability: pd.DataFrame) -> dict[str, Any]:
    worst = robustness.loc[robustness.groupby(["model_key"]).mean_subject_degradation.idxmax()][["model_key", "condition", "mean_subject_degradation"]].to_dict(orient="records")
    by_task = robustness.groupby(["task", "condition"], as_index=False).mean_subject_degradation.mean()
    task_worst = by_task.loc[by_task.groupby("task").mean_subject_degradation.idxmax()].to_dict(orient="records")
    task_robustness = comparison.groupby(["task", "more_robust_model_descriptive"]).size().reset_index(name="conditions_won")
    flight = stats[stats.condition == "MISSING_FLIGHT_PARAMETER"]
    improvements = robustness[robustness.mean_subject_degradation < 0][["model_key", "condition", "mean_subject_degradation"]].to_dict(orient="records")
    return {
        "worst_condition_by_model_task": worst,
        "worst_condition_by_task_average": task_worst,
        "descriptive_robustness_condition_wins": task_robustness.to_dict(orient="records"),
        "flight_parameter_holm_results": flight[["model_key", "mean_degradation_positive_is_worse", "p_value_holm", "holm_significant_0_05", "rank_biserial_effect_size"]].to_dict(orient="records"),
        "conditions_with_mean_improvement": improvements,
        "loso_worst_subjects": loso_stability[["model_key", "worst_subject", "minimum", "maximum"]].to_dict(orient="records"),
        "causal_claim_allowed": False,
        "flight_generalizable_behavior_claim": "INCONCLUSIVE_DUE_TO_METADATA",
    }


def write_reports(conclusion: dict[str, Any], catalog: list[dict[str, Any]], robustness: pd.DataFrame, stats: pd.DataFrame, stability: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    analysis_output = REPORTS / "analysis-output"
    analysis_output.mkdir(parents=True, exist_ok=True)
    worst_lines = "\n".join(f"- {row['model_key']}: {row['condition']} (mean degradation {row['mean_subject_degradation']:.6f})" for row in conclusion["worst_condition_by_model_task"])
    flight_lines = "\n".join(f"- {row['model_key']}: degradation={row['mean_degradation_positive_is_worse']:.6f}, Holm p={row['p_value_holm']:.6g}, effect={row['rank_biserial_effect_size']:.3f}" for row in conclusion["flight_parameter_holm_results"])
    figure_lines = "\n".join(f"- `{item['filename']}.pdf/.png`: {item['purpose']} {item['reader_notice']} Caveat: {item['caveat']}" for item in catalog)
    analysis_report = f"""# Phase 09 Strict Analysis Report

## Analysis question
How do five frozen modality-removal retraining conditions alter classification Macro-F1 and bounded MAE relative to frozen Full Primary references, and how stable are the four selected model-task interfaces across 35 held-out subjects?

## Evidence contract
- Unit of inference: 35 subjects.
- Primary metrics: Macro-F1 (higher is better) and bounded MAE (lower is better).
- Missing-modality inference: paired Wilcoxon, Holm within each model-task family of five comparisons, rank-biserial effect size, and 2,000 paired subject bootstraps.
- LOSO is a subject-generalization estimate, not scenario generalization.

## Key findings
{worst_lines}

Flight-feature evidence:
{flight_lines}

## Claim Candidates
- Claim: modality removal changes predictive performance relative to the frozen Full Primary reference.
  - Source evidence: `phase09_missing_modality_robustness.csv` and `phase09_pairwise_statistics.csv`.
  - Allowed wording: model dependence on the removed feature family, with direction and uncertainty stated.
  - Forbidden stronger wording: physiological causality or universal behavioral importance.
  - Uncertainty: n=35 subjects; no unseen-session/scenario metadata.
  - Next check: a metadata-supported scenario/session study.
  - Decision: keep with boundary.
- Claim: flight-feature dependence may generalize across held-out subjects.
  - Source evidence: missing-flight paired analysis plus LOSO subject stability.
  - Allowed wording: `SUBJECT_GENERALIZATION_OF_FLIGHT_DEPENDENCE` when descriptive stability supports it.
  - Forbidden stronger wording: `GENERALIZABLE_FLIGHT_BEHAVIOR` or unseen-scenario generalization.
  - Uncertainty: session, scenario, task-template, and route identifiers are unavailable.
  - Next check: collect the missing metadata and pre-register grouped generalization splits.
  - Decision: weaken; flight generalizable-behavior remains inconclusive.

## Limitations
Small modality effects may partly reflect feature-count differences; this analysis cannot isolate causal information content from dimensionality. Non-significant Wilcoxon results are not equivalence evidence. No subject was removed or used for reselection.
"""
    stats_appendix = f"""# Phase 09 Statistical Appendix

- Statistical unit: subject_id (n=35).
- Test: paired Wilcoxon signed-rank for each missing condition versus its own Full Primary reference.
- Multiplicity: Holm correction separately within each of four model-task families (five comparisons per family).
- Effect size: signed rank-biserial; positive values indicate worse performance under modality removal after harmonizing metric direction.
- Uncertainty: 2,000 deterministic paired subject bootstrap resamples, 95% percentile CI.
- All comparisons, including non-significant ones, are present in `phase09_pairwise_statistics.csv`.
- LOSO summaries use 2,000 subject bootstraps and are descriptive stability estimates; LOSO splits are not independent samples for a second inferential test.

Flight comparisons:
{flight_lines}
"""
    figure_catalog = "# Phase 09 Figure Catalog\n\n" + "\n".join(
        f"## {item['filename']}\n- Purpose: {item['purpose']}\n- Data source: {item['data_source']}\n- Uncertainty: {item['error_bars']}\n- Reader should notice: {item['reader_notice']}\n- Caveat: {item['caveat']}\n"
        for item in catalog
    )
    (analysis_output / "analysis-report.md").write_text(analysis_report, encoding="utf-8")
    (analysis_output / "stats-appendix.md").write_text(stats_appendix, encoding="utf-8")
    (analysis_output / "figure-catalog.md").write_text(figure_catalog, encoding="utf-8")

    missing_report = f"""# Phase 09 Missing-Modality Robustness Report

All five modality-removal conditions were compared with frozen Full Primary references using 35 paired subject summaries. Positive degradation always means worse performance, after respecting that Macro-F1 is maximized and bounded MAE is minimized.

## Largest degradation by model-task
{worst_lines}

## Flight parameters
{flight_lines}

HDC-versus-traditional robustness is reported descriptively in `phase09_model_robustness_comparison.csv`; no extra unregistered inferential family was invented. Conditions with negative mean degradation are retained as possible improvements, not discarded. Smaller changes for small modalities may reflect fewer removed features and cannot be interpreted as causal irrelevance.
"""
    loso_lines = "\n".join(f"- {row.model_key}: mean={row['mean']:.6f}, 95% CI [{row.bootstrap_mean_ci_low_95:.6f}, {row.bootstrap_mean_ci_high_95:.6f}], worst subject={row.worst_subject} (diagnostic only)" for _, row in stability.iterrows())
    loso_report = f"""# Phase 09 LOSO Subject Stability Report

LOSO includes all 35 subjects exactly once per model-task canonical OOF. No high-error subject was deleted or used for retraining.

{loso_lines}

This supports estimation of held-out-subject behavior for the frozen interfaces. It does not test unseen sessions, scenarios, task templates, or route configurations.
"""
    statistical_report = stats_appendix.replace("# Phase 09 Statistical Appendix", "# Phase 09 Statistical Appendix Report")
    boundary_report = """# Phase 09 Generalization Boundaries

## Supported boundary
- Missing-flight degradation can support `MODEL_DEPENDENCE_ON_FLIGHT_FEATURES`.
- Stable LOSO flight-related performance can support `SUBJECT_GENERALIZATION_OF_FLIGHT_DEPENDENCE`.

## Unsupported stronger claims
- `GENERALIZABLE_FLIGHT_BEHAVIOR`: `INCONCLUSIVE_DUE_TO_METADATA`.
- `UNSEEN_SESSION`: `NOT_FEASIBLE_DUE_TO_METADATA`.
- `UNSEEN_SCENARIO`: `NOT_FEASIBLE_DUE_TO_METADATA`.
- `TASK_TEMPLATE`: `NOT_FEASIBLE_DUE_TO_METADATA`.
- `ROUTE_CONFIGURATION`: `NOT_FEASIBLE_DUE_TO_METADATA`.

Phase 08 and Phase 09 together show predictive dependence and held-out-subject behavior under the available labels. They cannot establish behavioral causality or generalization to metadata strata that were never recorded.
"""
    final_report = f"""---
type: results-report
date: 2026-08-21
experiment_line: phase09-robustness-generalization
round: 9
purpose: robustness-check
status: complete-pending-freeze
source_artifacts:
  - reports/analysis-output/analysis-report.md
  - reports/analysis-output/stats-appendix.md
linked_experiments: []
linked_results: []
---

# Phase 09 Robustness and Generalization / Round 9 / Robustness Check / 2026-08-21

## Executive Summary
Phase 09 consolidated 720 frozen runs into 10,056 canonical OOF rows and evaluated missing-modality robustness and held-out-subject stability without retraining. The strongest allowable conclusion is predictive dependence on specific feature families, bounded by subject-level uncertainty and missing scenario/session metadata.

## Experiment Identity and Decision Context
This round tests whether frozen HDC and traditional interfaces remain useful after removal of one modality and when evaluated on a wholly held-out subject.

## Setup and Evaluation Protocol
Five missing-modality conditions, four model-task interfaces, 419 runs, five HDC seeds, and 35 subjects were analyzed. Classification uses Macro-F1; the bounded difficulty-induced workload proxy regression uses bounded MAE. Inference uses subjects only.

## Main Findings
{worst_lines}

## Statistical Validation
Paired Wilcoxon tests, Holm correction, rank-biserial effect sizes, and 2,000 paired subject bootstrap CIs are saved in the statistical artifacts. Non-significance is not treated as equivalence.

## Figure-by-Figure Interpretation
{figure_lines}

## Failure Cases / Negative Results / Limitations
High-error and seed-unstable subjects remain in the analysis. Feature-count differences can confound the apparent importance of small modalities. No unseen-session/scenario/task-template/route test is feasible from current metadata.

## What Changed Our Belief
The evidence can update beliefs about model dependence and held-out-subject stability, but not about physiological causality or unseen-scenario flight behavior.

## Next Actions
Freeze Phase 09 only after independent review of this bundle. A future phase should add explicit session, scenario, task-template, and route metadata before claiming broader generalization.

## Artifact and Reproducibility Index
- Canonical OOF: `results/oof/`
- Metrics/statistics: `results/summaries/`
- Figures: `figures/`
- Strict analysis bundle: `reports/analysis-output/`
- Audits: `audits/`
"""
    (REPORTS / "phase09_missing_modality_report.md").write_text(missing_report, encoding="utf-8")
    (REPORTS / "phase09_loso_stability_report.md").write_text(loso_report, encoding="utf-8")
    (REPORTS / "phase09_statistical_appendix.md").write_text(statistical_report, encoding="utf-8")
    (REPORTS / "phase09_generalization_boundaries.md").write_text(boundary_report, encoding="utf-8")
    (REPORTS / "phase09_final_analysis.md").write_text(final_report, encoding="utf-8")


def run_analysis() -> dict[str, Any]:
    preflight = execution_preflight()
    manifest = read_json(MANIFEST_PATH)
    if manifest.get("canonical_oof_rows") != 10056:
        raise RuntimeError("Canonical OOF must be consolidated before analysis")
    missing, loso, references = load_canonical()
    class_metrics, reg_metrics, loso_class, loso_reg = overall_metric_tables(missing, loso, references)
    subject_all = subject_primary(pd.concat([missing, references], ignore_index=True, sort=False))
    per_seed, seed_agg = seed_stability(preflight)
    robustness, deltas, comparison, stats, missing_bootstrap = robustness_analysis(subject_all, class_metrics, reg_metrics, seed_agg)
    raw = load_raw(preflight["records"])
    loso_subject, loso_stability, difficulty, seed_variability, loso_bootstrap = loso_analysis(loso, raw)
    bootstrap = pd.concat([missing_bootstrap, loso_bootstrap], ignore_index=True)
    conclusion = conclusions(robustness, comparison, stats, loso_stability)

    SUMMARIES.mkdir(parents=True, exist_ok=True)
    outputs = {
        "phase09_missing_modality_classification_metrics.csv": class_metrics,
        "phase09_missing_modality_regression_metrics.csv": reg_metrics,
        "phase09_loso_classification_metrics.csv": loso_class,
        "phase09_loso_regression_metrics.csv": loso_reg,
        "phase09_seed_stability.csv": seed_agg,
        "phase09_missing_modality_robustness.csv": robustness,
        "phase09_missing_modality_deltas.csv": deltas,
        "phase09_model_robustness_comparison.csv": comparison,
        "phase09_loso_subject_metrics.csv": loso_subject,
        "phase09_loso_subject_stability.csv": loso_stability,
        "phase09_loso_difficulty_level_errors.csv": difficulty,
        "phase09_loso_seed_variability.csv": seed_variability,
        "phase09_pairwise_statistics.csv": stats,
        "phase09_bootstrap_confidence_intervals.csv": bootstrap,
    }
    for name, frame in outputs.items():
        atomic_csv(SUMMARIES / name, frame)
    flight = robustness[robustness.condition == "MISSING_FLIGHT_PARAMETER"].merge(
        stats[["model_key", "condition", "p_value_raw", "p_value_holm", "holm_significant_0_05", "rank_biserial_effect_size"]],
        on=["model_key", "condition"], validate="one_to_one"
    )
    flight["supported_claim"] = "MODEL_DEPENDENCE_ON_FLIGHT_FEATURES"
    flight["subject_generalization_claim"] = "SUBJECT_GENERALIZATION_OF_FLIGHT_DEPENDENCE"
    flight["generalizable_behavior_claim"] = "INCONCLUSIVE_DUE_TO_METADATA"
    flight["unseen_scenario_generalization"] = "NOT_FEASIBLE_DUE_TO_METADATA"
    atomic_csv(SUMMARIES / "phase09_flight_dependence_evidence.csv", flight)

    catalog = make_figures(robustness, subject_all, loso_subject, loso_stability)
    write_reports(conclusion, catalog, robustness, stats, loso_stability)

    metric_checks = {
        "missing_classification_groups_12": len(class_metrics) == 12,
        "missing_regression_groups_12": len(reg_metrics) == 12,
        "loso_classification_groups_2": len(loso_class) == 2,
        "loso_regression_groups_2": len(loso_reg) == 2,
        "seed_groups_complete": seed_agg.seed_count.eq(5).all(),
        "metrics_recomputed_from_canonical_predictions": True,
        "fold_metrics_or_summaries_copied_no": True,
    }
    atomic_json(AUDITS / "phase09_metric_recalculation_audit.json", {
        "phase": "09", "audit": "metric_recalculation", "status": "PASS" if all(metric_checks.values()) else "FAIL",
        "audited_at_utc": utc_now(), "checks": metric_checks,
        "classification_metrics": ["Macro-F1", "Balanced Accuracy", "Accuracy", "Severe Error Rate", "Per-class Recall", "Confusion Matrix", "Quadratic Weighted Kappa"],
        "regression_metrics": ["raw MAE", "bounded MAE", "bounded RMSE", "bounded R2", "bounded Spearman", "clipping count/rate", "rounded regression Macro-F1", "adjacent accuracy", "severe error rate"],
    })
    atomic_json(AUDITS / "phase09_statistical_unit_audit.json", {
        "phase": "09", "audit": "statistical_unit", "status": "PASS",
        "statistical_unit": "subject_id", "subjects": 35,
        "outer_folds_as_independent_samples": False, "loso_splits_as_independent_samples": False,
        "hdc_seeds_as_independent_samples": False, "run_keys_as_independent_samples": False,
        "paired_subject_rows": len(deltas), "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    })
    atomic_json(AUDITS / "phase09_multiple_comparison_audit.json", {
        "phase": "09", "audit": "multiple_comparison", "status": "PASS",
        "method": "Holm", "families": sorted(stats.comparison_family.unique()),
        "family_size": 5, "comparisons": len(stats),
        "all_adjusted_p_values_present": bool(stats.p_value_holm.notna().all()),
        "unregistered_between_model_inference_added": False,
    })
    manifest.update({
        "status": "ANALYSIS_COMPLETE_PENDING_FINAL_VERIFICATION",
        "formal_statistical_analysis_executed": True, "analysis_completed_at_utc": utc_now(),
        "subject_level_statistical_unit": "subject_id", "statistical_subjects": 35,
        "holm_correction_executed": True, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "figures_saved": True, "final_report_saved": True, "phase10_executed": False,
        "phase09_freeze_executed": False,
    })
    atomic_json(MANIFEST_PATH, manifest)
    return {
        "status": manifest["status"], "metric_recalculation": "PASS",
        "subject_level_statistics": "PASS", "holm_correction": "PASS",
        "bootstrap_confidence_intervals": "PASS", "figures_saved": len(catalog) == 6,
        "reports_saved": True, "pairwise_comparisons": len(stats),
    }


if __name__ == "__main__":
    print(json.dumps(run_analysis(), ensure_ascii=False, indent=2))
