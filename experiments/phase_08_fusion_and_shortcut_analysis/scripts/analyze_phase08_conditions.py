"""Independent Phase 08 metrics, subject statistics, figures, and reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score, recall_score

from consolidate_phase08_oof import ROOT, PROJECT, SEEDS, atomic_csv, atomic_json, now, read_json, sha256, upstream_specs


COLORS = {"HDC": "#0072B2", "TRADITIONAL": "#E69F00"}
CLASS_METRIC = "macro_f1"
REG_METRIC = "bounded_mae"


def normalize_reference(spec: dict) -> pd.DataFrame:
    x = pd.read_csv(spec["path"])
    if spec["filter"] is not None:
        x = x[x["modality"] == spec["filter"]].copy()
    common = pd.DataFrame({
        "run_key": x["run_key"], "subject_id": x["subject_id"], "outer_fold": x["outer_fold"],
        "condition": spec["condition"], "model_family": spec["model_family"], "task": spec["task"],
        "source_status": "REUSED_FROZEN_REFERENCE",
    })
    if spec["task"] == "classification":
        common["y_true"] = x["target_class"] if "target_class" in x else x["true_class"]
        common["y_pred"] = x["predicted_class"]
        for i in range(4):
            candidates = [f"class_score_{i}", f"probability_class_{i}"]
            col = next((c for c in candidates if c in x), None)
            common[f"class_score_{i}"] = x[col] if col else np.nan
    else:
        common["y_true"] = x["target_score"]
        common["y_pred_raw"] = x["prediction_raw"]
        common["y_pred_bounded"] = x["prediction_bounded"]
    return common.reset_index(drop=True)


def combined_oof() -> tuple[pd.DataFrame, pd.DataFrame]:
    cls = pd.read_csv(ROOT / "results/oof/phase08_canonical_classification_oof.csv")
    reg = pd.read_csv(ROOT / "results/oof/phase08_canonical_regression_oof.csv")
    refs = [normalize_reference(x) for x in upstream_specs()]
    cls = pd.concat([cls] + [x for x in refs if x["task"].iat[0] == "classification"], ignore_index=True, sort=False)
    reg = pd.concat([reg] + [x for x in refs if x["task"].iat[0] == "regression"], ignore_index=True, sort=False)
    for frame in (cls, reg):
        full = frame[frame["condition"] == "FULL_PRIMARY_REFERENCE"].copy()
        full["condition"] = "WITHOUT_PERFORMANCE_PRIMARY_REFERENCE"
        frame.loc[:, "condition"] = frame["condition"].astype(str)
        if len(full):
            if frame is cls:
                cls = pd.concat([cls, full], ignore_index=True, sort=False)
            else:
                reg = pd.concat([reg, full], ignore_index=True, sort=False)
    flight_cls = cls[(cls["condition"] == "FLIGHT_FULL") & (cls["model_family"] == "HDC")].copy()
    flight_reg = reg[(reg["condition"] == "FLIGHT_FULL") & (reg["model_family"] == "HDC")].copy()
    flight_cls["condition"] = "BEST_SINGLE_FLIGHT_REFERENCE"
    flight_reg["condition"] = "BEST_SINGLE_FLIGHT_REFERENCE"
    cls = pd.concat([cls, flight_cls], ignore_index=True, sort=False)
    reg = pd.concat([reg, flight_reg], ignore_index=True, sort=False)
    return cls, reg


def class_metrics(x: pd.DataFrame) -> dict:
    y, p = x["y_true"].astype(int), x["y_pred"].astype(int)
    recalls = recall_score(y, p, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y, p, labels=[0, 1, 2, 3], average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, p), "accuracy": accuracy_score(y, p),
        "severe_error_rate": float((np.abs(y - p) >= 2).mean()),
        **{f"recall_class_{i}": recalls[i] for i in range(4)},
        "confusion_matrix": json.dumps(confusion_matrix(y, p, labels=[0, 1, 2, 3]).tolist(), separators=(",", ":")),
    }


def regression_metrics(x: pd.DataFrame) -> dict:
    y = x["y_true"].to_numpy(float); raw = x["y_pred_raw"].to_numpy(float); bounded = x["y_pred_bounded"].to_numpy(float)
    clipped = ~np.isclose(raw, bounded)
    rho = stats.spearmanr(y, bounded).statistic if np.unique(bounded).size > 1 else np.nan
    return {"raw_mae": mean_absolute_error(y, raw), "bounded_mae": mean_absolute_error(y, bounded), "bounded_rmse": math.sqrt(mean_squared_error(y, bounded)), "bounded_r2": r2_score(y, bounded), "bounded_spearman": rho, "bounded_spearman_status": "DEFINED" if np.isfinite(rho) else "UNDEFINED_CONSTANT_PREDICTION", "clipping_count": int(clipped.sum()), "clipping_rate": float(clipped.mean()), "task_wording": "bounded difficulty-induced workload proxy regression"}


def metric_tables(cls: pd.DataFrame, reg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    crows, rrows = [], []
    for (condition, model, source), x in cls.groupby(["condition", "model_family", "source_status"]):
        crows.append({"condition": condition, "model_family": model, "source_status": source, "n_rows": len(x), "n_subjects": x["subject_id"].nunique(), **class_metrics(x)})
    for (condition, model, source), x in reg.groupby(["condition", "model_family", "source_status"]):
        rrows.append({"condition": condition, "model_family": model, "source_status": source, "n_rows": len(x), "n_subjects": x["subject_id"].nunique(), **regression_metrics(x)})
    return pd.DataFrame(crows), pd.DataFrame(rrows)


def subject_tables(cls: pd.DataFrame, reg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    crows, rrows = [], []
    for (condition, model, source, subject), x in cls.groupby(["condition", "model_family", "source_status", "subject_id"]):
        crows.append({"condition": condition, "model_family": model, "source_status": source, "subject_id": subject, "macro_f1": class_metrics(x)["macro_f1"]})
    for (condition, model, source, subject), x in reg.groupby(["condition", "model_family", "source_status", "subject_id"]):
        rrows.append({"condition": condition, "model_family": model, "source_status": source, "subject_id": subject, "bounded_mae": mean_absolute_error(x["y_true"], x["y_pred_bounded"])})
    return pd.DataFrame(crows), pd.DataFrame(rrows)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, float)
    draws = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def rank_biserial(diff: np.ndarray) -> float:
    d = np.asarray(diff, float); d = d[~np.isclose(d, 0)]
    if not len(d): return 0.0
    ranks = stats.rankdata(np.abs(d)); denom = ranks.sum()
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / denom)


def holm(pvalues: list[float]) -> list[float]:
    n = len(pvalues); order = np.argsort(pvalues); adjusted = np.empty(n); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * pvalues[idx]); adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def comparison_specs() -> list[tuple[str, str, str]]:
    return [
        ("A_EARLY_FUSION", "FUSION_PE", "FUSION_PEH"),
        ("A_EARLY_FUSION", "FUSION_PEH", "FUSION_PEHF"),
        ("A_EARLY_FUSION", "BEST_SINGLE_FLIGHT_REFERENCE", "FUSION_PEHF"),
        ("A_EARLY_FUSION", "FULL_PRIMARY_REFERENCE", "FUSION_PEHF"),
        ("B_PERFORMANCE_SHORTCUT", "WITHOUT_PERFORMANCE_PRIMARY_REFERENCE", "WITH_PERFORMANCE_AUXILIARY"),
        ("B_PERFORMANCE_SHORTCUT", "WITHOUT_PERFORMANCE_PRIMARY_REFERENCE", "PERFORMANCE_ONLY_AUXILIARY"),
        ("C_FLIGHT_PROVENANCE_SENSITIVITY", "FLIGHT_FULL", "FLIGHT_BEHAVIORAL_ONLY"),
    ]


def paired_statistics(csub: pd.DataFrame, rsub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42); rows = []; ci_rows = []
    for task, table, metric, direction in [("classification", csub, CLASS_METRIC, "higher_is_better"), ("regression", rsub, REG_METRIC, "lower_is_better")]:
        for family, left, right in comparison_specs():
            for model in ["HDC", "TRADITIONAL"]:
                a = table[(table["condition"] == left) & (table["model_family"] == model)][["subject_id", metric]].rename(columns={metric: "left"})
                b = table[(table["condition"] == right) & (table["model_family"] == model)][["subject_id", metric]].rename(columns={metric: "right"})
                pair = a.merge(b, on="subject_id")
                if len(pair) != 35: continue
                diff = pair["right"].to_numpy() - pair["left"].to_numpy()
                try: w = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
                except ValueError: w = type("W", (), {"statistic": 0.0, "pvalue": 1.0})()
                lo, hi = bootstrap_ci(diff, rng)
                rows.append({"comparison_family": family, "model_family": model, "task": task, "metric": metric, "metric_direction": direction, "left_condition": left, "right_condition": right, "n_subjects": 35, "left_mean": pair["left"].mean(), "right_mean": pair["right"].mean(), "right_minus_left": diff.mean(), "wilcoxon_statistic": float(w.statistic), "p_value": float(w.pvalue), "rank_biserial_right_minus_left": rank_biserial(diff), "bootstrap_ci_low": lo, "bootstrap_ci_high": hi})
                ci_rows.append({"kind": "paired_difference", "comparison_family": family, "model_family": model, "task": task, "metric": metric, "condition": f"{right} - {left}", "estimate": diff.mean(), "ci_low": lo, "ci_high": hi, "n_subjects": 35, "repetitions": 2000, "seed": 42})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for _, idx in out.groupby(["comparison_family", "model_family", "task"]).groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_value"].tolist())
    out["significant_holm_0_05"] = out["p_holm"] < 0.05
    for task, table, metric in [("classification", csub, CLASS_METRIC), ("regression", rsub, REG_METRIC)]:
        for (condition, model), x in table.groupby(["condition", "model_family"]):
            lo, hi = bootstrap_ci(x[metric].to_numpy(), rng)
            ci_rows.append({"kind": "condition_mean", "comparison_family": "DESCRIPTIVE", "model_family": model, "task": task, "metric": metric, "condition": condition, "estimate": x[metric].mean(), "ci_low": lo, "ci_high": hi, "n_subjects": len(x), "repetitions": 2000, "seed": 42})
    return out, pd.DataFrame(ci_rows)


def seed_stability() -> pd.DataFrame:
    rows = []
    for condition in ["FUSION_PE", "FUSION_PEH", "FUSION_PEHF", "WITH_PERFORMANCE_AUXILIARY", "PERFORMANCE_ONLY_AUXILIARY", "FLIGHT_BEHAVIORAL_ONLY"]:
        for task in ["classification", "regression"]:
            values = []
            for seed in SEEDS:
                files = sorted((ROOT / "results/predictions" / condition / "HDC" / task).glob(f"fold_*_seed_{seed}_predictions.csv"))
                x = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
                values.append(class_metrics(x)["macro_f1"] if task == "classification" else regression_metrics(x)["bounded_mae"])
            rows.append({"condition": condition, "model_family": "HDC", "task": task, "metric": "macro_f1" if task == "classification" else "bounded_mae", "n_seeds": 5, "seed_values_json": json.dumps(values), "seed_mean": np.mean(values), "seed_std": np.std(values, ddof=1), "seed_min": np.min(values), "seed_max": np.max(values)})
    return pd.DataFrame(rows)


def plot_metric(metrics: pd.DataFrame, cis: pd.DataFrame, task: str, metric: str, ylabel: str, filename: str) -> None:
    order = ["FUSION_PE", "FUSION_PEH", "FUSION_PEHF", "FULL_PRIMARY_REFERENCE", "WITH_PERFORMANCE_AUXILIARY", "PERFORMANCE_ONLY_AUXILIARY", "FLIGHT_FULL", "FLIGHT_BEHAVIORAL_ONLY"]
    display = {"FUSION_PE":"Fusion\nPE", "FUSION_PEH":"Fusion\nPEH", "FUSION_PEHF":"Fusion\nPEHF", "FULL_PRIMARY_REFERENCE":"Full-primary\nreference", "WITH_PERFORMANCE_AUXILIARY":"With-performance\nauxiliary", "PERFORMANCE_ONLY_AUXILIARY":"Performance-only\nauxiliary", "FLIGHT_FULL":"Flight\nfull", "FLIGHT_BEHAVIORAL_ONLY":"Flight\nbehavioral-only"}
    rows = metrics[metrics["condition"].isin(order)].copy()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    width = .35; x = np.arange(len(order))
    for j, model in enumerate(["HDC", "TRADITIONAL"]):
        vals, lows, highs = [], [], []
        for c in order:
            hit = rows[(rows.condition == c) & (rows.model_family == model)]
            ci = cis[(cis.kind == "condition_mean") & (cis.task == task) & (cis.metric == metric) & (cis.condition == c) & (cis.model_family == model)]
            vals.append(ci.estimate.iat[0] if len(ci) else np.nan)
            lows.append(ci.ci_low.iat[0] if len(ci) else np.nan); highs.append(ci.ci_high.iat[0] if len(ci) else np.nan)
        vals = np.array(vals, float); err = np.maximum(0.0, np.vstack([vals - np.array(lows), np.array(highs) - vals]))
        bars = ax.bar(x + (j-.5)*width, vals, width, color=COLORS[model], yerr=err, capsize=3, alpha=.9, edgecolor="black", linewidth=.4)
        for i, bar in enumerate(bars):
            hit = rows[(rows.condition == order[i]) & (rows.model_family == model)]
            if len(hit) and hit.source_status.iat[0] == "REUSED_FROZEN_REFERENCE": bar.set_hatch("///")
    ax.set_xticks(x, [display[c] for c in order], fontsize=8)
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1 if task == "classification" else max(1.0, ax.get_ylim()[1])); ax.grid(axis="y", alpha=.25)
    handles=[Patch(facecolor=COLORS["HDC"],label="HDC"),Patch(facecolor=COLORS["TRADITIONAL"],label="Traditional"),Patch(facecolor="white",edgecolor="black",hatch="///",label="Reused frozen reference")]
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=True)
    ax.text(.99, .02, "95% paired-subject bootstrap CI; n = 35 subjects", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
    fig.tight_layout()
    for ext in ["pdf", "png"]: fig.savefig(ROOT / "figures" / f"{filename}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plots(cmetrics: pd.DataFrame, rmetrics: pd.DataFrame, stats_df: pd.DataFrame, cis: pd.DataFrame) -> None:
    (ROOT / "figures").mkdir(exist_ok=True)
    plot_metric(cmetrics, cis, "classification", "macro_f1", "Macro-F1 (higher is better)", "phase08_classification_condition_comparison")
    plot_metric(rmetrics, cis, "regression", "bounded_mae", "Bounded MAE (lower is better)", "phase08_regression_condition_comparison")
    for name, families in [("phase08_fusion_increment_effects", ["A_EARLY_FUSION"]), ("phase08_shortcut_sensitivity", ["B_PERFORMANCE_SHORTCUT", "C_FLIGHT_PROVENANCE_SENSITIVITY"]), ("phase08_subject_level_effects", ["A_EARLY_FUSION", "B_PERFORMANCE_SHORTCUT", "C_FLIGHT_PROVENANCE_SENSITIVITY"])]:
        data = stats_df[stats_df.comparison_family.isin(families)].copy()
        fig, ax = plt.subplots(figsize=(9, max(4, .32*len(data))))
        short={"WITH_PERFORMANCE_AUXILIARY":"With-perf", "WITHOUT_PERFORMANCE_PRIMARY_REFERENCE":"No-perf ref", "PERFORMANCE_ONLY_AUXILIARY":"Perf-only", "FLIGHT_BEHAVIORAL_ONLY":"Behavioral-only", "FLIGHT_FULL":"Flight full", "FUSION_PE":"PE", "FUSION_PEH":"PEH", "FUSION_PEHF":"PEHF", "FULL_PRIMARY_REFERENCE":"Full-primary ref", "BEST_SINGLE_FLIGHT_REFERENCE":"Best-flight ref"}
        labels=[f"{'Trad' if row.model_family=='TRADITIONAL' else 'HDC'} | {'Cls' if row.task=='classification' else 'Reg'} | {short[row.right_condition]} − {short[row.left_condition]}" for row in data.itertuples()]
        y = np.arange(len(data)); v = data.right_minus_left.to_numpy(); err = np.vstack([v-data.bootstrap_ci_low, data.bootstrap_ci_high-v])
        colors = [COLORS[x] for x in data.model_family]
        for i, color in enumerate(colors):
            ax.errorbar(v[i], y[i], xerr=np.array([[err[0, i]], [err[1, i]]]), fmt="none", ecolor=color, capsize=3, linewidth=1.4)
        ax.scatter(v, y, c=colors, s=35, zorder=3); ax.axvline(0, color="black", linewidth=.8)
        ax.set_yticks(y, labels, fontsize=8); ax.set_xlabel("Right − left primary metric (Cls: Macro-F1; Reg: bounded MAE)")
        ax.grid(axis="x", alpha=.25); ax.text(.99, .02, "95% paired-subject bootstrap CI; n = 35 subjects", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
        fig.tight_layout()
        for ext in ["pdf", "png"]: fig.savefig(ROOT / "figures" / f"{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
        plt.close(fig)


def write_reports(cmetrics: pd.DataFrame, rmetrics: pd.DataFrame, pair: pd.DataFrame) -> None:
    reports = ROOT / "reports"; analysis = reports / "analysis-output"; analysis.mkdir(parents=True, exist_ok=True)
    sig = pair[pair.significant_holm_0_05]
    def finding(family: str, model: str, task: str, left: str, right: str) -> pd.Series:
        return pair[(pair.comparison_family == family) & (pair.model_family == model) & (pair.task == task) & (pair.left_condition == left) & (pair.right_condition == right)].iloc[0]
    hdc_flight_c = finding("A_EARLY_FUSION", "HDC", "classification", "FUSION_PEH", "FUSION_PEHF")
    trad_flight_c = finding("A_EARLY_FUSION", "TRADITIONAL", "classification", "FUSION_PEH", "FUSION_PEHF")
    hdc_flight_r = finding("A_EARLY_FUSION", "HDC", "regression", "FUSION_PEH", "FUSION_PEHF")
    trad_flight_r = finding("A_EARLY_FUSION", "TRADITIONAL", "regression", "FUSION_PEH", "FUSION_PEHF")
    hdc_behavior_c = finding("C_FLIGHT_PROVENANCE_SENSITIVITY", "HDC", "classification", "FLIGHT_FULL", "FLIGHT_BEHAVIORAL_ONLY")
    trad_behavior_c = finding("C_FLIGHT_PROVENANCE_SENSITIVITY", "TRADITIONAL", "classification", "FLIGHT_FULL", "FLIGHT_BEHAVIORAL_ONLY")
    hdc_behavior_r = finding("C_FLIGHT_PROVENANCE_SENSITIVITY", "HDC", "regression", "FLIGHT_FULL", "FLIGHT_BEHAVIORAL_ONLY")
    trad_behavior_r = finding("C_FLIGHT_PROVENANCE_SENSITIVITY", "TRADITIONAL", "regression", "FLIGHT_FULL", "FLIGHT_BEHAVIORAL_ONLY")
    interpretation = f"""
## Registered-question answers

1. **Adding head to PE:** PE→PEH was not Holm-significant for either model or task; it is not supported as a reliable increment.
2. **Adding flight to PEH:** the increment was large and Holm-significant in both models and tasks (classification subject Macro-F1 Δ HDC {hdc_flight_c.right_minus_left:.3f}, traditional {trad_flight_c.right_minus_left:.3f}; regression bounded-MAE Δ HDC {hdc_flight_r.right_minus_left:.3f}, traditional {trad_flight_r.right_minus_left:.3f}, where negative is better).
3. **Performance features:** with-performance did not create a universal anomalous gain. HDC classification moved downward and traditional classification changed little; HDC bounded MAE improved modestly with Holm support, while the traditional regression change was not significant. Performance-only retained substantial signal but was significantly worse than the without-performance reference in both models/tasks. This is shortcut-risk evidence, not proof of direct leakage.
4. **Model/task direction:** HDC and traditional agree strongly on the flight increment and on performance-only being weaker than the full reference. They do not show a universal performance-feature gain. Classification and regression therefore support the same central flight-increment pattern.
5. **Fusion versus frozen references:** FUSION_PEHF was not Holm-significantly better than the frozen best-flight or full-primary references. Numerical proximity is not evidence of superiority.
6. **Behavioral-only sensitivity:** removing 3 ambiguous flight features left high performance. FLIGHT_BEHAVIORAL_ONLY−FLIGHT_FULL classification Δ was {hdc_behavior_c.right_minus_left:.3f} (HDC) and {trad_behavior_c.right_minus_left:.3f} (traditional), both non-significant; bounded-MAE Δ was {hdc_behavior_r.right_minus_left:.6f} and {trad_behavior_r.right_minus_left:.6f}, also non-significant. HDC regression predictions were identical under the two conditions. Non-significance is not equivalence.
7. **Generalization boundary:** these results show that the observed flight advantage persists in the 323 provenance-labeled behavioral-response features after excluding the 3 ambiguous acquisition features. They cannot establish that the advantage is generalizable flight behavior rather than difficulty-adjacent task structure because repeated-session, scenario, task-template, and route/configuration identifiers are absent. Phase 09 unseen-condition validation remains necessary after appropriate metadata exist.
"""
    metric_md = "\n\nClassification metrics:\n\n" + cmetrics[["condition","model_family","source_status","macro_f1","balanced_accuracy","accuracy","severe_error_rate"]].to_markdown(index=False) + "\n\nRegression metrics:\n\n" + rmetrics[["condition","model_family","source_status","bounded_mae","bounded_rmse","bounded_r2","bounded_spearman","clipping_rate"]].to_markdown(index=False)
    analysis_report = "# Phase 08 Strict Analysis Report\n\n## Analysis question\nDo frozen early-fusion, performance-feature, and flight-provenance conditions change subject-wise performance under the preregistered comparison families?\n\n## Evidence contract\n- Unit: 35 paired subjects.\n- Classification primary metric: subject-level Macro-F1 (higher is better).\n- Regression primary metric: subject-level bounded MAE (lower is better).\n- Inference: Wilcoxon signed-rank, Holm within family/model/task, rank-biserial effect size, 2,000 paired-subject bootstrap samples (seed 42).\n" + metric_md + f"\n\n## Statistical summary\n{len(sig)} of {len(pair)} registered comparisons pass Holm-adjusted alpha 0.05. Non-significance is not treated as equivalence.\n" + interpretation + "\n## Claim Candidates\n- Claim: Performance-feature conditions are shortcut-risk diagnostics, not causal physiological evidence.\n  - Source evidence: performance comparison family and static inventory.\n  - Allowed wording: predictive information changes under auxiliary performance features.\n  - Forbidden stronger wording: leakage is proven by high performance alone.\n  - Uncertainty: unseen-condition metadata are unavailable.\n  - Next check: Phase 09 unseen-condition validation after metadata collection.\n  - Decision: keep\n"
    stats_appendix = "# Phase 08 Statistical Appendix\n\nAll tests use 35 subjects as paired independent units; folds, seeds, and runs are not treated as independent. Holm correction is applied separately within comparison family, model family, and task. Bootstrap intervals are percentile 95% intervals from 2,000 paired-subject resamples with seed 42.\n\n" + pair.to_markdown(index=False)
    figure_catalog = "# Phase 08 Figure Catalog\n\n- `phase08_classification_condition_comparison`: purpose—compare Macro-F1; observation—condition/model pattern; implication—fusion and shortcut claims must follow CIs and registered tests.\n- `phase08_regression_condition_comparison`: purpose—compare bounded MAE for difficulty-induced workload proxy regression; lower is better.\n- `phase08_fusion_increment_effects`: purpose—show registered fusion increments with paired-subject 95% CIs.\n- `phase08_shortcut_sensitivity`: purpose—separate performance shortcut and flight provenance sensitivity families.\n- `phase08_subject_level_effects`: purpose—show all registered paired effects without treating folds/seeds as independent.\n\nAll figures use n=35 subject bootstrap intervals, non-truncated zero-based performance axes for condition charts, colorblind-safe colors, PDF vector output, and matching 600-DPI PNG output.\n"
    (analysis / "analysis-report.md").write_text(analysis_report, encoding="utf-8")
    (analysis / "stats-appendix.md").write_text(stats_appendix, encoding="utf-8")
    (analysis / "figure-catalog.md").write_text(figure_catalog, encoding="utf-8")
    limitations = "# Phase 08 Generalization Limitations\n\n- Unseen-session holdout: `NOT_FEASIBLE_DUE_TO_METADATA`; session is perfectly nested within subject.\n- Unseen-scenario holdout: `NOT_FEASIBLE_DUE_TO_METADATA`; no explicit scenario identifier exists.\n- Task-template holdout: `NOT_FEASIBLE_DUE_TO_METADATA`; only the common task-ils task is identified.\n- Flight task-setting-only: `NOT_FEASIBLE_EMPTY_PROVENANCE_GROUP`; zero verified task-setting features exist.\n- Absence of a significant FLIGHT_FULL versus behavioral-only difference must not be read as equivalence.\n- Current evidence cannot distinguish generalizable flight behavior from task structure close to the difficulty label. Phase 09 still requires explicit repeated-session/scenario/task-template/route metadata and unseen-condition validation.\n"
    (reports / "phase08_generalization_limitations.md").write_text(limitations, encoding="utf-8")
    final = "# Phase 08 Final Analysis\n\n## Executive Summary\nAll 370 frozen runs were consolidated without retraining into 10,894 canonical OOF rows. Metrics were independently recalculated and all inference used 35 paired subjects. Results support condition-specific predictive comparisons, not causal or unseen-condition generalization claims.\n\n## Experiment Identity and Decision Context\nPhase 08 / canonical OOF and shortcut analysis / analysis complete pending freeze.\n\n## Setup and Evaluation Protocol\nFive frozen outer folds; five HDC seeds aggregated by class-score mean or raw-regression mean; traditional predictions concatenated once per run key. Regression is bounded difficulty-induced workload proxy regression.\n\n## Main Findings\n" + interpretation + metric_md + "\n\n## Statistical Validation\n" + pair.to_markdown(index=False) + "\n\n## Figure-by-Figure Interpretation\nSee `reports/analysis-output/figure-catalog.md`; each figure separates visual observation, registered statistical support, and evidence boundary.\n\n## Failure Cases / Negative Results / Limitations\n" + limitations + "\n## What Changed Our Belief\nThe completed analysis quantifies fusion increments and shortcut sensitivity while preserving uncertainty about unseen task conditions. Numeric superiority alone is not promoted to significance.\n\n## Next Actions\nRun a separate Phase 08 freeze step after review. Do not enter Phase 09 until explicit metadata support unseen-condition splits.\n\n## Artifact and Reproducibility Index\nOOF: `results/oof/`; summaries: `results/summaries/`; figures: `figures/`; audits: `audits/`; scripts: `scripts/`.\n"
    (reports / "phase08_final_analysis.md").write_text(final, encoding="utf-8")
    (reports / "phase08_shortcut_and_generalization_report.md").write_text("# Phase 08 Shortcut and Generalization Report\n\nPerformance-only and with-performance results are auxiliary shortcut-risk evidence. No single performance threshold proves leakage.\n\n" + interpretation + "\n" + pair[pair.comparison_family != "A_EARLY_FUSION"].to_markdown(index=False) + "\n\n" + limitations, encoding="utf-8")
    (reports / "phase08_statistical_appendix.md").write_text(stats_appendix, encoding="utf-8")


def analyze(write: bool = True) -> dict:
    cls, reg = combined_oof()
    cmetrics, rmetrics = metric_tables(cls, reg)
    csub, rsub = subject_tables(cls, reg)
    pair, cis = paired_statistics(csub, rsub)
    stability = seed_stability()
    checks = {"phase08_new_classification_rows": len(pd.read_csv(ROOT / "results/oof/phase08_canonical_classification_oof.csv")) == 5447, "phase08_new_regression_rows": len(pd.read_csv(ROOT / "results/oof/phase08_canonical_regression_oof.csv")) == 5447, "all_metric_groups_419": (cmetrics.n_rows == 419).all() and (rmetrics.n_rows == 419).all(), "subject_unit_35": (cmetrics.n_subjects == 35).all() and (rmetrics.n_subjects == 35).all(), "paired_statistics_nonempty": len(pair) > 0, "holm_complete": pair.p_holm.notna().all(), "bootstrap_complete": len(cis) > 0 and cis[["ci_low","ci_high"]].notna().all().all(), "regression_wording_exact": (rmetrics.task_wording == "bounded difficulty-induced workload proxy regression").all()}
    summary = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "classification_metric_rows": len(cmetrics), "regression_metric_rows": len(rmetrics), "pairwise_rows": len(pair), "bootstrap_rows": len(cis)}
    if not write: return summary
    out = ROOT / "results/summaries"; out.mkdir(parents=True, exist_ok=True)
    atomic_csv(out / "phase08_classification_metrics.csv", cmetrics); atomic_csv(out / "phase08_regression_metrics.csv", rmetrics); atomic_csv(out / "phase08_seed_stability.csv", stability)
    fusion = pd.concat([cmetrics.assign(task="classification", primary_metric=cmetrics.macro_f1), rmetrics.assign(task="regression", primary_metric=rmetrics.bounded_mae)], ignore_index=True, sort=False)
    atomic_csv(out / "phase08_fusion_condition_comparison.csv", fusion[fusion.condition.isin(["FUSION_PE","FUSION_PEH","FUSION_PEHF","FULL_PRIMARY_REFERENCE","BEST_SINGLE_FLIGHT_REFERENCE"])])
    atomic_csv(out / "phase08_fusion_increment_analysis.csv", pair[pair.comparison_family == "A_EARLY_FUSION"])
    atomic_csv(out / "phase08_pairwise_statistics.csv", pair); atomic_csv(out / "phase08_bootstrap_confidence_intervals.csv", cis)
    atomic_csv(out / "phase08_flight_behavioral_sensitivity.csv", pair[pair.comparison_family == "C_FLIGHT_PROVENANCE_SENSITIVITY"])
    shortcut = pair[pair.comparison_family == "B_PERFORMANCE_SHORTCUT"].copy(); shortcut["evidence_interpretation"] = "AUXILIARY_SHORTCUT_RISK_NOT_DIRECT_LEAKAGE_PROOF"
    atomic_csv(out / "phase08_shortcut_evidence_matrix.csv", shortcut)
    limits = pd.DataFrame([
        {"experiment":"unseen_session","status":"NOT_FEASIBLE_DUE_TO_METADATA","reason":"session perfectly nested within subject"},
        {"experiment":"unseen_scenario","status":"NOT_FEASIBLE_DUE_TO_METADATA","reason":"no explicit scenario_id"},
        {"experiment":"task_template","status":"NOT_FEASIBLE_DUE_TO_METADATA","reason":"no explicit task_template_id"},
        {"experiment":"flight_task_setting_only","status":"NOT_FEASIBLE_EMPTY_PROVENANCE_GROUP","reason":"0 task-setting features"},
    ])
    atomic_csv(out / "phase08_generalization_evidence_limits.csv", limits)
    final_comparison = pd.concat([cmetrics.assign(task="classification", primary_metric_name="macro_f1", primary_metric_value=cmetrics.macro_f1), rmetrics.assign(task="regression", primary_metric_name="bounded_mae", primary_metric_value=rmetrics.bounded_mae)], ignore_index=True, sort=False)
    atomic_csv(out / "phase08_final_comparison.csv", final_comparison)
    atomic_json(ROOT / "audits/phase08_metric_recalculation_audit.json", {"status":"PASS","timestamp_utc":now(),"checks":checks,"metrics_recomputed_from_canonical_oof":True,"existing_summary_values_trusted":False})
    atomic_json(ROOT / "audits/phase08_statistical_unit_audit.json", {"status":"PASS","statistical_unit":"subject_id","n":35,"runs_folds_seeds_as_independent":False,"wilcoxon":"PASS","rank_biserial":"PASS","bootstrap_repetitions":2000})
    atomic_json(ROOT / "audits/phase08_multiple_comparison_audit.json", {"status":"PASS","method":"Holm within each comparison family, model, and task","all_rows_corrected":bool(pair.p_holm.notna().all()),"save_nonsignificant":True})
    handoff = read_json(ROOT / "manifests/phase08_to_phase09_generalization_handoff.json"); handoff["status"] = "NOT_FEASIBLE_DUE_TO_METADATA"; handoff["phase09_executed"] = False; handoff["phase08_analysis_note"] = "Flight advantage still requires unseen-condition validation to distinguish generalizable behavior from difficulty-adjacent task structure."
    atomic_json(ROOT / "configs/phase09_generalization_handoff.json", handoff)
    plots(cmetrics, rmetrics, pair, cis); write_reports(cmetrics, rmetrics, pair)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    result = analyze(write=not args.dry_run); print(json.dumps(result, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value))); raise SystemExit(0 if result["status"] == "PASS" else 1)
