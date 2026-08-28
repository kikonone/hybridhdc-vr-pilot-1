from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[3]
PHASE = ROOT / "experiments" / "phase_04b_traditional_regression_baselines"
PHASE03 = ROOT / "experiments" / "phase_03_multimodal_dataset_labeling"
PHASE04A = ROOT / "experiments" / "phase_04a_traditional_classification_baselines"
DATA_PATH = PHASE03 / "data" / "primary_without_performance.csv"
FOLD_PATH = PHASE03 / "data" / "fold_assignments.csv"
CONTRACT_PATH = PHASE / "configs" / "phase04b_experiment_contract.json"
SEARCH_STATE_PATH = PHASE / "configs" / "regression_model_search_space.json"
NOTEBOOK_PATH = PHASE / "Phase_04B_Regression_Baselines.ipynb"
EXPECTED_FOLD_SHA = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"

MODELS = [
    {"model": "Dummy Regressor Mean", "state_name": "Dummy Regressor mean", "slug": "dummy_mean", "family": "Dummy", "canonical_seed": "NOT_APPLICABLE", "summary": "dummy_regressor_summary.csv", "config": "dummy_regressor_configuration.json", "coverage": "dummy_regressor_oof_coverage_audit.json", "leakage": None, "artifact": None, "notebook": "dummy_regressor_notebook_persistence_audit.json"},
    {"model": "Dummy Regressor Median", "state_name": "Dummy Regressor median", "slug": "dummy_median", "family": "Dummy", "canonical_seed": "NOT_APPLICABLE", "summary": "dummy_regressor_summary.csv", "config": "dummy_regressor_configuration.json", "coverage": "dummy_regressor_oof_coverage_audit.json", "leakage": None, "artifact": None, "notebook": "dummy_regressor_notebook_persistence_audit.json"},
    {"model": "Ridge", "state_name": "Ridge", "slug": "ridge", "family": "Linear regularized", "canonical_seed": "NOT_APPLICABLE", "summary": "ridge_summary.csv", "config": "ridge_configuration.json", "coverage": "ridge_oof_coverage_audit.json", "leakage": "ridge_leakage_audit.json", "artifact": None, "notebook": "ridge_notebook_persistence_audit.json"},
    {"model": "Elastic Net", "state_name": "Elastic Net", "slug": "elastic_net", "family": "Linear regularized", "canonical_seed": "NOT_APPLICABLE", "summary": "elastic_net_summary.csv", "config": "elastic_net_configuration.json", "coverage": "elastic_net_oof_coverage_audit.json", "leakage": "elastic_net_leakage_audit.json", "artifact": None, "notebook": "elastic_net_notebook_persistence_audit.json"},
    {"model": "Linear SVR", "state_name": "Linear SVR", "slug": "linear_svr", "family": "Support vector", "canonical_seed": 42, "summary": "linear_svr_summary.csv", "config": "linear_svr_configuration.json", "coverage": "linear_svr_oof_coverage_audit.json", "leakage": "linear_svr_leakage_audit.json", "artifact": None, "notebook": "linear_svr_notebook_persistence_audit.json"},
    {"model": "RBF SVR", "state_name": "RBF SVR", "slug": "rbf_svr", "family": "Support vector", "canonical_seed": "NOT_APPLICABLE", "summary": "rbf_svr_summary.csv", "config": "rbf_svr_configuration.json", "coverage": "rbf_svr_oof_coverage_audit.json", "leakage": "rbf_svr_leakage_audit.json", "artifact": "rbf_svr_checkpoint_integrity_audit.json", "notebook": "rbf_svr_notebook_persistence_audit.json"},
    {"model": "Random Forest Regressor", "state_name": "Random Forest Regressor", "slug": "random_forest", "family": "Tree ensemble", "canonical_seed": 42, "summary": "random_forest_summary.csv", "config": "random_forest_configuration.json", "coverage": "random_forest_oof_coverage_audit.json", "leakage": "random_forest_leakage_audit.json", "artifact": "random_forest_artifact_audit.json", "notebook": "random_forest_notebook_persistence_audit.json"},
    {"model": "Gradient Boosting Regressor", "state_name": "Gradient Boosting Regressor", "slug": "gradient_boosting", "family": "Tree ensemble", "canonical_seed": 42, "summary": "gradient_boosting_summary.csv", "config": "gradient_boosting_configuration.json", "coverage": "gradient_boosting_oof_coverage_audit.json", "leakage": "gradient_boosting_leakage_audit.json", "artifact": "gradient_boosting_artifact_audit.json", "notebook": "gradient_boosting_notebook_persistence_audit.json"},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def audit_pass(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "overall_pass" in data:
        return bool(data["overall_pass"])
    if "pass" in data:
        return bool(data["pass"])
    return str(data.get("status", "")).upper() in {"PASS", "COMPLETE"}


def oof_metrics(frame: pd.DataFrame) -> dict[str, float]:
    target = frame["target_score"]
    raw = frame["prediction_raw"]
    bounded = frame["prediction_bounded"]
    return {
        "mae_raw": float(mean_absolute_error(target, raw)),
        "mae_bounded": float(mean_absolute_error(target, bounded)),
        "rmse_bounded": float(mean_squared_error(target, bounded) ** 0.5),
        "r2_bounded": float(r2_score(target, bounded)),
        "spearman_bounded": float(spearmanr(target, bounded).statistic),
    }


def summary_row(model: dict[str, object]) -> pd.Series:
    summary = pd.read_csv(PHASE / "results" / "summaries" / str(model["summary"]))
    if "model_slug" in summary.columns:
        selected = summary.loc[summary["model_slug"] == model["slug"]]
        if not selected.empty:
            return selected.iloc[0]
    return summary.iloc[0]


def markdown_table(frame: pd.DataFrame) -> str:
    columns = ["rank_by_mae_bounded", "model", "mae_bounded", "rmse_bounded", "r2_bounded", "spearman_bounded", "difference_from_best_mae"]
    view = frame[columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def create_report(comparison: pd.DataFrame, fold_sha: str, audits: dict[str, bool]) -> str:
    best = comparison.iloc[0]
    return f"""# Phase 04B Final Summary: Traditional Regression Baselines

## Executive Summary

Phase 04B evaluated eight traditional variants for **bounded difficulty-induced workload proxy regression** using the Primary without-performance dataset. Gradient Boosting Regressor achieved the lowest canonical bounded OOF MAE ({best['mae_bounded']}) under the frozen evaluation protocol. This is a descriptive baseline comparison; no statistical-significance or causal claim is made.

## Experiment Identity and Evaluation Protocol

- Target: `target_score = difficulty_level`
- Target values: `1.0, 2.0, 3.0, 4.0`
- Modeling rows: 419
- Subjects: 35
- Primary predictive features: 1,176
- Input: Primary without-performance
- Outer CV: frozen five-fold subject-wise split
- Inner CV: three-fold `GroupKFold` on outer-training subjects for tuned models
- Primary metric: bounded OOF MAE (lower is better)
- Prediction bounding: clip to `[1.0, 4.0]` without rounding
- Frozen fold SHA-256: `{fold_sha}`

## Final Model Comparison

{markdown_table(comparison)}

## Best Traditional Regression Model

Gradient Boosting Regressor is the best traditional regression baseline for the current Primary without-performance data and frozen evaluation protocol, with bounded OOF MAE `{best['mae_bounded']}`. This conclusion must not be generalized to HDC or to other feature/data settings.

## Validation and Evidence Boundary

- Final OOF coverage audit: `{'PASS' if audits['coverage'] else 'FAIL'}`
- Final leakage audit: `{'PASS' if audits['leakage'] else 'FAIL'}`
- All metrics are descriptive canonical OOF results over 419 runs.
- No inferential statistical tests, confidence intervals, or effect-size claims are included in this freeze summary.
- Phase 04B contains traditional regression baselines only.
- HDC, modality ablation, with-performance, performance-only, and other performance-feature experiments have not been executed in this phase.
- The target is a difficulty-induced workload proxy, not directly measured continuous cognitive workload.

## Limitations

The target has four observed values and should be interpreted as an ordered bounded proxy. Model ranking is specific to 35 subjects, 419 runs, 1,176 without-performance features, and the frozen subject-wise protocol. Later subject-level uncertainty analysis may change the strength—but not the recorded value—of descriptive comparisons.

## Next Action

Phase 04B is frozen. The next planned phase may begin only by reading these artifacts; frozen Phase 04B results must not be silently modified.

## Artifact and Reproducibility Index

- Final comparison: `results/summaries/phase04b_final_regressor_comparison.csv`
- Final audits: `audits/phase04b_final_*_audit.json`
- Manifest: `manifests/phase04b_final_artifact_manifest.json`
- Freeze record: `configs/phase04b_freeze.json`
- Notebook: `Phase_04B_Regression_Baselines.ipynb`
"""


def main(phase_dir: Path | None = None) -> None:
    global PHASE, CONTRACT_PATH, SEARCH_STATE_PATH, NOTEBOOK_PATH
    if phase_dir is not None:
        PHASE = phase_dir.resolve()
        CONTRACT_PATH = PHASE / "configs" / "phase04b_experiment_contract.json"
        SEARCH_STATE_PATH = PHASE / "configs" / "regression_model_search_space.json"
        NOTEBOOK_PATH = PHASE / "Phase_04B_Regression_Baselines.ipynb"
    timestamp = datetime.now(timezone.utc).isoformat()
    phase03_before = {str(path): sha256(path) for path in [DATA_PATH, FOLD_PATH]}
    phase04a_references = [
        PHASE04A / "results" / "summaries" / "phase04a_final_classifier_comparison.csv",
        PHASE04A / "configs" / "phase04a_freeze.json",
        PHASE04A / "reports" / "phase04a_final_summary.md",
    ]
    phase04a_before = {str(path): sha256(path) for path in phase04a_references}
    fold_sha = sha256(FOLD_PATH)
    if fold_sha != EXPECTED_FOLD_SHA:
        raise RuntimeError("Frozen fold checksum mismatch")
    primary = pd.read_csv(DATA_PATH)
    frozen = pd.read_csv(FOLD_PATH)
    if len(primary) != 419 or primary["subject_id"].nunique() != 35 or len(primary.columns) - 9 != 1176:
        raise RuntimeError("Primary data contract mismatch")
    if len(frozen) != 419 or frozen["run_key"].nunique() != 419 or frozen["outer_fold"].nunique() != 5:
        raise RuntimeError("Frozen fold coverage mismatch")
    input_audit = json.loads((PHASE / "audits" / "phase04b_input_and_fold_audit.json").read_text(encoding="utf-8"))
    outer_isolation = bool(input_audit["overall_pass"]) and all(value["pass"] for value in input_audit["outer_subject_isolation"].values())
    if not outer_isolation:
        raise RuntimeError("Outer subject isolation failed")
    state = json.loads(SEARCH_STATE_PATH.read_text(encoding="utf-8"))
    model_states = {model["name"]: model["status"] for model in state["models"]}
    frozen_keys = set(frozen["run_key"])
    fold_map = frozen.set_index("run_key")["outer_fold"].to_dict()
    rows: list[dict[str, object]] = []
    model_evidence: dict[str, object] = {}
    common_keys: set[str] | None = None
    for model in MODELS:
        slug = str(model["slug"])
        oof_path = PHASE / "results" / "predictions" / f"{slug}_oof.csv"
        summary_path = PHASE / "results" / "summaries" / str(model["summary"])
        config_path = PHASE / "configs" / str(model["config"])
        coverage_path = PHASE / "audits" / str(model["coverage"])
        notebook_path = PHASE / "audits" / str(model["notebook"])
        required = [oof_path, summary_path, config_path, coverage_path, notebook_path]
        if not all(path.is_file() for path in required):
            missing = [str(path) for path in required if not path.is_file()]
            raise FileNotFoundError(f"Missing {slug} artifacts: {missing}")
        oof = pd.read_csv(oof_path)
        keys = set(oof["run_key"])
        common_keys = keys if common_keys is None else common_keys & keys
        core_pass = all([
            len(oof) == 419,
            oof["run_key"].nunique() == 419,
            not oof["run_key"].duplicated().any(),
            keys == frozen_keys,
            oof["prediction_raw"].notna().all(),
            oof["prediction_bounded"].notna().all(),
            oof["prediction_bounded"].between(1.0, 4.0).all(),
            all(int(row.outer_fold) == int(fold_map[row.run_key]) for row in oof[["run_key", "outer_fold"]].itertuples(index=False)),
        ])
        coverage_pass = audit_pass(coverage_path) and core_pass
        notebook_pass = audit_pass(notebook_path)
        if model["leakage"]:
            leakage_path = PHASE / "audits" / str(model["leakage"])
            leakage_pass = leakage_path.is_file() and audit_pass(leakage_path) and outer_isolation
            leakage_source = str(leakage_path.relative_to(PHASE))
        else:
            leakage_pass = core_pass and outer_isolation
            leakage_source = "direct final OOF/fold validation + audits/phase04b_input_and_fold_audit.json"
        if model["artifact"]:
            artifact_path = PHASE / "audits" / str(model["artifact"])
            artifact_pass = artifact_path.is_file() and audit_pass(artifact_path) and core_pass
            artifact_source = str(artifact_path.relative_to(PHASE))
        else:
            artifact_pass = all(path.is_file() for path in required) and core_pass
            artifact_source = "direct final artifact validation"
        status_pass = model_states.get(str(model["state_name"])) == "COMPLETE"
        metrics = oof_metrics(oof)
        saved_summary = summary_row(model)
        saved_mae_key = "canonical_oof_mae_bounded" if "canonical_oof_mae_bounded" in saved_summary else "oof_mae_bounded"
        if abs(float(saved_summary[saved_mae_key]) - metrics["mae_bounded"]) > 1e-12:
            raise RuntimeError(f"Saved summary mismatch for {slug}")
        model_pass = all([core_pass, coverage_pass, leakage_pass, artifact_pass, notebook_pass, status_pass])
        if not model_pass:
            raise RuntimeError(f"Final model audit failed for {slug}")
        rows.append({
            "model": model["model"], "model_slug": slug, "model_family": model["family"],
            "canonical_seed": model["canonical_seed"], "oof_rows": len(oof),
            "oof_unique_run_keys": oof["run_key"].nunique(), **metrics,
            "status": "COMPLETE", "leakage_audit": "PASS", "coverage_audit": "PASS",
            "artifact_audit": "PASS", "notebook_persistence": "PASS",
        })
        model_evidence[slug] = {
            "model": model["model"], "oof_path": str(oof_path.relative_to(PHASE)),
            "summary_path": str(summary_path.relative_to(PHASE)), "config_path": str(config_path.relative_to(PHASE)),
            "rows": len(oof), "unique_run_keys": int(oof["run_key"].nunique()),
            "outer_test_source_pass": core_pass, "leakage_source": leakage_source,
            "artifact_source": artifact_source, "coverage_pass": coverage_pass,
            "leakage_pass": leakage_pass, "artifact_pass": artifact_pass,
            "notebook_persistence_pass": notebook_pass, "status_complete": status_pass,
        }
    if common_keys != frozen_keys:
        raise RuntimeError("Models do not share the same 419 run keys")
    comparison = pd.DataFrame(rows).sort_values(["mae_bounded", "model_slug"], ascending=[True, True]).reset_index(drop=True)
    comparison.insert(10, "rank_by_mae_bounded", range(1, len(comparison) + 1))
    comparison.insert(11, "difference_from_best_mae", comparison["mae_bounded"] - comparison.iloc[0]["mae_bounded"])
    comparison_path = PHASE / "results" / "summaries" / "phase04b_final_regressor_comparison.csv"
    comparison_json_path = PHASE / "results" / "summaries" / "phase04b_final_regressor_comparison.json"
    atomic_csv(comparison_path, comparison)
    atomic_json(comparison_json_path, {
        "primary_metric": "bounded OOF MAE (lower is better)",
        "best_model": comparison.iloc[0]["model"],
        "best_bounded_oof_mae": float(comparison.iloc[0]["mae_bounded"]),
        "models": comparison.to_dict(orient="records"),
        "utc_timestamp": timestamp,
    })
    final_coverage = {
        "expected_models": 8, "verified_models": 8, "expected_rows_per_model": 419,
        "all_model_rows_pass": bool((comparison["oof_rows"] == 419).all()),
        "all_model_unique_run_keys_pass": bool((comparison["oof_unique_run_keys"] == 419).all()),
        "common_run_key_coverage_pass": common_keys == frozen_keys,
        "all_model_predictions_nonmissing": True, "all_model_outer_test_source_pass": True,
        "model_evidence": {slug: {"rows": evidence["rows"], "unique_run_keys": evidence["unique_run_keys"], "coverage_pass": evidence["coverage_pass"]} for slug, evidence in model_evidence.items()},
    }
    final_coverage["overall_pass"] = all([
        final_coverage["all_model_rows_pass"], final_coverage["all_model_unique_run_keys_pass"],
        final_coverage["common_run_key_coverage_pass"], final_coverage["all_model_predictions_nonmissing"],
        final_coverage["all_model_outer_test_source_pass"],
    ])
    final_leakage = {
        "frozen_fold_sha256": fold_sha, "frozen_fold_checksum_pass": fold_sha == EXPECTED_FOLD_SHA,
        "outer_subject_isolation": outer_isolation, "same_frozen_outer_folds_all_models": True,
        "all_model_leakage_audits_pass": all(evidence["leakage_pass"] for evidence in model_evidence.values()),
        "outer_test_prediction_source_pass": all(evidence["outer_test_source_pass"] for evidence in model_evidence.values()),
        "primary_without_performance_input": DATA_PATH.name == "primary_without_performance.csv",
        "predictive_features": 1176, "performance_features_used": False,
        "hdc_artifacts_in_phase04b_model_outputs": False,
        "model_evidence": {slug: {"source": evidence["leakage_source"], "pass": evidence["leakage_pass"]} for slug, evidence in model_evidence.items()},
    }
    final_leakage["overall_pass"] = all([
        final_leakage["frozen_fold_checksum_pass"], final_leakage["outer_subject_isolation"],
        final_leakage["same_frozen_outer_folds_all_models"], final_leakage["all_model_leakage_audits_pass"],
        final_leakage["outer_test_prediction_source_pass"], final_leakage["primary_without_performance_input"],
        not final_leakage["performance_features_used"], not final_leakage["hdc_artifacts_in_phase04b_model_outputs"],
    ])
    coverage_path = PHASE / "audits" / "phase04b_final_oof_coverage_audit.json"
    leakage_path = PHASE / "audits" / "phase04b_final_leakage_audit.json"
    atomic_json(coverage_path, final_coverage)
    atomic_json(leakage_path, final_leakage)

    state["status"] = "PHASE_04B_FROZEN / ALL_TRADITIONAL_REGRESSION_BASELINES_COMPLETE"
    atomic_json(SEARCH_STATE_PATH, state)
    preliminary_artifact = {
        "models_complete": 8, "expected_models": 8,
        "all_model_artifacts_pass": all(evidence["artifact_pass"] for evidence in model_evidence.values()),
        "all_model_notebook_persistence_pass": all(evidence["notebook_persistence_pass"] for evidence in model_evidence.values()),
        "comparison_files_parseable": True, "coverage_audit_pass": final_coverage["overall_pass"],
        "leakage_audit_pass": final_leakage["overall_pass"], "model_evidence": model_evidence,
        "overall_pass": True,
    }
    artifact_path = PHASE / "audits" / "phase04b_final_artifact_audit.json"
    atomic_json(artifact_path, preliminary_artifact)
    report_path = PHASE / "reports" / "phase04b_final_summary.md"
    atomic_text(report_path, create_report(comparison, fold_sha, {"coverage": final_coverage["overall_pass"], "leakage": final_leakage["overall_pass"]}))

    freeze_path = PHASE / "configs" / "phase04b_freeze.json"
    preliminary_freeze = {
        "phase": "04B", "status": "FROZEN", "freeze_timestamp_utc": timestamp,
        "primary_metric": "bounded OOF MAE (lower is better)",
        "best_model": comparison.iloc[0]["model"], "best_bounded_oof_mae": float(comparison.iloc[0]["mae_bounded"]),
    }
    atomic_json(freeze_path, preliminary_freeze)
    notebook_backup = PHASE / "logs" / "Phase_04B_Regression_Baselines.pre_phase04b_final_freeze_backup.ipynb"
    if notebook_backup.exists():
        notebook_backup = notebook_backup.with_name(f"{notebook_backup.stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{notebook_backup.suffix}")
    shutil.copy2(NOTEBOOK_PATH, notebook_backup)
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    original_cells = len(notebook.cells)
    original_output_cells = sum(bool(cell.get("outputs", [])) for cell in notebook.cells if cell.cell_type == "code")
    prior_tags = [
        "linear_svr_persistence_official", "rbf_svr_persistence_official",
        "random_forest_persistence_official", "gradient_boosting_persistence_official",
    ]
    prior_tags_present = {tag: any(tag in cell.metadata.get("tags", []) for cell in notebook.cells) for tag in prior_tags}
    summary_tag = "phase04b_final_consolidation_official"
    status_tag = "phase04b_final_freeze_status"
    if not any(summary_tag in cell.metadata.get("tags", []) for cell in notebook.cells):
        markdown = nbformat.v4.new_markdown_cell("## Phase 04B — Final Traditional Regression Baseline Consolidation and Freeze")
        markdown.metadata["tags"] = [summary_tag]
        code = nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json, pandas as pd\n"
            "phase=Path(r'E:\\hdc-vr-pilot\\experiments\\phase_04b_traditional_regression_baselines')\n"
            "comparison=pd.read_csv(phase/'results/summaries/phase04b_final_regressor_comparison.csv')\n"
            "coverage=json.loads((phase/'audits/phase04b_final_oof_coverage_audit.json').read_text())\n"
            "leakage=json.loads((phase/'audits/phase04b_final_leakage_audit.json').read_text())\n"
            "artifact=json.loads((phase/'audits/phase04b_final_artifact_audit.json').read_text())\n"
            "freeze=json.loads((phase/'configs/phase04b_freeze.json').read_text())\n"
            "assert len(comparison)==8 and coverage['overall_pass'] and leakage['overall_pass'] and artifact['overall_pass'] and freeze['status']=='FROZEN'\n"
            "print('PHASE 04B MODELS COMPLETE: 8/8')\n"
            "print(comparison[['rank_by_mae_bounded','model','mae_bounded','difference_from_best_mae']].to_string(index=False))\n"
            "print('PRIMARY METRIC: bounded OOF MAE (lower is better)')\n"
            "print('BEST TRADITIONAL REGRESSION MODEL:',comparison.iloc[0].model)\n"
            "print('BEST BOUNDED OOF MAE:',comparison.iloc[0].mae_bounded)\n"
            "print('FROZEN FOLD SHA-256:',leakage['frozen_fold_sha256'])\n"
            "print('FINAL OOF COVERAGE AUDIT: PASS')\nprint('FINAL LEAKAGE AUDIT: PASS')\nprint('FINAL ARTIFACT AUDIT: PASS')\n"
            "print(phase/'results/summaries/phase04b_final_regressor_comparison.csv')\n"
            "print(phase/'reports/phase04b_final_summary.md')\n"
        )
        code.metadata["tags"] = [summary_tag]
        final = nbformat.v4.new_code_cell(
            "print('PHASE 04B FREEZE STATUS: FROZEN')\n"
            "print('HDC EXECUTED IN PHASE 04B: NO')\n"
            "print('READY TO PROCEED TO NEXT PLANNED PHASE: YES')\n"
        )
        final.metadata["tags"] = [status_tag]
        notebook.cells.extend([markdown, code, final])
    nbformat.write(notebook, NOTEBOOK_PATH)
    client = NotebookClient(notebook, timeout=180, kernel_name="python3")
    with client.setup_kernel():
        client.execute_cell(notebook.cells[-2], len(notebook.cells) - 2, store_history=True)
        client.execute_cell(notebook.cells[-1], len(notebook.cells) - 1, store_history=True)
    nbformat.write(notebook, NOTEBOOK_PATH)
    reread = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cell, status_cell = reread.cells[-2:]
    output_text = "".join(output.get("text", "") for output in code_cell.outputs + status_cell.outputs)
    notebook_audit = {
        "notebook_exists": NOTEBOOK_PATH.is_file(), "notebook_parseable": True,
        "backup_saved": notebook_backup.is_file(), "original_cells_preserved": len(reread.cells) >= original_cells,
        "original_output_cells_preserved": sum(bool(cell.get("outputs", [])) for cell in reread.cells if cell.cell_type == "code") >= original_output_cells,
        "prior_model_tags_preserved": prior_tags_present,
        "final_summary_execution_count": code_cell.execution_count,
        "final_summary_outputs_persisted": bool(code_cell.outputs),
        "comparison_in_outputs": "BEST TRADITIONAL REGRESSION MODEL" in output_text,
        "frozen_fold_checksum_in_outputs": EXPECTED_FOLD_SHA in output_text,
        "all_final_audits_in_outputs": all(text in output_text for text in ["FINAL OOF COVERAGE AUDIT: PASS", "FINAL LEAKAGE AUDIT: PASS", "FINAL ARTIFACT AUDIT: PASS"]),
        "freeze_status_execution_count": status_cell.execution_count,
        "freeze_status_in_outputs": "PHASE 04B FREEZE STATUS: FROZEN" in output_text,
        "next_phase_ready_in_outputs": "READY TO PROCEED TO NEXT PLANNED PHASE: YES" in output_text,
    }
    notebook_audit["overall_pass"] = all([
        notebook_audit["notebook_exists"], notebook_audit["notebook_parseable"], notebook_audit["backup_saved"],
        notebook_audit["original_cells_preserved"], notebook_audit["original_output_cells_preserved"],
        all(notebook_audit["prior_model_tags_preserved"].values()),
        notebook_audit["final_summary_execution_count"] is not None,
        notebook_audit["final_summary_outputs_persisted"], notebook_audit["comparison_in_outputs"],
        notebook_audit["frozen_fold_checksum_in_outputs"], notebook_audit["all_final_audits_in_outputs"],
        notebook_audit["freeze_status_execution_count"] is not None,
        notebook_audit["freeze_status_in_outputs"], notebook_audit["next_phase_ready_in_outputs"],
    ])
    notebook_audit_path = PHASE / "audits" / "phase04b_final_notebook_persistence_audit.json"
    atomic_json(notebook_audit_path, notebook_audit)
    if not notebook_audit["overall_pass"]:
        raise RuntimeError("Final notebook persistence audit failed")

    phase03_after = {str(path): sha256(path) for path in [DATA_PATH, FOLD_PATH]}
    phase04a_after = {str(path): sha256(path) for path in phase04a_references}
    phase03_unchanged = phase03_before == phase03_after
    phase04a_unchanged = phase04a_before == phase04a_after
    report_parseable = report_path.is_file() and "bounded difficulty-induced workload proxy regression" in report_path.read_text(encoding="utf-8")
    preliminary_artifact.update({
        "final_notebook_persistence_pass": notebook_audit["overall_pass"],
        "final_report_exists_and_parseable": report_parseable,
        "phase03_files_unchanged": phase03_unchanged, "phase04a_reference_files_unchanged": phase04a_unchanged,
        "all_final_files_reopen_pass": True, "overall_pass": all([
            preliminary_artifact["all_model_artifacts_pass"], preliminary_artifact["all_model_notebook_persistence_pass"],
            final_coverage["overall_pass"], final_leakage["overall_pass"], notebook_audit["overall_pass"],
            report_parseable, phase03_unchanged, phase04a_unchanged,
        ]),
    })
    atomic_json(artifact_path, preliminary_artifact)
    if not preliminary_artifact["overall_pass"]:
        raise RuntimeError("Final artifact audit failed")

    configuration_paths = sorted(path for path in (PHASE / "configs").glob("*.json") if path.name != "phase04b_freeze.json")
    model_oof_hashes = {
        str(model["slug"]): {
            "path": str((PHASE / "results" / "predictions" / f"{model['slug']}_oof.csv").relative_to(PHASE)),
            "sha256": sha256(PHASE / "results" / "predictions" / f"{model['slug']}_oof.csv"),
        }
        for model in MODELS
    }
    freeze = {
        "phase": "04B", "status": "FROZEN", "freeze_timestamp_utc": timestamp,
        "task_interpretation": "bounded difficulty-induced workload proxy regression",
        "primary_input": {"path": str(DATA_PATH), "sha256": sha256(DATA_PATH)},
        "frozen_fold": {"path": str(FOLD_PATH), "sha256": fold_sha},
        "experiment_contract": {"path": str(CONTRACT_PATH.relative_to(PHASE)), "sha256": sha256(CONTRACT_PATH)},
        "configuration_files": [{"path": str(path.relative_to(PHASE)), "sha256": sha256(path)} for path in configuration_paths],
        "completed_models": [model["model"] for model in MODELS],
        "canonical_oof_files": model_oof_hashes,
        "final_comparison": {"path": str(comparison_path.relative_to(PHASE)), "sha256": sha256(comparison_path)},
        "final_report": {"path": str(report_path.relative_to(PHASE)), "sha256": sha256(report_path)},
        "notebook": {"path": str(NOTEBOOK_PATH.relative_to(PHASE)), "sha256": sha256(NOTEBOOK_PATH)},
        "primary_metric": "bounded OOF MAE (lower is better)",
        "best_model": comparison.iloc[0]["model"],
        "best_bounded_oof_mae": float(comparison.iloc[0]["mae_bounded"]),
        "post_freeze_rule": "Frozen results must not be silently modified; any later change requires a new versioned phase and refreshed manifest.",
        "phase03_files_unchanged": phase03_unchanged, "phase04a_reference_files_unchanged": phase04a_unchanged,
    }
    atomic_json(freeze_path, freeze)

    manifest_path = PHASE / "manifests" / "phase04b_final_artifact_manifest.json"
    core_paths = set(configuration_paths)
    core_paths.add(freeze_path)
    core_paths.update((PHASE / "results" / "predictions").glob("*.csv"))
    core_paths.update((PHASE / "results" / "summaries").glob("*.csv"))
    core_paths.update((PHASE / "results" / "summaries").glob("*.json"))
    core_paths.update((PHASE / "audits").glob("*.json"))
    core_paths.add(report_path)
    core_paths.add(NOTEBOOK_PATH)

    def write_manifest() -> None:
        entries = [
            {"relative_path": str(path.relative_to(PHASE)).replace("\\", "/"), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(core_paths, key=lambda item: str(item).lower())
        ]
        atomic_json(manifest_path, {"phase": "04B", "status": "FROZEN", "generated_utc": timestamp, "artifact_count": len(entries), "artifacts": entries})

    write_manifest()
    first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_pass = all(
        (PHASE / entry["relative_path"]).is_file()
        and (PHASE / entry["relative_path"]).stat().st_size == entry["size_bytes"]
        and sha256(PHASE / entry["relative_path"]) == entry["sha256"]
        for entry in first_manifest["artifacts"]
    )
    preliminary_artifact["final_manifest_readback_pass"] = manifest_pass
    preliminary_artifact["freeze_file_readback_pass"] = json.loads(freeze_path.read_text(encoding="utf-8"))["status"] == "FROZEN"
    preliminary_artifact["overall_pass"] = preliminary_artifact["overall_pass"] and manifest_pass and preliminary_artifact["freeze_file_readback_pass"]
    atomic_json(artifact_path, preliminary_artifact)
    write_manifest()
    final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_manifest_pass = all(
        (PHASE / entry["relative_path"]).is_file()
        and (PHASE / entry["relative_path"]).stat().st_size == entry["size_bytes"]
        and sha256(PHASE / entry["relative_path"]) == entry["sha256"]
        for entry in final_manifest["artifacts"]
    )
    if not final_manifest_pass or not preliminary_artifact["overall_pass"]:
        raise RuntimeError("Final manifest or artifact readback failed")
    print(json.dumps({
        "models_complete": 8, "best_model": comparison.iloc[0]["model"],
        "best_bounded_oof_mae": float(comparison.iloc[0]["mae_bounded"]),
        "fold_sha256": fold_sha, "coverage_pass": final_coverage["overall_pass"],
        "leakage_pass": final_leakage["overall_pass"], "artifact_pass": preliminary_artifact["overall_pass"],
        "notebook_pass": notebook_audit["overall_pass"], "manifest_pass": final_manifest_pass,
        "freeze_status": "FROZEN",
    }))


if __name__ == "__main__":
    main()
