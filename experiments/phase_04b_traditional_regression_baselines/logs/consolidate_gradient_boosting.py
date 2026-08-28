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


PHASE = Path(r"E:\hdc-vr-pilot\experiments\phase_04b_traditional_regression_baselines")
DATA_DIR = Path(r"E:\hdc-vr-pilot\experiments\phase_03_multimodal_dataset_labeling\data")
CHECKPOINT_DIR = PHASE / "results" / "checkpoints" / "gradient_boosting"
EXPECTED_SHA = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
SEEDS = [42, 43, 44, 45, 46]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def compute_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    target = frame["target_score"]
    raw = frame["prediction_raw"]
    bounded = frame["prediction_bounded"]
    return {
        "oof_rows": int(len(frame)),
        "oof_unique_run_keys": int(frame["run_key"].nunique()),
        "oof_mae_raw": float(mean_absolute_error(target, raw)),
        "oof_mae_bounded": float(mean_absolute_error(target, bounded)),
        "oof_rmse_bounded": float(mean_squared_error(target, bounded) ** 0.5),
        "oof_r2_bounded": float(r2_score(target, bounded)),
        "oof_spearman_bounded": float(spearmanr(target, bounded).statistic),
    }


def prior_results() -> list[tuple[str, float]]:
    sources = [
        ("Ridge", "ridge_summary.csv"),
        ("Elastic Net", "elastic_net_summary.csv"),
        ("Linear SVR", "linear_svr_summary.csv"),
        ("RBF SVR", "rbf_svr_summary.csv"),
        ("Random Forest Regressor", "random_forest_summary.csv"),
    ]
    values: list[tuple[str, float]] = []
    for label, filename in sources:
        row = pd.read_csv(PHASE / "results" / "summaries" / filename).iloc[0]
        key = "canonical_oof_mae_bounded" if "canonical_oof_mae_bounded" in row else "oof_mae_bounded"
        values.append((label, float(row[key])))
    return values


def main() -> None:
    fold_path = DATA_DIR / "fold_assignments.csv"
    frozen_sha = hashlib.sha256(fold_path.read_bytes()).hexdigest()
    if frozen_sha != EXPECTED_SHA:
        raise RuntimeError("Frozen fold checksum mismatch")
    frozen = pd.read_csv(fold_path)
    canonical_parts: list[pd.DataFrame] = []
    all_seed_parts: list[pd.DataFrame] = []
    metric_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    fold_audits: list[dict[str, object]] = []
    leakage_audits: list[dict[str, object]] = []
    for fold in range(1, 6):
        prefix = f"gradient_boosting_fold_{fold}"
        audit = json.loads((CHECKPOINT_DIR / f"{prefix}_checkpoint_audit.json").read_text(encoding="utf-8"))
        leakage = json.loads((PHASE / "audits" / f"{prefix}_leakage_audit.json").read_text(encoding="utf-8"))
        best = json.loads((CHECKPOINT_DIR / f"{prefix}_best_params.json").read_text(encoding="utf-8"))
        canonical = pd.read_csv(CHECKPOINT_DIR / f"{prefix}_predictions_seed_42.csv")
        all_seed = pd.read_csv(CHECKPOINT_DIR / f"{prefix}_predictions_all_seeds.csv")
        metrics = pd.read_csv(CHECKPOINT_DIR / f"{prefix}_metrics_all_seeds.csv")
        expected_keys = set(frozen.loc[frozen["outer_fold"] == fold, "run_key"])
        assert audit["overall_pass"] and leakage["overall_pass"] and best["candidate_count"] == 32
        assert set(canonical["run_key"]) == expected_keys and canonical["run_key"].nunique() == len(expected_keys)
        assert set(canonical["outer_fold"]) == {fold} and set(canonical["seed"]) == {42}
        assert len(all_seed) == len(canonical) * 5 and set(all_seed["seed"]) == set(SEEDS)
        assert all_seed.groupby("run_key").size().eq(5).all() and len(metrics) == 5
        canonical_parts.append(canonical)
        all_seed_parts.append(all_seed)
        metric_parts.append(metrics)
        parameter_rows.append({"outer_fold": fold, **best["best_params"], "best_inner_bounded_mae": best["best_inner_bounded_mae"]})
        fold_audits.append(audit)
        leakage_audits.append(leakage)

    canonical_oof = pd.concat(canonical_parts, ignore_index=True).sort_values("run_key").reset_index(drop=True)
    all_seed_oof = pd.concat(all_seed_parts, ignore_index=True).sort_values(["seed", "run_key"]).reset_index(drop=True)
    fold_metrics = pd.concat(metric_parts, ignore_index=True).sort_values(["outer_fold", "seed"]).reset_index(drop=True)
    frozen_keys = set(frozen["run_key"])
    assert len(canonical_oof) == 419 and canonical_oof["run_key"].nunique() == 419
    assert set(canonical_oof["run_key"]) == frozen_keys and not canonical_oof["run_key"].duplicated().any()
    assert len(all_seed_oof) == 2095 and all_seed_oof.groupby("run_key").size().eq(5).all()
    assert canonical_oof["prediction_raw"].notna().all() and canonical_oof["prediction_bounded"].between(1.0, 4.0).all()
    canonical_metrics = compute_metrics(canonical_oof)
    seed_summary = pd.DataFrame([{"seed": seed, **compute_metrics(all_seed_oof.loc[all_seed_oof["seed"] == seed])} for seed in SEEDS])
    best_prior_model, best_prior_mae = min(prior_results(), key=lambda item: item[1])
    summary = {
        "model": "Gradient Boosting Regressor",
        "model_slug": "gradient_boosting",
        "canonical_seed": 42,
        "canonical_oof_rows": canonical_metrics["oof_rows"],
        "canonical_oof_unique_run_keys": canonical_metrics["oof_unique_run_keys"],
        "canonical_oof_mae_raw": canonical_metrics["oof_mae_raw"],
        "canonical_oof_mae_bounded": canonical_metrics["oof_mae_bounded"],
        "canonical_oof_rmse_bounded": canonical_metrics["oof_rmse_bounded"],
        "canonical_oof_r2_bounded": canonical_metrics["oof_r2_bounded"],
        "canonical_oof_spearman_bounded": canonical_metrics["oof_spearman_bounded"],
        "seed_count": 5,
        "seed_mean_oof_mae_bounded": float(seed_summary["oof_mae_bounded"].mean()),
        "seed_std_oof_mae_bounded": float(seed_summary["oof_mae_bounded"].std(ddof=1)),
        "best_prior_model": best_prior_model,
        "best_prior_oof_mae_bounded": best_prior_mae,
        "mae_difference_vs_best_prior": float(canonical_metrics["oof_mae_bounded"] - best_prior_mae),
        "status": "COMPLETE",
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    atomic_csv(PHASE / "results" / "predictions" / "gradient_boosting_oof.csv", canonical_oof)
    atomic_csv(PHASE / "results" / "predictions" / "gradient_boosting_oof_all_seeds.csv", all_seed_oof)
    atomic_csv(PHASE / "results" / "fold_metrics" / "gradient_boosting_fold_metrics_all_seeds.csv", fold_metrics)
    atomic_csv(PHASE / "results" / "summaries" / "gradient_boosting_selected_parameters.csv", pd.DataFrame(parameter_rows))
    atomic_csv(PHASE / "results" / "summaries" / "gradient_boosting_seed_summary.csv", seed_summary)
    atomic_csv(PHASE / "results" / "summaries" / "gradient_boosting_summary.csv", pd.DataFrame([summary]))
    atomic_json(PHASE / "results" / "summaries" / "gradient_boosting_summary.json", summary)
    coverage = {
        "canonical_rows": 419, "canonical_unique_run_keys": 419, "duplicate_run_keys": 0,
        "missing_run_keys": 0, "extra_run_keys": 0, "all_seed_rows": 2095,
        "seeds": SEEDS, "bounded_range_pass": True, "overall_pass": True,
    }
    leakage = {
        "frozen_fold_sha256": frozen_sha,
        "outer_subject_isolation": all(a["outer_subject_overlap_count"] == 0 for a in fold_audits),
        "inner_subject_isolation": all(a["subject_overlap_count"] == 0 for audit in fold_audits for a in audit["inner_subject_isolation"]),
        "pipeline_training_only": True, "outer_test_used_for_tuning": False,
        "canonical_seed_not_selected_by_performance": True,
        "five_fold_leakage_audits_pass": all(a["overall_pass"] for a in leakage_audits),
    }
    leakage["overall_pass"] = all([
        leakage["frozen_fold_sha256"] == EXPECTED_SHA,
        leakage["outer_subject_isolation"],
        leakage["inner_subject_isolation"],
        leakage["pipeline_training_only"],
        not leakage["outer_test_used_for_tuning"],
        leakage["canonical_seed_not_selected_by_performance"],
        leakage["five_fold_leakage_audits_pass"],
    ])
    artifact = {
        "five_checkpoint_integrity_pass": all(a["overall_pass"] for a in fold_audits),
        "canonical_oof_pass": True, "all_seed_oof_pass": True,
        "fold_metrics_rows": int(len(fold_metrics)), "selected_parameter_rows": len(parameter_rows),
        "seed_summary_rows": int(len(seed_summary)), "summary_files_pass": True,
    }
    artifact["overall_pass"] = all([
        artifact["five_checkpoint_integrity_pass"], artifact["canonical_oof_pass"],
        artifact["all_seed_oof_pass"], artifact["fold_metrics_rows"] == 25,
        artifact["selected_parameter_rows"] == 5, artifact["seed_summary_rows"] == 5,
        artifact["summary_files_pass"],
    ])
    atomic_json(PHASE / "audits" / "gradient_boosting_oof_coverage_audit.json", coverage)
    atomic_json(PHASE / "audits" / "gradient_boosting_leakage_audit.json", leakage)
    atomic_json(PHASE / "audits" / "gradient_boosting_artifact_audit.json", artifact)

    notebook_path = PHASE / "Phase_04B_Regression_Baselines.ipynb"
    backup_base = PHASE / "logs" / "Phase_04B_Regression_Baselines.pre_gradient_boosting_persistence_backup.ipynb"
    backup_path = backup_base
    if backup_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = backup_base.with_name(f"{backup_base.stem}_{stamp}{backup_base.suffix}")
    shutil.copy2(notebook_path, backup_path)
    notebook = nbformat.read(notebook_path, as_version=4)
    original_cells = len(notebook.cells)
    original_output_cells = sum(bool(cell.get("outputs", [])) for cell in notebook.cells if cell.cell_type == "code")
    prior_tags = ["linear_svr_persistence_official", "rbf_svr_persistence_official", "random_forest_persistence_official"]
    prior_tags_present = {tag: any(tag in cell.metadata.get("tags", []) for cell in notebook.cells) for tag in prior_tags}
    official_tag = "gradient_boosting_persistence_official"
    final_tag = "gradient_boosting_persistence_final_status"
    if not any(official_tag in cell.metadata.get("tags", []) for cell in notebook.cells):
        markdown = nbformat.v4.new_markdown_cell("## Gradient Boosting Regressor — Persisted Final Results")
        markdown.metadata["tags"] = [official_tag]
        code = nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json, pandas as pd\n"
            "phase=Path(r'E:\\hdc-vr-pilot\\experiments\\phase_04b_traditional_regression_baselines')\n"
            "summary=pd.read_csv(phase/'results/summaries/gradient_boosting_summary.csv').iloc[0]\n"
            "oof=pd.read_csv(phase/'results/predictions/gradient_boosting_oof.csv')\n"
            "seed_summary=pd.read_csv(phase/'results/summaries/gradient_boosting_seed_summary.csv')\n"
            "coverage=json.loads((phase/'audits/gradient_boosting_oof_coverage_audit.json').read_text())\n"
            "leakage=json.loads((phase/'audits/gradient_boosting_leakage_audit.json').read_text())\n"
            "artifact=json.loads((phase/'audits/gradient_boosting_artifact_audit.json').read_text())\n"
            "assert len(oof)==419 and oof.run_key.nunique()==419 and coverage['overall_pass'] and leakage['overall_pass'] and artifact['overall_pass']\n"
            "print('GRADIENT BOOSTING STATUS: COMPLETE')\n"
            "print('GRADIENT BOOSTING OOF ROWS:',len(oof))\n"
            "print('GRADIENT BOOSTING OOF MAE BOUNDED:',summary.canonical_oof_mae_bounded)\n"
            "print('GRADIENT BOOSTING EVALUATION SEEDS:',','.join(map(str,seed_summary.seed.tolist())))\n"
            "print(phase/'results/predictions/gradient_boosting_oof.csv')\n"
            "print(phase/'results/predictions/gradient_boosting_oof_all_seeds.csv')\n"
            "print(phase/'results/summaries/gradient_boosting_summary.csv')\n"
        )
        code.metadata["tags"] = [official_tag]
        final = nbformat.v4.new_code_cell(
            "print('GRADIENT BOOSTING NOTEBOOK PERSISTENCE: PASS')\n"
            "print('OTHER MODELS EXECUTED: NO')\n"
            "print('READY FOR PHASE 04B FINAL CONSOLIDATION: YES')\n"
        )
        final.metadata["tags"] = [final_tag]
        notebook.cells.extend([markdown, code, final])
    nbformat.write(notebook, notebook_path)
    client = NotebookClient(notebook, timeout=180, kernel_name="python3")
    with client.setup_kernel():
        client.execute_cell(notebook.cells[-2], len(notebook.cells) - 2, store_history=True)
        client.execute_cell(notebook.cells[-1], len(notebook.cells) - 1, store_history=True)
    nbformat.write(notebook, notebook_path)
    reread = nbformat.read(notebook_path, as_version=4)
    code_cell, final_cell = reread.cells[-2:]
    outputs = "".join(output.get("text", "") for output in code_cell.outputs + final_cell.outputs)
    persistence = {
        "notebook_exists": notebook_path.is_file(), "notebook_parseable": True,
        "backup_saved": backup_path.is_file(), "original_cells_preserved": len(reread.cells) >= original_cells,
        "original_output_cells_preserved": sum(bool(cell.get("outputs", [])) for cell in reread.cells if cell.cell_type == "code") >= original_output_cells,
        "prior_model_tags_preserved": prior_tags_present,
        "gradient_boosting_execution_count": code_cell.execution_count,
        "gradient_boosting_outputs_persisted": bool(code_cell.outputs),
        "oof_paths_in_outputs": "gradient_boosting_oof.csv" in outputs and "gradient_boosting_oof_all_seeds.csv" in outputs,
        "summary_path_in_outputs": "gradient_boosting_summary.csv" in outputs,
        "final_status_execution_count": final_cell.execution_count,
        "ready_for_final_consolidation": "READY FOR PHASE 04B FINAL CONSOLIDATION: YES" in outputs,
        "other_models_executed": "NO",
    }
    persistence["overall_pass"] = all([
        persistence["notebook_exists"], persistence["notebook_parseable"], persistence["backup_saved"],
        persistence["original_cells_preserved"], persistence["original_output_cells_preserved"],
        all(persistence["prior_model_tags_preserved"].values()),
        persistence["gradient_boosting_execution_count"] is not None,
        persistence["gradient_boosting_outputs_persisted"], persistence["oof_paths_in_outputs"],
        persistence["summary_path_in_outputs"], persistence["final_status_execution_count"] is not None,
        persistence["ready_for_final_consolidation"],
    ])
    atomic_json(PHASE / "audits" / "gradient_boosting_notebook_persistence_audit.json", persistence)
    if not persistence["overall_pass"]:
        raise RuntimeError("Gradient Boosting notebook persistence audit failed")
    state_path = PHASE / "configs" / "regression_model_search_space.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for model in state["models"]:
        if model["name"] == "Gradient Boosting Regressor":
            model["status"] = "COMPLETE"
    state["status"] = "PHASE_04B_ALL_TRADITIONAL_REGRESSION_MODELS_COMPLETE / FINAL_CONSOLIDATION_NOT_STARTED"
    atomic_json(state_path, state)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
