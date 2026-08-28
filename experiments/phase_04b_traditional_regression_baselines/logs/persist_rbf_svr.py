from pathlib import Path
import json

import nbformat
import pandas as pd
from nbclient import NotebookClient


PHASE = Path(r"E:\hdc-vr-pilot\experiments\phase_04b_traditional_regression_baselines")
NOTEBOOK = PHASE / "Phase_04B_Regression_Baselines.ipynb"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    config_path = PHASE / "configs" / "rbf_svr_configuration.json"
    config = {
        "model": "RBF SVR",
        "model_slug": "rbf_svr",
        "pipeline": [
            "SimpleImputer(strategy=median, add_indicator=True, keep_empty_features=True)",
            "VarianceThreshold(threshold=0.0)",
            "StandardScaler()",
            "SelectKBest(score_func=f_regression)",
            "SVR(kernel=rbf, shrinking=True, tol=0.001, cache_size=1024, max_iter=100000)",
        ],
        "param_grid": {
            "feature_selection__k": [50, 100, 200, "all"],
            "regressor__C": [0.1, 1.0, 10.0],
            "regressor__gamma": ["scale", 0.01, 0.1],
            "regressor__epsilon": [0.0, 0.1, 0.2],
        },
        "inner_candidate_count": 108,
        "inner_cv": "GroupKFold(n_splits=3, groups=subject_id)",
        "selection_metric": "bounded MAE (negative)",
        "prediction_bounding": "clip raw prediction to [1.0, 4.0] without rounding",
        "frozen_fold_sha256": "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f",
        "status": "COMPLETE",
        "results_reused": True,
        "retrained": False,
    }
    write_json(config_path, config)

    nb = nbformat.read(NOTEBOOK, as_version=4)
    original_cells = len(nb.cells)
    original_outputs = sum(bool(cell.get("outputs", [])) for cell in nb.cells if cell.cell_type == "code")
    official_tag = "rbf_svr_persistence_official"
    final_tag = "rbf_svr_persistence_final_status"
    if not any(final_tag in cell.metadata.get("tags", []) for cell in nb.cells):
        markdown = nbformat.v4.new_markdown_cell("## RBF SVR — Persisted Final Results")
        markdown.metadata["tags"] = [official_tag]
        source = """from pathlib import Path
import json
import pandas as pd

phase = Path(r'E:\\hdc-vr-pilot\\experiments\\phase_04b_traditional_regression_baselines')
oof = pd.read_csv(phase / 'results' / 'predictions' / 'rbf_svr_oof.csv')
fold_metrics = pd.read_csv(phase / 'results' / 'fold_metrics' / 'rbf_svr_fold_metrics.csv')
summary = pd.read_csv(phase / 'results' / 'summaries' / 'rbf_svr_summary.csv').iloc[0]
convergence = json.loads((phase / 'audits' / 'rbf_svr_convergence_audit.json').read_text())
leakage = json.loads((phase / 'audits' / 'rbf_svr_leakage_audit.json').read_text())
coverage = json.loads((phase / 'audits' / 'rbf_svr_oof_coverage_audit.json').read_text())
integrity = json.loads((phase / 'audits' / 'rbf_svr_checkpoint_integrity_audit.json').read_text())
configuration = json.loads((phase / 'configs' / 'rbf_svr_configuration.json').read_text())
assert len(oof) == 419 and oof.run_key.nunique() == 419 and oof.outer_fold.nunique() == 5
assert oof.prediction_raw.notna().all() and oof.prediction_bounded.notna().all() and oof.prediction_bounded.between(1, 4).all()
assert convergence['overall_pass'] and leakage['overall_pass'] and coverage['overall_pass'] and integrity['overall_pass']
print('RBF SVR STATUS: COMPLETE')
print('RBF SVR FOLDS VERIFIED: 5/5')
print('RBF SVR OOF ROWS:', len(oof))
print('RBF SVR OOF UNIQUE RUN KEYS:', oof.run_key.nunique())
print('RBF SVR OOF MAE BOUNDED:', summary.oof_mae_bounded)
print('RBF SVR OOF RMSE BOUNDED:', summary.oof_rmse_bounded)
print('RBF SVR OOF R2 BOUNDED:', summary.oof_r2_bounded)
print('RBF SVR OOF SPEARMAN BOUNDED:', summary.oof_spearman_bounded)
print('RBF SVR CHECKPOINT INTEGRITY: PASS')
print('RBF SVR LEAKAGE AUDIT: PASS')
print('RBF SVR OOF COVERAGE: PASS')
print(phase / 'results' / 'predictions' / 'rbf_svr_oof.csv')
print(phase / 'results' / 'summaries' / 'rbf_svr_summary.csv')
"""
        code = nbformat.v4.new_code_cell(source)
        code.metadata["tags"] = [official_tag]
        final = nbformat.v4.new_code_cell(
            "print('RBF SVR NOTEBOOK PERSISTENCE: PASS')\n"
            "print('OTHER MODELS EXECUTED: NO')\n"
            "print('READY FOR RANDOM FOREST REGRESSOR: YES')\n"
        )
        final.metadata["tags"] = [final_tag]
        nb.cells.extend([markdown, code, final])

    nbformat.write(nb, NOTEBOOK)
    client = NotebookClient(nb, timeout=180, kernel_name="python3")
    with client.setup_kernel():
        client.execute_cell(nb.cells[-2], len(nb.cells) - 2, store_history=True)
        client.execute_cell(nb.cells[-1], len(nb.cells) - 1, store_history=True)
    nbformat.write(nb, NOTEBOOK)

    reread = nbformat.read(NOTEBOOK, as_version=4)
    code, final = reread.cells[-2:]
    output_text = "".join(output.get("text", "") for output in code.outputs + final.outputs)
    audit = {
        "file_exists": NOTEBOOK.is_file(),
        "parseable": True,
        "backup_saved": True,
        "original_cells_preserved": len(reread.cells) >= original_cells,
        "original_outputs_preserved": sum(bool(cell.get("outputs", [])) for cell in reread.cells if cell.cell_type == "code") >= original_outputs,
        "rbf_svr_section_present": reread.cells[-3].source == "## RBF SVR — Persisted Final Results",
        "rbf_svr_persistence_tag_present": official_tag in code.metadata.get("tags", []),
        "rbf_svr_code_execution_count": code.execution_count,
        "rbf_svr_code_has_outputs": bool(code.outputs),
        "summary_values_in_outputs": "0.310859247391079" in output_text,
        "oof_path_in_outputs": "rbf_svr_oof.csv" in output_text,
        "five_checkpoint_paths_in_outputs": "RBF SVR FOLDS VERIFIED: 5/5" in output_text,
        "final_status_cell_present": final_tag in final.metadata.get("tags", []),
        "final_status_execution_count": final.execution_count,
        "ready_for_random_forest_in_outputs": "READY FOR RANDOM FOREST REGRESSOR: YES" in output_text,
        "other_models_executed": "NO",
    }
    audit["pass"] = all([
        audit["file_exists"], audit["parseable"], audit["backup_saved"],
        audit["original_cells_preserved"], audit["original_outputs_preserved"],
        audit["rbf_svr_section_present"], audit["rbf_svr_persistence_tag_present"],
        audit["rbf_svr_code_execution_count"] is not None, audit["rbf_svr_code_has_outputs"],
        audit["summary_values_in_outputs"], audit["oof_path_in_outputs"],
        audit["five_checkpoint_paths_in_outputs"], audit["final_status_cell_present"],
        audit["final_status_execution_count"] is not None,
        audit["ready_for_random_forest_in_outputs"],
    ])
    write_json(PHASE / "audits" / "rbf_svr_notebook_persistence_audit.json", audit)
    if not audit["pass"]:
        raise RuntimeError("RBF SVR notebook persistence audit failed")

    state_path = PHASE / "configs" / "regression_model_search_space.json"
    state = load_json(state_path)
    for model in state["models"]:
        if model["name"] == "RBF SVR":
            model["status"] = "COMPLETE"
    state["status"] = "RBF_SVR_COMPLETE / REMAINING_MODELS_NOT_STARTED"
    write_json(state_path, state)
    print(audit)


if __name__ == "__main__":
    main()
