from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold


PHASE = Path(r"E:\hdc-vr-pilot\experiments\phase_04b_traditional_regression_baselines")
DATA_DIR = Path(r"E:\hdc-vr-pilot\experiments\phase_03_multimodal_dataset_labeling\data")
CONFIG_PATH = PHASE / "configs" / "gradient_boosting_configuration.json"
CHECKPOINT_DIR = PHASE / "results" / "checkpoints" / "gradient_boosting"
AUDIT_DIR = PHASE / "audits"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
PARAM_GRID = CONFIG["param_grid"]
SEEDS = [int(seed) for seed in CONFIG["evaluation_seeds"]]
TUNING_SEED = int(CONFIG["tuning_seed"])
EXPECTED_SHA = str(CONFIG["frozen_fold_sha256"])
CANDIDATE_COUNT = len(list(ParameterGrid(PARAM_GRID)))
if CANDIDATE_COUNT != 32:
    raise RuntimeError(f"Frozen Gradient Boosting grid must contain 32 candidates, got {CANDIDATE_COUNT}")
ID_COLUMNS = ["subject_id", "session_id", "run_id", "difficulty_level_raw", "difficulty_level", "run_key", "target_class", "target_score", "outer_fold"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def bounded_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, np.clip(y_pred, 1.0, 4.0))


SCORER = make_scorer(bounded_mae, greater_is_better=False)


def metrics(y: pd.Series, raw: np.ndarray, bounded: np.ndarray) -> dict[str, float]:
    return {
        "mae_raw": float(mean_absolute_error(y, raw)),
        "mae_bounded": float(mean_absolute_error(y, bounded)),
        "rmse_raw": float(mean_squared_error(y, raw) ** 0.5),
        "rmse_bounded": float(mean_squared_error(y, bounded) ** 0.5),
        "r2_raw": float(r2_score(y, raw)),
        "r2_bounded": float(r2_score(y, bounded)),
        "spearman_raw": float(spearmanr(y, raw).statistic),
        "spearman_bounded": float(spearmanr(y, bounded).statistic),
    }


def build_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("feature_selection", SelectKBest(score_func=f_regression)),
        ("regressor", GradientBoostingRegressor(
            loss="squared_error", subsample=1.0, min_samples_leaf=1,
            max_features=None, random_state=seed,
        )),
    ])


def expected_run_keys(folds: pd.DataFrame, fold: int) -> set[str]:
    return set(folds.loc[folds.outer_fold == fold, "run_key"])


def checkpoint_is_reusable(fold: int, folds: pd.DataFrame, sha: str) -> bool:
    required = [
        CHECKPOINT_DIR / f"gradient_boosting_fold_{fold}_inner_search_results.csv",
        CHECKPOINT_DIR / f"gradient_boosting_fold_{fold}_best_params.json",
        CHECKPOINT_DIR / f"gradient_boosting_fold_{fold}_predictions_all_seeds.csv",
        CHECKPOINT_DIR / f"gradient_boosting_fold_{fold}_predictions_seed_42.csv",
        CHECKPOINT_DIR / f"gradient_boosting_fold_{fold}_metrics_all_seeds.csv",
        CHECKPOINT_DIR / f"gradient_boosting_fold_{fold}_checkpoint_audit.json",
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        audit = json.loads(required[-1].read_text(encoding="utf-8"))
        all_seed = pd.read_csv(required[2])
        canonical = pd.read_csv(required[3])
        return bool(audit.get("overall_pass")) and audit.get("frozen_fold_sha256") == sha and audit.get("candidate_count") == CANDIDATE_COUNT and set(canonical.run_key) == expected_run_keys(folds, fold) and len(all_seed) == len(canonical) * len(SEEDS) and set(all_seed.seed) == set(SEEDS)
    except (OSError, ValueError, KeyError):
        return False


def run_fold(fold: int, data: pd.DataFrame, folds: pd.DataFrame, sha: str) -> str:
    if checkpoint_is_reusable(fold, folds, sha):
        print(f"FOLD {fold}: REUSED", flush=True)
        return "REUSED"
    test_keys = expected_run_keys(folds, fold)
    test = data.loc[data.run_key.isin(test_keys)].copy()
    train = data.loc[~data.run_key.isin(test_keys)].copy()
    assert set(test.run_key) == test_keys and len(test) + len(train) == len(data)
    outer_overlap = set(train.subject_id) & set(test.subject_id)
    assert not outer_overlap
    x_train, x_test = train.drop(columns=ID_COLUMNS), test.drop(columns=ID_COLUMNS)
    y_train, y_test = train.target_score, test.target_score
    assert x_train.shape[1] == 1176 and x_test.shape[1] == 1176
    inner = list(GroupKFold(n_splits=3).split(x_train, y_train, groups=train.subject_id))
    inner_audit = []
    for index, (tr, va) in enumerate(inner, 1):
        overlap = set(train.iloc[tr].subject_id) & set(train.iloc[va].subject_id)
        assert not overlap
        inner_audit.append({"inner_fold": index, "subject_overlap_count": len(overlap)})
    search = GridSearchCV(
        estimator=build_pipeline(TUNING_SEED), param_grid=PARAM_GRID, cv=inner,
        scoring=SCORER, refit=True, n_jobs=1, return_train_score=False, error_score="raise",
    )
    start = time.perf_counter()
    search.fit(x_train, y_train)
    search_seconds = time.perf_counter() - start
    assert len(search.cv_results_["params"]) == CANDIDATE_COUNT
    cv = pd.DataFrame(search.cv_results_)
    search_results = pd.DataFrame({
        "candidate_index": np.arange(1, len(cv) + 1),
        "feature_selection__k": cv["param_feature_selection__k"],
        "regressor__n_estimators": cv["param_regressor__n_estimators"],
        "regressor__learning_rate": cv["param_regressor__learning_rate"],
        "regressor__max_depth": cv["param_regressor__max_depth"],
        "split_1_validation_bounded_mae": -cv["split0_test_score"],
        "split_2_validation_bounded_mae": -cv["split1_test_score"],
        "split_3_validation_bounded_mae": -cv["split2_test_score"],
        "mean_validation_bounded_mae": -cv["mean_test_score"],
        "std_validation_bounded_mae": cv["std_test_score"],
        "rank": cv["rank_test_score"],
    })
    best = dict(search.best_params_)
    best_params = {
        "outer_fold": fold, "best_params": best, "candidate_count": CANDIDATE_COUNT,
        "best_inner_bounded_mae": float(-search.best_score_), "tuning_seed": TUNING_SEED,
        "evaluation_seeds": SEEDS, "canonical_seed": TUNING_SEED,
        "frozen_fold_sha256": sha, "pipeline": "median imputer + indicator, variance filter, SelectKBest, GradientBoostingRegressor",
        "utc_timestamp": utc_now(),
    }
    atomic_csv(CHECKPOINT_DIR / f"gradient_boosting_fold_{fold}_inner_search_results.csv", search_results)
    atomic_json(CHECKPOINT_DIR / f"gradient_boosting_fold_{fold}_best_params.json", best_params)
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, object]] = []
    for seed in SEEDS:
        estimator = build_pipeline(seed).set_params(**best)
        fit_start = time.perf_counter()
        estimator.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        pred_start = time.perf_counter()
        raw = estimator.predict(x_test)
        prediction_seconds = time.perf_counter() - pred_start
        bounded = np.clip(raw, 1.0, 4.0)
        selected = {
            "selected_k": best["feature_selection__k"],
            "selected_n_estimators": best["regressor__n_estimators"],
            "selected_learning_rate": best["regressor__learning_rate"],
            "selected_max_depth": best["regressor__max_depth"],
        }
        prediction = test[["run_key", "subject_id", "session_id", "run_id", "outer_fold", "target_score"]].copy()
        prediction.insert(0, "model_slug", "gradient_boosting")
        prediction.insert(0, "model", "Gradient Boosting Regressor")
        prediction.insert(2, "seed", seed)
        prediction["prediction_raw"] = raw
        prediction["prediction_bounded"] = bounded
        prediction["absolute_error_raw"] = np.abs(y_test.to_numpy() - raw)
        prediction["absolute_error_bounded"] = np.abs(y_test.to_numpy() - bounded)
        for key, value in selected.items():
            prediction[key] = value
        predictions.append(prediction)
        row: dict[str, object] = {"outer_fold": fold, "seed": seed, **selected, "best_inner_bounded_mae": float(-search.best_score_), **metrics(y_test, raw, bounded), "fit_time_seconds": fit_seconds, "prediction_time_seconds": prediction_seconds}
        fold_metrics.append(row)
    all_seed = pd.concat(predictions, ignore_index=True)
    canonical = all_seed.loc[all_seed.seed == TUNING_SEED].copy()
    assert len(canonical) == len(test) and canonical.run_key.nunique() == len(test) and all_seed.prediction_raw.notna().all() and all_seed.prediction_bounded.between(1, 4).all()
    atomic_csv(CHECKPOINT_DIR / f"gradient_boosting_fold_{fold}_predictions_all_seeds.csv", all_seed)
    atomic_csv(CHECKPOINT_DIR / f"gradient_boosting_fold_{fold}_predictions_seed_42.csv", canonical)
    atomic_csv(CHECKPOINT_DIR / f"gradient_boosting_fold_{fold}_metrics_all_seeds.csv", pd.DataFrame(fold_metrics))
    audit = {
        "outer_fold": fold, "frozen_fold_sha256": sha, "candidate_count": CANDIDATE_COUNT,
        "tuning_seed": 42, "evaluation_seeds": SEEDS, "outer_subject_overlap_count": len(outer_overlap),
        "inner_subject_isolation": inner_audit, "test_rows": len(test), "canonical_unique_run_keys": int(canonical.run_key.nunique()),
        "all_seed_rows": len(all_seed), "all_seed_unique_run_keys": int(all_seed.run_key.nunique()),
        "all_predictions_nonmissing": bool(all_seed.prediction_raw.notna().all() and all_seed.prediction_bounded.notna().all()),
        "bounded_range_pass": bool(all_seed.prediction_bounded.between(1, 4).all()),
        "all_seed_same_best_params": True, "search_time_seconds": search_seconds,
        "overall_pass": True, "utc_timestamp": utc_now(),
    }
    atomic_json(CHECKPOINT_DIR / f"gradient_boosting_fold_{fold}_checkpoint_audit.json", audit)
    atomic_json(AUDIT_DIR / f"gradient_boosting_fold_{fold}_leakage_audit.json", {
        "outer_fold": fold,
        "frozen_fold_sha256": sha,
        "outer_subject_overlap_count": len(outer_overlap),
        "inner_subject_isolation": inner_audit,
        "outer_test_used_for_selection": False,
        "overall_pass": len(outer_overlap) == 0 and all(item["subject_overlap_count"] == 0 for item in inner_audit),
        "utc_timestamp": utc_now(),
    })
    print(f"FOLD {fold}: COMPLETE ({len(test)} test rows; {CANDIDATE_COUNT} candidates; {len(SEEDS)} seeds)", flush=True)
    return "COMPLETE"


def main() -> None:
    input_path = DATA_DIR / "primary_without_performance.csv"
    fold_path = DATA_DIR / "fold_assignments.csv"
    sha = hashlib.sha256(fold_path.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise RuntimeError("Frozen fold checksum mismatch")
    data = pd.read_csv(input_path)
    folds = pd.read_csv(fold_path)
    data = data.merge(folds[["run_key", "outer_fold"]], on="run_key", how="left", validate="one_to_one", suffixes=("", "_frozen"))
    if "outer_fold_frozen" in data:
        data["outer_fold"] = data["outer_fold_frozen"]
        data = data.drop(columns=["outer_fold_frozen"])
    assert len(data) == 419 and data.subject_id.nunique() == 35 and data.outer_fold.nunique() == 5
    start_fold = int(os.environ.get("GB_START_FOLD", "1"))
    end_fold = int(os.environ.get("GB_END_FOLD", str(start_fold)))
    if not 1 <= start_fold <= end_fold <= 5:
        raise ValueError("GB_START_FOLD and GB_END_FOLD must be in 1..5")
    for fold in range(start_fold, end_fold + 1):
        run_fold(fold, data, folds, sha)


if __name__ == "__main__":
    main()
