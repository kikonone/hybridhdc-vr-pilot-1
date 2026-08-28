from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"E:\hdc-vr-pilot")
PHASE = ROOT / "experiments" / "phase_04b_traditional_regression_baselines"
DATA = ROOT / "experiments" / "phase_03_multimodal_dataset_labeling" / "data" / "primary_without_performance.csv"
FOLDS = ROOT / "experiments" / "phase_03_multimodal_dataset_labeling" / "data" / "fold_assignments.csv"
EXPECTED_SHA = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
NON_FEATURES = ["subject_id", "session_id", "run_id", "difficulty_level_raw", "difficulty_level", "run_key", "target_class", "target_score", "outer_fold"]


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def bounded_mae(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return -mean_absolute_error(y_true, np.clip(prediction, 1.0, 4.0))


def safe_spearman(y_true: np.ndarray, prediction: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = spearmanr(y_true, prediction).statistic
    return float(value) if np.isfinite(value) else np.nan


parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True, choices=[3, 4, 5])
args = parser.parse_args()

actual_sha = hashlib.sha256(FOLDS.read_bytes()).hexdigest()
if actual_sha != EXPECTED_SHA:
    raise RuntimeError("Frozen fold checksum mismatch")
input_audit = json.loads((PHASE / "audits" / "phase04b_input_and_fold_audit.json").read_text(encoding="utf-8"))
if not input_audit["overall_pass"]:
    raise RuntimeError("Phase 04B input audit is not PASS")

data = pd.read_csv(DATA)
assignments = pd.read_csv(FOLDS)
features = [column for column in data.columns if column not in NON_FEATURES]
if len(data) != 419 or data.subject_id.nunique() != 35 or len(features) != 1176:
    raise RuntimeError("Frozen data contract mismatch")

fold = args.fold
test_mask = assignments.outer_fold.eq(fold).to_numpy()
train = data.loc[~test_mask].copy()
test = data.loc[test_mask].copy()
outer_overlap = set(train.subject_id) & set(test.subject_id)
if outer_overlap:
    raise RuntimeError(f"Outer subject leakage in fold {fold}")

inner_splits = []
inner_audit = []
for inner_fold, (train_idx, validation_idx) in enumerate(GroupKFold(n_splits=3).split(train[features], train.target_score, train.subject_id), start=1):
    overlap = set(train.iloc[train_idx].subject_id) & set(train.iloc[validation_idx].subject_id)
    if overlap:
        raise RuntimeError(f"Inner subject leakage in fold {fold}/{inner_fold}")
    inner_splits.append((train_idx, validation_idx))
    inner_audit.append({"inner_fold": inner_fold, "subject_overlap_count": 0})

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
    ("variance_filter", VarianceThreshold(threshold=0.0)),
    ("scaler", StandardScaler()),
    ("feature_selection", SelectKBest(score_func=f_regression)),
    ("regressor", ElasticNet(fit_intercept=True, max_iter=100000, tol=1e-4, selection="cyclic")),
])
grid = {
    "feature_selection__k": [50, 100, 200, "all"],
    "regressor__alpha": [0.001, 0.01, 0.1, 1.0],
    "regressor__l1_ratio": [0.1, 0.5, 0.9],
}
checkpoint = PHASE / "results" / "checkpoints" / "elastic_net"

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", ConvergenceWarning)
    started = time.perf_counter()
    search = GridSearchCV(pipeline, grid, cv=inner_splits, scoring=make_scorer(bounded_mae), refit=True, n_jobs=1, return_train_score=False, error_score="raise")
    search.fit(train[features], train.target_score)
    search_time = time.perf_counter() - started

warning_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
best = search.best_estimator_.named_steps["regressor"]
n_iter = int(np.max(np.atleast_1d(best.n_iter_)))
converged = n_iter < 100000
best_params = search.best_params_
cv = pd.DataFrame(search.cv_results_)
inner_results = pd.DataFrame({
    "candidate_index": range(48),
    "feature_selection__k": cv["param_feature_selection__k"].astype(str),
    "regressor__alpha": cv["param_regressor__alpha"].astype(float),
    "regressor__l1_ratio": cv["param_regressor__l1_ratio"].astype(float),
    "split_1_validation_bounded_mae": -cv["split0_test_score"],
    "split_2_validation_bounded_mae": -cv["split1_test_score"],
    "split_3_validation_bounded_mae": -cv["split2_test_score"],
    "mean_validation_bounded_mae": -cv["mean_test_score"],
    "std_validation_bounded_mae": cv["std_test_score"],
    "rank": cv["rank_test_score"],
    "search_convergence_warning_count_total": warning_count,
    "candidate_level_warning_mapping": "NOT_AVAILABLE_FROM_GRIDSEARCHCV",
    "status": "COMPLETE" if converged else "FAILED_CONVERGENCE_AT_100000",
})
atomic_csv(checkpoint / f"elastic_net_fold_{fold}_inner_search_results.csv", inner_results)
atomic_json(checkpoint / f"elastic_net_fold_{fold}_best_params.json", {"outer_fold": fold, "best_params": best_params, "candidate_count": 48, "best_inner_bounded_mae": float(-search.best_score_), "n_iter": n_iter, "max_iter_used": 100000, "tol": 1e-4, "converged": converged, "frozen_fold_sha256": actual_sha, "recovery_version": "V1"})
atomic_json(checkpoint / f"elastic_net_fold_{fold}_convergence_diagnostics.json", {"outer_fold": fold, "n_iter": n_iter, "max_iter": 100000, "search_convergence_warning_count_total": warning_count, "candidate_level_warning_mapping": "NOT_AVAILABLE_FROM_GRIDSEARCHCV", "converged": converged})
if not converged:
    raise RuntimeError(f"Fold {fold} FAILED_CONVERGENCE_AT_100000")

started = time.perf_counter()
raw = search.predict(test[features])
prediction_time = time.perf_counter() - started
bounded = np.clip(raw, 1.0, 4.0)
y = test.target_score.to_numpy(dtype=float)
predictions = test[["run_key", "subject_id", "session_id", "run_id", "outer_fold", "target_score"]].copy()
predictions.insert(0, "model_slug", "elastic_net")
predictions.insert(0, "model", "Elastic Net")
predictions["prediction_raw"] = raw
predictions["prediction_bounded"] = bounded
predictions["absolute_error_raw"] = np.abs(y - raw)
predictions["absolute_error_bounded"] = np.abs(y - bounded)
predictions["selected_k"] = best_params["feature_selection__k"]
predictions["selected_alpha"] = best_params["regressor__alpha"]
predictions["selected_l1_ratio"] = best_params["regressor__l1_ratio"]
predictions["nonzero_coefficient_count"] = int(np.count_nonzero(best.coef_))
predictions["max_iter_used"] = 100000
predictions["checkpoint_reused"] = False
predictions["recovery_version"] = "V1"
atomic_csv(checkpoint / f"elastic_net_fold_{fold}_predictions.csv", predictions)
metrics = {"model": "Elastic Net", "model_slug": "elastic_net", "outer_fold": fold, "train_rows": len(train), "test_rows": len(test), "train_subjects": train.subject_id.nunique(), "test_subjects": test.subject_id.nunique(), "subject_overlap_count": 0, "inner_candidate_count": 48, "selected_k": best_params["feature_selection__k"], "selected_alpha": best_params["regressor__alpha"], "selected_l1_ratio": best_params["regressor__l1_ratio"], "nonzero_coefficient_count": int(np.count_nonzero(best.coef_)), "best_inner_bounded_mae": float(-search.best_score_), "mae_raw": float(mean_absolute_error(y, raw)), "mae_bounded": float(mean_absolute_error(y, bounded)), "rmse_raw": float(np.sqrt(mean_squared_error(y, raw))), "rmse_bounded": float(np.sqrt(mean_squared_error(y, bounded))), "r2_raw": float(r2_score(y, raw)), "r2_bounded": float(r2_score(y, bounded)), "spearman_raw": safe_spearman(y, raw), "spearman_bounded": safe_spearman(y, bounded), "convergence_warning_count": warning_count, "best_estimator_n_iter": n_iter, "best_estimator_converged": converged, "fit_and_search_time_seconds": search_time, "prediction_time_seconds": prediction_time, "max_iter_used": 100000, "checkpoint_reused": False, "recovery_version": "V1", "inner_subject_isolation": inner_audit}
atomic_json(checkpoint / f"elastic_net_fold_{fold}_metrics.json", metrics)
print(json.dumps({"fold": fold, "converged": converged, "n_iter": n_iter, "best_params": best_params}, indent=2))
