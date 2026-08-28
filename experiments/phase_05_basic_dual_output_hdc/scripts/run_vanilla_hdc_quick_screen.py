"""Run one explicitly selected outer-fold Vanilla Prototype HDC quick screen."""

from __future__ import annotations

import csv
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, TextIO

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler


PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path.insert(0, str(PHASE / "src"))

from phase05_hdc_core import (  # noqa: E402
    EqualWidthQuantizer,
    build_prototypes,
    cosine_similarity_scores,
    incremental_encode_prefixes,
    predict_smallest_class_tie,
)


PRIMARY = ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv"
FOLDS = ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv"
EXPECTED_PRIMARY_SHA = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLD_SHA = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
SEED = 42
NON_PREDICTIVE = {
    "subject_id", "session_id", "run_id", "difficulty_level_raw", "difficulty_level",
    "run_key", "target_class", "target_score", "outer_fold",
}
CONTRACT_PATHS = [
    PHASE / "configs/phase05_hdc_encoding_contract.json",
    PHASE / "configs/phase05_hdc_model_selection_contract.json",
    PHASE / "configs/phase05_hdc_regression_heads_contract.json",
    PHASE / "configs/phase05_hdc_efficiency_protocol.json",
    PHASE / "configs/phase05_hdc_search_space.json",
]
class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def required_fold_artifacts(fold: int) -> list[str]:
    return [
        f"results/checkpoints/quick_screen/vanilla_hdc_quick_screen_fold_{fold}_checkpoint.json",
        f"results/summaries/vanilla_hdc_quick_screen_fold_{fold}_candidates.csv",
        f"results/summaries/vanilla_hdc_quick_screen_fold_{fold}_best_config.json",
        f"results/fold_metrics/vanilla_hdc_quick_screen_fold_{fold}_inner_metrics.csv",
        f"results/efficiency/vanilla_hdc_quick_screen_fold_{fold}_efficiency.csv",
        f"audits/vanilla_hdc_quick_screen_fold_{fold}_leakage_audit.json",
        f"audits/vanilla_hdc_quick_screen_fold_{fold}_artifact_audit.json",
        f"audits/vanilla_hdc_quick_screen_fold_{fold}_notebook_persistence_audit.json",
    ]


def validate_prior_fold_artifacts(outer_fold: int) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for prior_fold in range(1, outer_fold):
        artifacts = required_fold_artifacts(prior_fold)
        for relative in artifacts:
            path = PHASE / relative
            if not path.is_file():
                raise RuntimeError(f"required Fold {prior_fold} artifact is missing: {relative}")
            if path.suffix == ".json":
                load_json(path)
            hashes[relative] = sha256(path)
        checkpoint = load_json(PHASE / artifacts[0])
        leakage = load_json(PHASE / artifacts[5])
        artifact = load_json(PHASE / artifacts[6])
        persistence = load_json(PHASE / artifacts[7])
        if checkpoint.get("candidates_completed") != 16 or not checkpoint.get("overall_pass"):
            raise RuntimeError(f"Fold {prior_fold} checkpoint is not complete/pass")
        if leakage.get("result") != "PASS" or artifact.get("result") != "PASS":
            raise RuntimeError(f"Fold {prior_fold} leakage or artifact audit is not PASS")
        if persistence.get("persistence_result") != "PASS":
            raise RuntimeError(f"Fold {prior_fold} notebook persistence is not PASS")
    return hashes


def validate_gate(outer_fold: int) -> tuple[pd.DataFrame, set[str], list[str], dict[str, str]]:
    primary_sha = sha256(PRIMARY)
    fold_sha = sha256(FOLDS)
    if primary_sha != EXPECTED_PRIMARY_SHA or fold_sha != EXPECTED_FOLD_SHA:
        raise RuntimeError("frozen Phase 03 checksum mismatch")
    contracts = {path.name: load_json(path) for path in CONTRACT_PATHS}
    experiment = load_json(PHASE / "configs/phase05_experiment_contract.json")
    if experiment.get("status") != "CONTRACT_FROZEN_NOT_TRAINED":
        raise RuntimeError("Phase 05 status is not CONTRACT_FROZEN_NOT_TRAINED")
    search = contracts["phase05_hdc_search_space.json"]
    if search["status"] != "CONTRACT_FROZEN_NOT_EXECUTED":
        raise RuntimeError("HDC search contract is not frozen")
    if search["rapid_screening"]["candidate_count"] != 16:
        raise RuntimeError("frozen candidate count is not 16")

    metadata = pd.read_csv(PRIMARY, usecols=["run_key", "subject_id", "target_class", "outer_fold"])
    header = pd.read_csv(PRIMARY, nrows=0).columns.tolist()
    feature_names = [name for name in header if name not in NON_PREDICTIVE]
    if len(metadata) != 419 or metadata["subject_id"].nunique() != 35:
        raise RuntimeError("unexpected primary rows or subjects")
    if len(feature_names) != 1176 or sorted(metadata["target_class"].unique()) != [0, 1, 2, 3]:
        raise RuntimeError("unexpected primary feature or class contract")
    if not metadata["run_key"].is_unique:
        raise RuntimeError("primary run_key is not unique")

    folds = pd.read_csv(FOLDS)
    outer_train = folds.loc[folds["outer_fold"] != outer_fold].copy()
    outer_test = folds.loc[folds["outer_fold"] == outer_fold].copy()
    overlap = set(outer_train["subject_id"]) & set(outer_test["subject_id"])
    if overlap:
        raise RuntimeError(f"outer Fold {outer_fold} subject isolation failed")
    inner = list(GroupKFold(n_splits=3).split(outer_train, groups=outer_train["subject_id"]))
    if len(inner) != 3:
        raise RuntimeError("inner GroupKFold is not feasible")
    print(
        f"Gate PASS: fold={outer_fold} rows=419 subjects=35 features=1176 outer_train={len(outer_train)} "
        f"train_subjects={outer_train['subject_id'].nunique()} outer_test_count_from_assignments={len(outer_test)}"
    )
    hashes = {path.name: sha256(path) for path in CONTRACT_PATHS}
    hashes["primary_without_performance.csv"] = primary_sha
    hashes["fold_assignments.csv"] = fold_sha
    return outer_train, set(outer_test["run_key"]), feature_names, hashes


def load_outer_training_features(
    outer_train_assignments: pd.DataFrame,
    forbidden_test_keys: set[str],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Materialize feature values for allowed training keys only.

    The CSV is streamed because it is monolithic; rows assigned to the selected outer test are
    rejected by run_key before any of their feature fields are converted or stored.
    """
    allowed = set(outer_train_assignments["run_key"])
    assignment = outer_train_assignments.set_index("run_key")
    matrices: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    run_keys: list[str] = []
    with PRIMARY.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = {name: position for position, name in enumerate(header)}
        feature_indices = [index[name] for name in feature_names]
        run_key_index = index["run_key"]
        for row in reader:
            run_key = row[run_key_index]
            if run_key in forbidden_test_keys:
                continue
            if run_key not in allowed:
                continue
            values = [float(row[position]) if row[position].strip() else np.nan for position in feature_indices]
            matrices.append(values)
            labels.append(int(assignment.at[run_key, "target_class"]))
            groups.append(str(assignment.at[run_key, "subject_id"]))
            run_keys.append(run_key)
    if set(run_keys) != allowed or len(run_keys) != len(allowed):
        raise RuntimeError("outer-training feature extraction did not align one-to-one")
    return (
        np.asarray(matrices, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=object),
        run_keys,
    )


def fitted_preprocessing(
    train_values: np.ndarray,
    validation_values: np.ndarray,
    train_labels: np.ndarray,
    input_feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    train_imputed = imputer.fit_transform(train_values)
    validation_imputed = imputer.transform(validation_values)
    imputed_names = imputer.get_feature_names_out(input_feature_names)
    variance = VarianceThreshold(threshold=0.0)
    train_variable = variance.fit_transform(train_imputed)
    validation_variable = variance.transform(validation_imputed)
    variable_names = imputed_names[variance.get_support()]
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_variable)
    validation_scaled = scaler.transform(validation_variable)
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(train_scaled, train_labels)
    scores = np.nan_to_num(selector.scores_, nan=-np.inf)
    ranking = np.argsort(scores, kind="mergesort")[::-1]
    ranked_names = variable_names[ranking].astype(str).tolist()
    preprocessing_bytes = sum(
        array.nbytes
        for array in (
            np.asarray(imputer.statistics_), np.asarray(variance.variances_),
            np.asarray(scaler.mean_), np.asarray(scaler.scale_), np.asarray(scores),
        )
    )
    return train_scaled[:, ranking], validation_scaled[:, ranking], ranked_names, int(preprocessing_bytes)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    recalls = recall_score(labels, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "severe_error_rate": float(np.mean(np.abs(labels - predictions) >= 2)),
        "recall_class_0": float(recalls[0]), "recall_class_1": float(recalls[1]),
        "recall_class_2": float(recalls[2]), "recall_class_3": float(recalls[3]),
    }


def model_bytes(dimension: int, levels: int, effective_k: int, preprocessing_bytes: int) -> int:
    return int(
        effective_k * dimension  # identity int8
        + levels * dimension  # level int8
        + dimension  # tie int8
        + 4 * dimension * np.dtype(np.int32).itemsize  # prototypes
        + 2 * effective_k * np.dtype(np.float64).itemsize  # quantizer min/max
        + preprocessing_bytes
    )


def run_unit_tests() -> None:
    command = [sys.executable, "-m", "pytest", str(PHASE / "tests/test_phase05_hdc_core.py"), "-q"]
    completed = subprocess.run(command, cwd=PHASE, text=True, capture_output=True, check=False)
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"unit tests failed with exit code {completed.returncode}")
    print("Unit tests PASS; quick screening is authorized.")


def quick_screen(outer_fold: int) -> None:
    run_unit_tests()
    historical_hashes_before = validate_prior_fold_artifacts(outer_fold)
    outer_train, forbidden_test_keys, feature_names, frozen_hashes_before = validate_gate(outer_fold)
    values, labels, groups, _ = load_outer_training_features(
        outer_train, forbidden_test_keys, feature_names
    )
    if values.shape != (len(outer_train), 1176) or len(set(groups)) != outer_train["subject_id"].nunique():
        raise RuntimeError(f"unexpected outer Fold {outer_fold} training matrix")

    search = load_json(PHASE / "configs/phase05_hdc_search_space.json")["rapid_screening"]
    candidates = list(
        ParameterGrid(
            {
                "dimension": search["dimensions"], "levels": search["levels"],
                "k": search["feature_selection__k"], "seed": search["seeds"],
            }
        )
    )
    if len(candidates) != 16:
        raise RuntimeError("ParameterGrid did not produce 16 candidates")
    candidate_order = {
        (int(c["dimension"]), int(c["levels"]), str(c["k"]), int(c["seed"])): index
        for index, c in enumerate(candidates)
    }
    inner_rows: list[dict[str, Any]] = []
    leakage_splits: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=3)

    for inner_fold, (train_index, validation_index) in enumerate(
        splitter.split(values, labels, groups), start=1
    ):
        train_subjects = set(groups[train_index])
        validation_subjects = set(groups[validation_index])
        overlap = train_subjects & validation_subjects
        if overlap:
            raise RuntimeError(f"inner fold {inner_fold} has subject leakage")
        leakage_splits.append(
            {
                "inner_fold": inner_fold, "train_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)), "train_subjects": len(train_subjects),
                "validation_subjects": len(validation_subjects), "subject_overlap_count": 0,
            }
        )
        preprocessing_start = time.perf_counter()
        train_processed, validation_processed, ranked_names, preprocessing_bytes = fitted_preprocessing(
            values[train_index], values[validation_index], labels[train_index], feature_names
        )
        preprocessing_seconds = time.perf_counter() - preprocessing_start
        split_point = len(train_index)
        for levels in search["levels"]:
            quantizer = EqualWidthQuantizer(int(levels)).fit(train_processed)
            quantized = np.vstack(
                [quantizer.transform(train_processed), quantizer.transform(validation_processed)]
            )
            encoding = incremental_encode_prefixes(
                quantized, ranked_names, int(levels), SEED,
                search["feature_selection__k"], max(search["dimensions"]),
            )
            for k_value in search["feature_selection__k"]:
                k_label = str(k_value)
                effective_k = len(ranked_names) if k_value == "all" else int(k_value)
                encoded = encoding.samples_by_k[k_label]
                for dimension in search["dimensions"]:
                    train_hv = encoded[:split_point, : int(dimension)]
                    validation_hv = encoded[split_point:, : int(dimension)]
                    prototype_start = time.perf_counter()
                    prototypes = build_prototypes(train_hv, labels[train_index])
                    prototype_seconds = time.perf_counter() - prototype_start
                    inference_start = time.perf_counter()
                    similarities = cosine_similarity_scores(validation_hv, prototypes)
                    predictions = predict_smallest_class_tie(similarities)
                    inference_seconds = time.perf_counter() - inference_start
                    metrics = classification_metrics(labels[validation_index], predictions)
                    row: dict[str, Any] = {
                        "outer_fold": outer_fold, "inner_fold": inner_fold,
                        "dimension": int(dimension), "levels": int(levels), "k": k_label,
                        "effective_k": effective_k, "seed": SEED,
                        "train_rows": int(len(train_index)), "validation_rows": int(len(validation_index)),
                        "train_subjects": len(train_subjects), "validation_subjects": len(validation_subjects),
                        "subject_overlap_count": 0, "preprocessing_seconds": preprocessing_seconds,
                        "encoding_seconds": encoding.feature_completion_seconds[k_label],
                        "prototype_training_seconds": prototype_seconds,
                        "inference_seconds": inference_seconds,
                        "model_bytes": model_bytes(int(dimension), int(levels), effective_k, preprocessing_bytes),
                        "codebook_hashes": json.dumps(encoding.codebook_hashes, sort_keys=True),
                    }
                    row.update(metrics)
                    inner_rows.append(row)
        print(f"Progress: inner split {inner_fold}/3 complete; {len(inner_rows)} split-candidate evaluations saved in memory.")

    inner_frame = pd.DataFrame(inner_rows)
    summary_rows: list[dict[str, Any]] = []
    grouped = inner_frame.groupby(["dimension", "levels", "k", "seed"], sort=False)
    for completed_count, (key, group) in enumerate(grouped, start=1):
        dimension, levels, k_label, seed = key
        effective_k = int(group["effective_k"].max())
        ordered = group.sort_values("inner_fold")
        summary: dict[str, Any] = {
            "outer_fold": outer_fold, "dimension": int(dimension), "levels": int(levels),
            "k": str(k_label), "effective_k": int(effective_k), "seed": int(seed),
            "effective_k_by_inner_fold": json.dumps(group.sort_values("inner_fold")["effective_k"].astype(int).tolist()),
            "inner_fold_1_macro_f1": float(ordered.iloc[0]["macro_f1"]),
            "inner_fold_2_macro_f1": float(ordered.iloc[1]["macro_f1"]),
            "inner_fold_3_macro_f1": float(ordered.iloc[2]["macro_f1"]),
            "mean_macro_f1": float(group["macro_f1"].mean()),
            "std_macro_f1": float(group["macro_f1"].std(ddof=0)),
            "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
            "mean_accuracy": float(group["accuracy"].mean()),
            "mean_severe_error_rate": float(group["severe_error_rate"].mean()),
            "mean_recall_class_0": float(group["recall_class_0"].mean()),
            "mean_recall_class_1": float(group["recall_class_1"].mean()),
            "mean_recall_class_2": float(group["recall_class_2"].mean()),
            "mean_recall_class_3": float(group["recall_class_3"].mean()),
            "encoding_time_seconds": float(group["encoding_seconds"].sum()),
            "prototype_training_time_seconds": float(group["prototype_training_seconds"].sum()),
            "inference_time_seconds": float(group["inference_seconds"].sum()),
            "model_bytes": int(group["model_bytes"].max()),
            "codebook_hashes": json.dumps(group["codebook_hashes"].tolist()),
            "parameter_grid_order": candidate_order[(int(dimension), int(levels), str(k_label), int(seed))],
        }
        summary_rows.append(summary)
        if completed_count % 4 == 0:
            print(f"Progress: {completed_count}/16 candidates aggregated.")

    summary_frame = pd.DataFrame(summary_rows)
    if len(summary_frame) != 16 or len(inner_frame) != 48:
        raise RuntimeError("not all candidate/inner-fold evaluations completed")
    ranked = summary_frame.assign(
        _all_after_finite=(summary_frame["k"] == "all").astype(int),
    ).sort_values(
        ["mean_macro_f1", "std_macro_f1", "mean_severe_error_rate", "dimension", "_all_after_finite", "effective_k", "levels", "parameter_grid_order"],
        ascending=[False, True, True, True, True, True, True, True], kind="mergesort",
    )
    best = ranked.iloc[0].drop(labels=["_all_after_finite"]).to_dict()

    summary_path = PHASE / f"results/summaries/vanilla_hdc_quick_screen_fold_{outer_fold}_candidates.csv"
    inner_path = PHASE / f"results/fold_metrics/vanilla_hdc_quick_screen_fold_{outer_fold}_inner_metrics.csv"
    best_path = PHASE / f"results/summaries/vanilla_hdc_quick_screen_fold_{outer_fold}_best_config.json"
    efficiency_path = PHASE / f"results/efficiency/vanilla_hdc_quick_screen_fold_{outer_fold}_efficiency.csv"
    checkpoint_path = PHASE / f"results/checkpoints/quick_screen/vanilla_hdc_quick_screen_fold_{outer_fold}_checkpoint.json"
    leakage_path = PHASE / f"audits/vanilla_hdc_quick_screen_fold_{outer_fold}_leakage_audit.json"
    artifact_path = PHASE / f"audits/vanilla_hdc_quick_screen_fold_{outer_fold}_artifact_audit.json"
    summary_frame.to_csv(summary_path, index=False)
    inner_frame.to_csv(inner_path, index=False)
    write_json(best_path, best)
    summary_frame[[
        "dimension", "levels", "k", "seed", "encoding_time_seconds",
        "prototype_training_time_seconds", "inference_time_seconds", "model_bytes",
    ]].to_csv(efficiency_path, index=False)

    frozen_hashes_after = {path.name: sha256(path) for path in CONTRACT_PATHS}
    contract_unchanged = all(
        frozen_hashes_before[name] == digest for name, digest in frozen_hashes_after.items()
    )
    historical_hashes_after = {
        relative: sha256(PHASE / relative) for relative in historical_hashes_before
    }
    historical_preserved = historical_hashes_before == historical_hashes_after
    if not historical_preserved:
        raise RuntimeError("one or more prior-fold artifacts changed during execution")
    leakage_audit = {
        "outer_fold": outer_fold, "outer_training_rows": int(len(outer_train)),
        "outer_training_subjects": int(outer_train["subject_id"].nunique()),
        "outer_test_rows_from_fold_assignments_only": int(len(forbidden_test_keys)),
        "outer_test_feature_access": False, "outer_test_feature_matrix_materialized": False,
        "outer_test_prediction_generated": False, "inner_subject_isolation": all(item["subject_overlap_count"] == 0 for item in leakage_splits),
        "inner_splits": leakage_splits, "preprocessing_fit_scope": "each inner-training split independently",
        "fold_local_quantization": True, "contract_files_unchanged": contract_unchanged,
        "historical_artifacts_preserved": historical_preserved,
        "similarity_regression_executed": False, "ridge_readout_executed": False, "oof_generated": False,
        "result": "PASS" if contract_unchanged and historical_preserved else "FAIL",
    }
    write_json(leakage_path, leakage_audit)

    checkpoint = {
        "phase": "05", "model": "Vanilla Prototype HDC", "stage": "quick_screen",
        "outer_fold": outer_fold, "outer_training_rows": int(len(outer_train)),
        "outer_training_subjects": int(outer_train["subject_id"].nunique()),
        "outer_test_rows_from_fold_assignments_only": int(len(forbidden_test_keys)),
        "contract_sha256": {name: value for name, value in frozen_hashes_before.items() if name.endswith(".json")},
        "primary_sha256": frozen_hashes_before["primary_without_performance.csv"],
        "fold_sha256": frozen_hashes_before["fold_assignments.csv"],
        "candidates_completed": 16, "candidates_expected": 16, "best_candidate": best,
        "inner_subject_isolation": leakage_audit["inner_subject_isolation"],
        "outer_test_feature_access": False, "unit_tests": "PASS",
        "historical_artifact_sha256_before": historical_hashes_before,
        "historical_artifact_sha256_after": historical_hashes_after,
        "historical_artifacts_preserved": historical_preserved,
        "overall_pass": bool(contract_unchanged and historical_preserved and leakage_audit["inner_subject_isolation"]),
    }
    write_json(checkpoint_path, checkpoint)

    required = [checkpoint_path, summary_path, best_path, inner_path, efficiency_path, leakage_path]
    artifact_records = [
        {"path": str(path.relative_to(PHASE)), "exists": path.is_file(), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in required
    ]
    artifact_audit = {
        "outer_fold": outer_fold, "all_required_artifacts_exist": all(item["exists"] for item in artifact_records),
        "candidate_rows": int(len(summary_frame)), "inner_metric_rows": int(len(inner_frame)),
        "checkpoint_overall_pass": checkpoint["overall_pass"], "artifacts": artifact_records,
        "result": "PASS" if all(item["exists"] for item in artifact_records) and len(summary_frame) == 16 and len(inner_frame) == 48 else "FAIL",
    }
    write_json(artifact_path, artifact_audit)
    print(
        f"COMPLETE: 16/16 candidates; best dimension={best['dimension']} levels={best['levels']} "
        f"k={best['k']} mean Macro-F1={best['mean_macro_f1']:.9f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-fold", type=int, choices=[1, 2, 3, 4, 5], required=True)
    arguments = parser.parse_args()
    outer_fold = int(arguments.outer_fold)
    stdout_path = PHASE / f"logs/vanilla_hdc_quick_screen_fold_{outer_fold}_stdout.log"
    stderr_path = PHASE / f"logs/vanilla_hdc_quick_screen_fold_{outer_fold}_stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as stdout_file, stderr_path.open("a", encoding="utf-8") as stderr_file:
        with redirect_stdout(Tee(sys.__stdout__, stdout_file)), redirect_stderr(Tee(sys.__stderr__, stderr_file)):
            try:
                quick_screen(outer_fold)
                return 0
            except Exception:
                traceback.print_exc()
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
