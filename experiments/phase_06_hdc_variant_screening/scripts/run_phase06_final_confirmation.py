"""Strict, resumable Phase 06 Final Confirmation for the three frozen HDC variants."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path.insert(0, str(PHASE / "src"))

from phase06_hybrid import predict_hybrid, train_hybrid  # noqa: E402
from phase06_multicentroid import predict_multicentroid, train_multicentroid  # noqa: E402
from phase06_onlinehd import predict_onlinehd, train_onlinehd  # noqa: E402
from phase06_variant_common import (  # noqa: E402
    EqualWidthQuantizer,
    canonical_json,
    fitted_preprocessing,
    incremental_encode_prefixes,
    load_outer_training_features,
)

PRIMARY = ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv"
FOLDS = ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv"
FEATURE_MANIFEST = ROOT / "experiments/phase_03_multimodal_dataset_labeling/manifests/primary_feature_manifest.json"
EXPECTED_PRIMARY = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLDS = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
VARIANTS = ["onlinehd", "multicentroid", "hybrid"]
DIMENSIONS = [1000, 2000, 5000, 10000]
SEEDS = [42, 43, 44, 45, 46]
TEMPERATURES = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
LEVELS = 51
FEATURE_K = 50


class Tee:
    def __init__(self, *streams: Any): self.streams = streams
    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value); stream.flush()
        return len(value)
    def flush(self) -> None:
        for stream in self.streams: stream.flush()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def checkpoint_digest(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def quick_manifest_records() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = PHASE / "manifests/phase06_quick_screen_artifact_manifest.json"
    manifest = read_json(manifest_path)
    records = []
    for item in manifest["artifacts"]:
        relative = item.get("relative_path") or item.get("path")
        path = Path(relative)
        if not path.is_absolute(): path = PHASE / path
        records.append({
            "relative_path": str(path.relative_to(PHASE)),
            "file_size_bytes": path.stat().st_size if path.exists() else -1,
            "sha256": sha256(path) if path.exists() else None,
            "expected_size_bytes": int(item["file_size_bytes"]),
            "expected_sha256": item["sha256"],
            "result": "PASS" if path.exists() and path.stat().st_size == int(item["file_size_bytes"]) and sha256(path) == item["sha256"] else "FAIL",
        })
    return manifest, records


def run_preflight() -> dict[str, Any]:
    required = [
        "configs/phase06_hdc_variant_contract.json", "configs/phase06_variant_search_spaces.json",
        "configs/phase06_model_selection_rules.json", "audits/phase06_contract_freeze_audit.json",
        "manifests/phase06_contract_manifest.json", "results/summaries/phase06_onlinehd_quick_screen_all_folds.csv",
        "results/summaries/phase06_multicentroid_quick_screen_all_folds.csv",
        "results/summaries/phase06_hybrid_quick_screen_all_folds.csv",
        "results/summaries/phase06_all_variants_quick_screen_summary.csv",
        "manifests/phase06_quick_screen_artifact_manifest.json",
        "audits/phase06_quick_screen_all_folds_audit.json",
        "audits/phase06_quick_screen_notebook_persistence_audit.json",
        "audits/phase06_quick_screen_artifact_audit.json", "audits/phase06_quick_screen_leakage_audit.json",
        "audits/phase06_unit_test_audit.json", "Phase_06_HDC_Variant_Screening.ipynb",
    ]
    missing = [value for value in required if not (PHASE / value).exists()]
    checks: dict[str, Any] = {
        "required_artifacts_present": not missing, "missing": missing,
        "primary_sha256": sha256(PRIMARY), "primary_checksum": sha256(PRIMARY) == EXPECTED_PRIMARY,
        "fold_sha256": sha256(FOLDS), "fold_checksum": sha256(FOLDS) == EXPECTED_FOLDS,
    }
    pass_audits = [
        "audits/phase06_contract_freeze_audit.json", "audits/phase06_quick_screen_all_folds_audit.json",
        "audits/phase06_quick_screen_notebook_persistence_audit.json", "audits/phase06_quick_screen_artifact_audit.json",
        "audits/phase06_quick_screen_leakage_audit.json", "audits/phase06_unit_test_audit.json",
    ]
    checks["audits"] = {path: read_json(PHASE / path).get("result") for path in pass_audits if (PHASE / path).exists()}
    for variant, expected in [("onlinehd", 24), ("multicentroid", 6), ("hybrid", 32)]:
        checks[variant] = {"folds": {}}
        for fold in range(1, 6):
            best = PHASE / f"results/summaries/{variant}_quick_screen_fold_{fold}_best_config.json"
            candidates = PHASE / f"results/summaries/{variant}_quick_screen_fold_{fold}_candidates.csv"
            audits = [PHASE / f"audits/{variant}_quick_screen_fold_{fold}_{kind}_audit.json" for kind in ["leakage", "coverage", "artifact"]]
            count = len(pd.read_csv(candidates)) if candidates.exists() else -1
            checks[variant]["folds"][str(fold)] = {
                "best_result": read_json(best).get("result") if best.exists() else "FAIL",
                "candidate_count": count, "expected_candidate_count": expected,
                "audits": [read_json(path).get("result") if path.exists() else "FAIL" for path in audits],
            }
    manifest, records = quick_manifest_records()
    checks["quick_manifest_result"] = manifest.get("result")
    checks["quick_artifacts_verified"] = len(records)
    checks["quick_artifacts_match"] = all(item["result"] == "PASS" for item in records)
    audit_values_ok = len(checks["audits"]) == len(pass_audits) and all(v == "PASS" for v in checks["audits"].values())
    fold_values_ok = all(
        value["best_result"] == "PASS" and value["candidate_count"] == value["expected_candidate_count"] and all(a == "PASS" for a in value["audits"])
        for variant in VARIANTS for value in checks[variant]["folds"].values()
    )
    checks["result"] = "PASS" if (not missing and checks["primary_checksum"] and checks["fold_checksum"] and audit_values_ok and fold_values_ok and checks["quick_artifacts_match"] and manifest.get("result") == "PASS") else "FAIL"
    atomic_json(PHASE / "audits/phase06_final_confirmation_preflight_audit.json", {"phase": "06", "audit": "final_confirmation_preflight", "timestamp_utc": now(), **checks})
    atomic_json(PHASE / "audits/phase06_quick_screen_pre_final_confirmation_snapshot.json", {
        "phase": "06", "snapshot": "pre_final_confirmation", "timestamp_utc": now(),
        "manifest_sha256": sha256(PHASE / "manifests/phase06_quick_screen_artifact_manifest.json"),
        "artifact_count": len(records), "artifacts": records, "authorized_future_mutation": ["Phase_06_HDC_Variant_Screening.ipynb"],
        "result": "PASS" if checks["quick_artifacts_match"] else "FAIL",
    })
    if checks["result"] != "PASS": raise RuntimeError(f"Final Confirmation preflight failed: {checks}")
    return checks


def sealed_assignments(outer_fold: int) -> tuple[pd.DataFrame, list[dict[str, str]], set[str], set[str]]:
    training: list[dict[str, Any]] = []
    test_meta: list[dict[str, str]] = []
    test_keys, test_subjects = set(), set()
    with FOLDS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["outer_fold"]) == outer_fold:
                test_keys.add(row["run_key"]); test_subjects.add(row["subject_id"])
                test_meta.append({"run_key": row["run_key"], "subject_id": row["subject_id"]})
            else:
                training.append({"run_key": row["run_key"], "subject_id": row["subject_id"], "outer_fold": int(row["outer_fold"]), "target_class": int(row["target_class"])})
    frame = pd.DataFrame(training)
    if set(frame.subject_id) & test_subjects or len(frame) + len(test_meta) != 419:
        raise RuntimeError("Outer subject isolation or coverage failed")
    return frame, test_meta, test_keys, test_subjects


def load_test_targets_after_prediction(outer_fold: int, predicted_keys: list[str]) -> dict[str, dict[str, Any]]:
    expected = set(predicted_keys); result = {}
    with FOLDS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["outer_fold"]) == outer_fold and row["run_key"] in expected:
                result[row["run_key"]] = {"true_class": int(row["target_class"]), "target_score": float(row["target_score"]), "subject_id": row["subject_id"]}
    if set(result) != expected: raise RuntimeError("Deferred outer-test label join failed")
    return result


def structural_config(variant: str, outer_fold: int) -> dict[str, Any]:
    source = read_json(PHASE / f"results/summaries/{variant}_quick_screen_fold_{outer_fold}_best_config.json")["best_config"]
    keys = {"onlinehd": ["epochs", "learning_rate", "margin_threshold"], "multicentroid": ["centroids_per_class"], "hybrid": ["centroids_per_class", "epochs", "learning_rate", "margin_threshold"]}[variant]
    return {key: source[key] for key in keys}


def encode_inner_splits(outer_fold: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignment, _, test_keys, test_subjects = sealed_assignments(outer_fold)
    feature_names = read_json(FEATURE_MANIFEST)["features"]
    values, labels, groups, run_keys = load_outer_training_features(assignment, test_keys, feature_names)
    if set(run_keys) != set(assignment.run_key): raise RuntimeError("Outer-training materialization mismatch")
    splits, audit = [], []
    for inner_fold, (train_idx, val_idx) in enumerate(GroupKFold(n_splits=3).split(values, labels, groups), 1):
        train_subjects, val_subjects = set(groups[train_idx]), set(groups[val_idx])
        if train_subjects & val_subjects or train_subjects & test_subjects or val_subjects & test_subjects:
            raise RuntimeError("Inner or outer subject isolation failed")
        start = time.perf_counter()
        train_x, val_x, ranked, prep_bytes = fitted_preprocessing(values[train_idx], values[val_idx], labels[train_idx], feature_names)
        prep_seconds = time.perf_counter() - start
        quantizer = EqualWidthQuantizer(LEVELS).fit(train_x)
        quantized = np.vstack([quantizer.transform(train_x), quantizer.transform(val_x)])
        encoding = incremental_encode_prefixes(quantized, ranked, LEVELS, seed, [FEATURE_K], max(DIMENSIONS))
        encoded = encoding.samples_by_k[str(FEATURE_K)]
        splits.append({"inner_fold": inner_fold, "train_hv": encoded[:len(train_idx)], "val_hv": encoded[len(train_idx):], "train_y": labels[train_idx], "val_y": labels[val_idx]})
        audit.append({"inner_fold": inner_fold, "train_rows": len(train_idx), "validation_rows": len(val_idx), "train_subjects": len(train_subjects), "validation_subjects": len(val_subjects), "subject_overlap": 0, "outer_test_subject_overlap": 0, "preprocessing_fit_rows": len(train_idx), "preprocessing_seconds": prep_seconds, "encoding_seconds": float(encoding.feature_completion_seconds[str(FEATURE_K)]), "preprocessing_bytes": int(prep_bytes), "quantizer_state_sha256": quantizer.state_digest()})
    return splits, {"outer_training_rows": len(assignment), "outer_test_rows_sealed": len(test_keys), "outer_subject_overlap": 0, "inner_splits": audit, "outer_test_feature_access": False, "outer_test_label_access": False}


def train_predict(variant: str, train_hv: np.ndarray, train_y: np.ndarray, predict_hv: np.ndarray, structure: dict[str, Any], seed: int, stream: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if variant == "onlinehd":
        model, info = train_onlinehd(train_hv, train_y, epochs=structure["epochs"], learning_rate=structure["learning_rate"], margin_threshold=structure["margin_threshold"], seed=seed, stream_identifier=stream)
        prediction, scores = predict_onlinehd(predict_hv, model)
    elif variant == "multicentroid":
        model, info = train_multicentroid(train_hv, train_y, centroids_per_class=structure["centroids_per_class"], seed=seed, stream_identifier=stream)
        prediction, scores = predict_multicentroid(predict_hv, model)
    else:
        model, info = train_hybrid(train_hv, train_y, centroids_per_class=structure["centroids_per_class"], epochs=structure["epochs"], learning_rate=structure["learning_rate"], margin_threshold=structure["margin_threshold"], seed=seed, stream_identifier=stream)
        prediction, scores = predict_hybrid(predict_hv, model)
    return prediction, scores, {**info, "model_bytes": int(model.nbytes)}


def similarity_decode(scores: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(scores, dtype=np.float64) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits); probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities @ np.asarray([1.0, 2.0, 3.0, 4.0])


def select_temperatures(variant: str, outer_fold: int, seed: int, structure: dict[str, Any], splits: list[dict[str, Any]], needed: list[int]) -> dict[int, dict[str, Any]]:
    selected = {}
    for dimension in needed:
        rows = []
        for split in splits:
            start = time.perf_counter()
            _, scores, info = train_predict(variant, split["train_hv"][:, :dimension], split["train_y"], split["val_hv"][:, :dimension], structure, seed, f"final|variant={variant}|outer={outer_fold}|inner={split['inner_fold']}|dimension={dimension}|seed={seed}")
            model_seconds = time.perf_counter() - start
            truth = split["val_y"].astype(float) + 1.0
            for temperature in TEMPERATURES:
                prediction = np.clip(similarity_decode(scores, temperature), 1.0, 4.0)
                rows.append({"inner_fold": split["inner_fold"], "temperature": temperature, "bounded_mae": float(mean_absolute_error(truth, prediction)), "bounded_rmse": float(mean_squared_error(truth, prediction) ** 0.5), "model_seconds": model_seconds, "model_bytes": info["model_bytes"]})
        aggregate = []
        for temperature in TEMPERATURES:
            group = [row for row in rows if row["temperature"] == temperature]
            aggregate.append({"temperature": temperature, "mean_bounded_mae": float(np.mean([r["bounded_mae"] for r in group])), "std_bounded_mae_sample": float(np.std([r["bounded_mae"] for r in group], ddof=1)), "mean_bounded_rmse": float(np.mean([r["bounded_rmse"] for r in group]))})
        best = min(aggregate, key=lambda row: (row["mean_bounded_mae"], row["std_bounded_mae_sample"], row["mean_bounded_rmse"], row["temperature"]))
        selected[dimension] = {"selected_temperature": best["temperature"], "temperature_aggregate": aggregate, "inner_rows": rows, "selection_rule": "mean_bounded_mae,std_bounded_mae_sample,mean_bounded_rmse,temperature"}
    return selected


def load_outer_encoded_after_selection(outer_fold: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[dict[str, str]], dict[str, Any]]:
    assignment, test_meta, _, _ = sealed_assignments(outer_fold)
    feature_names = read_json(FEATURE_MANIFEST)["features"]
    columns = ["run_key", *feature_names]
    features = pd.read_csv(PRIMARY, usecols=columns).set_index("run_key")
    train_keys = assignment.run_key.tolist(); test_keys = [row["run_key"] for row in test_meta]
    train_values = features.loc[train_keys, feature_names].to_numpy(dtype=np.float64)
    test_values = features.loc[test_keys, feature_names].to_numpy(dtype=np.float64)
    train_y = assignment.target_class.to_numpy(dtype=np.int64)
    start = time.perf_counter()
    train_x, test_x, ranked, prep_bytes = fitted_preprocessing(train_values, test_values, train_y, feature_names)
    prep_seconds = time.perf_counter() - start
    quantizer = EqualWidthQuantizer(LEVELS).fit(train_x)
    quantized = np.vstack([quantizer.transform(train_x), quantizer.transform(test_x)])
    encoding = incremental_encode_prefixes(quantized, ranked, LEVELS, seed, [FEATURE_K], max(DIMENSIONS))
    encoded = encoding.samples_by_k[str(FEATURE_K)]
    return encoded[:len(train_keys)], encoded[len(train_keys):], train_y, test_keys, test_meta, {"preprocessing_seconds": prep_seconds, "encoding_seconds": float(encoding.feature_completion_seconds[str(FEATURE_K)]), "preprocessing_bytes": int(prep_bytes), "quantizer_state_sha256": quantizer.state_digest(), "outer_test_feature_access_after_inner_selection": True, "outer_test_label_access_before_prediction": False}


def classification_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(truth, prediction, labels=[0, 1, 2, 3])
    result: dict[str, Any] = {"accuracy": float(accuracy_score(truth, prediction)), "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)), "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)), "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)), "severe_error_rate": float(np.mean(np.abs(truth - prediction) >= 2))}
    for i in range(4):
        for j in range(4): result[f"confusion_{i}_{j}"] = int(matrix[i, j])
    return result


def regression_metrics(truth: np.ndarray, raw: np.ndarray) -> dict[str, Any]:
    bounded = np.clip(raw, 1.0, 4.0); correlation = spearmanr(truth, bounded).statistic
    return {"mae_raw": float(mean_absolute_error(truth, raw)), "mae_bounded": float(mean_absolute_error(truth, bounded)), "rmse_bounded": float(mean_squared_error(truth, bounded) ** 0.5), "r2_bounded": float(r2_score(truth, bounded)), "spearman_bounded": float(correlation) if np.isfinite(correlation) else 0.0}


def checkpoint_path(variant: str, outer_fold: int, dimension: int, seed: int) -> Path:
    return PHASE / f"results/checkpoints/final_confirmation/{variant}/fold_{outer_fold}/dimension_{dimension}_seed_{seed}.json"


def valid_checkpoint(path: Path, variant: str, outer_fold: int, dimension: int, seed: int, test_count: int) -> bool:
    try:
        payload = read_json(path)
        return payload.get("result") == "PASS" and payload.get("checkpoint_sha256") == checkpoint_digest(payload) and payload["variant"] == variant and payload["outer_fold"] == outer_fold and payload["dimension"] == dimension and payload["seed"] == seed and len(payload["predictions"]) == test_count and payload["primary_sha256"] == EXPECTED_PRIMARY and payload["fold_sha256"] == EXPECTED_FOLDS
    except Exception:
        return False


def run_fold(variant: str, outer_fold: int) -> None:
    stdout_path = PHASE / f"logs/{variant}_final_confirmation_fold_{outer_fold}_stdout.log"
    stderr_path = PHASE / f"logs/{variant}_final_confirmation_fold_{outer_fold}_stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as stdout_file, stderr_path.open("a", encoding="utf-8") as stderr_file:
        with redirect_stdout(Tee(sys.__stdout__, stdout_file)), redirect_stderr(Tee(sys.__stderr__, stderr_file)):
            assignment, test_meta, _, test_subjects = sealed_assignments(outer_fold)
            structure = structural_config(variant, outer_fold); test_count = len(test_meta)
            print(f"{now()} START {variant} outer fold {outer_fold}; test rows={test_count}; structure={structure}")
            for seed in SEEDS:
                needed = [d for d in DIMENSIONS if not valid_checkpoint(checkpoint_path(variant, outer_fold, d, seed), variant, outer_fold, d, seed, test_count)]
                if not needed:
                    print(f"{variant} fold {outer_fold} seed {seed}: all checkpoints reused")
                    continue
                splits, isolation = encode_inner_splits(outer_fold, seed)
                selections = select_temperatures(variant, outer_fold, seed, structure, splits, needed)
                # This is the first point at which outer-test features are materialized.
                train_hv, test_hv, train_y, test_keys, test_meta_ordered, outer_info = load_outer_encoded_after_selection(outer_fold, seed)
                generated: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any], float]] = {}
                for dimension in needed:
                    start = time.perf_counter()
                    predicted, scores, info = train_predict(variant, train_hv[:, :dimension], train_y, test_hv[:, :dimension], structure, seed, f"final|variant={variant}|outer={outer_fold}|dimension={dimension}|seed={seed}")
                    generated[dimension] = (predicted, scores, info, time.perf_counter() - start)
                # Labels are deliberately joined only after every pending prediction for this seed exists.
                labels = load_test_targets_after_prediction(outer_fold, test_keys)
                for dimension in needed:
                    predicted, scores, info, model_seconds = generated[dimension]
                    temperature = selections[dimension]["selected_temperature"]
                    raw = similarity_decode(scores, temperature); bounded = np.clip(raw, 1.0, 4.0)
                    truth_class = np.asarray([labels[key]["true_class"] for key in test_keys], dtype=np.int64)
                    truth_score = np.asarray([labels[key]["target_score"] for key in test_keys], dtype=float)
                    predictions = []
                    structure_json = canonical_json(structure)
                    for index, key in enumerate(test_keys):
                        row = {"run_key": key, "subject_id": labels[key]["subject_id"], "outer_fold": outer_fold, "variant": variant, "dimension": dimension, "seed": seed, "levels": LEVELS, "feature_k": FEATURE_K, "selected_structure_parameters": structure_json, "true_class": int(truth_class[index]), "predicted_class": int(predicted[index]), "target_score": float(truth_score[index]), "selected_temperature": float(temperature), "similarity_prediction_raw": float(raw[index]), "similarity_prediction_bounded": float(bounded[index])}
                        for class_id in range(4): row[f"class_score_{class_id}"] = float(scores[index, class_id])
                        predictions.append(row)
                    payload = {"phase": "06", "stage": "final_confirmation", "timestamp_utc": now(), "variant": variant, "outer_fold": outer_fold, "dimension": dimension, "seed": seed, "levels": LEVELS, "feature_k": FEATURE_K, "structure": structure, "primary_sha256": EXPECTED_PRIMARY, "fold_sha256": EXPECTED_FOLDS, "ridge_handling": "COMMON_ENCODER_READOUT_BASELINE", "inner_selection": selections[dimension], "isolation": isolation, "outer_fit": outer_info, "model_info": info, "classification_metrics": classification_metrics(truth_class, predicted), "similarity_regression_metrics": regression_metrics(truth_score, raw), "efficiency": {"model_training_and_inference_seconds": model_seconds, "model_bytes": info["model_bytes"], **outer_info}, "predictions": predictions, "test_rows": test_count, "test_subjects": len(test_subjects), "outer_test_features_loaded_after_temperature_fixed": True, "outer_test_labels_loaded_after_predictions_generated": True, "outer_test_used_for_tuning": False, "result": "PASS"}
                    payload["checkpoint_sha256"] = checkpoint_digest(payload)
                    atomic_json(checkpoint_path(variant, outer_fold, dimension, seed), payload)
                    print(f"CHECKPOINT PASS {variant} fold={outer_fold} dimension={dimension} seed={seed} T={temperature}")
            consolidate_fold(variant, outer_fold)


def consolidate_fold(variant: str, outer_fold: int) -> None:
    _, test_meta, _, test_subjects = sealed_assignments(outer_fold); count = len(test_meta)
    checkpoints = []
    for dimension in DIMENSIONS:
        for seed in SEEDS:
            path = checkpoint_path(variant, outer_fold, dimension, seed)
            if not valid_checkpoint(path, variant, outer_fold, dimension, seed, count): raise RuntimeError(f"Invalid checkpoint {path}")
            checkpoints.append(read_json(path))
    predictions = [row for payload in checkpoints for row in payload["predictions"]]
    class_rows, reg_rows, inner_rows, efficiency_rows = [], [], [], []
    for payload in checkpoints:
        base = {"variant": variant, "outer_fold": outer_fold, "dimension": payload["dimension"], "seed": payload["seed"], "levels": LEVELS, "feature_k": FEATURE_K, "selected_structure_parameters": canonical_json(payload["structure"]), "selected_temperature": payload["inner_selection"]["selected_temperature"]}
        class_rows.append({**base, **payload["classification_metrics"]}); reg_rows.append({**base, **payload["similarity_regression_metrics"]})
        for row in payload["inner_selection"]["inner_rows"]: inner_rows.append({**base, **row})
        efficiency_rows.append({**base, **payload["efficiency"]})
    outputs = {
        "predictions": PHASE / f"results/predictions/{variant}_final_confirmation_fold_{outer_fold}_predictions.csv",
        "classification": PHASE / f"results/fold_metrics/{variant}_final_confirmation_fold_{outer_fold}_classification_metrics.csv",
        "regression": PHASE / f"results/fold_metrics/{variant}_final_confirmation_fold_{outer_fold}_similarity_regression_metrics.csv",
        "inner": PHASE / f"results/fold_metrics/{variant}_final_confirmation_fold_{outer_fold}_inner_selection.csv",
        "efficiency": PHASE / f"results/efficiency/{variant}_final_confirmation_fold_{outer_fold}_efficiency.csv",
    }
    for key, rows in [("predictions", predictions), ("classification", class_rows), ("regression", reg_rows), ("inner", inner_rows), ("efficiency", efficiency_rows)]: atomic_csv(outputs[key], rows)
    expected_keys = {row["run_key"] for row in test_meta}
    coverage_ok = len(checkpoints) == 20 and len(predictions) == 20 * count and all(len({r["run_key"] for r in predictions if r["dimension"] == d and r["seed"] == s}) == count for d in DIMENSIONS for s in SEEDS) and set(r["run_key"] for r in predictions) == expected_keys
    leakage = {"phase": "06", "variant": variant, "outer_fold": outer_fold, "outer_train_test_subject_overlap": 0, "inner_split_subject_overlap": 0, "temperature_inner_cv_only": True, "outer_test_used_for_tuning": False, "outer_test_features_loaded_after_inner_selection": True, "outer_test_labels_loaded_after_predictions": True, "result": "PASS"}
    coverage = {"phase": "06", "variant": variant, "outer_fold": outer_fold, "expected_configurations": 20, "completed_configurations": len(checkpoints), "actual_test_rows": count, "expected_prediction_rows": 20 * count, "actual_prediction_rows": len(predictions), "test_subjects": len(test_subjects), "run_key_duplicates_per_configuration": 0, "result": "PASS" if coverage_ok else "FAIL"}
    atomic_json(PHASE / f"audits/{variant}_final_confirmation_fold_{outer_fold}_leakage_audit.json", leakage)
    atomic_json(PHASE / f"audits/{variant}_final_confirmation_fold_{outer_fold}_coverage_audit.json", coverage)
    artifacts = [{"path": str(path.relative_to(PHASE)), "file_size_bytes": path.stat().st_size, "sha256": sha256(path), "result": "PASS"} for path in outputs.values()]
    atomic_json(PHASE / f"audits/{variant}_final_confirmation_fold_{outer_fold}_artifact_audit.json", {"phase": "06", "variant": variant, "outer_fold": outer_fold, "artifacts": artifacts, "checkpoint_count": len(checkpoints), "checkpoint_integrity": all(p["checkpoint_sha256"] == checkpoint_digest(p) for p in checkpoints), "result": "PASS" if coverage_ok else "FAIL"})
    if not coverage_ok: raise RuntimeError(f"Coverage failed for {variant} fold {outer_fold}")


def consolidate_all() -> dict[str, Any]:
    fold_map = pd.read_csv(FOLDS).set_index("run_key")
    summary, variant_audits = [], {}
    for variant in VARIANTS:
        all_predictions = []
        for fold in range(1, 6):
            for dimension in DIMENSIONS:
                for seed in SEEDS:
                    payload = read_json(checkpoint_path(variant, fold, dimension, seed))
                    summary.append({"variant": variant, "outer_fold": fold, "dimension": dimension, "seed": seed, "selected_temperature": payload["inner_selection"]["selected_temperature"], "macro_f1": payload["classification_metrics"]["macro_f1"], "balanced_accuracy": payload["classification_metrics"]["balanced_accuracy"], "mae_bounded": payload["similarity_regression_metrics"]["mae_bounded"], "rmse_bounded": payload["similarity_regression_metrics"]["rmse_bounded"], "model_bytes": payload["efficiency"]["model_bytes"], "checkpoint_sha256": payload["checkpoint_sha256"], "status": "COMPLETE"})
                    all_predictions.extend(payload["predictions"])
        config_checks = []
        for dimension in DIMENSIONS:
            for seed in SEEDS:
                rows = [r for r in all_predictions if r["dimension"] == dimension and r["seed"] == seed]
                keys = [r["run_key"] for r in rows]
                assignment_mismatch = sum(int(fold_map.loc[r["run_key"], "outer_fold"]) != int(r["outer_fold"]) for r in rows)
                subject_mismatch = sum(str(fold_map.loc[r["run_key"], "subject_id"]) != str(r["subject_id"]) for r in rows)
                target_mismatch = sum(int(fold_map.loc[r["run_key"], "target_class"]) != int(r["true_class"]) or abs(float(fold_map.loc[r["run_key"], "target_score"]) - float(r["target_score"])) > 1e-12 for r in rows)
                config_checks.append({"dimension": dimension, "seed": seed, "rows": len(rows), "unique_run_keys": len(set(keys)), "duplicate_run_keys": len(keys) - len(set(keys)), "assignment_mismatch": assignment_mismatch, "subject_mismatch": subject_mismatch, "target_mismatch": target_mismatch, "classification_missing": sum(pd.isna(r["predicted_class"]) for r in rows), "similarity_missing": sum(pd.isna(r["similarity_prediction_bounded"]) for r in rows), "result": "PASS" if len(rows) == 419 and len(set(keys)) == 419 and assignment_mismatch == subject_mismatch == target_mismatch == 0 else "FAIL"})
        variant_audits[variant] = {"folds_completed": 5, "fold_config_runs": 100, "configuration_combinations": 20, "configurations": config_checks, "result": "PASS" if all(row["result"] == "PASS" for row in config_checks) else "FAIL"}
    atomic_csv(PHASE / "results/summaries/phase06_final_confirmation_execution_summary.csv", summary)
    result = "PASS" if len(summary) == 300 and all(v["result"] == "PASS" for v in variant_audits.values()) else "FAIL"
    audit = {"phase": "06", "audit": "final_confirmation_all_folds", "timestamp_utc": now(), "completed_fold_config_runs": len(summary), "expected_fold_config_runs": 300, "variants": variant_audits, "classification_predictions_generated": True, "similarity_regression_predictions_generated": True, "ridge_handling": "COMMON_ENCODER_READOUT_BASELINE", "outer_subject_isolation": "PASS", "inner_subject_isolation": "PASS", "temperature_inner_cv_only": "PASS", "outer_test_used_for_tuning": False, "best_hdc_selected": False, "ready_for_final_oof_consolidation": result == "PASS", "result": result}
    atomic_json(PHASE / "audits/phase06_final_confirmation_all_folds_audit.json", audit)
    if result != "PASS": raise RuntimeError("All-fold coverage audit failed")
    return audit


def quick_preservation_before_notebook() -> dict[str, Any]:
    snapshot = read_json(PHASE / "audits/phase06_quick_screen_pre_final_confirmation_snapshot.json")
    failures = []
    for item in snapshot["artifacts"]:
        if item["relative_path"].replace("\\", "/") == "Phase_06_HDC_Variant_Screening.ipynb": continue
        path = PHASE / item["relative_path"]
        if not path.exists() or sha256(path) != item["sha256"] or path.stat().st_size != item["file_size_bytes"]: failures.append(item["relative_path"])
    audit = {"phase": "06", "audit": "quick_screen_preservation_pre_notebook", "checked_artifacts": len(snapshot["artifacts"]) - 1, "authorized_notebook_append_excluded": True, "failures": failures, "result": "PASS" if not failures else "FAIL"}
    atomic_json(PHASE / "audits/phase06_quick_screen_post_final_confirmation_preservation_audit.json", audit)
    if failures: raise RuntimeError(f"Quick Screen artifacts changed: {failures}")
    return audit


def write_report() -> None:
    summary = pd.read_csv(PHASE / "results/summaries/phase06_final_confirmation_execution_summary.csv")
    lines = ["# Phase 06 Final Confirmation Execution Report", "", "Status: FINAL_CONFIRMATION_COMPLETE", "", "All 300 fold-config runs completed with strict nested inner-CV temperature selection. The reported regression target is the bounded difficulty-induced workload proxy regression.", "", "Ridge handling: `COMMON_ENCODER_READOUT_BASELINE`. Ridge consumes sample hypervectors, while these variants change only prototype/centroid learning; repeating an identical Ridge fit per variant would be pseudo-replication.", "", "Outer-test features were not loaded until inner selections were fixed. Outer-test labels were joined only after predictions were generated. Outer-test data were not used for tuning.", "", "| Variant | Runs | Mean Macro-F1 | Mean bounded MAE |", "|---|---:|---:|---:|"]
    for variant in VARIANTS:
        rows = summary[summary.variant == variant]
        lines.append(f"| {variant} | {len(rows)} | {rows.macro_f1.mean():.6f} | {rows.mae_bounded.mean():.6f} |")
    lines.extend(["", "No final HDC variant, dimension, or seed was selected. Final OOF Consolidation has not been executed.", ""])
    path = PHASE / "reports/phase06_final_confirmation_execution_report.md"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_manifest() -> None:
    roots = [PHASE / "results/checkpoints/final_confirmation", PHASE / "results/predictions", PHASE / "results/fold_metrics", PHASE / "results/efficiency", PHASE / "audits", PHASE / "logs", PHASE / "reports"]
    paths = []
    for root in roots:
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file() and ("final_confirmation" in path.name.lower() or "final_confirmation" in str(path.parent).lower()))
    paths.extend([
        PHASE / "configs/phase06_final_confirmation_contract.json",
        PHASE / "scripts/run_phase06_final_confirmation.py",
        PHASE / "scripts/persist_phase06_final_confirmation_notebook.py",
        PHASE / "scripts/verify_phase06_final_confirmation.py",
        PHASE / "final_confirmation_task_plan.md", PHASE / "final_confirmation_notes.md",
        PHASE / "Phase_06_HDC_Variant_Screening.ipynb",
    ])
    manifest_path = PHASE / "manifests/phase06_final_confirmation_artifact_manifest.json"
    unique = sorted({path.resolve() for path in paths if path.exists() and path.resolve() != manifest_path.resolve()}, key=str)
    artifacts = [{"relative_path": str(path.relative_to(PHASE)), "file_size_bytes": path.stat().st_size, "sha256": sha256(path), "result": "PASS"} for path in unique]
    atomic_json(manifest_path, {"phase": "06", "manifest": "final_confirmation_artifacts", "timestamp_utc": now(), "artifact_count": len(artifacts), "artifacts": artifacts, "fold_config_runs": 300, "result": "PASS"})


def main() -> int:
    failure_path = PHASE / "audits/phase06_final_confirmation_failure.json"
    try:
        run_preflight()
        for variant in VARIANTS:
            for outer_fold in range(1, 6): run_fold(variant, outer_fold)
        consolidate_all(); quick_preservation_before_notebook(); write_report(); build_manifest()
        print("PHASE 06 FINAL CONFIRMATION CORE EXECUTION COMPLETE")
        return 0
    except Exception as error:
        atomic_json(failure_path, {"phase": "06", "stage": "final_confirmation", "timestamp_utc": now(), "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "result": "FAIL"})
        traceback.print_exc(); return 1


if __name__ == "__main__": raise SystemExit(main())
