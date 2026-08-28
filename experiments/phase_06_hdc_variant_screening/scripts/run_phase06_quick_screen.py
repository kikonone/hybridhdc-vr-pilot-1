"""Run resumable Phase 06 inner-CV-only quick screening without outer-test feature access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, TextIO

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
PHASE05 = ROOT / "experiments" / "phase_05_basic_dual_output_hdc"
sys.path.insert(0, str(PHASE / "src"))

from phase06_hybrid import predict_hybrid, train_hybrid  # noqa: E402
from phase06_multicentroid import predict_multicentroid, train_multicentroid  # noqa: E402
from phase06_onlinehd import predict_onlinehd, train_onlinehd  # noqa: E402
from phase06_variant_common import (  # noqa: E402
    EqualWidthQuantizer,
    best_candidate,
    candidate_grid,
    canonical_json,
    classification_metrics,
    expected_candidate_count,
    fitted_preprocessing,
    incremental_encode_prefixes,
    load_outer_training_features,
)


PRIMARY = ROOT / "experiments" / "phase_03_multimodal_dataset_labeling" / "data" / "primary_without_performance.csv"
FOLDS = ROOT / "experiments" / "phase_03_multimodal_dataset_labeling" / "data" / "fold_assignments.csv"
FEATURE_MANIFEST = ROOT / "experiments" / "phase_03_multimodal_dataset_labeling" / "manifests" / "primary_feature_manifest.json"
CONTRACTS = [
    PHASE / "configs" / "phase06_hdc_variant_contract.json",
    PHASE / "configs" / "phase06_variant_search_spaces.json",
    PHASE / "configs" / "phase06_model_selection_rules.json",
]
EXPECTED_PRIMARY = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLDS = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"


class Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "file_size_bytes": path.stat().st_size, "sha256": sha256(path), "result": "PASS"}


def contract_hashes() -> dict[str, str]:
    audit = read_json(PHASE / "audits" / "phase06_contract_freeze_audit.json")
    if audit.get("result") != "PASS":
        raise RuntimeError("Phase 06 contract freeze audit is not PASS")
    values = {path.name: sha256(path) for path in CONTRACTS}
    manifest = read_json(PHASE / "manifests" / "phase06_contract_manifest.json")
    expected = {Path(item["path"]).name: item["sha256"] for item in manifest["artifacts"]}
    if values != expected:
        raise RuntimeError("Frozen Phase 06 contract hash mismatch")
    if sha256(PRIMARY) != EXPECTED_PRIMARY or sha256(FOLDS) != EXPECTED_FOLDS:
        raise RuntimeError("Frozen Phase 03 input checksum mismatch")
    return {**values, "primary_without_performance.csv": EXPECTED_PRIMARY, "fold_assignments.csv": EXPECTED_FOLDS}


def run_test_gate() -> None:
    command = [sys.executable, "-m", "pytest", str(PHASE / "tests"), "-q"]
    completed = subprocess.run(command, cwd=PHASE, text=True, capture_output=True, check=False)
    audit = {
        "phase": "06", "audit": "unit_tests", "command": command,
        "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr,
        "result": "PASS" if completed.returncode == 0 else "FAIL",
    }
    write_json(PHASE / "audits" / "phase06_unit_test_audit.json", audit)
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        raise RuntimeError("Phase 06 unit-test gate failed")


def sealed_fold_assignments(outer_fold: int) -> tuple[pd.DataFrame, set[str], set[str]]:
    training_rows: list[dict[str, Any]] = []
    forbidden_keys: set[str] = set()
    forbidden_subjects: set[str] = set()
    with FOLDS.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fold = int(row["outer_fold"])
            if fold == outer_fold:
                forbidden_keys.add(row["run_key"])
                forbidden_subjects.add(row["subject_id"])
                continue
            training_rows.append({
                "run_key": row["run_key"], "subject_id": row["subject_id"],
                "outer_fold": fold, "target_class": int(row["target_class"]),
            })
    training = pd.DataFrame(training_rows)
    if set(training["subject_id"]) & forbidden_subjects:
        raise RuntimeError("Outer subject isolation failed")
    if len(training) + len(forbidden_keys) != 419:
        raise RuntimeError("Fold coverage failed")
    return training, forbidden_keys, forbidden_subjects


def prepare_inner_cache(outer_fold: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outer_train, forbidden_keys, forbidden_subjects = sealed_fold_assignments(outer_fold)
    feature_names = read_json(FEATURE_MANIFEST)["features"]
    values, labels, groups, run_keys = load_outer_training_features(outer_train, forbidden_keys, feature_names)
    if values.shape != (len(outer_train), 1176) or set(run_keys) != set(outer_train["run_key"]):
        raise RuntimeError("Outer-training feature materialization failed")
    cache: list[dict[str, Any]] = []
    split_audit: list[dict[str, Any]] = []
    for inner_fold, (train_index, validation_index) in enumerate(GroupKFold(n_splits=3).split(values, labels, groups), start=1):
        train_subjects = set(groups[train_index])
        validation_subjects = set(groups[validation_index])
        if train_subjects & validation_subjects:
            raise RuntimeError("Inner subject isolation failed")
        preprocessing_start = time.perf_counter()
        train_processed, validation_processed, ranked_names, preprocessing_bytes = fitted_preprocessing(
            values[train_index], values[validation_index], labels[train_index], feature_names
        )
        preprocessing_seconds = time.perf_counter() - preprocessing_start
        quantizer = EqualWidthQuantizer(51).fit(train_processed)
        quantized = np.vstack([quantizer.transform(train_processed), quantizer.transform(validation_processed)])
        encoding = incremental_encode_prefixes(
            quantized, ranked_names, 51, 42, [50], 5000
        )
        encoded = encoding.samples_by_k["50"]
        split_point = len(train_index)
        cache.append({
            "inner_fold": inner_fold,
            "train_hv_5000": encoded[:split_point].copy(),
            "validation_hv_5000": encoded[split_point:].copy(),
            "train_labels": labels[train_index].copy(),
            "validation_labels": labels[validation_index].copy(),
            "train_subjects": sorted(train_subjects),
            "validation_subjects": sorted(validation_subjects),
            "preprocessing_seconds": preprocessing_seconds,
            "encoding_seconds": float(encoding.feature_completion_seconds["50"]),
            "preprocessing_bytes": preprocessing_bytes,
            "quantizer_state_sha256": quantizer.state_digest(),
            "codebook_hashes": encoding.codebook_hashes,
            "hybrid_initializations": {},
        })
        split_audit.append({
            "inner_fold": inner_fold, "train_rows": int(len(train_index)), "validation_rows": int(len(validation_index)),
            "train_subjects": len(train_subjects), "validation_subjects": len(validation_subjects),
            "subject_overlap": [], "preprocessing_fit_rows": int(len(train_index)), "result": "PASS",
        })
        print(f"Prepared outer Fold {outer_fold} inner split {inner_fold}/3 without outer-test feature or label access.")
    context = {
        "outer_training_rows": len(outer_train), "outer_training_subjects": int(outer_train["subject_id"].nunique()),
        "outer_test_rows_sealed": len(forbidden_keys), "outer_test_subjects_sealed": len(forbidden_subjects),
        "outer_test_feature_access": False, "outer_test_label_access": False,
        "inner_splits": split_audit,
    }
    return cache, context


def model_bytes(variant: str, config: dict[str, Any]) -> int:
    dimension = int(config["dimension"])
    if variant == "onlinehd":
        return 4 * dimension * np.dtype(np.float32).itemsize
    return 4 * int(config["centroids_per_class"]) * dimension * np.dtype(np.float32).itemsize


def evaluate_candidate(
    variant: str, config: dict[str, Any], candidate_id: str, cache: list[dict[str, Any]], outer_fold: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_training = 0.0
    total_inference = 0.0
    for split in cache:
        dimension = int(config["dimension"])
        train_hv = split["train_hv_5000"][:, :dimension]
        validation_hv = split["validation_hv_5000"][:, :dimension]
        train_before = hashlib.sha256(train_hv.tobytes()).hexdigest()
        validation_before = hashlib.sha256(validation_hv.tobytes()).hexdigest()
        stream = f"outer={outer_fold}|inner={split['inner_fold']}|dimension={dimension}|candidate={candidate_id}"
        start = time.perf_counter()
        if variant == "onlinehd":
            model, training_info = train_onlinehd(
                train_hv, split["train_labels"], epochs=config["epochs"], learning_rate=config["learning_rate"],
                margin_threshold=config["margin_threshold"], seed=config["seed"], stream_identifier=stream,
            )
        elif variant == "multicentroid":
            model, training_info = train_multicentroid(
                train_hv, split["train_labels"], centroids_per_class=config["centroids_per_class"],
                seed=config["seed"], stream_identifier=stream,
            )
        else:
            initialization_key = f"dimension={config['dimension']}|centroids={config['centroids_per_class']}"
            initialization_cache = split["hybrid_initializations"]
            if initialization_key not in initialization_cache:
                initialization_start = time.perf_counter()
                initial_centroids, initialization_info = train_multicentroid(
                    train_hv, split["train_labels"], centroids_per_class=config["centroids_per_class"],
                    seed=config["seed"],
                    stream_identifier=f"outer={outer_fold}|inner={split['inner_fold']}|dimension={dimension}",
                )
                initialization_cache[initialization_key] = {
                    "centroids": initial_centroids,
                    "info": initialization_info,
                    "seconds": time.perf_counter() - initialization_start,
                }
            frozen_initialization = initialization_cache[initialization_key]
            model, training_info = train_hybrid(
                train_hv, split["train_labels"], centroids_per_class=config["centroids_per_class"],
                epochs=config["epochs"], learning_rate=config["learning_rate"], margin_threshold=config["margin_threshold"],
                seed=config["seed"], stream_identifier=stream,
                initial_centroids=frozen_initialization["centroids"],
                initialization_info=frozen_initialization["info"],
            )
        training_seconds = time.perf_counter() - start
        if variant == "hybrid":
            training_seconds += float(frozen_initialization["seconds"])
        start = time.perf_counter()
        if variant == "onlinehd":
            predictions, _ = predict_onlinehd(validation_hv, model)
        elif variant == "multicentroid":
            predictions, _ = predict_multicentroid(validation_hv, model)
        else:
            predictions, _ = predict_hybrid(validation_hv, model)
        inference_seconds = time.perf_counter() - start
        if train_before != hashlib.sha256(train_hv.tobytes()).hexdigest() or validation_before != hashlib.sha256(validation_hv.tobytes()).hexdigest():
            raise RuntimeError("Variant modified an input array in place")
        metrics = classification_metrics(split["validation_labels"], predictions)
        row = {
            "variant": variant, "outer_fold": outer_fold, "inner_fold": split["inner_fold"], "candidate_id": candidate_id,
            **config, **metrics, "training_seconds": training_seconds, "inference_seconds": inference_seconds,
            "preprocessing_seconds_shared": split["preprocessing_seconds"], "encoding_seconds_shared": split["encoding_seconds"],
            "model_bytes": model_bytes(variant, config), "model_dtype": str(model.dtype),
            "sample_hv_dtype": str(train_hv.dtype), "subject_overlap_count": 0,
            "training_info_json": canonical_json(training_info),
        }
        rows.append(row)
        total_training += training_seconds
        total_inference += inference_seconds
    efficiency = {
        "candidate_id": candidate_id, "variant": variant, "outer_fold": outer_fold,
        "training_seconds": total_training, "inference_seconds": total_inference,
        "model_bytes": model_bytes(variant, config),
    }
    return rows, efficiency


def aggregate_candidate(config: dict[str, Any], candidate_id: str, rows: list[dict[str, Any]], efficiency: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame(rows).sort_values("inner_fold")
    result: dict[str, Any] = {
        "candidate_id": candidate_id, **config,
        "canonical_config_json": canonical_json(config),
        "inner_fold_1_macro_f1": float(frame.iloc[0]["macro_f1"]),
        "inner_fold_2_macro_f1": float(frame.iloc[1]["macro_f1"]),
        "inner_fold_3_macro_f1": float(frame.iloc[2]["macro_f1"]),
        "mean_macro_f1": float(frame["macro_f1"].mean()),
        "std_macro_f1_sample": float(frame["macro_f1"].std(ddof=1)),
        "mean_balanced_accuracy": float(frame["balanced_accuracy"].mean()),
        "mean_accuracy": float(frame["accuracy"].mean()),
        "mean_severe_error_rate": float(frame["severe_error_rate"].mean()),
        "training_seconds": efficiency["training_seconds"],
        "inference_seconds": efficiency["inference_seconds"],
        "model_bytes": efficiency["model_bytes"],
        "status": "COMPLETE",
    }
    return result


def valid_checkpoint(path: Path, config: dict[str, Any], frozen_hashes: dict[str, str]) -> bool:
    if not path.is_file():
        return False
    checkpoint = read_json(path)
    if checkpoint.get("status") != "COMPLETE":
        return False
    if checkpoint.get("canonical_config_json") != canonical_json(config):
        raise RuntimeError(f"Invalid checkpoint configuration: {path}")
    if checkpoint.get("frozen_hashes") != frozen_hashes:
        raise RuntimeError(f"Invalid checkpoint contract/input hashes: {path}")
    if len(checkpoint.get("inner_metrics", [])) != 3:
        raise RuntimeError(f"Invalid checkpoint inner metrics: {path}")
    return True


def run_fold(variant: str, outer_fold: int, frozen_hashes: dict[str, str]) -> None:
    stdout_path = PHASE / "logs" / f"{variant}_quick_screen_fold_{outer_fold}_stdout.log"
    stderr_path = PHASE / "logs" / f"{variant}_quick_screen_fold_{outer_fold}_stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as stdout_file, stderr_path.open("a", encoding="utf-8") as stderr_file:
        with redirect_stdout(Tee(sys.__stdout__, stdout_file)), redirect_stderr(Tee(sys.__stderr__, stderr_file)):
            candidates = candidate_grid(variant)
            expected = expected_candidate_count(variant)
            if len(candidates) != expected:
                raise RuntimeError("Frozen candidate count mismatch")
            checkpoint_dir = PHASE / "results" / "checkpoints" / "quick_screen" / variant / f"fold_{outer_fold}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            completed: dict[str, dict[str, Any]] = {}
            needs_cache = False
            for index, config in enumerate(candidates, start=1):
                candidate_id = f"candidate_{index:03d}"
                path = checkpoint_dir / f"{candidate_id}.json"
                if valid_checkpoint(path, config, frozen_hashes):
                    completed[candidate_id] = read_json(path)
                else:
                    needs_cache = True
            cache: list[dict[str, Any]] = []
            context: dict[str, Any]
            if needs_cache:
                cache, context = prepare_inner_cache(outer_fold)
            else:
                _, forbidden_keys, forbidden_subjects = sealed_fold_assignments(outer_fold)
                context = {
                    "outer_training_rows": 419 - len(forbidden_keys), "outer_training_subjects": 35 - len(forbidden_subjects),
                    "outer_test_rows_sealed": len(forbidden_keys), "outer_test_subjects_sealed": len(forbidden_subjects),
                    "outer_test_feature_access": False, "outer_test_label_access": False,
                    "inner_splits": completed[next(iter(completed))]["split_audit"],
                }
            for index, config in enumerate(candidates, start=1):
                candidate_id = f"candidate_{index:03d}"
                checkpoint_path = checkpoint_dir / f"{candidate_id}.json"
                if candidate_id in completed:
                    print(f"RESUME: {variant} Fold {outer_fold} {candidate_id} verified and reused.")
                    continue
                inner_rows, efficiency = evaluate_candidate(variant, config, candidate_id, cache, outer_fold)
                summary = aggregate_candidate(config, candidate_id, inner_rows, efficiency)
                checkpoint = {
                    "phase": "06", "stage": "quick_screen", "variant": variant, "outer_fold": outer_fold,
                    "candidate_id": candidate_id, "canonical_config_json": canonical_json(config), "config": config,
                    "inner_metrics": inner_rows, "summary": summary, "efficiency": efficiency,
                    "split_audit": context["inner_splits"], "frozen_hashes": frozen_hashes,
                    "outer_test_feature_access": False, "outer_test_label_access": False,
                    "outer_test_prediction_generated": False, "status": "COMPLETE", "result": "PASS",
                }
                write_json(checkpoint_path, checkpoint)
                completed[candidate_id] = checkpoint
                print(f"CHECKPOINT: {variant} Fold {outer_fold} {candidate_id} complete ({index}/{expected}).")
            if len(completed) != expected:
                raise RuntimeError("Not all candidate checkpoints completed")
            summary_rows = [completed[f"candidate_{index:03d}"]["summary"] for index in range(1, expected + 1)]
            inner_rows = [row for index in range(1, expected + 1) for row in completed[f"candidate_{index:03d}"]["inner_metrics"]]
            efficiency_rows = [completed[f"candidate_{index:03d}"]["efficiency"] for index in range(1, expected + 1)]
            best = best_candidate(summary_rows)
            summary_path = PHASE / "results" / "summaries" / f"{variant}_quick_screen_fold_{outer_fold}_candidates.csv"
            best_path = PHASE / "results" / "summaries" / f"{variant}_quick_screen_fold_{outer_fold}_best_config.json"
            inner_path = PHASE / "results" / "fold_metrics" / f"{variant}_quick_screen_fold_{outer_fold}_inner_metrics.csv"
            efficiency_path = PHASE / "results" / "efficiency" / f"{variant}_quick_screen_fold_{outer_fold}_efficiency.csv"
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            pd.DataFrame(inner_rows).to_csv(inner_path, index=False)
            pd.DataFrame(efficiency_rows).to_csv(efficiency_path, index=False)
            write_json(best_path, {"phase": "06", "variant": variant, "outer_fold": outer_fold, "selection_rule": "phase06_model_selection_rules_v1", "best_config": best, "result": "PASS"})
            leakage_path = PHASE / "audits" / f"{variant}_quick_screen_fold_{outer_fold}_leakage_audit.json"
            coverage_path = PHASE / "audits" / f"{variant}_quick_screen_fold_{outer_fold}_coverage_audit.json"
            artifact_path = PHASE / "audits" / f"{variant}_quick_screen_fold_{outer_fold}_artifact_audit.json"
            leakage = {
                "phase": "06", "variant": variant, "outer_fold": outer_fold, "result": "PASS",
                **context, "inner_subject_isolation": all(not item["subject_overlap"] for item in context["inner_splits"]),
                "preprocessing_fit_scope": "inner-training only", "outer_test_feature_access": False,
                "outer_test_label_access": False, "outer_test_prediction_generated": False,
                "similarity_regression_executed": False, "ridge_readout_executed": False,
                "oof_generated": False, "final_confirmation_executed": False,
            }
            coverage = {
                "phase": "06", "variant": variant, "outer_fold": outer_fold,
                "candidates_expected": expected, "candidates_completed": len(summary_rows),
                "inner_metric_rows_expected": expected * 3, "inner_metric_rows_actual": len(inner_rows),
                "checkpoint_count": len(list(checkpoint_dir.glob("candidate_*.json"))),
                "all_best_configs_reproducible": best == best_candidate(summary_rows),
                "result": "PASS" if len(summary_rows) == expected and len(inner_rows) == expected * 3 else "FAIL",
            }
            write_json(leakage_path, leakage)
            write_json(coverage_path, coverage)
            required = [summary_path, best_path, inner_path, efficiency_path, leakage_path, coverage_path, stdout_path, stderr_path, *sorted(checkpoint_dir.glob("candidate_*.json"))]
            artifact = {
                "phase": "06", "variant": variant, "outer_fold": outer_fold,
                "required_artifact_count": len(required), "artifacts": [file_record(path) for path in required],
                "result": "PASS" if coverage["result"] == leakage["result"] == "PASS" else "FAIL",
            }
            write_json(artifact_path, artifact)
            print(f"COMPLETE: {variant} Fold {outer_fold}, {expected}/{expected} candidates; best mean Macro-F1={best['mean_macro_f1']:.9f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["onlinehd", "multicentroid", "hybrid", "all"], default="all")
    parser.add_argument("--outer-fold", type=int, choices=[1, 2, 3, 4, 5])
    arguments = parser.parse_args()
    try:
        frozen_hashes = contract_hashes()
        run_test_gate()
        variants = ["onlinehd", "multicentroid", "hybrid"] if arguments.variant == "all" else [arguments.variant]
        folds = [arguments.outer_fold] if arguments.outer_fold else [1, 2, 3, 4, 5]
        for variant in variants:
            for outer_fold in folds:
                run_fold(variant, int(outer_fold), frozen_hashes)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
