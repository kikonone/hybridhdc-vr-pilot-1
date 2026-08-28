"""Build the anonymous dual-task UI package from frozen canonical references.

The script validates upstream provenance and aggregation, aligns records on the
real run key, and only then removes research identifiers. It never trains a
model, generates a prediction, selects a model, or recomputes a scientific
metric.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

UI = Path(__file__).resolve().parents[1]
EXPERIMENTS = UI.parents[1]
P03 = EXPERIMENTS / "phase_03_multimodal_dataset_labeling"
P05 = EXPERIMENTS / "phase_05_basic_dual_output_hdc"
P06 = EXPERIMENTS / "phase_06_hdc_variant_screening"
P07 = EXPERIMENTS / "phase_07_unimodal_contribution"
P10 = EXPERIMENTS / "phase_10_final_synthesis_and_demo_ui"

SOURCES = {
    "fold_assignments": P03 / "data/fold_assignments.csv",
    "classification_selection": P06 / "configs/phase06_best_classification_hdc.json",
    "regression_selection": P06 / "configs/phase06_best_regression_hdc.json",
    "phase06_oof_seal": P06 / "manifests/phase06_preselection_outer_oof_seal.json",
    "classification_seed_oof": P06 / "results/oof/phase06_hybrid_final_oof.csv",
    "regression_seed_oof": P05 / "results/oof/vanilla_hdc_ridge_regression_oof.csv",
    "dual_task_contract": P07 / "configs/phase07_frozen_unimodal_contract.json",
    "classification_canonical_oof": P07 / "results/oof/phase07_readonly_multimodal_classification_reference.csv",
    "regression_canonical_oof": P07 / "results/oof/phase07_readonly_multimodal_regression_reference.csv",
    "phase10_selected_interface": P10 / "configs/phase10_best_dual_task_hdc_interface.json",
    "phase10_prediction_index": P10 / "results/final_prediction_library/final_prediction_library_index.csv",
}

EXPECTED = {
    "fold_assignments": "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f",
    "classification_selection": "174a99de2d993acdea49fdebc9647b28db4648ada2bea7a33f620f4677f031a4",
    "regression_selection": "acde51709971d57c76eefaffcf1ecd571a4d4c5c36f8d76edf39841c5e7065b8",
    "classification_seed_oof": "ff619baf4be600279482c9e1f4f4139000fc05c1dfaf41555d644674b45d875a",
    "regression_seed_oof": "a449d8f43a0935f0a3fcf8cf901894e426a83e552807dcef9551bc983ba22758",
    "classification_canonical_oof": "5933f705875e205a31c487384bbc6bb0460fe36076a1dafa01825859759c42a8",
    "regression_canonical_oof": "6af63df19a14c652b991230833a5d9bdf264cb1c5c387823f42d6e5cd30e0c7a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assert_close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{context}: {actual!r} != {expected!r}")


def validate_source_hashes() -> None:
    for role, expected in EXPECTED.items():
        actual = sha256(SOURCES[role])
        if actual != expected:
            raise AssertionError(f"Frozen source hash mismatch for {role}: {actual}")


def validate_selection_contracts() -> tuple[dict, dict, dict]:
    classification = read_json(SOURCES["classification_selection"])
    regression = read_json(SOURCES["regression_selection"])
    contract = read_json(SOURCES["dual_task_contract"])
    assert classification["selected_variant_name"] == "HDC+OnlineHD Hybrid"
    assert classification["selected_variant"] == "hybrid"
    assert classification["selected_fixed_dimension"] == 5000
    assert classification["feature_k"] == 50 and classification["levels"] == 51
    assert regression["selected_regression_head"] == "COMMON_ENCODER_READOUT_BASELINE"
    assert regression["selected_variant"] == "common_ridge"
    assert regression["selected_fixed_dimension"] == 10000
    assert regression["feature_k"] == 50 and regression["levels"] == 51
    for policy in regression["fold_parameter_policy"]:
        parameters = json.loads(policy["parameter_policy_json"])
        assert {float(item["ridge_alpha"]) for item in parameters} == {0.01}
        assert {int(item["seed"]) for item in parameters} == {42, 43, 44, 45, 46}
    assert contract["regression"]["target_description"] == "bounded difficulty-induced workload proxy regression"
    assert contract["oof_aggregation"]["classification"]["rule"] == "arithmetic mean of the five seed class scores, then argmax"
    assert contract["oof_aggregation"]["regression"]["rule"] == "arithmetic mean of five prediction_raw values, then clip to [1.0, 4.0]"
    return classification, regression, contract


def validate_prediction_index() -> None:
    rows = read_csv(SOURCES["phase10_prediction_index"])
    expected = {
        str(SOURCES["classification_canonical_oof"].resolve()): EXPECTED["classification_canonical_oof"],
        str(SOURCES["regression_canonical_oof"].resolve()): EXPECTED["regression_canonical_oof"],
    }
    selected = {row["source_path"]: row for row in rows if row["source_path"] in expected}
    assert set(selected) == set(expected)
    for path, row in selected.items():
        assert row["prediction_level"] == "CANONICAL_OOF"
        assert row["canonical_status"] == "CANONICAL"
        assert row["seed_coverage"] == "NO_SINGLE_SEED"
        assert row["row_count"] == row["unique_run_keys"] == "419"
        assert row["outer_fold_coverage"] == "1;2;3;4;5"
        assert row["source_sha256"] == expected[path]


def validate_canonical_values(class_rows: list[dict[str, str]], reg_rows: list[dict[str, str]]) -> None:
    class_seed_rows = [row for row in read_csv(SOURCES["classification_seed_oof"]) if int(row["dimension"]) == 5000]
    reg_seed_rows = [row for row in read_csv(SOURCES["regression_seed_oof"]) if int(row["dimension"]) == 10000]
    assert len(class_seed_rows) == len(reg_seed_rows) == 419 * 5

    class_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    reg_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in class_seed_rows:
        class_groups[row["run_key"]].append(row)
    for row in reg_seed_rows:
        reg_groups[row["run_key"]].append(row)
    assert set(class_groups) == set(reg_groups) == {row["run_key"] for row in class_rows} == {row["run_key"] for row in reg_rows}

    class_lookup = {row["run_key"]: row for row in class_rows}
    reg_lookup = {row["run_key"]: row for row in reg_rows}
    for run_key in sorted(class_groups):
        source_class = class_groups[run_key]
        canonical_class = class_lookup[run_key]
        assert len(source_class) == 5 and {int(row["seed"]) for row in source_class} == {42, 43, 44, 45, 46}
        means = []
        for class_index in range(4):
            mean_score = sum(float(row[f"class_score_{class_index}"]) for row in source_class) / 5
            assert_close(mean_score, float(canonical_class[f"class_score_{class_index}"]), f"classification {run_key} score {class_index}")
            means.append(mean_score)
        assert means.index(max(means)) == int(canonical_class["predicted_class"])

        source_reg = reg_groups[run_key]
        canonical_reg = reg_lookup[run_key]
        assert len(source_reg) == 5 and {int(row["seed"]) for row in source_reg} == {42, 43, 44, 45, 46}
        assert {float(row["ridge_alpha"]) for row in source_reg} == {0.01}
        raw_mean = sum(float(row["ridge_prediction_raw"]) for row in source_reg) / 5
        bounded = min(4.0, max(1.0, raw_mean))
        assert_close(raw_mean, float(canonical_reg["prediction_raw"]), f"regression {run_key} raw")
        assert_close(bounded, float(canonical_reg["prediction_bounded"]), f"regression {run_key} bounded")


def build_data() -> None:
    validate_source_hashes()
    classification, regression, contract = validate_selection_contracts()
    validate_prediction_index()

    class_rows = read_csv(SOURCES["classification_canonical_oof"])
    reg_rows = read_csv(SOURCES["regression_canonical_oof"])
    folds = {row["run_key"]: row for row in read_csv(SOURCES["fold_assignments"])}
    assert len(class_rows) == len(reg_rows) == len(folds) == 419
    validate_canonical_values(class_rows, reg_rows)

    class_by_key = {row["run_key"]: row for row in class_rows}
    reg_by_key = {row["run_key"]: row for row in reg_rows}
    aligned_keys = sorted(class_by_key)
    assert set(aligned_keys) == set(reg_by_key) == set(folds)
    assert len(aligned_keys) == 419

    demo_rows = []
    for index, run_key in enumerate(aligned_keys, 1):
        class_row = class_by_key[run_key]
        reg_row = reg_by_key[run_key]
        fold_row = folds[run_key]
        assert class_row["outer_fold"] == reg_row["outer_fold"] == fold_row["outer_fold"]
        assert int(class_row["target_class"]) + 1 == int(float(reg_row["target_score"]))
        bounded = float(reg_row["prediction_bounded"])
        target = float(reg_row["target_score"])
        demo_rows.append({
            "demo_id": f"DEMO-{index:04d}",
            "fold": int(class_row["outer_fold"]),
            "true_difficulty": int(class_row["target_class"]) + 1,
            "predicted_difficulty": int(class_row["predicted_class"]) + 1,
            "classification_correct": str(class_row["target_class"] == class_row["predicted_class"]).lower(),
            "difficulty_1_cosine": class_row["class_score_0"],
            "difficulty_2_cosine": class_row["class_score_1"],
            "difficulty_3_cosine": class_row["class_score_2"],
            "difficulty_4_cosine": class_row["class_score_3"],
            "true_difficulty_score": reg_row["target_score"],
            "raw_frozen_prediction": reg_row["prediction_raw"],
            "bounded_frozen_prediction": reg_row["prediction_bounded"],
            "absolute_error": format(abs(target - bounded), ".16g"),
        })

    data_dir = UI / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "frozen_dual_task_oof.csv"
    with data_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(demo_rows[0]))
        writer.writeheader()
        writer.writerows(demo_rows)

    model_path = data_dir / "frozen_dual_task_model.json"
    model_payload = {
        "title": "Frozen HDC Classification and Proxy-Regression Demonstration",
        "classification": {
            "model": "HDC+OnlineHD Hybrid", "variant": "hybrid", "dimension": 5000,
            "feature_k": 50, "levels": 51,
            "aggregation": contract["oof_aggregation"]["classification"]["rule"],
            "score_semantics": "cosine similarities, not probabilities",
        },
        "regression": {
            "model": "COMMON_ENCODER_READOUT_BASELINE", "variant": "common_ridge", "dimension": 10000,
            "feature_k": 50, "levels": 51, "ridge_alpha": 0.01,
            "ridge_alpha_policy": "0.01 for every frozen fold and seed configuration",
            "aggregation": contract["oof_aggregation"]["regression"]["rule"],
            "interpretation": "bounded difficulty-induced workload proxy regression",
        },
        "records": 419, "folds": 5, "frozen_evaluation_seeds": [42, 43, 44, 45, 46],
    }
    write_json(model_path, model_payload)

    source_entries = [
        {"role": role, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for role, path in SOURCES.items()
    ]
    output_entries = [
        {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in (data_path, model_path)
    ]
    alignment_digest = hashlib.sha256("\n".join(aligned_keys).encode("utf-8")).hexdigest()
    manifest = {
        "package": "phase10_frozen_dual_task_demo_data", "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_resolution": "Phase10 canonical prediction index -> Phase07 read-only references -> Phase06 hybrid classification / Phase05 common ridge regression OOF",
        "source_files": source_entries, "output_files": output_entries,
        "row_counts": {"classification": 419, "regression": 419, "aligned": 419},
        "unique_demo_ids": {"classification": 419, "regression": 419, "shared": 419},
        "coverage": {
            "classification_missing_target": 0, "classification_missing_prediction": 0,
            "regression_missing_target": 0, "regression_missing_prediction": 0,
        },
        "fold_coverage": [1, 2, 3, 4, 5],
        "alignment": {"performed_before_anonymization": True, "real_key_sets_equal": True,
                      "ordered_real_key_sha256": alignment_digest,
                      "stable_demo_id_range": ["DEMO-0001", "DEMO-0419"]},
        "presentation_only_transformations": [
            "aligned both canonical tasks by real run key before anonymization",
            "removed run key, subject, and session identifiers",
            "mapped source classification labels 0-3 to display Difficulty 1-4",
            "derived classification correctness from frozen labels",
            "derived absolute error from frozen regression target and bounded prediction",
        ],
        "scientific_transformations": "NONE", "ui_clipping_executed": False,
        "canonical_aggregation_policy": {
            "classification": contract["oof_aggregation"]["classification"]["rule"],
            "regression": contract["oof_aggregation"]["regression"]["rule"],
        },
        "training_executed": False, "new_predictions_generated": False,
        "statistics_recomputed": False, "model_selection_executed": False,
    }
    write_json(data_dir / "demo_data_manifest.json", manifest)


if __name__ == "__main__":
    build_data()
    print("Frozen dual-task UI package built and verified.")
