from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


PHASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PHASE_DIR.parents[1]
sys.path.insert(0, str(PHASE_DIR / "scripts"))

from freeze_phase07_contract import (  # noqa: E402
    EXPECTED_UPSTREAM_HASHES,
    MODALITY_COUNTS,
    P5_ENCODING,
    P6_CLASSIFICATION,
    P6_FREEZE,
    P6_REGRESSION,
    P6_VARIANT,
    SEEDS,
    effective_feature_k,
    result_inventory,
)


def load(relative: str) -> dict:
    return json.loads((PHASE_DIR / relative).read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def contract() -> dict:
    return load("configs/phase07_frozen_unimodal_contract.json")


def test_all_json_parseable() -> None:
    files = list((PHASE_DIR / "configs").glob("*.json")) + list((PHASE_DIR / "manifests").glob("*.json")) + list((PHASE_DIR / "audits").glob("*.json"))
    assert files
    for path in files:
        json.loads(path.read_text(encoding="utf-8"))


def test_modalities_counts_disjoint_union() -> None:
    manifest = load("manifests/phase07_modality_feature_manifest.json")
    counts = {item["name"]: item["feature_count"] for item in manifest["modalities"]}
    feature_lists = [item["features"] for item in manifest["modalities"]]
    flat = [feature for features in feature_lists for feature in features]
    assert counts == MODALITY_COUNTS
    assert len(flat) == len(set(flat)) == 1176


def test_upstream_checksums() -> None:
    paths = {
        "primary_data": PROJECT_ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv",
        "frozen_folds": PROJECT_ROOT / "experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv",
        "phase06_freeze": P6_FREEZE,
        "phase06_classification": P6_CLASSIFICATION,
        "phase06_regression": P6_REGRESSION,
        "phase06_variant_contract": P6_VARIANT,
        "phase05_encoding_contract": P5_ENCODING,
    }
    assert {name: digest(path) for name, path in paths.items()} == EXPECTED_UPSTREAM_HASHES


def test_classification_interface_and_fold_structures(contract: dict) -> None:
    frozen = json.loads(P6_CLASSIFICATION.read_text(encoding="utf-8"))
    expected = [{"outer_fold": item["outer_fold"], **json.loads(item["selected_structure_json"])} for item in frozen["fold_selected_structures"]]
    classification = contract["classification"]
    assert classification["model"] == "HDC+OnlineHD Hybrid"
    assert classification["dimension"] == 5000
    assert classification["levels"] == 51
    assert classification["seeds"] == SEEDS
    assert classification["fold_structures"] == expected


def test_regression_interface_alpha_and_seeds(contract: dict) -> None:
    regression = contract["regression"]
    assert regression["head"] == "COMMON_ENCODER_READOUT_BASELINE"
    assert regression["dimension"] == 10000
    assert regression["ridge_alpha"] == 0.01
    assert regression["seeds"] == SEEDS
    for fold in regression["fold_seed_parameter_policy"]:
        assert [item["seed"] for item in fold["seed_parameters"]] == SEEDS
        assert all(item["ridge_alpha"] == 0.01 for item in fold["seed_parameters"])


def test_effective_feature_k_handles_body_movement(contract: dict) -> None:
    assert effective_feature_k(50, 42) == 42
    assert effective_feature_k(50, 65) == 50
    assert contract["preprocessing"]["effective_feature_k_rule"] == "min(50, post_variance_feature_count)"


def test_full_cohort_and_missingness_policy(contract: dict) -> None:
    assert contract["sample_and_fold_policy"]["modeling_rows"] == 419
    assert contract["sample_and_fold_policy"]["retain_all_modeling_rows"] is True
    policy = contract["missingness_policy"]
    assert policy["status"] == "FROZEN"
    assert policy["fully_missing_rows_retained"] is True
    assert policy["missing_indicators_required"] is True
    assert policy["fully_missing_rows_by_modality"] == {"physiological_features": 0, "eye_tracking_features": 14, "head_movement_features": 0, "flight_parameter_features": 0, "body_movement": 29}


def test_seed_aggregation_and_rankings_frozen(contract: dict) -> None:
    assert contract["oof_aggregation"]["status"] == "FROZEN"
    assert contract["oof_aggregation"]["classification"]["argmax_tie"] == "smaller target_class"
    assert contract["oof_aggregation"]["regression"]["canonical_metrics_recomputed_from_aggregated_predictions"] is True
    assert contract["ranking"]["classification"]["status"] == "FROZEN"
    assert contract["ranking"]["regression"]["status"] == "FROZEN"
    assert contract["ranking"]["combined_best_modality_prohibited"] is True


def test_statistical_rules() -> None:
    stats = load("configs/phase07_statistical_analysis_contract.json")
    assert stats["statistical_unit"] == "subject" and stats["subjects"] == 35
    assert stats["bootstrap"] == {"repetitions": 2000, "seed": 42, "confidence_interval": "percentile 95%", "shared_subject_resamples_for_all_models": True, "resampled_subject_includes_all_oof_runs": True, "classification_metrics": ["macro_f1", "balanced_accuracy", "severe_error_rate"], "regression_metrics": ["bounded_mae", "bounded_rmse"]}
    assert stats["overall_unimodal_comparison"]["test"] == "Friedman"
    assert stats["preregistered_pairwise"]["test"] == "Wilcoxon signed-rank"
    assert "Holm" in stats["preregistered_pairwise"]["multiplicity"]
    assert stats["preregistered_pairwise"]["effect_size"] == "rank-biserial correlation"


def test_execution_manifest_and_no_results() -> None:
    execution = load("configs/phase07_execution_manifest.json")
    assert execution["modalities"] == execution["outer_folds"] == execution["evaluation_seed_count"] == 5
    assert execution["classification_runs"] == execution["regression_runs"] == 125
    assert execution["total_model_runs"] == 250
    assert execution["completed_runs"] in {0, 250}
    assert execution["training_executed"] is (execution["completed_runs"] == 250)
    inventory = result_inventory()
    if execution["completed_runs"] == 0:
        assert all(not files for files in inventory.values())
    else:
        assert inventory["checkpoints"] and inventory["predictions"]
        if execution.get("status") == "FROZEN":
            assert execution["canonical_oof_generated"] is True
            assert len(inventory["oof"]) == 6
            assert set(Path(path).name for path in inventory["oof"]) == {
                "phase07_unimodal_classification_seed_level_oof.csv",
                "phase07_unimodal_classification_canonical_oof.csv",
                "phase07_unimodal_regression_seed_level_oof.csv",
                "phase07_unimodal_regression_canonical_oof.csv",
                "phase07_readonly_multimodal_classification_reference.csv",
                "phase07_readonly_multimodal_regression_reference.csv",
            }
        else:
            assert not inventory["oof"]


def test_status_and_prohibitions(contract: dict) -> None:
    assert contract["status"] == "CONTRACT_FROZEN_NOT_TRAINED"
    assert contract["training_executed"] is False
    assert contract["outer_test_predictions_generated"] is False
