from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "src"))

from phase06_preflight import (  # noqa: E402
    CANONICAL_SELECTION,
    EXPECTED_FOLD_SHA256,
    EXPECTED_PRIMARY_SHA256,
    REQUIRED_VARIANTS,
    run_preflight,
    sha256,
)


def prepare_isolated_phase(tmp_path: Path) -> Path:
    isolated = tmp_path / "phase06"
    for relative in ["README.md", "src/phase06_preflight.py", "scripts/initialize_phase06.py", "scripts/build_phase06_notebook.py", "tests/test_phase06_preflight.py"]:
        destination = isolated / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PHASE_DIR / relative, destination)
    for relative in ["data", "manifests", "audits", "configs", "src", "scripts", "tests", "figures", "logs", "reports", "results/checkpoints", "results/predictions", "results/fold_metrics", "results/oof", "results/summaries", "results/efficiency"]:
        (isolated / relative).mkdir(parents=True, exist_ok=True)
    return isolated


def test_preflight_all_required_gates_pass(tmp_path: Path) -> None:
    isolated = prepare_isolated_phase(tmp_path)
    result = run_preflight(PHASE_DIR, output_dir=isolated)
    assert result["primary_sha256"] == EXPECTED_PRIMARY_SHA256
    assert result["frozen_fold_sha256"] == EXPECTED_FOLD_SHA256
    assert result["outer_subject_isolation"] == "PASS"
    assert result["inner_groupkfold_3_feasibility"] == "PASS"
    assert result["phase05"]["freeze_interface"] == "PASS"
    assert result["phase04"] == {"phase04a": "PASS", "phase04b": "PASS"}
    assert result["required_hdc_variants"] == REQUIRED_VARIANTS
    assert result["model_training_executed"] == "NO"
    assert result["phase06_status"] == "PENDING_CONTRACT_FREEZE"
    assert not list((isolated / "results" / "oof").glob("*.csv"))


def test_contract_defers_unfrozen_algorithm_choices() -> None:
    contract = json.loads((PHASE_DIR / "configs" / "phase06_experiment_contract.json").read_text(encoding="utf-8"))
    assert contract["status"] == "PENDING_CONTRACT_FREEZE"
    assert contract["model_training_executed"] is False
    assert contract["contract_freeze_required_before_modeling"] is True
    assert contract["canonical_phase05_configuration_selection_record"] == CANONICAL_SELECTION
    assert len(contract["deferred_to_contract_freeze"]) == 11


def test_phase06_contains_no_copied_upstream_csv() -> None:
    assert not (PHASE_DIR / "data" / "primary_without_performance.csv").exists()
    assert not (PHASE_DIR / "data" / "fold_assignments.csv").exists()
    frozen_oof = {path.name: len(pd.read_csv(path)) for path in (PHASE_DIR / "results" / "oof").glob("*.csv")}
    assert frozen_oof == {
        "phase06_four_variant_classification_oof_long.csv": 33520,
        "phase06_four_variant_similarity_regression_oof_long.csv": 33520,
        "phase06_hybrid_final_oof.csv": 8380,
        "phase06_multicentroid_final_oof.csv": 8380,
        "phase06_new_variants_final_oof_long.csv": 25140,
        "phase06_onlinehd_final_oof.csv": 8380,
    }


def test_every_json_evidence_record_reopens_with_matching_hash_and_size(tmp_path: Path) -> None:
    isolated = prepare_isolated_phase(tmp_path)
    run_preflight(PHASE_DIR, output_dir=isolated)
    json_paths = [
        isolated / "configs" / "phase06_experiment_contract.json",
        isolated / "configs" / "phase06_environment.json",
        isolated / "configs" / "phase06_upstream_interface.json",
        isolated / "manifests" / "phase06_input_manifest.json",
        isolated / "audits" / "phase06_input_and_fold_audit.json",
        isolated / "audits" / "phase06_phase05_freeze_interface_audit.json",
        isolated / "audits" / "phase06_phase04_baseline_interface_audit.json",
        isolated / "audits" / "phase06_initialization_artifact_audit.json",
    ]

    def records(value):
        if isinstance(value, dict):
            if {"path", "file_size_bytes", "sha256"}.issubset(value):
                yield value
            for child in value.values():
                yield from records(child)
        elif isinstance(value, list):
            for child in value:
                yield from records(child)

    checked = 0
    for json_path in json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["result"] == "PASS"
        for record in records(payload):
            evidence_path = Path(record["path"])
            assert evidence_path.is_file()
            assert evidence_path.stat().st_size == record["file_size_bytes"]
            assert sha256(evidence_path) == record["sha256"]
            checked += 1
    assert checked >= 20


def test_initialization_notebook_did_not_execute_training() -> None:
    prohibited = (".fit(", ".fit_predict(", ".partial_fit(", "predict_proba(")
    sources = [PHASE_DIR / "src" / "phase06_preflight.py", PHASE_DIR / "scripts" / "initialize_phase06.py", PHASE_DIR / "scripts" / "build_phase06_notebook.py"]
    notebook = json.loads((PHASE_DIR / "Phase_06_HDC_Variant_Screening.ipynb").read_text(encoding="utf-8"))
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    text += "\n" + "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", ""))
        for cell in notebook["cells"]
    )
    assert not any(token in text for token in prohibited)
