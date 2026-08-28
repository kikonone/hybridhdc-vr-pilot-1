import json
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]


def load(relative):
    return json.loads((BASE / relative).read_text(encoding="utf-8-sig"))


def test_ui_was_deferred_at_core_preflight_and_later_delivery_is_isolated():
    status = load("configs/phase10_deferred_ui_status.json")
    assert status == {"status":"DEFERRED_BY_USER_NOT_EXECUTED","ui_build_authorized":False,"ui_dependencies_installed":False,"ui_files_created":False,"ui_server_started":False,"effect_on_core_phase10_completion":"NONE"}
    assert not (BASE / "best_hdc_demo_ui").exists()

    # The frozen status above records the core-preflight boundary. A later,
    # independently authorized read-only display layer now lives under ui/.
    ui = BASE / "ui"
    assert (ui / "app.py").exists()
    delivery_audit = json.loads(
        (ui / "audits" / "ui_final_dual_task_audit.json").read_text(
            encoding="utf-8-sig"
        )
    )
    assert delivery_audit["status"] == "PASS"
    assert delivery_audit["phase00_10_files_modified"] == 0
    assert delivery_audit["ready_for_defense_demonstration"] is True


def test_contract_is_core_freeze_pending_only():
    contract = load("configs/phase10_experiment_contract_draft.json")
    assert contract["status"] == "PENDING_CONTRACT_FREEZE"
    assert contract["core_scope"] == "FINAL_SYNTHESIS_AND_REPRODUCIBILITY"
    assert contract["next_action"] == "PHASE_10_CORE_SYNTHESIS_CONTRACT_FREEZE"
    assert contract["onlinehd_replay"] == "OPTIONAL_NOT_EXECUTED"
    for key in ("model_training_authorized","prediction_generation_authorized","statistics_recomputation_authorized","ui_build_authorized"):
        assert contract[key] is False


def test_reproducibility_package_is_index_only():
    package = BASE / "reproducibility_package"
    for name in ("README.md","environment_inventory.json","notebook_index.csv","config_index.csv","manifest_index.csv","checksum_index.csv","read_only_verification_plan.md"):
        assert (package / name).exists(), name
    assert not list(package.rglob("*.py"))


def test_core_audits_pass():
    for name in ("phase10_plan_scope_audit.json","phase10_upstream_freeze_audit.json","phase10_prediction_inventory_audit.json","phase10_statistics_inventory_audit.json","phase10_paper_artifact_inventory_audit.json","phase10_rq_matrix_initialization_audit.json","phase10_reproducibility_package_initialization_audit.json","phase10_cross_phase_consistency_preflight_audit.json","phase10_deferred_ui_audit.json"):
        assert load("audits/" + name)["status"] == "PASS", name


def test_contract_freeze_materializes_indices_only():
    prediction_files = {p.relative_to(BASE / "final_prediction_library").as_posix() for p in (BASE / "final_prediction_library").rglob("*") if p.is_file()}
    statistics_files = {p.relative_to(BASE / "final_statistics_bundle").as_posix() for p in (BASE / "final_statistics_bundle").rglob("*") if p.is_file()}
    assert prediction_files == {"README.md", "index.csv"}
    assert statistics_files == {"README.md", "index.csv"}
    assert not list((BASE / "final_paper_tables").rglob("*.*"))
    assert not list((BASE / "final_paper_figures").rglob("*.*"))


def test_core_contract_freeze_audits_pass():
    for name in (
        "phase10_core_contract_freeze_audit.json", "phase10_source_of_truth_contract_audit.json",
        "phase10_prediction_library_contract_audit.json", "phase10_statistics_bundle_contract_audit.json",
        "phase10_paper_artifact_contract_audit.json", "phase10_rq_contract_audit.json",
        "phase10_reproducibility_contract_audit.json", "phase10_cross_phase_contract_audit.json",
        "phase10_deferred_ui_contract_audit.json", "phase10_core_contract_artifact_audit.json",
        "phase10_core_contract_notebook_persistence_audit.json",
    ):
        assert load("audits/" + name)["status"] == "PASS", name


def test_execution_manifest_has_no_forbidden_operations():
    manifest = load("configs/phase10_core_execution_manifest.json")
    operations = [item["operation"] for item in manifest["work_items"]]
    assert manifest["contains_model_runs"] is False
    assert all(item["status"] == "AUTHORIZED_NOT_EXECUTED" for item in manifest["work_items"])
    assert not set(operations).intersection({"TRAIN", "PREDICT", "TUNE", "RESELECT_MODEL", "RECOMPUTE_STATISTICS", "BUILD_UI"})


def test_terminal_contract_status():
    freeze = load("configs/phase10_core_contract_freeze.json")
    audit = load("audits/phase10_core_contract_freeze_audit.json")
    assert freeze["status"] == "CORE_CONTRACT_FROZEN_NOT_SYNTHESIZED"
    assert audit["phase10_status"] == freeze["status"]
    assert audit["ready_for_phase10_final_synthesis"] is True
    assert audit["phase00_09_files_modified"] == 0
    assert audit["scientific_source_conflicts"] == 0
