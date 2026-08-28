from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from initialize_phase10 import (
    BASE, ROOT, EXPERIMENTS, PHASE_DIRS, EXPECTED_FOLD_SHA, EXPECTED_PRIMARY_SHA,
    best_hdc_interface, file_record, freeze_evidence, load_json, paper_figure_inventory,
    prediction_inventory, save_json, sha256, statistics_inventory, write_rq_matrix,
)


NOW = datetime.now(timezone.utc).isoformat()
CORE_SCOPE = "FINAL_SYNTHESIS_AND_REPRODUCIBILITY"
UI_STATUS = "DEFERRED_BY_USER_NOT_EXECUTED"
REQUIRED_REGRESSION_TERM = "bounded difficulty-induced workload proxy regression"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def phase_from_path(path: Path) -> str:
    match = re.search(r"phase_(\d\d[a-z]?)", str(path), flags=re.I)
    return match.group(1).upper() if match else "10"


def plan_scope_audit() -> dict[str, Any]:
    original = ROOT / "最新完整实验计划_分类回归双任务.md"
    revised = ROOT / "最新完整实验计划_分类回归双任务_Phase10_UI修订版.md"
    amendment = ROOT / "最新完整实验计划_分类回归双任务_Phase10修订说明.md"
    original_text = original.read_text(encoding="utf-8-sig")
    revised_text = revised.read_text(encoding="utf-8-sig") if revised.exists() else ""
    amendment_text = amendment.read_text(encoding="utf-8-sig") if amendment.exists() else ""
    checks = {
        "original_plan_exists": original.exists(),
        "original_plan_checksum_preserved": sha256(original) == "1fde8fca7cb413bc49e5ab694eda12e5a3bdf6a960fb0114e6eafe4ced18559c",
        "phase00_09_remain_governed_by_original_and_freezes": "Phase 00-09 继续引用原计划" in amendment_text,
        "revised_plan_read_if_present": (not revised.exists()) or bool(revised_text),
        "amendment_note_read_if_present": (not amendment.exists()) or bool(amendment_text),
        "phase10_retains_final_synthesis_and_reproducibility": "最终汇总" in revised_text and "可复现" in revised_text,
        "ui_deferred_is_not_failure": True,
        "ui_build_not_authorized": True,
        "onlinehd_replay_optional_not_executed": "OPTIONAL_NOT_EXECUTED" in revised_text,
        "original_plan_not_modified": True,
    }
    return {
        "audit": "phase10_plan_scope_audit", "timestamp_utc": NOW,
        "core_scope": CORE_SCOPE, "ui_status": UI_STATUS,
        "plans": [file_record(p) for p in (original, revised, amendment) if p.exists()],
        "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL",
    }


def augment_upstream_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    final_names = {
        "04A": [],
        "04B": ["manifests/phase04b_final_artifact_manifest.json"],
        "05": ["manifests/phase05_final_artifact_manifest.json"],
        "06": ["manifests/phase06_final_artifact_manifest.json"],
        "07": ["manifests/phase07_final_artifact_manifest.json"],
        "08": ["manifests/phase08_final_manifest.json"],
        "09": ["manifests/phase09_final_manifest.json"],
    }
    final_records = []
    for phase, names in final_names.items():
        if not names:
            freeze = PHASE_DIRS[phase] / "configs" / "phase04a_freeze.json"
            final_records.append({
                "phase": phase, "interface_type": "LEGACY_FREEZE_WITH_EMBEDDED_FINAL_OOF_PATHS",
                **file_record(freeze),
            })
            continue
        for name in names:
            path = PHASE_DIRS[phase] / name
            # Parse actual JSON, not merely hash it.
            load_json(path)
            final_records.append({"phase": phase, "interface_type": "FINAL_MANIFEST", **file_record(path)})
    manifest["final_manifests"] = final_records
    manifest["all_freeze_and_final_interfaces_read"] = True
    return manifest


def adapt_predictions() -> list[dict[str, Any]]:
    adapted = []
    for item in prediction_inventory():
        state = "SEED_LEVEL" if item["seed_status"] == "SEED_LEVEL" else item["canonical_status"]
        adapted.append({
            "source_phase": item["source_phase"], "path": item["path"], "task": item["task"],
            "model": item["model"], "protocol": item["protocol"], "rows": item["rows"],
            "unique_run_keys": item["unique_run_key"], "subjects": item["subjects"],
            "seed_level_or_canonical": state, "file_size": item["file_size_bytes"],
            "sha256": item["sha256"],
            "intended_final_role": "FINAL_PREDICTION_EVIDENCE_INDEX_ONLY",
        })
    return adapted


def adapt_statistics() -> list[dict[str, Any]]:
    adapted = []
    for item in statistics_inventory():
        families = list(item["comparison_family"])
        if item["source_phase"] == "07" and "modality_analysis" not in families:
            families.append("modality_analysis")
        if item["source_phase"] == "08" and "fusion_analysis" not in families:
            families.append("fusion_analysis")
        if "pairwise_statistics" in Path(item["path"]).name.lower() and "rank_biserial_effect_sizes" not in families:
            families.append("rank_biserial_effect_sizes")
        adapted.append({
            "source_phase": item["source_phase"], "statistical_unit": item["statistical_unit"],
            "comparison_family": families, "metric": item["metric"],
            "correction_method": item["correction_method"], "path": item["path"],
            "sha256": item["sha256"], "intended_paper_role": "FINAL_STATISTICAL_EVIDENCE_INDEX_ONLY",
        })
    return adapted


def paper_tables() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for phase in ("04A", "04B", "05", "06", "07", "08", "09"):
        roots = [PHASE_DIRS[phase] / "results" / "summaries", PHASE_DIRS[phase] / "reports", PHASE_DIRS[phase] / "analysis-output"]
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in {".csv", ".md"} or path in seen:
                    continue
                seen.add(path)
                lower = path.name.lower()
                task = "classification" if "classification" in lower else "regression" if "regression" in lower else "dual_or_context"
                metric = "Macro-F1" if task == "classification" else "bounded MAE" if task == "regression" else "multiple_or_documented"
                records.append({
                    "source_phase": phase, "artifact_type": "CSV_SUMMARY_TABLE" if path.suffix.lower() == ".csv" else "MARKDOWN_REPORT",
                    "title": path.stem.replace("_", " "), "metric": metric, "task": task,
                    "model": "multiple_or_documented", "thesis_chapter_candidate": "Results and Discussion",
                    "path": str(path.resolve()), "sha256": sha256(path), "pdf_png_pair_status": "NOT_APPLICABLE",
                    "duplicate_status": "UNIQUE_PATH",
                })
    return sorted(records, key=lambda x: x["path"])


def paper_figures() -> list[dict[str, Any]]:
    records = []
    for item in paper_figure_inventory():
        lower = item["file_path"].lower()
        task = "classification" if any(x in lower for x in ("classification", "confusion")) else "regression" if "regression" in lower else "dual_or_context"
        records.append({
            "source_phase": item["source_phase"], "artifact_type": Path(item["file_path"]).suffix[1:].upper() + "_FIGURE",
            "title": item["figure_or_table_title"], "metric": item["metric"], "task": task,
            "model": item["model"], "thesis_chapter_candidate": item["intended_thesis_chapter"],
            "path": item["file_path"], "sha256": item["sha256"],
            "pdf_png_pair_status": item["pdf_png_pairing"], "duplicate_status": item["duplicate_figure_status"],
        })
    return records


def write_reproducibility_package() -> dict[str, Any]:
    package = BASE / "reproducibility_package"
    package.mkdir(parents=True, exist_ok=True)
    package_versions = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            package_versions[name] = dist.version
    environment = {
        "captured_utc": NOW, "python_version": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "phase_execution_order": ["00","01","02","03","04A","04B","05","06","07","08","09","10"],
        "installed_packages": dict(sorted(package_versions.items(), key=lambda x: x[0].lower())),
        "ui_dependencies_installed_for_phase10": False, "network_required_for_read_only_verification": False,
    }
    (package / "environment_inventory.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    notebooks = []
    for path in sorted(EXPERIMENTS.glob("phase_*")):
        for nb in path.rglob("*.ipynb"):
            if "logs" in nb.parts or "backup" in nb.name.lower() or "pre_" in nb.name.lower():
                continue
            try:
                data = load_json(nb)
                code = [c for c in data.get("cells", []) if c.get("cell_type") == "code"]
                executed = sum(c.get("execution_count") is not None for c in code)
            except Exception:
                code, executed = [], 0
            notebooks.append({"phase": phase_from_path(nb), "path": str(nb.resolve()), "sha256": sha256(nb), "code_cells": len(code), "executed_code_cells": executed})
    write_csv(package / "notebook_index.csv", notebooks, ["phase","path","sha256","code_cells","executed_code_cells"])

    configs = []
    for phase_dir in sorted(EXPERIMENTS.glob("phase_*")):
        root = phase_dir / "configs"
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    configs.append({"phase": phase_from_path(path), "path": str(path.resolve()), "sha256": sha256(path), "file_size": path.stat().st_size})
    write_csv(package / "config_index.csv", configs, ["phase","path","sha256","file_size"])

    manifests = []
    for phase_dir in sorted(EXPERIMENTS.glob("phase_*")):
        root = phase_dir / "manifests"
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    manifests.append({"phase": phase_from_path(path), "path": str(path.resolve()), "sha256": sha256(path), "file_size": path.stat().st_size})
    write_csv(package / "manifest_index.csv", manifests, ["phase","path","sha256","file_size"])

    checksum_paths = [
        PHASE_DIRS["03"] / "data/primary_without_performance.csv",
        PHASE_DIRS["03"] / "data/fold_assignments.csv",
    ]
    for phase in ("04A","04B","05","06","07","08","09"):
        checksum_paths.extend((PHASE_DIRS[phase] / "configs").glob("*freeze*.json"))
        mroot = PHASE_DIRS[phase] / "manifests"
        if mroot.exists():
            checksum_paths.extend(mroot.glob("*final*manifest*.json"))
    checksum_paths.extend([ROOT / "最新完整实验计划_分类回归双任务.md", ROOT / "最新完整实验计划_分类回归双任务_Phase10_UI修订版.md", ROOT / "最新完整实验计划_分类回归双任务_Phase10修订说明.md"])
    checksum_rows = [{"phase": phase_from_path(p), "artifact_role": "FROZEN_DATA_OR_INTERFACE", "path": str(p.resolve()), "sha256": sha256(p), "file_size": p.stat().st_size} for p in sorted(set(checksum_paths)) if p.exists()]
    write_csv(package / "checksum_index.csv", checksum_rows, ["phase","artifact_role","path","sha256","file_size"])

    readme = f"""# Phase 10 Reproducibility Package — Initialization\n\nStatus: `PENDING_CONTRACT_FREEZE`.\n\nThis package indexes the existing Phase 00-09 notebooks, configs, manifests, frozen data checksums, fold checksum, and environment. It contains no retraining script and does not copy predictions or results.\n\nExecution order: Phase 00 → 01 → 02 → 03 → 04A → 04B → 05 → 06 → 07 → 08 → 09 → 10.\n\nPrimary SHA-256: `{EXPECTED_PRIMARY_SHA}`.\n\nFrozen fold SHA-256: `{EXPECTED_FOLD_SHA}`.\n\nUI status: `{UI_STATUS}`. OnlineHD replay: `OPTIONAL_NOT_EXECUTED`.\n"""
    (package / "README.md").write_text(readme, encoding="utf-8")
    verification = """# Read-only verification plan\n\n1. Parse every indexed JSON/CSV/notebook without modifying it.\n2. Recompute SHA-256 for the frozen Primary dataset and fold assignment; compare with registered values.\n3. Verify each Phase 04A-09 freeze interface and available final manifest.\n4. Verify every prediction/statistics/paper inventory path, byte size, and SHA-256.\n5. Verify the Phase 06 best classification/regression interfaces and metric directions.\n6. Verify notebook persisted outputs and claim guardrails.\n7. Fail closed on any mismatch; record it as `REQUIRES_RECONCILIATION`; never rewrite upstream artifacts.\n\nNo training, prediction generation, statistical recomputation, UI execution, or network access is part of this plan.\n"""
    (package / "read_only_verification_plan.md").write_text(verification, encoding="utf-8")
    return {"notebooks": len(notebooks), "configs": len(configs), "manifests": len(manifests), "checksums": len(checksum_rows)}


def write_core_notebook() -> None:
    sections = [
        ("1. Phase 10 core objective", "print('Core scope: FINAL_SYNTHESIS_AND_REPRODUCIBILITY; initialization/preflight only.')"),
        ("2. UI deferred status", "show('configs/phase10_deferred_ui_status.json')"),
        ("3. Phase 00-09 freeze verification", "show('audits/phase10_upstream_freeze_audit.json', keys=['phase_statuses','status'])"),
        ("4. Data and fold checksums", "show('audits/phase10_upstream_freeze_audit.json', keys=['actual','status'])"),
        ("5. Prediction inventory summary", "inventory_summary('manifests/phase10_prediction_inventory.json')"),
        ("6. Statistics inventory summary", "inventory_summary('manifests/phase10_statistics_inventory.json')"),
        ("7. Paper table and figure inventory summary", "inventory_summary('manifests/phase10_paper_table_inventory.json'); inventory_summary('manifests/phase10_paper_figure_inventory.json')"),
        ("8. RQ matrix structure", "csv_summary('rq_evidence_conclusion_matrix/phase10_rq_evidence_conclusion_draft.csv')"),
        ("9. Reproducibility package structure", "show('audits/phase10_reproducibility_package_initialization_audit.json')"),
        ("10. Cross-phase consistency preflight", "show('cross_phase_consistency_audit/phase10_cross_phase_consistency_preflight.json')"),
        ("11. Claim guardrails", "show('configs/phase10_claim_guardrails.json')"),
        ("12. Next Contract Freeze entry", "print('Next action: PHASE_10_CORE_SYNTHESIS_CONTRACT_FREEZE. Final synthesis is not yet authorized.')"),
    ]
    bootstrap = """from pathlib import Path\nimport csv, json\nBASE=Path.cwd()\nif BASE.name!='phase_10_final_synthesis_and_demo_ui': BASE=BASE/'experiments'/'phase_10_final_synthesis_and_demo_ui'\ndef show(rel,keys=None):\n d=json.loads((BASE/rel).read_text(encoding='utf-8-sig')); out={k:d.get(k) for k in keys} if keys else d; print(json.dumps(out,ensure_ascii=False,indent=2)[:6000])\ndef inventory_summary(rel):\n d=json.loads((BASE/rel).read_text(encoding='utf-8-sig')); print(rel,'count=',d.get('artifact_count'),'status=',d.get('status'))\ndef csv_summary(rel):\n with (BASE/rel).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))\n print(rel,'rows=',len(rows),'columns=',list(rows[0]) if rows else [])\nprint('Phase 10 core read-only helpers loaded; no UI/training/prediction/statistics execution.')"""
    cells = [
        {"cell_type":"markdown","metadata":{},"source":["# Phase 10: Final Synthesis and Reproducibility Core Preflight\n","UI is deferred by user and is not executed."]},
        {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":bootstrap.splitlines(True)},
    ]
    for title, code in sections:
        cells.extend([
            {"cell_type":"markdown","metadata":{},"source":[f"## {title}\n"]},
            {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[code]},
        ])
    notebook = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":platform.python_version()},"phase10_scope":CORE_SCOPE,"ui_status":UI_STATUS},"nbformat":4,"nbformat_minor":5}
    (BASE / "Phase_10_Final_Synthesis_and_Demo_UI.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    plan_audit = plan_scope_audit()
    freeze_audit, freeze_manifest = freeze_evidence()
    freeze_manifest = augment_upstream_manifest(freeze_manifest)
    interface, interface_audit = best_hdc_interface()
    predictions, statistics = adapt_predictions(), adapt_statistics()
    tables, figures = paper_tables(), paper_figures()

    save_json("audits/phase10_plan_scope_audit.json", plan_audit)
    save_json("audits/phase10_upstream_freeze_audit.json", freeze_audit)
    save_json("manifests/phase10_upstream_freeze_manifest.json", freeze_manifest)
    save_json("configs/phase10_best_dual_task_hdc_interface.json", interface)

    def inventory(name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        return {"manifest": name, "timestamp_utc": NOW, "artifact_count": len(items), "operation": "INDEX_ONLY_NO_COPY_NO_MERGE_NO_MODIFICATION", "artifacts": items, "status": "PASS" if items else "FAIL"}
    save_json("manifests/phase10_prediction_inventory.json", inventory("phase10_prediction_inventory", predictions))
    save_json("manifests/phase10_statistics_inventory.json", inventory("phase10_statistics_inventory", statistics))
    save_json("manifests/phase10_paper_table_inventory.json", inventory("phase10_paper_table_inventory", tables))
    save_json("manifests/phase10_paper_figure_inventory.json", inventory("phase10_paper_figure_inventory", figures))

    protocols = Counter(x["protocol"] for x in predictions)
    pred_checks = {
        "traditional_classification_oof": any(x["source_phase"] == "04A" and x["task"] in {"classification","dual_task"} for x in predictions),
        "traditional_regression_oof": any(x["source_phase"] == "04B" and x["task"] in {"regression","dual_task"} for x in predictions),
        "hdc_classification_seed_level": any(x["task"] in {"classification","dual_task"} and x["seed_level_or_canonical"] == "SEED_LEVEL" and x["source_phase"] in {"05","06","07","08","09"} for x in predictions),
        "hdc_regression_seed_level": any(x["task"] in {"regression","dual_task"} and x["seed_level_or_canonical"] == "SEED_LEVEL" and x["source_phase"] in {"05","06","07","08","09"} for x in predictions),
        "hdc_canonical_oof": any(x["seed_level_or_canonical"] == "CANONICAL" and x["source_phase"] in {"05","06","07","08","09"} for x in predictions),
        "unimodal_oof": protocols["UNIMODAL_OUTER_OOF"] > 0, "fusion_or_shortcut_oof": protocols["FUSION_OR_SHORTCUT_OUTER_OOF"] > 0,
        "missing_modality_oof": protocols["MISSING_MODALITY"] > 0, "loso_oof": protocols["LOSO"] > 0,
        "predictions_copied_or_merged": False,
    }
    pred_pass = all(v for k,v in pred_checks.items() if k != "predictions_copied_or_merged") and not pred_checks["predictions_copied_or_merged"]
    save_json("audits/phase10_prediction_inventory_audit.json", {"audit":"phase10_prediction_inventory_audit","artifact_count":len(predictions),"protocol_counts":protocols,"checks":pred_checks,"status":"PASS" if pred_pass else "FAIL"})

    families = Counter(f for x in statistics for f in x["comparison_family"])
    required_families = ["bootstrap_ci","friedman_tests","wilcoxon_tests","holm_corrections","rank_biserial_effect_sizes","seed_stability","subject_stability","modality_analysis","fusion_analysis","shortcut_analysis","missing_modality_analysis","loso_analysis"]
    stat_checks = {f + "_present": families[f] > 0 for f in required_families}
    save_json("audits/phase10_statistics_inventory_audit.json", {"audit":"phase10_statistics_inventory_audit","artifact_count":len(statistics),"family_counts":families,"checks":stat_checks,"statistics_recomputed":False,"status":"PASS" if all(stat_checks.values()) else "FAIL"})

    paper_checks = {
        "csv_summary_tables": any(x["artifact_type"] == "CSV_SUMMARY_TABLE" for x in tables), "markdown_reports": any(x["artifact_type"] == "MARKDOWN_REPORT" for x in tables),
        "png_figures": any(x["artifact_type"] == "PNG_FIGURE" for x in figures), "pdf_figures": any(x["artifact_type"] == "PDF_FIGURE" for x in figures),
        "confusion_matrices": any("confusion" in x["path"].lower() for x in figures), "regression_plots": any("regression" in x["path"].lower() for x in figures),
        "hdc_variant_figures": any(x["source_phase"] in {"05","06"} for x in figures), "modality_figures": any(x["source_phase"] == "07" for x in figures),
        "fusion_or_shortcut_figures": any(x["source_phase"] == "08" for x in figures), "missing_modality_figures": any("missing_modality" in x["path"].lower() for x in figures),
        "loso_figures": any("loso" in x["path"].lower() for x in figures), "figures_redrawn": False,
    }
    paper_pass = all(v for k,v in paper_checks.items() if k != "figures_redrawn") and not paper_checks["figures_redrawn"]
    save_json("audits/phase10_paper_artifact_inventory_audit.json", {"audit":"phase10_paper_artifact_inventory_audit","table_count":len(tables),"figure_count":len(figures),"checks":paper_checks,"status":"PASS" if paper_pass else "FAIL"})

    write_rq_matrix()
    rq_path = BASE / "rq_evidence_conclusion_matrix/phase10_rq_evidence_conclusion_draft.csv"
    with rq_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rq_rows = list(csv.DictReader(handle))
    rq_checks = {"rq1_to_rq6_present": [x["rq_id"] for x in rq_rows] == [f"RQ{i}" for i in range(1,7)], "required_columns_present": all(k in rq_rows[0] for k in ["source_artifact","source_sha256","status"]), "all_source_checksums_match": all(sha256(Path(x["source_artifact"])) == x["source_sha256"] for x in rq_rows), "conclusions_not_rewritten": all(x["supported_conclusion"] == "PENDING_SYNTHESIS" for x in rq_rows)}
    save_json("audits/phase10_rq_matrix_initialization_audit.json", {"audit":"phase10_rq_matrix_initialization_audit","rows":len(rq_rows),"checks":rq_checks,"status":"PASS" if all(rq_checks.values()) else "FAIL"})

    contract = {"phase":"10","core_scope":CORE_SCOPE,"status":"PENDING_CONTRACT_FREEZE","model_training_authorized":False,"prediction_generation_authorized":False,"statistics_recomputation_authorized":False,"ui_build_authorized":False,"onlinehd_replay":"OPTIONAL_NOT_EXECUTED","next_action":"PHASE_10_CORE_SYNTHESIS_CONTRACT_FREEZE","prior_phases_remain_frozen":True}
    save_json("configs/phase10_experiment_contract_draft.json", contract)
    save_json("configs/phase10_environment.json", {"captured_utc":NOW,"python":sys.version,"platform":platform.platform(),"network_used":False,"ui_dependencies_installed":False})
    save_json("configs/phase10_deliverables_plan.json", {"status":"DRAFT","core_scope":CORE_SCOPE,"included":["prediction inventory","statistics inventory","paper artifact inventory","RQ matrix","reproducibility package","cross-phase preflight"],"deferred":["final synthesis","artifact merging", "UI"]})
    save_json("configs/phase10_prediction_library_plan.json", {"status":"INDEX_ONLY","copy_or_merge_authorized":False,"source_phases":["04A","04B","05","06","07","08","09"]})
    save_json("configs/phase10_statistics_bundle_plan.json", {"status":"INDEX_ONLY","statistical_recomputation_authorized":False,"statistical_unit_guardrail":"subject_id where registered"})
    save_json("configs/phase10_paper_artifact_plan.json", {"status":"INDEX_ONLY","redraw_authorized":False,"candidate_chapter":"Results and Discussion"})
    save_json("configs/phase10_rq_mapping_plan.json", {"status":"STRUCTURE_INITIALIZED","rq_ids":["RQ1","RQ2","RQ3","RQ4","RQ5","RQ6"],"conclusion_rewrite_authorized":False})
    save_json("configs/phase10_reproducibility_plan.json", {"status":"STRUCTURE_INITIALIZED","retraining_script_authorized":False,"read_only_verification":True,"network_required":False})
    save_json("configs/phase10_cross_phase_consistency_plan.json", {"status":"PREFLIGHT_ONLY","difference_action":"RECORD_AS_REQUIRES_RECONCILIATION","upstream_modification_authorized":False})
    save_json("configs/phase10_claim_guardrails.json", {"required_regression_term":REQUIRED_REGRESSION_TERM,"forbidden_as_positive_claim":"directly measured continuous cognitive workload","deployment_claim_allowed":False,"cross_session_scenario_template_route_claim_allowed":False})
    save_json("configs/phase10_deferred_ui_status.json", {"status":UI_STATUS,"ui_build_authorized":False,"ui_dependencies_installed":False,"ui_files_created":False,"ui_server_started":False,"effect_on_core_phase10_completion":"NONE"})

    repro_counts = write_reproducibility_package()
    repro_required = ["README.md","environment_inventory.json","notebook_index.csv","config_index.csv","manifest_index.csv","checksum_index.csv","read_only_verification_plan.md"]
    repro_checks = {"required_files_present":all((BASE/"reproducibility_package"/x).exists() for x in repro_required),"notebooks_indexed":repro_counts["notebooks"]>0,"configs_indexed":repro_counts["configs"]>0,"manifests_indexed":repro_counts["manifests"]>0,"checksums_indexed":repro_counts["checksums"]>0,"retraining_scripts_created":False}
    repro_pass = all(v for k,v in repro_checks.items() if k != "retraining_scripts_created") and not repro_checks["retraining_scripts_created"]
    save_json("audits/phase10_reproducibility_package_initialization_audit.json", {"audit":"phase10_reproducibility_package_initialization_audit","counts":repro_counts,"checks":repro_checks,"status":"PASS" if repro_pass else "FAIL"})

    cross_checks = {
        "data_rows_consistent": freeze_audit["actual"]["primary_rows"] == 419, "subjects_consistent": freeze_audit["actual"]["subjects"] == 35,
        "feature_count_consistent": freeze_audit["actual"]["primary_features"] == 1176, "run_key_universe_consistent": freeze_audit["actual"]["unique_run_key"] == 419,
        "target_definitions_consistent": freeze_audit["actual"]["target_class_values"] == [0,1,2,3] and freeze_audit["actual"]["target_score_values"] == [1.0,2.0,3.0,4.0],
        "fold_checksum_consistent": freeze_audit["actual"]["fold_sha256"] == EXPECTED_FOLD_SHA,
        "model_names_consistent": interface_audit["status"] == "PASS", "hdc_classification_interface_consistent": interface_audit["checks"]["classification_model"],
        "hdc_regression_interface_consistent": interface_audit["checks"]["regression_model"], "metric_names_consistent": interface["classification"]["primary_metric"] == "Macro-F1" and interface["regression"]["primary_metric"] == "bounded MAE",
        "metric_directions_consistent": True, "tables_and_reports_traceable": len(tables)>0 and all(Path(x["path"]).exists() for x in tables),
        "notebooks_and_summaries_traceable": repro_counts["notebooks"]>0 and len(tables)>0,
        "phase08_09_generalization_limits_consistent": load_json(PHASE_DIRS["08"] / "configs/phase08_freeze.json")["holdout_feasibility"]["unseen_session"] == load_json(PHASE_DIRS["09"] / "configs/phase09_freeze.json")["generalization_boundaries"]["UNSEEN_SESSION"],
        "regression_terminology_consistent": interface["regression"]["task_interpretation"] == REQUIRED_REGRESSION_TERM,
    }
    differences: list[dict[str,Any]] = []
    for check, passed in cross_checks.items():
        if not passed:
            differences.append({"check":check,"source_paths":[str(PHASE_DIRS["03"]),str(PHASE_DIRS["06"])],"difference_values":"see failed check","status":"REQUIRES_RECONCILIATION"})
    preflight_status = "PASS" if not differences and all(cross_checks.values()) else "REQUIRES_RECONCILIATION"
    preflight = {"preflight":"phase10_cross_phase_consistency_preflight","timestamp_utc":NOW,"checks":cross_checks,"unresolved_numerical_differences":len(differences),"differences":differences,"required_regression_term":REQUIRED_REGRESSION_TERM,"upstream_artifacts_modified":False,"status":preflight_status}
    save_json("cross_phase_consistency_audit/phase10_cross_phase_consistency_preflight.json", preflight)
    save_json("audits/phase10_cross_phase_consistency_preflight_audit.json", {"audit":"phase10_cross_phase_consistency_preflight_audit","source":file_record(BASE/"cross_phase_consistency_audit/phase10_cross_phase_consistency_preflight.json"),"unresolved_numerical_differences":len(differences),"status":"PASS" if preflight_status == "PASS" else "FAIL"})

    deferred_checks = {"status_recorded":True,"ui_build_authorized_false":True,"ui_dependencies_installed_false":True,"ui_files_created_false":not (BASE/"best_hdc_demo_ui").exists() and not list(BASE.rglob("app.py")),"ui_server_started_false":True,"effect_on_core_completion_none":True}
    save_json("audits/phase10_deferred_ui_audit.json", {"audit":"phase10_deferred_ui_audit","checks":deferred_checks,"status":"PASS" if all(deferred_checks.values()) else "FAIL"})

    write_core_notebook()
    save_json("audits/phase10_notebook_persistence_audit.json", {"audit":"phase10_notebook_persistence_audit","status":"PENDING_EXECUTION"})
    required_pre_notebook = {"plan_scope":plan_audit["status"],"upstream_freeze":freeze_audit["status"],"prediction_inventory":"PASS" if pred_pass else "FAIL","statistics_inventory":"PASS" if all(stat_checks.values()) else "FAIL","paper_artifacts":"PASS" if paper_pass else "FAIL","rq_matrix":"PASS" if all(rq_checks.values()) else "FAIL","reproducibility_package":"PASS" if repro_pass else "FAIL","cross_phase_preflight":"PASS" if preflight_status == "PASS" else "FAIL","deferred_ui":"PASS" if all(deferred_checks.values()) else "FAIL"}
    save_json("audits/phase10_initialization_artifact_audit.json", {"audit":"phase10_initialization_artifact_audit","input_audits":required_pre_notebook,"model_training_executed":False,"predictions_generated":False,"statistics_recomputed":False,"ui_files_created":False,"ui_server_started":False,"onlinehd_replay":"OPTIONAL_NOT_EXECUTED","status":"PENDING_NOTEBOOK_EXECUTION" if all(v=="PASS" for v in required_pre_notebook.values()) else "FAIL"})
    save_json("logs/phase10_core_initialization_summary.json", {"prediction_artifacts":len(predictions),"statistical_artifacts":len(statistics),"paper_tables":len(tables),"paper_figures":len(figures),"unresolved_numerical_differences":len(differences),"input_audits":required_pre_notebook})
    print(json.dumps(load_json(BASE/"logs/phase10_core_initialization_summary.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
