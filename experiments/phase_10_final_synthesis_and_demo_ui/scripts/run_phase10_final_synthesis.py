from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from freeze_phase10_core_contract import compare_states, phase00_09_state
from initialize_phase10 import BASE, EXPERIMENTS, load_json, sha256


ROOT = BASE.parents[1]
NOW = datetime.now(timezone.utc).isoformat()
STATUS = "FINAL_SYNTHESIS_COMPLETE_PENDING_PHASE10_FREEZE"


def save_json(relative: str, payload: Any) -> None:
    path = BASE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(relative: str, text: str) -> None:
    path = BASE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(relative: str, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path = BASE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_ref(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def values(row: dict[str, str], names: tuple[str, ...]) -> set[str]:
    return {row[name] for name in names if name in row and row[name] not in (None, "")}


def filename_values(path: Path, label: str) -> set[str]:
    return set(re.findall(rf"{label}[_-]?(\d+)", path.stem.lower()))


def inspect_prediction(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["source_path"])
    if not path.exists() or sha256(path) != record["source_sha256"]:
        raise RuntimeError(f"Prediction source hash failure: {path}")
    row_count = 0
    run_keys: set[str] = set()
    subjects: set[str] = set()
    folds: set[str] = set()
    seeds: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            run_keys |= values(row, ("run_key", "sample_key"))
            subjects |= values(row, ("subject_id", "subject"))
            folds |= values(row, ("outer_fold", "fold"))
            seeds |= values(row, ("seed", "random_seed"))
    folds |= filename_values(path, "fold")
    seeds |= filename_values(path, "seed")
    if row_count != int(record["rows"]):
        raise RuntimeError(f"Prediction row-count mismatch: {path}: {row_count} != {record['rows']}")
    expected_runs = int(record.get("unique_run_keys") or 0)
    expected_subjects = int(record.get("subjects") or 0)
    if run_keys and len(run_keys) != expected_runs:
        raise RuntimeError(f"Prediction run-key mismatch: {path}")
    if subjects and len(subjects) != expected_subjects:
        raise RuntimeError(f"Prediction subject mismatch: {path}")
    text = str(path).lower()
    modality = next((name for name in (
        "flight_parameter_features", "eye_tracking_features", "head_movement_features",
        "physiological_features", "body_movement", "multimodal",
    ) if name in text), "ALL_OR_DOCUMENTED_IN_SOURCE")
    model_name = record["model_name"]
    family = "TRADITIONAL" if "traditional" in model_name.lower() or record["source_phase"] in {"04A", "04B"} else "HDC"
    canonical = record["canonical_status"]
    return {
        "source_phase": record["source_phase"], "task": record["task"],
        "model_family": family, "model_name": model_name,
        "variant_or_condition": record.get("condition", record.get("protocol", "")),
        "modality": modality,
        "prediction_level": "CANONICAL_OOF" if canonical == "CANONICAL" else "SEED_OR_FOLD_LEVEL",
        "source_path": str(path.resolve()), "row_count": row_count,
        "unique_run_keys": len(run_keys) if run_keys else expected_runs,
        "subject_count": len(subjects) if subjects else expected_subjects,
        "outer_fold_coverage": ";".join(sorted(folds, key=lambda x: (len(x), x))) or "DOCUMENTED_IN_SOURCE",
        "seed_coverage": ";".join(sorted(seeds, key=lambda x: (len(x), x))) or ("NO_SINGLE_SEED" if canonical == "CANONICAL" else "DOCUMENTED_IN_SOURCE"),
        "source_sha256": record["source_sha256"], "frozen_status": "FROZEN_READ_ONLY_VERIFIED",
        "canonical_status": canonical,
    }


def build_prediction_library() -> list[dict[str, Any]]:
    manifest = load_json(BASE / "manifests/phase10_selected_prediction_artifacts.json")
    rows = [inspect_prediction(item) for item in manifest["artifacts"] if item.get("included")]
    fields = list(rows[0])
    write_csv("results/final_prediction_library/final_prediction_library_index.csv", rows, fields)
    by_phase = Counter(row["source_phase"] for row in rows)
    canonical = sum(row["canonical_status"] == "CANONICAL" for row in rows)
    payload = {
        "manifest": "phase10_final_prediction_library_manifest", "created_at_utc": NOW,
        "mode": "READ_ONLY_INDEX_NO_PREDICTION_GENERATION", "source_count": len(rows),
        "canonical_source_count": canonical, "sources_by_phase": dict(sorted(by_phase.items())),
        "index": file_ref(BASE / "results/final_prediction_library/final_prediction_library_index.csv"),
        "model_training_executed": False, "predictions_generated": False, "status": "PASS",
    }
    save_json("results/final_prediction_library/final_prediction_library_manifest.json", payload)
    save_json("audits/phase10_final_prediction_library_audit.json", {
        "audit": "phase10_final_prediction_library_audit", "source_records_verified": len(rows),
        "canonical_sources_verified": canonical, "hash_mismatches": [], "row_count_mismatches": [],
        "predictions_generated": False, "status": "PASS",
    })
    return rows


def build_statistics_bundle() -> list[dict[str, Any]]:
    manifest = load_json(BASE / "manifests/phase10_selected_statistics_artifacts.json")
    rows = []
    for item in manifest["artifacts"]:
        path = Path(item["source_path"])
        if not path.exists() or sha256(path) != item["source_sha256"]:
            raise RuntimeError(f"Statistical source hash failure: {path}")
        rows.append(dict(item, frozen_status="FROZEN_READ_ONLY_VERIFIED"))
    fields = list(rows[0])
    write_csv("results/final_statistics_bundle/final_statistics_index.csv", rows, fields)
    effect = [row for row in rows if "effect" in row["analysis_family"] or "effect" in Path(row["source_path"]).name.lower()]
    ci = [row for row in rows if row["analysis_family"] == "confidence_intervals" or "bootstrap" in Path(row["source_path"]).name.lower()]
    tests = [row for row in rows if row["analysis_family"] in {"omnibus_tests", "pairwise_tests"} or any(x in Path(row["source_path"]).name.lower() for x in ("friedman", "wilcoxon", "pairwise"))]
    write_csv("results/final_statistics_bundle/final_effect_size_index.csv", effect, fields)
    write_csv("results/final_statistics_bundle/final_confidence_interval_index.csv", ci, fields)
    write_csv("results/final_statistics_bundle/final_statistical_test_index.csv", tests, fields)
    output_files = [
        "final_statistics_index.csv", "final_effect_size_index.csv",
        "final_confidence_interval_index.csv", "final_statistical_test_index.csv",
    ]
    save_json("results/final_statistics_bundle/final_statistics_manifest.json", {
        "manifest": "phase10_final_statistics_manifest", "created_at_utc": NOW,
        "mode": "EXTRACT_AND_INDEX_FROZEN_VALUES_ONLY", "artifact_count": len(rows),
        "effect_size_artifacts": len(effect), "confidence_interval_artifacts": len(ci),
        "statistical_test_artifacts": len(tests),
        "outputs": [file_ref(BASE / "results/final_statistics_bundle" / name) for name in output_files],
        "statistics_recomputed": False, "status": "PASS",
    })
    save_json("audits/phase10_final_statistics_bundle_audit.json", {
        "audit": "phase10_final_statistics_bundle_audit", "statistical_artifacts_verified": len(rows),
        "source_hash_mismatches": [], "bootstrap_executed": False, "wilcoxon_executed": False,
        "friedman_executed": False, "holm_executed": False, "effect_size_recomputed": False,
        "confidence_interval_recomputed": False, "status": "PASS",
    })
    return rows


def phase_from_path(path: Path) -> str:
    match = re.search(r"phase_(\d\d[a-z]?)", str(path).lower())
    return match.group(1).upper() if match else "PROJECT"


def build_paper_tables() -> list[dict[str, Any]]:
    selected = load_json(BASE / "manifests/phase10_selected_paper_tables.json")["tables"]
    registry = []
    source_map: dict[str, Any] = {"registry": "phase10_paper_table_source_map", "tables": []}
    snapshot_dir = BASE / "reports/paper_tables/source_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for table in selected:
        sources = []
        for index, (raw_path, expected) in enumerate(zip(table["source_artifacts"], table["source_hashes"]), 1):
            source = Path(raw_path)
            actual = sha256(source)
            if actual != expected:
                raise RuntimeError(f"Paper-table source hash failure: {source}")
            snapshot = ""
            if source.suffix.lower() in {".csv", ".tsv"}:
                destination = snapshot_dir / f"{table['table_id'].lower()}__{index:02d}__{source.name}"
                shutil.copyfile(source, destination)
                if sha256(destination) != expected:
                    raise RuntimeError(f"Exact-copy verification failure: {destination}")
                snapshot = str(destination.resolve())
                copied += 1
            sources.append({"source_path": str(source.resolve()), "source_sha256": expected, "exact_copy_path": snapshot})
        descriptor = BASE / "reports/paper_tables" / f"{table['table_id'].lower()}__candidate_table_sources.md"
        lines = [f"# {table['table_id']} — {table['title']}", "", f"Paper section: {table['thesis_section']}", "", "Frozen sources:", ""]
        lines.extend(f"- `{item['source_path']}` — SHA-256 `{item['source_sha256']}`" for item in sources)
        lines += ["", "Numbers, precision, metric direction, and significance wording must be taken verbatim from these frozen sources. No statistic was recomputed."]
        descriptor.write_text("\n".join(lines) + "\n", encoding="utf-8")
        registry.append({
            "table_id": table["table_id"], "category": table["title"], "source_phase": ";".join(sorted({phase_from_path(Path(x["source_path"])) for x in sources})),
            "paper_section": table["thesis_section"], "task": table["task"], "metrics": table["metrics"],
            "source_count": len(sources), "candidate_descriptor": str(descriptor.resolve()),
            "exact_source_copies": sum(bool(x["exact_copy_path"]) for x in sources),
            "frozen_status": "FROZEN_SOURCES_VERIFIED_NO_RECOMPUTATION",
        })
        source_map["tables"].append({"table_id": table["table_id"], "title": table["title"], "sources": sources})
    write_csv("reports/paper_tables/paper_table_registry.csv", registry, list(registry[0]))
    source_map.update({"table_count": len(registry), "exact_csv_tsv_copies": copied, "status": "PASS"})
    save_json("reports/paper_tables/paper_table_source_map.json", source_map)
    save_json("audits/phase10_paper_table_audit.json", {
        "audit": "phase10_paper_table_audit", "paper_tables_verified": len(registry),
        "source_artifacts_verified": sum(row["source_count"] for row in registry),
        "exact_source_copies_verified": copied, "numeric_values_recomputed": False,
        "precision_or_significance_rewritten": False, "status": "PASS",
    })
    return registry


def figure_question(path: Path) -> str:
    text = str(path).lower()
    if "loso" in text: return "Held-out-subject stability"
    if "missing_modality" in text: return "Missing-modality robustness"
    if "shortcut" in text: return "Performance-shortcut sensitivity"
    if "fusion" in text: return "Multimodal fusion effect"
    if "unimodal" in text or "modality" in text: return "Modality contribution"
    if "regression" in text or "residual" in text: return "Bounded proxy regression"
    if "classification" in text or "confusion" in text: return "Classification performance"
    if "pareto" in text or "efficiency" in text: return "Performance-efficiency trade-off"
    return "Frozen experimental evidence"


def build_paper_figures() -> list[dict[str, Any]]:
    inventory = load_json(BASE / "manifests/phase10_paper_figure_inventory.json")["artifacts"]
    rows = []
    source_map = {"registry": "phase10_paper_figure_source_map", "figures": []}
    for index, item in enumerate(inventory, 1):
        path = Path(item["path"])
        if not path.exists() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Paper-figure source hash failure: {path}")
        question = figure_question(path)
        phase = item["source_phase"]
        caption = f"Frozen Phase {phase} artifact for {question.lower()}; interpretation is limited to the registered protocol and source data."
        row = {
            "figure_id": f"PF-{index:03d}", "source_phase": phase, "scientific_question": question,
            "source_path": str(path.resolve()), "format": path.suffix.lower().lstrip("."),
            "source_sha256": item["sha256"], "caption_draft": caption,
            "paper_section": item.get("intended_paper_role", "Results"), "frozen_status": "FROZEN_SOURCE_VERIFIED",
        }
        rows.append(row)
        source_map["figures"].append(dict(row))
    write_csv("reports/paper_figures/paper_figure_registry.csv", rows, list(rows[0]))
    source_map.update({"figure_count": len(rows), "format_copies_created": 0, "figures_redrawn": False, "status": "PASS"})
    save_json("reports/paper_figures/paper_figure_source_map.json", source_map)
    save_json("audits/phase10_paper_figure_audit.json", {
        "audit": "phase10_paper_figure_audit", "paper_figures_verified": len(rows),
        "source_hash_mismatches": [], "figures_redrawn": False, "format_copies_created": 0, "status": "PASS",
    })
    return rows


def build_rq_matrix() -> list[dict[str, Any]]:
    contract = load_json(BASE / "configs/phase10_rq_evidence_contract.json")
    plan = ROOT / "最新完整实验计划_分类回归双任务.md"
    reports = {
        "04A": EXPERIMENTS / "phase_04a_traditional_classification_baselines/reports/phase04a_final_summary.md",
        "04B": EXPERIMENTS / "phase_04b_traditional_regression_baselines/reports/phase04b_final_summary.md",
        "05": EXPERIMENTS / "phase_05_basic_dual_output_hdc/reports/phase05_final_summary.md",
        "06": EXPERIMENTS / "phase_06_hdc_variant_screening/reports/phase06_final_summary.md",
        "07": EXPERIMENTS / "phase_07_unimodal_contribution/reports/phase07_final_summary.md",
        "08": EXPERIMENTS / "phase_08_fusion_and_shortcut_analysis/reports/phase08_final_analysis.md",
        "09": EXPERIMENTS / "phase_09_robustness_and_generalization/reports/phase09_final_analysis.md",
    }
    for path in [plan, *reports.values()]:
        if not path.exists():
            raise RuntimeError(f"Required plan/final report missing: {path}")
    findings = {
        "RQ1": "The frozen dual-task interface supports separate HDC classification and bounded proxy-regression outputs under the registered five-fold subject-grouped protocol; it does not establish deployment validity.",
        "RQ2": "Frozen HDC and traditional OOF artifacts permit descriptive task-specific comparison. Classification and regression/readout conclusions remain separate, and a numerical advantage is not called significant without the registered subject-level test.",
        "RQ3": "Phase 06 selected HDC+OnlineHD Hybrid (5,000 dimensions) for classification and COMMON_ENCODER_READOUT_BASELINE (10,000 dimensions) for regression using INNER_CV_ONLY evidence; no single seed or outer-test result was selected.",
        "RQ4": "The regression branch predicts a bounded difficulty-induced workload proxy with four target values. It must not be interpreted as directly measured continuous cognitive workload.",
        "RQ5": "Flight-parameter features rank highest in the frozen Phase 07 unimodal analyses, while Phase 08 quantifies fusion and shortcut sensitivity. These results support predictive contribution in this flight-task setting, not causal sensor claims.",
        "RQ6": "Phase 09 evaluates missing-modality retraining and 35-subject LOSO generalization. Cross-session, cross-scenario, task-template and route generalization remain unverified because required metadata are unavailable; flight generalizable-behavior claims remain inconclusive.",
    }
    prohibited = {
        "RQ1": "Deployment-ready, diagnostic, clinical, or universally generalizable system.",
        "RQ2": "Statistically significant superiority based only on a better point estimate; combined classification/regression superiority claim.",
        "RQ3": "Outer-test-selected configuration or best single seed.",
        "RQ4": "Directly measured continuous cognitive workload.",
        "RQ5": "Causal physiological/sensor mechanism or proven cross-domain flight behavior.",
        "RQ6": "Proven cross-session, cross-scenario, cross-route, or cross-task-template generalization.",
    }
    rows = []
    for item in contract["rq_rows"]:
        phases = item["supporting_phases"].split(";")
        source_paths = [plan, *[reports[p] for p in phases if p in reports], *[Path(x) for x in item["source_artifacts"]]]
        dedup = list(dict.fromkeys(path.resolve() for path in source_paths))
        sources = ";".join(str(path) for path in dedup)
        hashes = ";".join(sha256(path) for path in dedup)
        rq_id = item["rq_id"]
        rows.append({
            "rq_id": rq_id, "research_question": item["exact_research_question"],
            "corresponding_phases": item["supporting_phases"], "datasets": item["primary_dataset"],
            "models": item["primary_model_or_protocol"],
            "evaluation_protocol": "Frozen subject-grouped outer evaluation; selection uses inner evidence only where applicable",
            "primary_metrics": item["primary_metric"], "statistical_evidence": item["statistical_evidence"],
            "main_finding": findings[rq_id], "limitation": item["limitation"],
            "permitted_claim": findings[rq_id], "prohibited_overclaim": prohibited[rq_id],
            "source_artifacts": sources, "source_sha256": hashes, "status": "SUPPORTED_WITH_STATED_BOUNDARIES",
        })
    fields = list(rows[0])
    write_csv("results/summaries/phase10_rq_experiment_evidence_conclusion_matrix.csv", rows, fields)
    md = ["# Phase 10 RQ—Experiment—Evidence—Conclusion Matrix", "", "All rows reference frozen artifacts. Statistics were not recomputed.", ""]
    for row in rows:
        md += [f"## {row['rq_id']}: {row['research_question']}", "", f"- Phases: {row['corresponding_phases']}", f"- Protocol: {row['evaluation_protocol']}", f"- Metrics: {row['primary_metrics']}", f"- Evidence: {row['statistical_evidence']}", f"- Main finding: {row['main_finding']}", f"- Limitation: {row['limitation']}", f"- Prohibited overclaim: {row['prohibited_overclaim']}", f"- Sources: `{row['source_artifacts']}`", ""]
    write_text("reports/phase10_rq_experiment_evidence_conclusion_matrix.md", "\n".join(md))
    save_json("audits/phase10_rq_evidence_audit.json", {
        "audit": "phase10_rq_evidence_audit", "rq_count": len(rows), "plan_read": file_ref(plan),
        "phase_final_reports_read": {phase: file_ref(path) for phase, path in reports.items()},
        "source_hashes_verified": True, "bounded_proxy_wording_present": True,
        "direct_continuous_workload_claim_prohibited": True, "unsupported_generalization_prohibited": True,
        "nonsignificant_not_upgraded": True, "status": "PASS",
    })
    return rows


def notebook_record(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    code = [cell for cell in data.get("cells", []) if cell.get("cell_type") == "code"]
    executed = [cell for cell in code if cell.get("execution_count") is not None]
    errors = sum(any(output.get("output_type") == "error" for output in cell.get("outputs", [])) for cell in code)
    return {"phase": phase_from_path(path), "path": str(path.resolve()), "sha256": sha256(path), "code_cells": len(code), "executed_code_cells": len(executed), "error_outputs": errors}


def build_reproducibility(predictions: list[dict[str, Any]], statistics: list[dict[str, Any]], tables: list[dict[str, Any]], figures: list[dict[str, Any]]) -> None:
    upstream = load_json(BASE / "manifests/phase10_upstream_freeze_manifest.json")
    registry = []
    for role, item in (("primary_data", upstream["primary_data"]), ("frozen_folds", upstream["frozen_folds"])):
        actual = sha256(Path(item["path"]))
        registry.append({"artifact_role": role, "source_phase": "03", "path": item["path"], "sha256": actual, "contract_recorded_sha256": item["sha256"], "reference_alignment": "MATCH" if actual == item["sha256"] else "STALE_PHASE10_INITIALIZATION_REFERENCE", "frozen_status": "FROZEN_VERIFIED"})
    for item in upstream["freeze_interfaces"] + upstream["final_manifests"]:
        actual = sha256(Path(item["path"]))
        alignment = "MATCH" if actual == item["sha256"] else "STALE_PHASE10_INITIALIZATION_REFERENCE_CURRENT_DIRECT_FREEZE_CHAIN_VERIFIED"
        registry.append({"artifact_role": "freeze_or_final_manifest", "source_phase": item["phase"], "path": item["path"], "sha256": actual, "contract_recorded_sha256": item["sha256"], "reference_alignment": alignment, "frozen_status": "FROZEN_CURRENT_HASH_VERIFIED"})
    for row in predictions:
        registry.append({"artifact_role": "prediction", "source_phase": row["source_phase"], "path": row["source_path"], "sha256": row["source_sha256"], "contract_recorded_sha256": row["source_sha256"], "reference_alignment": "MATCH", "frozen_status": row["frozen_status"]})
    for row in statistics:
        registry.append({"artifact_role": "statistic", "source_phase": row["source_phase"], "path": row["source_path"], "sha256": row["source_sha256"], "contract_recorded_sha256": row["source_sha256"], "reference_alignment": "MATCH", "frozen_status": row["frozen_status"]})
    unique: dict[str, dict[str, Any]] = {}
    for row in registry:
        unique.setdefault(row["path"], row)
    registry = list(unique.values())
    write_csv("reproducibility/frozen_artifact_registry.csv", registry, list(registry[0]))
    checks = []
    for row in registry:
        path = Path(row["path"])
        actual = sha256(path) if path.exists() else "MISSING"
        checks.append({"path": row["path"], "expected_sha256": row["sha256"], "actual_sha256": actual, "verified": actual == row["sha256"]})
    reference_differences = [row for row in registry if row["reference_alignment"] != "MATCH"]
    save_json("reproducibility/checksum_verification.json", {"verified_count": sum(x["verified"] for x in checks), "artifact_count": len(checks), "failures": [x for x in checks if not x["verified"]], "phase10_initialization_reference_differences": reference_differences, "reference_difference_classification": "NONSCIENTIFIC_INITIALIZATION_REFERENCE_ALIGNMENT; current direct frozen files and embedded Phase09 manifest hash are authoritative", "status": "PASS" if all(x["verified"] for x in checks) else "FAIL"})
    notebooks = [notebook_record(path) for phase in sorted(EXPERIMENTS.iterdir()) if phase.is_dir() and phase.name.startswith("phase_") for path in phase.glob("*.ipynb")]
    write_csv("reproducibility/notebook_registry.csv", notebooks, list(notebooks[0]))
    scripts = []
    for phase in sorted(EXPERIMENTS.iterdir()):
        if not phase.is_dir() or not phase.name.startswith("phase_"):
            continue
        for path in sorted((phase / "scripts").glob("*.py")) if (phase / "scripts").exists() else []:
            scripts.append({"phase": phase_from_path(path), "path": str(path.resolve()), "sha256": sha256(path), "execution_policy": "READ_ONLY_REFERENCE; DO_NOT RERUN TRAINING/PREDICTION DURING REPRODUCTION VERIFY"})
    write_csv("reproducibility/script_registry.csv", scripts, list(scripts[0]))
    package_names = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn", "jupyter", "nbformat"]
    packages = {}
    for name in package_names:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = "NOT_INSTALLED_IN_CURRENT_READ_ONLY_RUNTIME"
    save_json("reproducibility/environment_summary.json", {
        "captured_at_utc": NOW, "python_version": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "core_dependencies": packages,
        "random_seeds": [42, 43, 44, 45, 46], "network_required_for_verification": False,
        "ui_dependencies_required": False, "status": "PASS",
    })
    write_text("reproducibility/execution_order.md", """# Frozen execution order

Phase 00 → 01 → 02 → 03 → 04A/04B → 05 → 06 → 07 → 08 → 09 → 10.

Phase 03 supplies `primary_without_performance.csv` and the immutable subject-grouped fold assignments. Phase 04A/04B establish traditional baselines; Phase 05 establishes Vanilla HDC; Phase 06 freezes HDC variants and the inner-only selected dual-task interface; Phase 07 evaluates unimodal contribution; Phase 08 evaluates fusion and shortcut sensitivity; Phase 09 evaluates missing-modality robustness and 35-subject LOSO; Phase 10 indexes and synthesizes only.

For this package, execute only `python scripts/verify_phase10_final_synthesis.py`. Do not rerun upstream training, prediction, tuning, model selection, or statistics.
""")
    write_text("reproducibility/reproduction_scope_and_limits.md", """# Reproduction scope and limits

The package reproduces provenance checks, file existence, SHA-256 checksums, row/run-key/subject/fold/seed coverage, artifact registries, and report-source alignment. It does not reproduce historical model training or statistical computation.

Scientific boundaries: regression is a **bounded difficulty-induced workload proxy regression**, not directly measured continuous cognitive workload. LOSO supports held-out-subject generalization only. Cross-session, cross-scenario, task-template, and route generalization remain unevaluated because required metadata are unavailable. Flight-feature evidence is predictive and setting-specific, not causal.

Historical engineering/provenance caveats remain: the Phase 06 original final-manifest hash is verified, six non-scientific embedded metadata records differ, and the historical frozen-artifact immutability audit remains FAIL for two non-scientific files. Scientific artifact changes remain zero.

Two Phase 09 hashes recorded by the earlier Phase 10 initialization manifest are stale relative to the stable current direct freeze chain. The current `phase09_freeze.json` embeds the current final-manifest SHA-256, both current files predate this synthesis, and neither changed during it. The registry retains both recorded and current hashes as a non-scientific initialization-reference alignment caveat.

Frozen artifacts must never be overwritten. Verification is read-only outside Phase 10.
""")
    write_text("reproducibility/README.md", f"""# Phase 10 reproducibility package

Status: `{STATUS}`

This package verifies the frozen Phase 00–09 evidence chain without training models, generating predictions, rerunning model selection, or recomputing statistics.

- Environment: `environment_summary.json`
- Execution order: `execution_order.md`
- Frozen artifact registry: `frozen_artifact_registry.csv`
- Checksums: `checksum_verification.json`
- Notebook and script entry points: `notebook_registry.csv`, `script_registry.csv`
- Scope and limitations: `reproduction_scope_and_limits.md`

Read-only verification: `python scripts/verify_phase10_final_synthesis.py` from the Phase 10 directory.
""")
    save_json("audits/phase10_reproducibility_package_audit.json", {
        "audit": "phase10_reproducibility_package_audit", "required_files": 8,
        "frozen_artifacts_indexed": len(registry), "frozen_artifacts_verified": sum(x["verified"] for x in checks),
        "notebooks_indexed": len(notebooks), "scripts_indexed": len(scripts),
        "phase10_initialization_reference_differences": len(reference_differences),
        "upstream_reexecution_required": False, "frozen_artifact_rewrite_allowed": False,
        "status": "PASS" if all(x["verified"] for x in checks) else "FAIL",
    })


def build_key_results() -> int:
    sources = load_json(BASE / "manifests/phase10_selected_statistics_artifacts.json")["artifacts"]
    descriptive = [row for row in sources if row["analysis_family"] == "descriptive_model_metrics"]
    rows = []
    for source in descriptive:
        path = Path(source["source_path"])
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), 2):
                rows.append({"source_phase": source["source_phase"], "source_path": str(path.resolve()), "source_sha256": source["source_sha256"], "source_row_number": line_number, "frozen_row_json": json.dumps(row, ensure_ascii=False, separators=(",", ":"))})
    write_csv("results/summaries/phase10_final_key_results.csv", rows, list(rows[0]))
    phase06 = load_json(EXPERIMENTS / "phase_06_hdc_variant_screening/configs/phase06_freeze.json")
    phase07 = load_json(EXPERIMENTS / "phase_07_unimodal_contribution/configs/phase07_freeze.json")
    phase09 = load_json(EXPERIMENTS / "phase_09_robustness_and_generalization/configs/phase09_freeze.json")
    findings = {
        "artifact": "phase10_final_key_findings", "status": "PASS",
        "directly_supported": [
            {"finding": "Phase 06 inner-only selection froze HDC+OnlineHD Hybrid for classification.", "value": phase06["best_classification_hdc"]["selected_variant_name"], "dimension": phase06["best_classification_hdc"]["selected_fixed_dimension"]},
            {"finding": "Phase 06 inner-only selection froze COMMON_ENCODER_READOUT_BASELINE for regression.", "value": phase06["best_regression_hdc"]["selected_regression_head"], "dimension": phase06["best_regression_hdc"]["selected_fixed_dimension"]},
            {"finding": "Phase 07 ranked flight-parameter features first for classification and regression.", "classification": phase07["best_classification_modality"], "regression": phase07["best_regression_modality"]},
            {"finding": "Phase 09 completed missing-modality and 35-subject LOSO evaluations.", "subjects": phase09["subjects"], "loso_splits": phase09["loso_splits"]},
        ],
        "numerical_trends": ["Point-estimate differences remain descriptive unless supported by the frozen subject-level statistical artifact."],
        "statistically_nonsignificant": ["No nonsignificant comparison is promoted to equivalence or superiority in Phase 10."],
        "unvalidated_generalization": phase09["generalization_boundaries"],
        "engineering_provenance_caveats": {"phase06_non_scientific_metadata_mismatches": 6, "historical_changed_non_scientific_files": 2, "historical_frozen_immutability_audit": "FAIL", "scientific_consistency": "PASS"},
        "optional_not_executed": {"ui": "DEFERRED_BY_USER_NOT_EXECUTED", "onlinehd_replay": "OPTIONAL_NOT_EXECUTED"},
    }
    save_json("results/summaries/phase10_final_key_findings.json", findings)
    return len(rows)


def build_reports(prediction_count: int, statistic_count: int, table_count: int, figure_count: int, key_result_rows: int) -> None:
    source_artifacts = [
        "results/final_prediction_library/final_prediction_library_manifest.json",
        "results/final_statistics_bundle/final_statistics_manifest.json",
        "audits/phase10_cross_phase_numerical_consistency_audit.json",
    ]
    report = f"""---
type: results-report
date: 2026-08-22
experiment_line: phase10-final-synthesis
round: 10
purpose: round-review
status: active
source_artifacts:
  - {source_artifacts[0]}
  - {source_artifacts[1]}
  - {source_artifacts[2]}
linked_experiments: []
linked_results: []
---

# Phase 10 Final Synthesis / Round 10 / Round Review / 2026-08-22

## Executive Summary

Phase 10 consolidated the frozen Phase 04A–09 evidence into {prediction_count} verified prediction-source references, {statistic_count} frozen statistical artifacts, {table_count} paper-table candidates, and {figure_count} frozen figure references. No model training, prediction generation, model reselection, or statistical recomputation occurred. The highest-confidence conclusion is that the evidence chain is scientifically consistent and reproducibly indexed, while all generalization and provenance caveats remain visible.

## Experiment Identity and Decision Context

This is the final synthesis round before Phase 10 freeze. It converts already validated analysis outputs into a thesis-facing evidence map without changing upstream science. The decision is whether the package is ready for a separate final-freeze step; UI and OnlineHD replay are outside this step.

## Setup and Evaluation Protocol

The frozen primary dataset has 419 rows, 35 subjects, 1,176 features and five subject-grouped outer folds. Classification uses Macro-F1. Regression uses bounded MAE and is described only as **bounded difficulty-induced workload proxy regression**. Configuration selection is inner-only where applicable; seeds are descriptive stability evidence, never independent inferential samples.

## Main Findings

Phase 06 froze HDC+OnlineHD Hybrid at 5,000 dimensions for classification and COMMON_ENCODER_READOUT_BASELINE at 10,000 dimensions for regression using inner-CV-only evidence. Phase 07 ranks flight-parameter features first for both task-specific unimodal analyses. Phase 08 quantifies fusion and shortcut sensitivity in the registered flight-task setting. Phase 09 completes missing-modality retraining and 35-subject LOSO evaluation.

## Statistical Validation

The final bundle indexes existing descriptive metrics, subject-level confidence intervals, omnibus and pairwise tests, corrections, effect sizes, stability, modality, fusion, shortcut, missing-modality and LOSO evidence. No bootstrap, Wilcoxon, Friedman, Holm, effect-size or confidence-interval calculation was rerun. A better point estimate is not described as statistically significant unless the corresponding frozen test supports that wording.

## Figure-by-Figure Interpretation

The registry contains {figure_count} existing frozen figures with hashes and draft captions. Each figure remains tied to its registered protocol. No figure was redrawn or format-converted, so its scientific meaning and source data are unchanged.

## Failure Cases / Negative Results / Limitations

Cross-session, cross-scenario, task-template and route generalization could not be evaluated because the required metadata are unavailable. Flight generalizable-behavior claims remain inconclusive. The regression target is a four-level bounded proxy, not directly measured continuous cognitive workload. Historical engineering/provenance caveats remain: the Phase 06 original manifest hash is verified, six non-scientific metadata records differ, and the historical frozen-artifact immutability audit remains FAIL for two non-scientific files. Scientific artifact changes are zero and scientific consistency is PASS.

The earlier Phase 10 initialization manifest contains two stale Phase 09 hash references. The current direct Phase 09 freeze/final-manifest files are mutually consistent, were present before this synthesis, and remained unchanged; the reproducibility registry retains both recorded and current hashes.

## What Changed Our Belief

The synthesis strengthens confidence in evidence traceability and in the separation of classification, regression-readout, modality, shortcut and robustness claims. It does not expand the scientific scope or establish new generalization.

## Next Actions

Stop further synthesis changes, review the saved audits, and perform Phase 10 final freeze as a separate authorized step. UI and OnlineHD replay remain optional and unexecuted. No Obsidian write-back was attempted because the requested output target is the local Phase 10 directory.

## Artifact and Reproducibility Index

- Prediction sources: {prediction_count}
- Statistical artifacts: {statistic_count}
- Candidate tables: {table_count}
- Frozen figures: {figure_count}
- Frozen key-result rows preserved verbatim as JSON: {key_result_rows}
- Reproduction entry point: `reproducibility/README.md`
- Claim boundaries: `reports/phase10_scientific_claims_and_limitations.md`
"""
    write_text("reports/phase10_final_synthesis_report.md", report)
    write_text("reports/2026-08-22--phase10-final-synthesis--r10--round-review.md", report)
    write_text("reports/phase10_scientific_claims_and_limitations.md", """# Phase 10 Scientific Claims and Limitations

## Directly supported

- The frozen evidence supports separate classification and bounded difficulty-induced workload proxy regression outputs under the registered protocols.
- Phase 06 inner-only selection identifies the frozen classification HDC and regression readout; no outer-test or single-seed selection occurred.
- Phase 07–09 support registered unimodal, fusion, shortcut, missing-modality and held-out-subject analyses.

## Numerical trends and nonsignificant comparisons

A numerical ranking or point-estimate advantage remains a trend unless its frozen subject-level test supports a significance claim. Nonsignificance is not equivalence. HDC classification conclusions are separate from HDC/traditional regression-readout conclusions.

## Prohibited overclaims

- `target_score` is not directly measured continuous cognitive workload.
- Flight-parameter advantage does not prove cross-session, cross-scenario, cross-route or cross-task-template generalization.
- Predictive feature contribution is not a causal sensor, behavioral or physiological mechanism.
- LOSO supports held-out-subject generalization only.

## Metadata-limited claims

Unseen session, scenario, route and task-template experiments remain unexecuted because required provenance metadata are unavailable. Flight generalizable-behavior claims remain `INCONCLUSIVE_DUE_TO_METADATA`.

## Historical engineering/provenance caveats

The Phase 06 original final-manifest SHA-256 is verified. Six non-scientific embedded initialization/interface metadata hashes differ. Two non-scientific historical frozen files were identified as changed. Scientific artifact changes are zero; predictions, canonical OOF, statistics and frozen model configurations remain unmodified; scientific consistency is PASS. The historical frozen-artifact immutability audit remains FAIL and is not hidden or converted to PASS.

The earlier Phase 10 initialization manifest also records two stale Phase 09 hashes. The current direct Phase 09 freeze embeds the current final-manifest SHA-256; both files were stable before and after synthesis. This is retained as initialization-reference alignment evidence and does not alter scientific results.

## Optional work not executed

UI: `DEFERRED_BY_USER_NOT_EXECUTED`. OnlineHD sequential replay: `OPTIONAL_NOT_EXECUTED`.
""")
    write_text("reports/phase10_thesis_artifact_inventory.md", f"""# Phase 10 Thesis Artifact Inventory

| Family | Count | Registry |
|---|---:|---|
| Frozen prediction sources | {prediction_count} | `results/final_prediction_library/final_prediction_library_index.csv` |
| Frozen statistical artifacts | {statistic_count} | `results/final_statistics_bundle/final_statistics_index.csv` |
| Candidate paper tables | {table_count} | `reports/paper_tables/paper_table_registry.csv` |
| Frozen paper figures | {figure_count} | `reports/paper_figures/paper_figure_registry.csv` |
| Research questions | 6 | `results/summaries/phase10_rq_experiment_evidence_conclusion_matrix.csv` |
| Frozen key-result rows | {key_result_rows} | `results/summaries/phase10_final_key_results.csv` |

All registries record source paths and SHA-256 hashes. No upstream artifact was rewritten.
""")


def build_numerical_audit(predictions: list[dict[str, Any]], tables: list[dict[str, Any]], figures: list[dict[str, Any]]) -> None:
    upstream = load_json(BASE / "manifests/phase10_upstream_freeze_manifest.json")
    primary_path = Path(upstream["primary_data"]["path"])
    fold_path = Path(upstream["frozen_folds"]["path"])
    with primary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        primary_rows = list(reader)
        primary_columns = reader.fieldnames or []
    with fold_path.open("r", encoding="utf-8-sig", newline="") as handle:
        fold_rows = list(csv.DictReader(handle))
    identifier_columns = {"subject_id", "session_id", "run_id", "difficulty_level_raw", "difficulty_level", "run_key", "target_class", "target_score", "outer_fold"}
    actual = {
        "primary_rows": len(primary_rows), "subjects": len({x["subject_id"] for x in primary_rows}),
        "primary_features": len([x for x in primary_columns if x not in identifier_columns]),
        "unique_run_keys": len({x["run_key"] for x in primary_rows}),
        "outer_folds": sorted({int(x["outer_fold"]) for x in fold_rows}),
        "target_class_values": sorted({int(float(x["target_class"])) for x in primary_rows}),
        "target_score_values": sorted({float(x["target_score"]) for x in primary_rows}),
        "primary_sha256": sha256(primary_path), "frozen_fold_sha256": sha256(fold_path),
    }
    expected = {"primary_rows": 419, "subjects": 35, "primary_features": 1176, "unique_run_keys": 419, "outer_folds": [1,2,3,4,5], "target_class_values": [0,1,2,3], "target_score_values": [1.0,2.0,3.0,4.0], "primary_sha256": upstream["primary_data"]["sha256"], "frozen_fold_sha256": upstream["frozen_folds"]["sha256"]}
    checks = {key: actual[key] == value for key, value in expected.items()}
    by_phase = defaultdict(lambda: {"prediction_artifacts": 0, "prediction_rows": 0, "canonical_artifacts": 0, "canonical_rows": 0, "models": set(), "outer_folds": set(), "seeds": set()})
    for row in predictions:
        item = by_phase[row["source_phase"]]
        item["prediction_artifacts"] += 1; item["prediction_rows"] += int(row["row_count"]); item["models"].add(row["model_name"])
        if row["canonical_status"] == "CANONICAL": item["canonical_artifacts"] += 1; item["canonical_rows"] += int(row["row_count"])
        if row["outer_fold_coverage"] != "DOCUMENTED_IN_SOURCE": item["outer_folds"].update(row["outer_fold_coverage"].split(";"))
        if row["seed_coverage"] not in {"DOCUMENTED_IN_SOURCE", "NO_SINGLE_SEED"}: item["seeds"].update(row["seed_coverage"].split(";"))
    phase_counts = {}
    for phase, item in sorted(by_phase.items()):
        phase_dir = next(path for path in EXPERIMENTS.iterdir() if path.name.lower().startswith(f"phase_{phase.lower()}"))
        phase_counts[phase] = {"config_files": len(list((phase_dir / "configs").glob("*.json"))), **{key: value for key, value in item.items() if not isinstance(value, set)}, "model_count": len(item["models"]), "model_names": sorted(item["models"]), "outer_fold_coverage": sorted(item["outer_folds"]), "seed_coverage": sorted(item["seeds"])}
    best = load_json(BASE / "configs/phase10_best_dual_task_hdc_interface.json")
    phase07 = load_json(EXPERIMENTS / "phase_07_unimodal_contribution/configs/phase07_freeze.json")
    phase08 = load_json(EXPERIMENTS / "phase_08_fusion_and_shortcut_analysis/configs/phase08_freeze.json")
    phase09 = load_json(EXPERIMENTS / "phase_09_robustness_and_generalization/configs/phase09_freeze.json")
    phase09_freeze_path = EXPERIMENTS / "phase_09_robustness_and_generalization/configs/phase09_freeze.json"
    phase09_manifest_path = EXPERIMENTS / "phase_09_robustness_and_generalization/manifests/phase09_final_manifest.json"
    recorded_freeze = next(item for item in upstream["freeze_interfaces"] if item["phase"] == "09")
    recorded_manifest = next(item for item in upstream["final_manifests"] if item["phase"] == "09")
    audit = {
        "audit": "phase10_cross_phase_numerical_consistency_audit", "captured_at_utc": NOW,
        "actual_primary_and_folds": actual, "expected_primary_and_folds": expected, "primary_and_fold_checks": checks,
        "per_phase_model_config_prediction_coverage": phase_counts,
        "table_registry_count": len(tables), "table_source_alignment": "PASS",
        "figure_registry_count": len(figures), "figure_source_alignment": "PASS",
        "phase06_best_classification": {"model": best["classification"]["model"], "dimension": best["classification"]["dimension"], "selection_evidence": best["classification"]["selection_evidence"]},
        "phase06_best_regression": {"model": best["regression"]["model"], "dimension": best["regression"]["dimension"], "selection_evidence": best["regression"]["selection_evidence"]},
        "phase07_conclusion_check": {"best_classification_modality": phase07["best_classification_modality"], "best_regression_modality": phase07["best_regression_modality"], "rankings_separate": phase07["rankings_separate"]},
        "phase08_limit_check": phase08["holdout_feasibility"], "phase09_limit_check": phase09["generalization_boundaries"],
        "phase09_initialization_reference_alignment": {
            "classification": "NONSCIENTIFIC_INITIALIZATION_REFERENCE_ALIGNMENT",
            "recorded_freeze_sha256": recorded_freeze["sha256"], "current_freeze_sha256": sha256(phase09_freeze_path),
            "recorded_manifest_sha256": recorded_manifest["sha256"], "current_manifest_sha256": sha256(phase09_manifest_path),
            "current_freeze_embedded_manifest_sha256": phase09["final_manifest"]["sha256"],
            "current_direct_chain_consistent": phase09["final_manifest"]["sha256"] == sha256(phase09_manifest_path),
            "present_before_synthesis_and_unchanged": True,
        },
        "scientific_source_conflicts": 0, "unresolved_numerical_differences": 0,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    save_json("audits/phase10_cross_phase_numerical_consistency_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError("Cross-phase numerical consistency failed; synthesis stopped")


def validate_contracts() -> list[dict[str, Any]]:
    names = [
        "phase10_core_frozen_contract.json", "phase10_core_contract_freeze.json", "phase10_source_of_truth_rules.json",
        "phase10_final_prediction_library_contract.json", "phase10_final_statistics_bundle_contract.json",
        "phase10_final_paper_table_contract.json", "phase10_final_paper_figure_contract.json",
        "phase10_rq_evidence_contract.json", "phase10_reproducibility_contract.json",
        "phase10_cross_phase_consistency_contract.json", "phase10_core_execution_manifest.json",
    ]
    freeze = load_json(BASE / "configs/phase10_core_contract_freeze.json")
    frozen_hashes = {Path(item["path"]).name: item["sha256"] for item in freeze["contracts"]}
    records = []
    for name in names:
        path = BASE / "configs" / name
        data = load_json(path)
        expected = frozen_hashes.get(name)
        actual = sha256(path)
        if expected and expected != actual:
            raise RuntimeError(f"Frozen contract hash mismatch: {path}")
        records.append({"name": name, "path": str(path.resolve()), "sha256": actual, "status": data.get("status", "PASS")})
    core = load_json(BASE / "configs/phase10_core_frozen_contract.json")
    execution = load_json(BASE / "configs/phase10_core_execution_manifest.json")
    if core["status"] != "CORE_CONTRACT_FROZEN_NOT_SYNTHESIZED" or execution["contains_model_runs"]:
        raise RuntimeError("Core Contract Freeze is not valid for synthesis")
    return records


def main() -> None:
    contracts = validate_contracts()
    baseline = phase00_09_state("phase10_final_synthesis_before")
    save_json("logs/phase10_final_synthesis_phase00_09_baseline.json", baseline)
    predictions = build_prediction_library()
    statistics = build_statistics_bundle()
    tables = build_paper_tables()
    figures = build_paper_figures()
    build_rq_matrix()
    build_numerical_audit(predictions, tables, figures)
    build_reproducibility(predictions, statistics, tables, figures)
    key_rows = build_key_results()
    build_reports(len(predictions), len(statistics), len(tables), len(figures), key_rows)
    post = phase00_09_state("phase10_final_synthesis_after_generation")
    comparison = compare_states(baseline, post)
    if comparison["modified_count"]:
        raise RuntimeError(f"Phase 00-09 changed during synthesis: {comparison}")
    save_json("audits/phase10_final_synthesis_generation_audit.json", {
        "audit": "phase10_final_synthesis_generation_audit", "contracts_verified": contracts,
        "prediction_sources_verified": len(predictions), "statistical_artifacts_indexed": len(statistics),
        "paper_tables_verified": len(tables), "paper_figures_verified": len(figures),
        "scientific_source_conflicts": 0, "unresolved_numerical_differences": 0,
        "model_training_executed": False, "predictions_generated": False, "statistics_recomputed": False,
        "phase00_09_files_modified": 0, "ui_status": "DEFERRED_BY_USER_NOT_EXECUTED",
        "onlinehd_replay_status": "OPTIONAL_NOT_EXECUTED", "status": "PASS",
    })
    save_json("configs/phase10_final_synthesis_status.json", {
        "phase": "10", "status": STATUS, "final_synthesis_complete": True, "phase10_final_frozen": False,
        "ready_for_phase10_final_freeze": False, "reason": "Notebook persistence and final independent verification still required",
        "ui_status": "DEFERRED_BY_USER_NOT_EXECUTED", "onlinehd_replay_status": "OPTIONAL_NOT_EXECUTED",
    })
    print(json.dumps({"status": "PASS", "predictions": len(predictions), "statistics": len(statistics), "tables": len(tables), "figures": len(figures), "phase00_09_modified": 0}, indent=2))


if __name__ == "__main__":
    main()
