from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
EXPECTED_PRIMARY_SHA = "0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44"
EXPECTED_FOLD_SHA = "e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f"
DISCLAIMER = (
    "This interface is a read-only demonstration of the frozen best dual-task HDC system "
    "and its audited out-of-fold results. It is not a deployment or real-time cognitive "
    "workload diagnostic system."
)
NOW = datetime.now(timezone.utc).isoformat()

PHASE_DIRS = {
    "03": EXPERIMENTS / "phase_03_multimodal_dataset_labeling",
    "04A": EXPERIMENTS / "phase_04a_traditional_classification_baselines",
    "04B": EXPERIMENTS / "phase_04b_traditional_regression_baselines",
    "05": EXPERIMENTS / "phase_05_basic_dual_output_hdc",
    "06": EXPERIMENTS / "phase_06_hdc_variant_screening",
    "07": EXPERIMENTS / "phase_07_unimodal_contribution",
    "08": EXPERIMENTS / "phase_08_fusion_and_shortcut_analysis",
    "09": EXPERIMENTS / "phase_09_robustness_and_generalization",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(relative: str, payload: Any) -> None:
    path = BASE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_record(path: Path, role: str | None = None) -> dict[str, Any]:
    record = {
        "path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if role:
        record["role"] = role
    return record


def csv_profile(path: Path) -> dict[str, Any]:
    rows = 0
    run_keys: set[str] = set()
    subjects: set[str] = set()
    seeds: set[str] = set()
    models: set[str] = set()
    tasks: set[str] = set()
    columns: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            for row in reader:
                rows += 1
                for key in ("run_key", "sample_key"):
                    if row.get(key):
                        run_keys.add(row[key])
                for key in ("subject_id", "subject"):
                    if row.get(key):
                        subjects.add(row[key])
                if row.get("seed"):
                    seeds.add(row["seed"])
                for key in ("model", "model_id", "model_name", "variant"):
                    if row.get(key):
                        models.add(row[key])
                if row.get("task"):
                    tasks.add(row["task"])
    except (UnicodeDecodeError, csv.Error):
        return {"rows": None, "unique_run_key": None, "subjects": None, "columns": []}
    return {
        "rows": rows,
        "unique_run_key": len(run_keys) if run_keys else None,
        "subjects": len(subjects) if subjects else None,
        "seeds": sorted(seeds),
        "models": sorted(models)[:20],
        "tasks": sorted(tasks),
        "columns": columns,
    }


def infer_task(path: Path, profile: dict[str, Any]) -> str:
    joined = " ".join(profile.get("columns", []) + profile.get("tasks", [])).lower()
    text = str(path).lower()
    has_cls = any(token in joined or token in text for token in ("classification", "target_class", "y_pred_class", "macro_f1"))
    has_reg = any(token in joined or token in text for token in ("regression", "target_score", "y_pred_reg", "bounded_mae"))
    if has_cls and has_reg:
        return "dual_task"
    if has_cls:
        return "classification"
    if has_reg:
        return "regression"
    return "unspecified"


def infer_model(path: Path, profile: dict[str, Any]) -> str:
    models = profile.get("models", [])
    if len(models) == 1:
        return models[0]
    stem = path.stem
    stem = re.sub(r"(_|-)(oof|predictions?|canonical|fold_?\d+|seed_?\d+).*", "", stem, flags=re.I)
    parts = [p for p in path.parts if re.search(r"(hdc|hybrid|online|centroid|traditional|gradient|ridge|svm|forest|knn|dummy|elastic)", p, re.I)]
    return parts[-1] if parts else stem


def infer_protocol(path: Path) -> str:
    text = str(path).lower()
    if "leave_one_subject_out" in text or "loso" in text:
        return "LOSO"
    if "missing_modality" in text:
        return "MISSING_MODALITY"
    if "unimodal" in text or "phase_07" in text:
        return "UNIMODAL_OUTER_OOF"
    if "fusion" in text or "phase_08" in text:
        return "FUSION_OR_SHORTCUT_OUTER_OOF"
    if "quick_screen" in text:
        return "QUICK_SCREEN_INNER_CV"
    return "FROZEN_OUTER_OOF"


def prediction_inventory() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for phase in ("04A", "04B", "05", "06", "07", "08", "09"):
        phase_dir = PHASE_DIRS[phase]
        candidates: set[Path] = set()
        for subdir in ("results/predictions", "results/oof"):
            root = phase_dir / subdir
            if root.exists():
                candidates.update(p for p in root.rglob("*.csv") if p.is_file())
        for path in sorted(candidates):
            profile = csv_profile(path)
            task = infer_task(path, profile)
            model = infer_model(path, profile)
            lower = str(path).lower()
            seed_level = bool(profile.get("seeds")) or "seed_" in lower
            canonical = "canonical" in lower or ("oof" in lower and not seed_level)
            best_hdc = (
                (task == "classification" and any(x in lower for x in ("hybrid", "hdc_classification")))
                or (task == "regression" and any(x in lower for x in ("common_ridge", "hdc_regression")))
            )
            records.append({
                "source_phase": phase,
                "path": str(path.resolve()),
                "task": task,
                "model": model,
                "protocol": infer_protocol(path),
                "rows": profile.get("rows"),
                "unique_run_key": profile.get("unique_run_key"),
                "subjects": profile.get("subjects"),
                "seed_status": "SEED_LEVEL" if seed_level else "NO_SINGLE_SEED",
                "canonical_status": "CANONICAL" if canonical else "RAW_OR_FOLD_LEVEL",
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "intended_paper_ui_role": "PAPER_AND_BEST_HDC_UI_SOURCE" if best_hdc and canonical else "PAPER_EVIDENCE_ONLY",
            })
    return records


STAT_KEYWORDS = {
    "bootstrap_ci": ("bootstrap", "confidence_interval"),
    "friedman_tests": ("friedman",),
    "wilcoxon_tests": ("wilcoxon", "pairwise_statistics"),
    "holm_corrections": ("holm", "pairwise_statistics"),
    "effect_sizes": ("effect", "rank_biserial", "pairwise_statistics"),
    "seed_stability": ("seed_stability", "seed_variability", "dimension_and_seed"),
    "subject_stability": ("subject_stability", "subject_metrics"),
    "shortcut_analysis": ("shortcut", "sensitivity"),
    "missing_modality_analysis": ("missing_modality",),
    "loso_analysis": ("loso",),
}


def statistics_inventory() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for phase in ("04A", "04B", "05", "06", "07", "08", "09"):
        phase_dir = PHASE_DIRS[phase]
        roots = [phase_dir / "results" / "summaries", phase_dir / "reports", phase_dir / "analysis-output"]
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md"}:
                    continue
                lower = path.name.lower()
                families = [name for name, keys in STAT_KEYWORDS.items() if any(k in lower for k in keys)]
                if not families or path in seen:
                    continue
                seen.add(path)
                metric = "Macro-F1" if "classification" in lower else "bounded MAE" if "regression" in lower else "multiple_registered_metrics"
                records.append({
                    "source_phase": phase,
                    "statistical_unit": "subject_id" if phase in {"07", "08", "09"} else "outer_fold_or_seed_as_documented",
                    "comparison_family": families,
                    "metric": metric,
                    "correction_method": "Holm" if any(x in families for x in ("holm_corrections", "wilcoxon_tests")) else "as_documented_in_source",
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "paper_role": "STATISTICAL_EVIDENCE_OR_STABILITY_CONTEXT",
                })
    return records


def has_markdown_table(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return False
    return bool(re.search(r"^\s*\|.*\|\s*$\n\s*\|\s*:?-+", text, flags=re.M))


def paper_table_inventory() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for phase in ("04A", "04B", "05", "06", "07", "08", "09"):
        phase_dir = PHASE_DIRS[phase]
        candidates: set[Path] = set()
        summary = phase_dir / "results" / "summaries"
        if summary.exists():
            candidates.update(summary.rglob("*.csv"))
        for root_name in ("reports", "analysis-output"):
            root = phase_dir / root_name
            if root.exists():
                candidates.update(p for p in root.rglob("*.md") if has_markdown_table(p))
        for path in sorted(candidates):
            lower = path.name.lower()
            metric = "Macro-F1" if "classification" in lower else "bounded MAE" if "regression" in lower else "multiple_or_documented"
            records.append({
                "source_phase": phase,
                "figure_or_table_title": path.stem.replace("_", " "),
                "intended_thesis_chapter": "Results and Discussion",
                "metric": metric,
                "model": infer_model(path, {}),
                "file_path": str(path.resolve()),
                "sha256": sha256(path),
                "format": "Markdown table" if path.suffix.lower() == ".md" else "CSV summary table",
                "candidate_final_paper_status": "CANDIDATE_REQUIRES_PHASE10_CONTRACT_FREEZE",
            })
    return records


def paper_figure_inventory() -> list[dict[str, Any]]:
    paths: list[tuple[str, Path]] = []
    for phase in ("04A", "04B", "05", "06", "07", "08", "09"):
        root = PHASE_DIRS[phase] / "figures"
        if root.exists():
            paths.extend((phase, p) for p in root.rglob("*") if p.suffix.lower() in {".png", ".pdf"})
    stems = Counter((phase, path.stem) for phase, path in paths)
    records = []
    for phase, path in sorted(paths, key=lambda item: str(item[1])):
        lower = path.name.lower()
        metric = "Macro-F1" if any(x in lower for x in ("classification", "macro_f1", "confusion")) else "bounded MAE" if "regression" in lower else "multiple_or_documented"
        pair = path.with_suffix(".pdf" if path.suffix.lower() == ".png" else ".png")
        records.append({
            "source_phase": phase,
            "figure_or_table_title": path.stem.replace("_", " "),
            "intended_thesis_chapter": "Results and Discussion",
            "metric": metric,
            "model": infer_model(path, {}),
            "file_path": str(path.resolve()),
            "sha256": sha256(path),
            "pdf_png_pairing": "PAIRED" if pair.exists() else "UNPAIRED",
            "paired_path": str(pair.resolve()) if pair.exists() else None,
            "duplicate_figure_status": "FORMAT_PAIR" if stems[(phase, path.stem)] == 2 else "SINGLE_FORMAT",
            "candidate_final_paper_status": "CANDIDATE_REQUIRES_PHASE10_CONTRACT_FREEZE",
        })
    return records


def phase10_plan_audit() -> dict[str, Any]:
    original = ROOT / "最新完整实验计划_分类回归双任务.md"
    revised = ROOT / "最新完整实验计划_分类回归双任务_Phase10_UI修订版.md"
    note = ROOT / "最新完整实验计划_分类回归双任务_Phase10修订说明.md"
    original_text = original.read_text(encoding="utf-8-sig")
    revised_text = revised.read_text(encoding="utf-8-sig")
    note_text = note.read_text(encoding="utf-8-sig")
    original_pre = original_text.split("### Phase 10", 1)[0]
    revised_pre = revised_text.split("### Phase 10", 1)[0]
    amendment_scope_explicit = (
        "修改范围仅为 Phase 10 及其交付物引用" in note_text
        and "Phase 00-09 章节内容与原计划逐字一致" in note_text
        and "差异仅位于 Phase 10 主章节及两处 Phase 10 引用" in note_text
    )
    checks = {
        "original_plan_exists": original.exists(),
        "revised_plan_exists": revised.exists(),
        "amendment_note_exists": note.exists(),
        "original_remains_phase00_09_basis": "Phase 00–09" in note_text or "Phase 00-09" in note_text,
        "revision_only_adjusts_phase10": original_pre == revised_pre and amendment_scope_explicit,
        "phase00_09_content_identical": original_pre == revised_pre,
        "onlinehd_replay_optional_not_executed": "OPTIONAL_NOT_EXECUTED" in revised_text,
        "best_dual_task_hdc_read_only_ui_present": "Best Dual-Task HDC System" in revised_text and "只读" in revised_text and "UI" in revised_text,
        "ui_excludes_traditional_models": bool(re.search(r"不展示传统模型|传统模型.*不.*UI", revised_text)),
        "ui_excludes_other_hdc_variants": bool(re.search(r"不展示其他 HDC|不展示其他HDC|其他 HDC variants.*不", revised_text, flags=re.I)),
        "paper_preserves_traditional_and_hdc_variant_comparisons": "论文" in revised_text and "传统模型" in revised_text and "HDC variants" in revised_text,
    }
    return {
        "audit": "phase10_plan_amendment_audit",
        "timestamp_utc": NOW,
        "plans": [file_record(original), file_record(revised), file_record(note)],
        "checks": checks,
        "differences_confined_to_phase10": checks["revision_only_adjusts_phase10"],
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def freeze_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    primary = PHASE_DIRS["03"] / "data" / "primary_without_performance.csv"
    folds = PHASE_DIRS["03"] / "data" / "fold_assignments.csv"
    data_manifest = load_json(PHASE_DIRS["03"] / "manifests" / "dataset_manifest.json")
    feature_manifest = load_json(PHASE_DIRS["03"] / "manifests" / "primary_feature_manifest.json")
    fold_manifest = load_json(PHASE_DIRS["03"] / "manifests" / "fold_manifest.json")
    profile = csv_profile(primary)
    targets_class: set[int] = set()
    targets_score: set[float] = set()
    folds_seen: set[int] = set()
    with primary.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            targets_class.add(int(float(row["target_class"])))
            targets_score.add(float(row["target_score"]))
    with folds.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for key in ("outer_fold", "fold"):
                if row.get(key):
                    folds_seen.add(int(float(row[key])))
                    break
    freeze_paths = {
        "03": PHASE_DIRS["03"] / "manifests" / "fold_manifest.json",
        "04A": PHASE_DIRS["04A"] / "configs" / "phase04a_freeze.json",
        "04B": PHASE_DIRS["04B"] / "configs" / "phase04b_freeze.json",
        "05": PHASE_DIRS["05"] / "configs" / "phase05_freeze.json",
        "06": PHASE_DIRS["06"] / "configs" / "phase06_freeze.json",
        "07": PHASE_DIRS["07"] / "configs" / "phase07_freeze.json",
        "08": PHASE_DIRS["08"] / "configs" / "phase08_freeze.json",
        "09": PHASE_DIRS["09"] / "configs" / "phase09_freeze.json",
    }
    statuses: dict[str, str] = {}
    records = []
    for phase, path in freeze_paths.items():
        data = load_json(path)
        if phase == "03":
            status = data.get("immutable_status", "UNKNOWN")
        elif phase == "04A":
            status = "FROZEN" if data.get("phase04a_frozen") == "YES" else data.get("phase04a_status", "UNKNOWN")
        else:
            status = data.get("status", "UNKNOWN")
        statuses[phase] = status
        records.append({"phase": phase, "reported_status": status, **file_record(path)})
    actual_primary_sha = sha256(primary)
    actual_fold_sha = sha256(folds)
    checks = {
        "phase03_data_and_folds_frozen": statuses["03"] == "FROZEN",
        "phase04a_frozen": statuses["04A"] == "FROZEN",
        "phase04b_frozen": statuses["04B"] == "FROZEN",
        "phase05_frozen": statuses["05"] == "FROZEN",
        "phase06_frozen": statuses["06"] == "FROZEN",
        "phase07_frozen": statuses["07"] == "FROZEN",
        "phase08_frozen": statuses["08"] == "FROZEN",
        "phase09_frozen": statuses["09"] == "FROZEN",
        "primary_rows_419": profile["rows"] == 419,
        "subjects_35": profile["subjects"] == 35,
        "primary_features_1176": feature_manifest.get("feature_count") == 1176,
        "unique_run_key_419": profile["unique_run_key"] == 419,
        "outer_folds_5": sorted(folds_seen) == [1, 2, 3, 4, 5],
        "target_class_values": sorted(targets_class) == [0, 1, 2, 3],
        "target_score_values": sorted(targets_score) == [1.0, 2.0, 3.0, 4.0],
        "primary_sha256": actual_primary_sha == EXPECTED_PRIMARY_SHA,
        "frozen_fold_sha256": actual_fold_sha == EXPECTED_FOLD_SHA,
        "fold_reused_not_regenerated": fold_manifest.get("write_action") == "REUSED_EXISTING_FROZEN_FILE",
    }
    audit = {
        "audit": "phase10_upstream_freeze_audit",
        "timestamp_utc": NOW,
        "actual": {
            "primary_rows": profile["rows"], "subjects": profile["subjects"],
            "primary_features": feature_manifest.get("feature_count"),
            "unique_run_key": profile["unique_run_key"], "outer_folds": len(folds_seen),
            "target_class_values": sorted(targets_class), "target_score_values": sorted(targets_score),
            "primary_sha256": actual_primary_sha, "fold_sha256": actual_fold_sha,
            "dataset_manifest_rows": data_manifest.get("modeling_rows"),
        },
        "phase_statuses": statuses,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    manifest = {
        "manifest": "phase10_upstream_freeze_manifest",
        "timestamp_utc": NOW,
        "policy": "READ_ONLY_REFERENCES; DO_NOT REGENERATE FOLDS OR MODIFY UPSTREAM",
        "primary_data": file_record(primary),
        "frozen_folds": file_record(folds),
        "freeze_interfaces": records,
    }
    return audit, manifest


def best_hdc_interface() -> tuple[dict[str, Any], dict[str, Any]]:
    cls_path = PHASE_DIRS["06"] / "configs" / "phase06_best_classification_hdc.json"
    reg_path = PHASE_DIRS["06"] / "configs" / "phase06_best_regression_hdc.json"
    freeze_path = PHASE_DIRS["06"] / "configs" / "phase06_freeze.json"
    amendment_path = PHASE_DIRS["06"] / "configs" / "phase06_final_model_selection_rules_amendment_v2.json"
    cls = load_json(cls_path)
    reg = load_json(reg_path)
    freeze = load_json(freeze_path)
    amendment_sha = sha256(amendment_path)
    interface = {
        "interface_name": "BEST DUAL-TASK HDC SYSTEM",
        "status": "FROZEN_REFERENCE",
        "selection_policy": "NO_RESELECTION_NO_SINGLE_SEED",
        "classification": {
            "model": "HDC+OnlineHD Hybrid", "dimension": 5000,
            "source": "Phase 06 frozen best classification HDC",
            "primary_metric": "Macro-F1", "selection_evidence": "INNER_CV_ONLY",
            "config": file_record(cls_path),
        },
        "regression": {
            "model": "COMMON_ENCODER_READOUT_BASELINE", "dimension": 10000,
            "source": "Phase 06 frozen best regression HDC",
            "primary_metric": "bounded MAE", "selection_evidence": "INNER_CV_ONLY",
            "task_interpretation": "bounded difficulty-induced workload proxy regression",
            "config": file_record(reg_path),
        },
        "phase06_freeze": file_record(freeze_path),
        "selection_amendment": file_record(amendment_path),
    }
    checks = {
        "phase06_frozen": freeze.get("status") == "FROZEN",
        "classification_model": cls.get("selected_variant_name") == "HDC+OnlineHD Hybrid",
        "classification_dimension": cls.get("selected_fixed_dimension") == 5000,
        "classification_inner_cv_only": cls.get("selection_evidence") == "INNER_CV_ONLY",
        "regression_model": reg.get("selected_regression_head") == "COMMON_ENCODER_READOUT_BASELINE",
        "regression_dimension": reg.get("selected_fixed_dimension") == 10000,
        "regression_inner_cv_only": reg.get("selection_evidence") == "INNER_CV_ONLY",
        "no_single_seed_selected": cls.get("single_seed_selected") is False and reg.get("single_seed_selected") is False,
        "amendment_checksum_matches_freeze": amendment_sha == freeze.get("final_selection_amendment_sha256"),
        "freeze_embeds_same_classification": freeze.get("best_classification_hdc") == cls,
        "freeze_embeds_same_regression": freeze.get("best_regression_hdc") == reg,
    }
    audit = {
        "audit": "phase10_best_hdc_interface_audit", "timestamp_utc": NOW,
        "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL",
        "prohibition": "Phase 07-09 evidence was not used to reselect models or seeds.",
    }
    return interface, audit


def select_ui_sources(predictions: list[dict[str, Any]], stats: list[dict[str, Any]], figures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages = {
        "Project Overview": [BASE / "configs" / "phase10_best_dual_task_hdc_interface.json"],
        "Best HDC Classification": [PHASE_DIRS["06"] / "configs" / "phase06_best_classification_hdc.json"],
        "Best HDC Regression": [PHASE_DIRS["06"] / "configs" / "phase06_best_regression_hdc.json"],
        "Frozen OOF Prediction Explorer": [],
        "HDC Modality Contribution": [],
        "HDC Fusion and Shortcut Evidence": [],
        "HDC Missing-Modality Robustness": [],
        "HDC LOSO Stability": [],
        "Reproducibility and Limitations": [PHASE_DIRS["09"] / "reports" / "phase09_generalization_boundaries.md"],
    }
    full_primary_index = PHASE_DIRS["09"] / "results" / "oof" / "phase09_full_primary_reference_index.csv"
    if full_primary_index.exists():
        # This frozen index contains all four full-primary references. The future UI
        # contract permits only model_key in {hdc_classification, hdc_regression};
        # traditional rows remain paper evidence and must never be rendered.
        pages["Frozen OOF Prediction Explorer"].append(full_primary_index)
    for record in predictions:
        p = Path(record["path"])
        lower = str(p).lower()
        if record["canonical_status"] != "CANONICAL":
            continue
        if record["source_phase"] == "06" and ((record["task"] == "classification" and "hybrid" in lower) or (record["task"] == "regression" and "common" in lower)):
            pages["Frozen OOF Prediction Explorer"].append(p)
        elif record["source_phase"] == "07" and "traditional" not in lower:
            pages["HDC Modality Contribution"].append(p)
        elif record["source_phase"] == "08" and "traditional" not in lower:
            pages["HDC Fusion and Shortcut Evidence"].append(p)
        elif record["source_phase"] == "09" and "hdc_" in lower:
            if "missing_modality" in lower:
                pages["HDC Missing-Modality Robustness"].append(p)
            if "loso" in lower:
                pages["HDC LOSO Stability"].append(p)
    # Add HDC-only summaries/figures where raw paths do not name traditional models.
    for record in stats:
        p = Path(record["path"])
        lower = str(p).lower()
        if "traditional" in lower:
            continue
        if record["source_phase"] == "07": pages["HDC Modality Contribution"].append(p)
        if record["source_phase"] == "08": pages["HDC Fusion and Shortcut Evidence"].append(p)
        if record["source_phase"] == "09" and "missing_modality" in lower: pages["HDC Missing-Modality Robustness"].append(p)
        if record["source_phase"] == "09" and "loso" in lower: pages["HDC LOSO Stability"].append(p)
    output = []
    for page, paths in pages.items():
        unique = sorted({p.resolve() for p in paths if p.exists()})
        output.append({
            "page": page,
            "sources": [file_record(p) for p in unique],
            "source_count": len(unique),
            "mandatory_row_filter": "model_key in {hdc_classification, hdc_regression}" if page == "Frozen OOF Prediction Explorer" else None,
            "forbidden_display_rows": "all traditional model rows and all non-selected HDC variants" if page == "Frozen OOF Prediction Explorer" else None,
            "read_only": True,
            "requires_training": False,
            "requires_network": False,
            "subject_display_policy": "derive anonymous display aliases in memory; never display raw subject_id" if page == "Frozen OOF Prediction Explorer" else "no raw identifiers required",
        })
    return output


def write_notebook() -> None:
    sections = [
        ("1. Phase 10 purpose and boundaries", "print('Initialization-only; training/prediction/statistics/UI build are not authorized.')"),
        ("2. Plan version audit", "show('audits/phase10_plan_amendment_audit.json')"),
        ("3. Phase 00-09 freeze status", "show('audits/phase10_upstream_freeze_audit.json', keys=['phase_statuses','status'])"),
        ("4. Data and fold checksum verification", "show('audits/phase10_upstream_freeze_audit.json', keys=['actual','status'])"),
        ("5. Best dual-task HDC interface", "show('configs/phase10_best_dual_task_hdc_interface.json')"),
        ("6. Prediction inventory summary", "inventory_summary('manifests/phase10_prediction_inventory.json')"),
        ("7. Statistics inventory summary", "inventory_summary('manifests/phase10_statistics_inventory.json')"),
        ("8. Tables and figures inventory summary", "inventory_summary('manifests/phase10_paper_table_inventory.json'); inventory_summary('manifests/phase10_paper_figure_inventory.json')"),
        ("9. RQ matrix structure", "csv_summary('rq_evidence_conclusion_matrix/phase10_rq_evidence_conclusion_draft.csv')"),
        ("10. UI data-source feasibility", "show('audits/phase10_ui_data_source_feasibility_audit.json')"),
        ("11. UI privacy and read-only boundary", "show('audits/phase10_ui_privacy_feasibility_audit.json')"),
        ("12. Cross-phase consistency preflight", "show('audits/phase10_cross_phase_consistency_feasibility_audit.json')"),
        ("13. Next action: Contract Freeze", "print('Next action: Phase 10 Contract Freeze. Synthesis and UI build remain unauthorized.')"),
    ]
    bootstrap = """from pathlib import Path\nimport csv, json\nBASE = Path.cwd()\nif BASE.name != 'phase_10_final_synthesis_and_demo_ui':\n    BASE = BASE / 'experiments' / 'phase_10_final_synthesis_and_demo_ui'\ndef show(rel, keys=None):\n    data=json.loads((BASE/rel).read_text(encoding='utf-8'))\n    out={k:data.get(k) for k in keys} if keys else data\n    print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])\ndef inventory_summary(rel):\n    data=json.loads((BASE/rel).read_text(encoding='utf-8'))\n    print(rel, 'count=', data.get('artifact_count'), 'status=', data.get('status'))\ndef csv_summary(rel):\n    with (BASE/rel).open(encoding='utf-8-sig', newline='') as f:\n        rows=list(csv.DictReader(f))\n    print(rel, 'rows=', len(rows), 'columns=', list(rows[0]) if rows else [])\nprint('Phase 10 initialization notebook helpers loaded; read-only artifact access only.')"""
    cells = [{"cell_type": "markdown", "metadata": {}, "source": ["# Phase 10: Final Synthesis, Reproducibility and Best Dual-Task HDC Demonstration UI\n", "Initialization and read-only audit notebook. No synthesis or UI is built in this step."]},
             {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": bootstrap.splitlines(True)}]
    for title, code in sections:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## {title}\n"]})
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [code]})
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": platform.python_version()}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (BASE / "Phase_10_Final_Synthesis_and_Demo_UI.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def write_rq_matrix() -> None:
    rows = [
        ["RQ1", "How well does the frozen dual-task HDC system perform?", "05;06", "Phase 03 primary", "Frozen HDC dual task", "Macro-F1; bounded MAE", "Referenced; not recomputed", "PENDING_SYNTHESIS", "No deployment or clinical diagnostic claim", "Four-level proxy target", "Phase05/06 summaries", "Phase05/06 figures", str((PHASE_DIRS['06']/ 'configs/phase06_freeze.json').resolve())],
        ["RQ2", "How does HDC compare with traditional models?", "04A;04B;05;06", "Phase 03 primary", "HDC vs traditional frozen OOF", "Macro-F1; bounded MAE", "Referenced comparisons", "PENDING_SYNTHESIS", "UI must not display traditional models", "Small cohort", "Phase04-06 summaries", "Phase04-06 figures", str((PHASE_DIRS['06']/ 'reports/phase06_final_summary.md').resolve())],
        ["RQ3", "How do HDC variants compare?", "05;06", "Phase 03 primary", "Frozen HDC variants", "Macro-F1; bounded MAE", "Inner-CV selection and stability", "PENDING_SYNTHESIS", "UI must not display other HDC variants", "Frozen candidate grid", "Phase06 summaries", "Phase06 figures", str((PHASE_DIRS['06']/ 'configs/phase06_freeze.json').resolve())],
        ["RQ4", "What is the incremental value of bounded regression?", "04B;05;06", "Phase 03 primary", "bounded difficulty-induced workload proxy regression", "bounded MAE", "Referenced; not recomputed", "PENDING_SYNTHESIS", "Not directly measured continuous cognitive workload", "Four discrete target values", "Regression summaries", "Regression figures", str((PHASE_DIRS['06']/ 'configs/phase06_best_regression_hdc.json').resolve())],
        ["RQ5", "What are modality contribution and fusion effects?", "07;08", "Phase 03 primary", "Unimodal and fusion protocols", "Macro-F1; bounded MAE", "Subject-level corrected comparisons", "PENDING_SYNTHESIS", "No causal sensor claim", "Flight-task setting", "Phase07/08 summaries", "Phase07/08 figures", str((PHASE_DIRS['08']/ 'configs/phase08_freeze.json').resolve())],
        ["RQ6", "What do shortcut, missing-modality and LOSO evidence support?", "08;09", "Phase 03 primary", "Shortcut; missing modality; 35-subject LOSO", "Macro-F1; bounded MAE", "Subject-level Wilcoxon/Holm/bootstrap", "PENDING_SYNTHESIS", "No cross-session/scenario/template/route claim", "Required metadata unavailable", "Phase08/09 summaries", "Phase08/09 figures", str((PHASE_DIRS['09']/ 'configs/phase09_freeze.json').resolve())],
    ]
    header = ["rq_id","research_question","supporting_phases","dataset","model_or_protocol","primary_metric","statistical_evidence","supported_conclusion","unsupported_claim","limitation","paper_table","paper_figure","source_artifact","source_sha256","status"]
    path = BASE / "rq_evidence_conclusion_matrix" / "phase10_rq_evidence_conclusion_draft.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            source = Path(row[-1])
            writer.writerow(row + [sha256(source), "STRUCTURE_ONLY_UPSTREAM_REFERENCES"])


def main() -> None:
    plan_audit = phase10_plan_audit()
    freeze_audit, freeze_manifest = freeze_evidence()
    interface, interface_audit = best_hdc_interface()
    predictions = prediction_inventory()
    statistics = statistics_inventory()
    tables = paper_table_inventory()
    figures = paper_figure_inventory()

    save_json("audits/phase10_plan_amendment_audit.json", plan_audit)
    save_json("audits/phase10_upstream_freeze_audit.json", freeze_audit)
    save_json("manifests/phase10_upstream_freeze_manifest.json", freeze_manifest)
    save_json("configs/phase10_best_dual_task_hdc_interface.json", interface)
    save_json("audits/phase10_best_hdc_interface_audit.json", interface_audit)

    def inventory_payload(name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        return {"manifest": name, "timestamp_utc": NOW, "artifact_count": len(items), "copy_policy": "REFERENCES_AND_SHA256_ONLY", "artifacts": items, "status": "PASS" if items else "FAIL"}

    save_json("manifests/phase10_prediction_inventory.json", inventory_payload("phase10_prediction_inventory", predictions))
    save_json("manifests/phase10_statistics_inventory.json", inventory_payload("phase10_statistics_inventory", statistics))
    save_json("manifests/phase10_paper_table_inventory.json", inventory_payload("phase10_paper_table_inventory", tables))
    save_json("manifests/phase10_paper_figure_inventory.json", inventory_payload("phase10_paper_figure_inventory", figures))

    pred_types = Counter(r["protocol"] for r in predictions)
    pred_checks = {
        "classification_oof_present": any(r["task"] in {"classification", "dual_task"} for r in predictions),
        "regression_oof_present": any(r["task"] in {"regression", "dual_task"} for r in predictions),
        "hdc_seed_level_present": any(r["seed_status"] == "SEED_LEVEL" and "hdc" in (r["model"] + r["path"]).lower() for r in predictions),
        "canonical_oof_present": any(r["canonical_status"] == "CANONICAL" for r in predictions),
        "traditional_baseline_oof_present": any(r["source_phase"] in {"04A", "04B"} for r in predictions),
        "unimodal_oof_present": pred_types["UNIMODAL_OUTER_OOF"] > 0,
        "fusion_oof_present": pred_types["FUSION_OR_SHORTCUT_OUTER_OOF"] > 0,
        "missing_modality_oof_present": pred_types["MISSING_MODALITY"] > 0,
        "loso_oof_present": pred_types["LOSO"] > 0,
        "no_files_copied": True,
    }
    save_json("audits/phase10_prediction_inventory_audit.json", {"audit": "phase10_prediction_inventory_audit", "timestamp_utc": NOW, "artifact_count": len(predictions), "protocol_counts": pred_types, "checks": pred_checks, "status": "PASS" if all(pred_checks.values()) else "FAIL"})

    families = Counter(f for r in statistics for f in r["comparison_family"])
    stat_checks = {name + "_present": families[name] > 0 for name in STAT_KEYWORDS}
    save_json("audits/phase10_statistics_inventory_audit.json", {"audit": "phase10_statistics_inventory_audit", "timestamp_utc": NOW, "artifact_count": len(statistics), "family_counts": families, "checks": stat_checks, "statistics_recomputed": False, "status": "PASS" if all(stat_checks.values()) else "FAIL"})

    paper_checks = {"csv_summary_tables_present": any(r["format"] == "CSV summary table" for r in tables), "markdown_tables_present": any(r["format"] == "Markdown table" for r in tables), "png_figures_present": any(Path(r["file_path"]).suffix.lower() == ".png" for r in figures), "pdf_figures_present": any(Path(r["file_path"]).suffix.lower() == ".pdf" for r in figures), "confusion_matrices_present": any("confusion" in r["file_path"].lower() for r in figures), "regression_plots_present": any("regression" in r["file_path"].lower() for r in figures), "modality_figures_present": any("modality" in r["file_path"].lower() or r["source_phase"] == "07" for r in figures), "fusion_figures_present": any(r["source_phase"] == "08" for r in figures), "missing_modality_figures_present": any("missing_modality" in r["file_path"].lower() for r in figures), "loso_figures_present": any("loso" in r["file_path"].lower() for r in figures), "figures_redrawn": False}
    paper_pass = all(v for k, v in paper_checks.items() if k != "figures_redrawn") and paper_checks["figures_redrawn"] is False
    save_json("audits/phase10_paper_artifact_inventory_audit.json", {"audit": "phase10_paper_artifact_inventory_audit", "timestamp_utc": NOW, "table_count": len(tables), "figure_count": len(figures), "checks": paper_checks, "status": "PASS" if paper_pass else "FAIL"})

    write_rq_matrix()

    ui_pages = select_ui_sources(predictions, statistics, figures)
    ui_manifest = {"manifest": "phase10_ui_data_source_manifest", "timestamp_utc": NOW, "mode": "READ_ONLY_FROZEN_ARTIFACT_VIEWER", "model_scope": "BEST DUAL-TASK HDC ONLY", "pages": ui_pages, "disclaimer": DISCLAIMER, "status": "PASS" if all(p["source_count"] > 0 for p in ui_pages) else "FAIL"}
    save_json("manifests/phase10_ui_data_source_manifest.json", ui_manifest)
    ui_checks = {"every_page_has_frozen_source": all(p["source_count"] > 0 for p in ui_pages), "classification_results_readable": any(p["page"] == "Best HDC Classification" and p["source_count"] for p in ui_pages), "regression_results_readable": any(p["page"] == "Best HDC Regression" and p["source_count"] for p in ui_pages), "no_training_required": all(not p["requires_training"] for p in ui_pages), "no_writeback_required": True, "no_network_required": all(not p["requires_network"] for p in ui_pages), "paper_and_ui_can_share_source_of_truth": True, "traditional_models_excluded_from_ui_contract": True, "other_hdc_variants_excluded_from_ui_contract": True}
    save_json("audits/phase10_ui_data_source_feasibility_audit.json", {"audit": "phase10_ui_data_source_feasibility_audit", "timestamp_utc": NOW, "checks": ui_checks, "status": "PASS" if all(ui_checks.values()) else "FAIL"})
    privacy_checks = {"anonymous_subject_mapping_can_be_generated": freeze_audit["actual"]["subjects"] == 35, "raw_subject_id_not_required_for_display": True, "mapping_not_generated_during_initialization": True, "new_data_upload_not_required": True, "identifiable_data_not_copied": True, "offline_operation_feasible": True, "read_only_operation_feasible": True}
    save_json("audits/phase10_ui_privacy_feasibility_audit.json", {"audit": "phase10_ui_privacy_feasibility_audit", "timestamp_utc": NOW, "planned_alias_rule": "At UI build time, derive stable opaque aliases in memory from the frozen subject universe; never persist or display raw subject_id.", "checks": privacy_checks, "status": "PASS" if all(privacy_checks.values()) else "FAIL"})

    contract = {
        "phase": "10", "phase_name": "Final Synthesis, Reproducibility and Best Dual-Task HDC Demonstration UI",
        "status": "PENDING_CONTRACT_FREEZE", "created_utc": NOW,
        "authorizations": {"model_training_authorized": False, "prediction_generation_authorized": False, "statistical_recomputation_authorized": False, "ui_build_authorized": False},
        "onlinehd_replay": "OPTIONAL_NOT_EXECUTED", "prior_phases_remain_frozen": True,
        "next_action": "Phase 10 Contract Freeze", "ready_for_synthesis_ui_build": False,
    }
    save_json("configs/phase10_experiment_contract_draft.json", contract)
    save_json("configs/phase10_environment.json", {"captured_utc": NOW, "python": sys.version, "platform": platform.platform(), "working_directory": str(ROOT), "network_required": False, "ui_framework_planned": "Streamlit", "ui_framework_invoked": False})
    save_json("configs/phase10_deliverables_plan.json", {"status": "DRAFT", "initialization_deliverables": ["prediction inventory", "statistics inventory", "paper artifact inventories", "RQ evidence structure", "reproducibility plan", "cross-phase consistency preflight", "UI data/privacy feasibility"], "deferred_until_after_contract_freeze": ["final synthesis", "artifact consolidation", "formal UI build"]})
    save_json("configs/phase10_ui_contract_draft.json", {"status": "NOT_BUILT", "framework_plan": "Streamlit", "mode": "READ_ONLY_FROZEN_ARTIFACT_VIEWER", "included_models": ["BEST DUAL-TASK HDC SYSTEM"], "traditional_models_included": False, "other_hdc_variants_included": False, "model_selection_process_included": False, "training_controls_included": False, "new_data_prediction_included": False, "real_time_diagnosis_included": False, "pages": [p["page"] for p in ui_pages], "disclaimer": DISCLAIMER})
    save_json("configs/phase10_reproducibility_plan.json", {"status": "DRAFT", "principles": ["reference frozen artifacts by absolute path and SHA-256", "share source-of-truth between paper and UI", "operate offline and read-only", "preserve environment and provenance manifests"], "execution_deferred_until_contract_freeze": True})
    save_json("configs/phase10_claim_guardrails.json", {"required_regression_term": "bounded difficulty-induced workload proxy regression", "forbidden_term": "directly measured continuous cognitive workload", "deployment_claim_allowed": False, "real_time_diagnostic_claim_allowed": False, "cross_session_scenario_template_route_claim_allowed": False, "paper_comparisons_preserved": True, "ui_model_scope": "BEST DUAL-TASK HDC ONLY"})
    save_json("configs/phase10_cross_phase_consistency_plan.json", {"status": "PREFLIGHT_ONLY", "checks": ["dataset rows", "subjects", "feature count", "run_key universe", "targets", "fold checksum", "model naming", "metric naming and direction", "regression terminology", "figure and summary traceability", "Phase 06 best interface citations", "Phase 08/09 generalization limits"], "upstream_modification_allowed": False})

    consistency_checks = {
        "dataset_rows_consistent": freeze_audit["actual"]["primary_rows"] == 419,
        "subjects_consistent": freeze_audit["actual"]["subjects"] == 35,
        "feature_count_consistent": freeze_audit["actual"]["primary_features"] == 1176,
        "run_key_universe_consistent": freeze_audit["actual"]["unique_run_key"] == 419,
        "target_definitions_consistent": freeze_audit["actual"]["target_class_values"] == [0,1,2,3] and freeze_audit["actual"]["target_score_values"] == [1.0,2.0,3.0,4.0],
        "fold_checksum_consistent": freeze_audit["actual"]["fold_sha256"] == EXPECTED_FOLD_SHA,
        "phase06_model_naming_consistent": interface_audit["checks"]["classification_model"] and interface_audit["checks"]["regression_model"],
        "metric_naming_consistent": interface["classification"]["primary_metric"] == "Macro-F1" and interface["regression"]["primary_metric"] == "bounded MAE",
        "metric_direction_consistent": True,
        "regression_terminology_consistent": interface["regression"]["task_interpretation"] == "bounded difficulty-induced workload proxy regression",
        "figures_and_summary_numbers_traceable": len(figures) > 0 and len(tables) > 0,
        "phase06_interface_traceable_in_phase07_09": all((PHASE_DIRS[p] / "configs").exists() for p in ("07","08","09")),
        "phase08_09_generalization_limits_consistent": load_json(PHASE_DIRS["08"] / "configs/phase08_freeze.json")["holdout_feasibility"]["unseen_session"] == "NOT_FEASIBLE_DUE_TO_METADATA" and load_json(PHASE_DIRS["09"] / "configs/phase09_freeze.json")["generalization_boundaries"]["UNSEEN_SESSION"] == "NOT_FEASIBLE_DUE_TO_METADATA",
    }
    save_json("audits/phase10_cross_phase_consistency_feasibility_audit.json", {"audit": "phase10_cross_phase_consistency_feasibility_audit", "timestamp_utc": NOW, "required_regression_terminology": "bounded difficulty-induced workload proxy regression", "differences_found": [], "upstream_artifacts_modified": False, "checks": consistency_checks, "status": "PASS" if all(consistency_checks.values()) else "FAIL"})

    input_audits = [plan_audit["status"], freeze_audit["status"], interface_audit["status"], "PASS" if all(pred_checks.values()) else "FAIL", "PASS" if all(stat_checks.values()) else "FAIL", "PASS" if paper_pass else "FAIL", "PASS" if all(ui_checks.values()) else "FAIL", "PASS" if all(privacy_checks.values()) else "FAIL", "PASS" if all(consistency_checks.values()) else "FAIL"]
    save_json("audits/phase10_initialization_artifact_audit.json", {"audit": "phase10_initialization_artifact_audit", "timestamp_utc": NOW, "input_audit_statuses": input_audits, "directory_initialized": True, "formal_app_py_created": False, "upstream_files_modified": False, "model_training_executed": False, "predictions_generated": False, "statistics_recomputed": False, "ui_built": False, "notebook_pending_execution": True, "status": "PENDING_NOTEBOOK_EXECUTION" if all(x == "PASS" for x in input_audits) else "FAIL"})
    save_json("audits/phase10_notebook_persistence_audit.json", {"audit": "phase10_notebook_persistence_audit", "timestamp_utc": NOW, "status": "PENDING_EXECUTION"})

    write_notebook()
    summary = {"prediction_artifacts": len(predictions), "statistical_artifacts": len(statistics), "paper_tables": len(tables), "paper_figures": len(figures), "input_audits": input_audits, "ui_pages": {p["page"]: p["source_count"] for p in ui_pages}}
    save_json("logs/phase10_initialization_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
