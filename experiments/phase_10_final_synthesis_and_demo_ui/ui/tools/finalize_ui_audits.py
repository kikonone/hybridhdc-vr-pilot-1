"""Finalize required dual-task UI audits without changing frozen upstream files."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

UI = Path(__file__).resolve().parents[1]
EXPERIMENTS = UI.parents[1]
AUDITS = UI / "audits"
REPORTS = UI / "tests/reports"
DATA = UI / "data"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(name: str, payload: dict) -> None:
    AUDITS.mkdir(exist_ok=True)
    (AUDITS / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def current_upstream() -> dict[str, dict]:
    result = {}
    for phase in sorted(EXPERIMENTS.glob("phase_*")):
        for path in sorted(item for item in phase.rglob("*") if item.is_file()):
            if UI == path or UI in path.parents:
                continue
            result[str(path.resolve())] = {"size": path.stat().st_size, "sha256": digest(path)}
    return result


now = datetime.now(timezone.utc).isoformat()
manifest = load(DATA / "demo_data_manifest.json")
model = load(DATA / "frozen_dual_task_model.json")
with (DATA / "frozen_dual_task_oof.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

source_checks = []
for item in manifest["source_files"]:
    path = Path(item["path"])
    actual_hash = digest(path) if path.is_file() else None
    actual_size = path.stat().st_size if path.is_file() else None
    source_checks.append({
        **item, "actual_size_bytes": actual_size, "actual_sha256": actual_hash,
        "status": "PASS" if actual_hash == item["sha256"] and actual_size == item["size_bytes"] else "FAIL",
    })

regression_roles = {
    "regression_selection", "phase06_oof_seal", "regression_seed_oof",
    "dual_task_contract", "regression_canonical_oof", "phase10_selected_interface", "phase10_prediction_index",
}
regression_checks = [item for item in source_checks if item["role"] in regression_roles]
regression_pass = all(item["status"] == "PASS" for item in regression_checks)
write("ui_regression_source_integrity_audit.json", {
    "audit": "ui_regression_source_integrity", "status": "PASS" if regression_pass else "FAIL",
    "checked_at_utc": now, "canonical_source_role": "regression_canonical_oof",
    "selected_head": model["regression"]["model"], "selected_variant": model["regression"]["variant"],
    "dimension": model["regression"]["dimension"], "feature_k": model["regression"]["feature_k"],
    "levels": model["regression"]["levels"], "ridge_alpha": model["regression"]["ridge_alpha"],
    "similarity_regression_used": False, "ui_clipping_executed": False,
    "checks": regression_checks,
})

ids = [row["demo_id"] for row in rows]
alignment_checks = {
    "classification_rows": len(rows), "regression_rows": len(rows),
    "aligned_rows": manifest["row_counts"]["aligned"], "unique_demo_ids": len(set(ids)),
    "stable_id_sequence": ids == [f"DEMO-{index:04d}" for index in range(1, 420)],
    "alignment_before_anonymization": manifest["alignment"]["performed_before_anonymization"],
    "real_key_sets_equal": manifest["alignment"]["real_key_sets_equal"],
    "fold_coverage": sorted({int(row["fold"]) for row in rows}),
    "missing_targets": sum(not row["true_difficulty"] or not row["true_difficulty_score"] for row in rows),
    "missing_predictions": sum(not row["predicted_difficulty"] or not row["bounded_frozen_prediction"] for row in rows),
}
alignment_pass = (
    alignment_checks["classification_rows"] == alignment_checks["regression_rows"] == alignment_checks["aligned_rows"] == 419
    and alignment_checks["unique_demo_ids"] == 419 and alignment_checks["stable_id_sequence"]
    and alignment_checks["alignment_before_anonymization"] and alignment_checks["real_key_sets_equal"]
    and alignment_checks["fold_coverage"] == [1, 2, 3, 4, 5]
    and alignment_checks["missing_targets"] == alignment_checks["missing_predictions"] == 0
)
write("ui_dual_task_alignment_audit.json", {
    "audit": "ui_dual_task_alignment", "status": "PASS" if alignment_pass else "FAIL",
    "checked_at_utc": now, "checks": alignment_checks,
    "ordered_real_key_sha256": manifest["alignment"]["ordered_real_key_sha256"],
})

runtime_files = [
    UI / "app.py", *(UI / "components").glob("*.py"), UI / "assets/aviation_console.css",
    UI / "configs/ui_contract.json", DATA / "frozen_dual_task_model.json",
    UI / "README.md", UI / "pages/README.md", UI / "start_ui.ps1", UI / "start_ui.bat",
]
cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
language_violations = []
for path in runtime_files:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if cjk.search(line):
            language_violations.append({"file": str(path.relative_to(UI)), "line": line_number})
write("ui_english_language_audit.json", {
    "audit": "ui_english_language", "status": "PASS" if not language_violations else "FAIL",
    "checked_at_utc": now, "files_scanned": len(runtime_files),
    "cjk_visible_text_violations": language_violations,
})

contract = load(UI / "configs/ui_contract.json")
active_python_pages = list((UI / "pages").glob("*.py"))
runtime_text = "\n".join(path.read_text(encoding="utf-8") for path in runtime_files)
legacy_page_labels = ["任务总览", "方法流程", "冻结样例演示", "模型性能", "多模态输入说明", "结论与局限"]
scope_checks = {
    "single_page": contract["single_page"], "allowed_tasks": contract["allowed_tasks"],
    "active_additional_python_pages": [str(path.relative_to(UI)) for path in active_python_pages],
    "legacy_page_label_hits": [label for label in legacy_page_labels if label in runtime_text],
    "classification_model": model["classification"]["model"],
    "classification_dimension": model["classification"]["dimension"],
    "regression_model": model["regression"]["model"],
    "regression_variant": model["regression"]["variant"],
    "regression_dimension": model["regression"]["dimension"],
    "uploads_present": "st.file_uploader" in runtime_text,
    "network_client_present": any(token in runtime_text for token in ("requests.", "httpx.", "urllib.")),
}
scope_pass = (
    scope_checks["single_page"] and scope_checks["allowed_tasks"] == ["Classification", "Regression"]
    and not active_python_pages and not scope_checks["legacy_page_label_hits"]
    and scope_checks["classification_model"] == "HDC+OnlineHD Hybrid" and scope_checks["classification_dimension"] == 5000
    and scope_checks["regression_model"] == "COMMON_ENCODER_READOUT_BASELINE"
    and scope_checks["regression_variant"] == "common_ridge" and scope_checks["regression_dimension"] == 10000
    and not scope_checks["uploads_present"] and not scope_checks["network_client_present"]
)
write("ui_scope_reduction_audit.json", {
    "audit": "ui_scope_reduction", "status": "PASS" if scope_pass else "FAIL",
    "checked_at_utc": now, "checks": scope_checks,
})

baseline = load(AUDITS / "frozen_phase00_10_pre_ui_baseline.json")
before = {item["path"]: {"size": item["size"], "sha256": item["sha256"]} for item in baseline["files"]}
after = current_upstream()
modified = sorted(path for path in before.keys() & after.keys() if before[path] != after[path])
added = sorted(after.keys() - before.keys())
removed = sorted(before.keys() - after.keys())
immutability_pass = not modified and not added and not removed
write("ui_post_change_frozen_immutability_audit.json", {
    "audit": "ui_post_change_frozen_immutability", "status": "PASS" if immutability_pass else "FAIL",
    "checked_at_utc": now, "scope": baseline["scope"],
    "baseline_file_count": len(before), "current_file_count": len(after),
    "modified_count": len(modified), "added_count": len(added), "removed_count": len(removed),
    "modified": modified, "added": added, "removed": removed,
})

required_audits = {
    "regression_source_integrity": AUDITS / "ui_regression_source_integrity_audit.json",
    "alignment": AUDITS / "ui_dual_task_alignment_audit.json",
    "english_language": AUDITS / "ui_english_language_audit.json",
    "scope_reduction": AUDITS / "ui_scope_reduction_audit.json",
    "playwright_e2e": AUDITS / "ui_playwright_e2e_audit.json",
    "stress": AUDITS / "ui_stress_test_report.json",
    "memory_soak": AUDITS / "ui_memory_soak_audit.json",
    "immutability": AUDITS / "ui_post_change_frozen_immutability_audit.json",
    "unit_tests": AUDITS / "ui_unit_test_audit.json",
}
statuses = {name: load(path).get("status", "MISSING") if path.is_file() else "MISSING" for name, path in required_audits.items()}
screenshots = [
    REPORTS / "screenshots/classification_1366x768.png", REPORTS / "screenshots/regression_1366x768.png",
    REPORTS / "screenshots/classification_1920x1080.png", REPORTS / "screenshots/regression_1920x1080.png",
]
screenshots_ok = all(path.is_file() and path.stat().st_size > 0 for path in screenshots)
final_pass = all(status == "PASS" for status in statuses.values()) and screenshots_ok
write("ui_final_dual_task_audit.json", {
    "audit": "ui_final_dual_task", "status": "PASS" if final_pass else "FAIL",
    "checked_at_utc": now, "component_statuses": statuses,
    "screenshots_present": screenshots_ok, "phase00_10_files_modified": len(modified) + len(added) + len(removed),
    "classification_records": 419, "regression_records": 419,
    "classification_model": "HDC+OnlineHD Hybrid", "classification_dimension": 5000,
    "regression_model": "COMMON_ENCODER_READOUT_BASELINE", "regression_variant": "common_ridge",
    "regression_dimension": 10000, "ready_for_defense_demonstration": final_pass,
})

unit = load(AUDITS / "ui_unit_test_audit.json") if (AUDITS / "ui_unit_test_audit.json").is_file() else {}
playwright = load(AUDITS / "ui_playwright_e2e_audit.json") if (AUDITS / "ui_playwright_e2e_audit.json").is_file() else {}
stress = load(AUDITS / "ui_stress_test_report.json") if (AUDITS / "ui_stress_test_report.json").is_file() else {}
REPORTS.mkdir(parents=True, exist_ok=True)
(REPORTS / "ui_test_summary.md").write_text(
    "# UI Test Summary\n\n"
    f"- Final status: {'PASS' if final_pass else 'FAIL'}\n"
    f"- Compile/import/pytest: {unit.get('status', 'MISSING')} ({unit.get('tests_passed', 0)} passed)\n"
    f"- Playwright E2E: {playwright.get('status', 'MISSING')} at 1366x768 and 1920x1080\n"
    f"- Console errors: {len(playwright.get('console_errors', []))}\n"
    f"- Stress/soak: {stress.get('status', 'MISSING')} ({stress.get('duration_seconds', 0)} seconds, "
    f"{stress.get('concurrent_browser_sessions', 0)} sessions, {stress.get('http_page_interactions', 0)} HTTP/page interactions)\n"
    f"- Rapid switches: {stress.get('total_rapid_switches', 0)}\n"
    f"- Reloads: {stress.get('reloads', 0)}\n"
    f"- Phase 00-10 upstream changes: {len(modified) + len(added) + len(removed)}\n",
    encoding="utf-8",
)
print("ui final dual-task audit:", "PASS" if final_pass else "FAIL")
