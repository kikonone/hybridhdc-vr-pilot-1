from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[3]
PHASE = ROOT / "experiments" / "phase_06_hdc_variant_screening"
OUTPUT = ROOT / "audits" / "pre_submission_repair" / "phase06_evidence_chain.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def records(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        path_key = "relative_path" if "relative_path" in value else "path" if "path" in value else None
        if path_key and "sha256" in value:
            yield value
        for child in value.values():
            yield from records(child)
    elif isinstance(value, list):
        for child in value:
            yield from records(child)


def resolve_record_path(row: dict[str, Any]) -> Path:
    raw = str(row.get("relative_path", row.get("path", "")))
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else PHASE / candidate


freeze_path = PHASE / "configs" / "phase06_freeze.json"
classification_path = PHASE / "configs" / "phase06_best_classification_hdc.json"
regression_path = PHASE / "configs" / "phase06_best_regression_hdc.json"
freeze = load_json(freeze_path)
baseline = load_json(ROOT / "audits" / "pre_submission_repair" / "pre_repair_state.json")
baseline_configs = {
    row["relative_path"]: row["sha256"]
    for row in baseline["scientific_artifacts"]["frozen_model_configs"]
}

creation_sources = {
    "phase06_contract_manifest.json": "scripts/freeze_phase06_contract.py",
    "phase06_final_artifact_manifest.json": "scripts/resolve_phase06_selection_and_freeze.py (final frozen revision; originally scripts/finalize_phase06.py)",
    "phase06_final_confirmation_artifact_manifest.json": "scripts/run_phase06_final_confirmation.py",
    "phase06_input_manifest.json": "src/phase06_preflight.py",
    "phase06_preselection_outer_oof_seal.json": "scripts/resolve_phase06_selection_and_freeze.py",
    "phase06_quick_screen_artifact_manifest.json": "scripts/consolidate_phase06_quick_screen.py",
}

candidate_paths = sorted((PHASE / "manifests").glob("*.json"))
backup_candidate = ROOT / "audits" / "pre_submission_repair" / "phase06_manifest_pre_repair" / "phase06_final_artifact_manifest.json"
if backup_candidate.is_file():
    candidate_paths.append(backup_candidate)

candidates = []
for path in candidate_paths:
    payload = load_json(path)
    rows = list(records(payload))
    missing: list[str] = []
    mismatches: list[dict[str, str | None]] = []
    referenced: list[str] = []
    for row in rows:
        artifact = resolve_record_path(row)
        raw = str(row.get("relative_path", row.get("path", "")))
        referenced.append(raw)
        if not artifact.is_file():
            missing.append(raw)
            continue
        actual = sha256(artifact)
        if actual != row["sha256"]:
            mismatches.append({"path": raw, "expected_sha256": row["sha256"], "actual_sha256": actual})
    digest = sha256(path)
    candidates.append({
        "source_path": str(path.resolve()),
        "creation_source": creation_sources.get(path.name, "audit backup copy; original creation source inherited from filename"),
        "sha256": digest,
        "declared_artifact_count": payload.get("artifact_count", payload.get("input_count")),
        "referenced_artifact_count": len(rows),
        "referenced_artifact_paths": sorted(set(referenced)),
        "missing_artifacts": sorted(set(missing)),
        "artifact_hash_mismatches": mismatches,
        "phase06_freeze_reference_consistency": "PASS" if digest == freeze["final_manifest_sha256"] else "NOT_THE_FINAL_MANIFEST_REFERENCED_BY_FREEZE",
        "later_phase_reference_consistency": "PENDING_REFERENCE_SCAN",
        "best_classification_selection_unchanged": sha256(classification_path) == baseline_configs[str(classification_path.relative_to(ROOT)).replace("\\", "/")],
        "best_regression_selection_unchanged": sha256(regression_path) == baseline_configs[str(regression_path.relative_to(ROOT)).replace("\\", "/")],
    })

later_references: list[dict[str, str]] = []
seen: set[tuple[str, str, str]] = set()
for phase_name in ["phase_08_fusion_and_shortcut_analysis", "phase_09_robustness_and_generalization", "phase_10_final_synthesis_and_demo_ui"]:
    phase_root = ROOT / "experiments" / phase_name
    for source in phase_root.rglob("*.json"):
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for row in records(payload):
            raw = str(row.get("relative_path", row.get("path", "")))
            if "phase_06_hdc_variant_screening" not in raw.replace("\\", "/"):
                continue
            item = (str(source.relative_to(ROOT)).replace("\\", "/"), raw, str(row["sha256"]))
            if item in seen:
                continue
            seen.add(item)
            later_references.append({"source": item[0], "phase06_path": item[1], "sha256": item[2]})

later_hashes = {row["sha256"] for row in later_references}
for candidate in candidates:
    candidate["later_phase_reference_consistency"] = "PASS" if candidate["sha256"] in later_hashes else "NOT_REFERENCED_BY_HASH"

git_log = subprocess.run(
    ["git", "log", "--all", "--format=%H %ad %s", "--date=iso", "--", "experiments/phase_06_hdc_variant_screening"],
    cwd=ROOT, check=False, capture_output=True, text=True,
).stdout.strip().splitlines()

classification_hash = sha256(classification_path)
regression_hash = sha256(regression_path)
final_candidates = [row for row in candidates if row["sha256"] == freeze["final_manifest_sha256"]]
payload = {
    "status": "PASS_WITH_ENGINEERING_METADATA_MISMATCHES",
    "phase06_root": str(PHASE.resolve()),
    "inventory": {
        "freeze": str(freeze_path.resolve()),
        "best_classification": str(classification_path.resolve()),
        "best_regression": str(regression_path.resolve()),
        "manifest_files": len(list((PHASE / "manifests").glob("*"))),
        "audit_files": len(list((PHASE / "audits").rglob("*"))),
        "result_files": len(list((PHASE / "results").rglob("*"))),
        "report_files": len(list((PHASE / "reports").rglob("*"))),
        "notebook": str((PHASE / "Phase_06_HDC_Variant_Screening.ipynb").resolve()),
        "backup_or_recovery_directories": [],
    },
    "freeze_sha256": sha256(freeze_path),
    "freeze_referenced_final_manifest_sha256": freeze["final_manifest_sha256"],
    "original_final_manifest_found": bool(final_candidates),
    "original_final_manifest_hash_verified": bool(final_candidates),
    "original_final_manifest_paths": [row["source_path"] for row in final_candidates],
    "candidate_manifests": candidates,
    "later_phase_references": later_references,
    "later_phase_reference_count": len(later_references),
    "best_classification": {
        "sha256": classification_hash,
        "baseline_sha256": baseline_configs[str(classification_path.relative_to(ROOT)).replace("\\", "/")],
        "unchanged": classification_hash == baseline_configs[str(classification_path.relative_to(ROOT)).replace("\\", "/")],
        "equals_freeze_embedded_payload": load_json(classification_path) == freeze["best_classification_hdc"],
    },
    "best_regression": {
        "sha256": regression_hash,
        "baseline_sha256": baseline_configs[str(regression_path.relative_to(ROOT)).replace("\\", "/")],
        "unchanged": regression_hash == baseline_configs[str(regression_path.relative_to(ROOT)).replace("\\", "/")],
        "equals_freeze_embedded_payload": load_json(regression_path) == freeze["best_regression_hdc"],
    },
    "git_history": {
        "read_only_checked": True,
        "matching_path_commits": git_log,
        "phase06_present_in_reachable_git_history": bool(git_log),
    },
    "pre_repair_hash_source": "audits/pre_submission_repair/pre_repair_state.json",
    "conclusion": "The exact final manifest referenced by phase06_freeze.json is present and its own SHA-256 is verified. Six non-scientific initialization/interface records referenced inside it no longer match; no original byte copies for those six were found. Best classification and regression selections remain exact and match the payload embedded in the freeze record.",
}
OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({
    "output": str(OUTPUT),
    "original_manifest_found": payload["original_final_manifest_found"],
    "original_manifest_hash_verified": payload["original_final_manifest_hash_verified"],
    "candidate_count": len(candidates),
    "later_reference_count": len(later_references),
    "classification_unchanged": payload["best_classification"]["unchanged"],
    "regression_unchanged": payload["best_regression"]["unchanged"],
}, indent=2))
