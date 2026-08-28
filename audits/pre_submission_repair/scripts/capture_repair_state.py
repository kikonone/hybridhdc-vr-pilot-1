from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_ROOT = PROJECT_ROOT / "audits" / "pre_submission_repair"
EXPERIMENTS = PROJECT_ROOT / "experiments"
PRIMARY = EXPERIMENTS / "phase_03_multimodal_dataset_labeling" / "data" / "primary_without_performance.csv"
FOLDS = EXPERIMENTS / "phase_03_multimodal_dataset_labeling" / "data" / "fold_assignments.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "last_write_time_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
    }


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def phase_directories() -> list[Path]:
    return sorted(path for path in EXPERIMENTS.glob("phase_*") if path.is_dir() and path.name.startswith(tuple(f"phase_{index:02d}" for index in range(10))))


def freeze_and_manifest_files() -> tuple[list[Path], list[Path]]:
    freezes: list[Path] = []
    manifests: list[Path] = []
    for phase in phase_directories():
        freezes.extend(path for path in phase.rglob("*freeze*.json") if path.is_file())
        manifests.extend(path for path in phase.rglob("*manifest*.json") if path.is_file() and "final" in path.name.lower())
    return sorted(set(freezes)), sorted(set(manifests))


def scientific_files() -> dict[str, list[dict[str, Any]]]:
    categories: dict[str, list[Path]] = {
        "primary_data": [PRIMARY],
        "frozen_folds": [FOLDS],
        "predictions": [],
        "canonical_oof": [],
        "statistics": [],
        "notebooks": [],
        "frozen_model_configs": [],
    }
    for phase in phase_directories():
        categories["notebooks"].extend(phase.glob("*.ipynb"))
        for path in phase.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(phase).as_posix().lower()
            name = path.name.lower()
            if "/predictions/" in f"/{relative}" or ("oof" in name and path.suffix.lower() == ".csv"):
                categories["predictions"].append(path)
            if "canonical" in name and "oof" in name:
                categories["canonical_oof"].append(path)
            if any(token in relative for token in ("statistics", "bootstrap", "pairwise")) and path.suffix.lower() in {".csv", ".json"}:
                categories["statistics"].append(path)
            if path.parent.name == "configs" and any(token in name for token in ("best_", "selected_model", "selection_rule")):
                categories["frozen_model_configs"].append(path)
    return {key: [record(path) for path in sorted(set(paths)) if path.is_file()] for key, paths in categories.items()}


def phase04b_script_hashes() -> list[dict[str, Any]]:
    phase = EXPERIMENTS / "phase_04b_traditional_regression_baselines"
    return [record(path) for path in sorted(phase.rglob("*.py"))]


def phase06_status() -> dict[str, Any]:
    phase = EXPERIMENTS / "phase_06_hdc_variant_screening"
    targets = [
        phase / "configs" / "phase06_freeze.json",
        phase / "manifests" / "phase06_final_artifact_manifest.json",
        phase / "audits" / "phase06_final_artifact_audit.json",
        phase / "audits" / "phase06_model_selection_resolution_audit.json",
    ]
    payload: dict[str, Any] = {"files": [record(path) for path in targets if path.is_file()]}
    for path in targets:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                payload[path.name] = {key: data.get(key) for key in ("status", "phase_status", "artifact_count", "final_manifest_sha256") if key in data}
            except json.JSONDecodeError as exc:
                payload[path.name] = {"parse_error": str(exc)}
    return payload


def capture(label: str) -> dict[str, Any]:
    freezes, manifests = freeze_and_manifest_files()
    git_status = run_git("status", "--short", "--ignored")
    tracked = run_git("ls-files")
    ignored = run_git("ls-files", "--others", "--ignored", "--exclude-standard")
    return {
        "schema_version": 1,
        "label": label,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python": {"version": sys.version, "executable": sys.executable},
        "git": {
            "is_repository": run_git("rev-parse", "--is-inside-work-tree").stdout.strip() == "true",
            "status_short_ignored": git_status.stdout,
            "status_stderr": git_status.stderr,
            "tracked_file_count": len([line for line in tracked.stdout.splitlines() if line]),
            "ignored_file_count": len([line for line in ignored.stdout.splitlines() if line]),
        },
        "checksums": {"primary": record(PRIMARY), "frozen_folds": record(FOLDS)},
        "freeze_files": [record(path) for path in freezes],
        "final_manifests": [record(path) for path in manifests],
        "phase04b_scripts": phase04b_script_hashes(),
        "phase06": phase06_status(),
        "scientific_artifacts": scientific_files(),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: capture_repair_state.py <label> <output-json>")
    label, output = sys.argv[1], Path(sys.argv[2])
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    if AUDIT_ROOT not in output.resolve().parents:
        raise SystemExit("output must be inside audits/pre_submission_repair")
    payload = capture(label)
    write_json(output, payload)
    if label == "pre_repair":
        write_json(
            AUDIT_ROOT / "pre_repair_freeze_hashes.json",
            {
                "captured_at_utc": payload["captured_at_utc"],
                "freeze_files": payload["freeze_files"],
                "final_manifests": payload["final_manifests"],
                "primary": payload["checksums"]["primary"],
                "frozen_folds": payload["checksums"]["frozen_folds"],
            },
        )
    print(json.dumps({"output": str(output), "freeze_files": len(payload["freeze_files"]), "final_manifests": len(payload["final_manifests"]), "scientific_counts": {key: len(value) for key, value in payload["scientific_artifacts"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
