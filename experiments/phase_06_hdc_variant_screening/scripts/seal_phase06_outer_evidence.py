"""Create or verify the immutable outer-evidence seal used by the inner-only selector."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
PHASE05 = ROOT / "experiments/phase_05_basic_dual_output_hdc"
SEAL = PHASE / "manifests/phase06_preselection_outer_oof_seal.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def evidence_paths() -> list[tuple[str, Path]]:
    records: set[tuple[str, Path]] = set()
    for path in (PHASE05 / "results/oof").glob("*.csv"): records.add(("phase05_outer_oof", path))
    for path in (PHASE / "results/oof").glob("*.csv"): records.add(("phase06_outer_oof", path))
    for path in (PHASE05 / "results/predictions").glob("*final_confirmation*.csv"): records.add(("phase05_final_predictions", path))
    for path in (PHASE / "results/predictions").glob("*final_confirmation*.csv"): records.add(("phase06_final_predictions", path))
    for path in (PHASE05 / "results/fold_metrics").glob("*final_confirmation_fold_*_metrics.csv"): records.add(("phase05_outer_fold_metrics", path))
    for pattern in ["*_final_confirmation_fold_*_classification_metrics.csv", "*_final_confirmation_fold_*_similarity_regression_metrics.csv"]:
        for path in (PHASE / "results/fold_metrics").glob(pattern): records.add(("phase06_outer_fold_metrics", path))
    for path in (PHASE05 / "results/summaries").glob("*oof_metrics_by_config.csv"): records.add(("phase05_outer_oof_metrics", path))
    for name in ["phase06_classification_metrics_by_config.csv", "phase06_similarity_regression_metrics_by_config.csv", "phase06_common_ridge_metrics_by_config.csv"]:
        path = PHASE / "results/summaries" / name
        if path.exists(): records.add(("phase06_outer_oof_metrics", path))
    return sorted(records, key=lambda item: str(item[1]))


def record(category: str, path: Path) -> dict[str, Any]:
    return {"project_relative_path": str(path.relative_to(ROOT)), "category": category, "file_size_bytes": path.stat().st_size, "sha256": sha256(path), "status": "SEALED"}


def create() -> int:
    if SEAL.exists(): raise RuntimeError("Preselection outer-evidence seal already exists; use --verify")
    best_paths = [PHASE / "configs/phase06_best_classification_hdc.json", PHASE / "configs/phase06_best_regression_hdc.json"]
    if any(path.exists() for path in best_paths): raise RuntimeError("Best-model JSON exists before outer-evidence sealing")
    artifacts = [record(category, path) for category, path in evidence_paths()]
    payload = {"phase": "06", "seal": "preselection_outer_oof_and_final_confirmation_evidence", "created_before_model_selection": True, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "artifact_count": len(artifacts), "artifacts": artifacts, "best_model_json_present_at_seal_time": False, "self_hash_excluded": True, "result": "PASS"}
    atomic_json(SEAL, payload)
    print(f"OUTER EVIDENCE SEALED: {len(artifacts)} artifacts")
    return 0


def verify() -> int:
    seal = json.loads(SEAL.read_text(encoding="utf-8")); mismatches = []
    for item in seal["artifacts"]:
        path = ROOT / item["project_relative_path"]
        if not path.exists() or path.stat().st_size != item["file_size_bytes"] or sha256(path) != item["sha256"]: mismatches.append(item["project_relative_path"])
    audit = {"phase": "06", "audit": "outer_oof_seal_integrity", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seal": str(SEAL), "seal_sha256": sha256(SEAL), "artifacts_verified": len(seal["artifacts"]), "mismatches": mismatches, "outer_oof_read_by_selector": False, "result": "PASS" if not mismatches else "FAIL"}
    atomic_json(PHASE / "audits/phase06_outer_oof_seal_integrity_audit.json", audit)
    print(f"OUTER EVIDENCE SEAL VERIFY: {audit['result']}")
    return 0 if not mismatches else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--verify", action="store_true")
    return verify() if parser.parse_args().verify else create()


if __name__ == "__main__": raise SystemExit(main())
