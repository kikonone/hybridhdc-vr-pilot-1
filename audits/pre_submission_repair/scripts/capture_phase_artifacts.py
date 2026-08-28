from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "experiments"
AUDIT_ROOT = ROOT / "audits" / "pre_submission_repair"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: capture_phase_artifacts.py <label> <output-json>")
    label = sys.argv[1]
    output = Path(sys.argv[2])
    if not output.is_absolute():
        output = ROOT / output
    if AUDIT_ROOT not in output.resolve().parents:
        raise SystemExit("output must be inside audits/pre_submission_repair")
    phases = [
        path for path in sorted(EXPERIMENTS.iterdir())
        if path.is_dir() and any(path.name.startswith(f"phase_{index:02d}") for index in range(10))
    ]
    records = []
    for phase in phases:
        for path in sorted(phase.rglob("*")):
            if not path.is_file() or EXCLUDED_PARTS.intersection(path.parts):
                continue
            records.append({
                "relative_path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    payload = {
        "label": label,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(records),
        "artifacts": records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "file_count": len(records)}))


if __name__ == "__main__":
    main()
