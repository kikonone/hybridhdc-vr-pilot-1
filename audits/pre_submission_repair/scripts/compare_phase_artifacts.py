from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AUDIT_ROOT = ROOT / "audits" / "pre_submission_repair"


def load(raw: str) -> dict:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: compare_phase_artifacts.py <before> <after> <output>")
    before, after = load(sys.argv[1]), load(sys.argv[2])
    left = {row["relative_path"]: row for row in before["artifacts"]}
    right = {row["relative_path"]: row for row in after["artifacts"]}
    result = {
        "status": "PASS",
        "before_file_count": len(left),
        "after_file_count": len(right),
        "added": sorted(set(right) - set(left)),
        "removed": sorted(set(left) - set(right)),
        "modified": [
            {"path": path, "before_sha256": left[path]["sha256"], "after_sha256": right[path]["sha256"]}
            for path in sorted(set(left) & set(right))
            if left[path]["sha256"] != right[path]["sha256"]
        ],
    }
    result["production_files_modified_by_tests"] = len(result["added"]) + len(result["removed"]) + len(result["modified"])
    result["status"] = "PASS" if result["production_files_modified_by_tests"] == 0 else "FAIL"
    output = Path(sys.argv[3])
    if not output.is_absolute():
        output = ROOT / output
    if AUDIT_ROOT not in output.resolve().parents:
        raise SystemExit("output must be inside audits/pre_submission_repair")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
