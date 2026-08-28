from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_ROOT = PROJECT_ROOT / "audits" / "pre_submission_repair"


def load(path: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return json.loads(candidate.read_text(encoding="utf-8"))


def index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["relative_path"]: record for record in records}


def differences(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, list[Any]]:
    left, right = index(before), index(after)
    return {
        "added": sorted(set(right) - set(left)),
        "removed": sorted(set(left) - set(right)),
        "modified": [
            {"relative_path": path, "before_sha256": left[path]["sha256"], "after_sha256": right[path]["sha256"]}
            for path in sorted(set(left) & set(right))
            if left[path]["sha256"] != right[path]["sha256"]
        ],
    }


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    sections: dict[str, Any] = {
        "freeze_files": differences(before["freeze_files"], after["freeze_files"]),
        "final_manifests": differences(before["final_manifests"], after["final_manifests"]),
        "phase04b_scripts": differences(before["phase04b_scripts"], after["phase04b_scripts"]),
        "phase06_files": differences(before["phase06"]["files"], after["phase06"]["files"]),
    }
    for category, records in before["scientific_artifacts"].items():
        sections[f"scientific_artifacts.{category}"] = differences(records, after["scientific_artifacts"][category])
    modified_paths = sorted({
        item["relative_path"]
        for section in sections.values()
        for item in section["modified"]
    })
    return {
        "before_label": before["label"],
        "after_label": after["label"],
        "sections": sections,
        "modified_paths": modified_paths,
        "modified_path_count": len(modified_paths),
        "primary_checksum_unchanged": before["checksums"]["primary"]["sha256"] == after["checksums"]["primary"]["sha256"],
        "frozen_folds_checksum_unchanged": before["checksums"]["frozen_folds"]["sha256"] == after["checksums"]["frozen_folds"]["sha256"],
    }


def main() -> int:
    if len(sys.argv) not in {3, 4}:
        raise SystemExit("usage: compare_repair_states.py <before-json> <after-json> [output-json]")
    result = compare(load(sys.argv[1]), load(sys.argv[2]))
    if len(sys.argv) == 4:
        output = Path(sys.argv[3])
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        if AUDIT_ROOT not in output.resolve().parents:
            raise SystemExit("output must be inside audits/pre_submission_repair")
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
