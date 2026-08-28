from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path


PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "logs"))

from finalize_phase04b import main  # noqa: E402


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def test_phase04b_finalization_writes_only_to_temp_directory(tmp_path: Path) -> None:
    production_before = tree_hashes(PHASE)
    isolated = tmp_path / PHASE.name
    shutil.copytree(PHASE, isolated, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    main(phase_dir=isolated)

    assert (isolated / "configs/phase04b_freeze.json").is_file()
    assert (isolated / "manifests/phase04b_final_artifact_manifest.json").is_file()
    assert tree_hashes(PHASE) == production_before
