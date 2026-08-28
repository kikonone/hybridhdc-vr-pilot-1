"""CLI for Phase 06 preflight generation and post-Notebook finalization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PHASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_DIR / "src"))

from phase06_preflight import finalize_after_notebook, run_preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-notebook", action="store_true")
    args = parser.parse_args()
    result = finalize_after_notebook(PHASE_DIR) if args.finalize_notebook else run_preflight(PHASE_DIR)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if "FAIL" not in json.dumps(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
