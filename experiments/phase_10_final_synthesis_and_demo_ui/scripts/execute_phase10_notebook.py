from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import traceback
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
NOTEBOOK = BASE / "Phase_10_Final_Synthesis_and_Demo_UI.ipynb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_reproducibility_notebook_index(executed_cells: int) -> None:
    index = BASE / "reproducibility_package" / "notebook_index.csv"
    if not index.exists():
        return
    with index.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fieldnames = ["phase", "path", "sha256", "code_cells", "executed_code_cells"]
    target = str(NOTEBOOK.resolve())
    updated = False
    for row in rows:
        if row.get("path") == target:
            row.update({"phase":"10","sha256":sha256(NOTEBOOK),"code_cells":str(executed_cells),"executed_code_cells":str(executed_cells)})
            updated = True
    if not updated:
        rows.append({"phase":"10","path":target,"sha256":sha256(NOTEBOOK),"code_cells":str(executed_cells),"executed_code_cells":str(executed_cells)})
    with index.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8-sig"))
    namespace = {"__name__": "__phase10_notebook__"}
    execution_count = 0
    failed = False
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        execution_count += 1
        source = "".join(cell.get("source", []))
        stdout = io.StringIO()
        outputs = []
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
                exec(compile(source, f"<phase10-cell-{execution_count}>", "exec"), namespace, namespace)
        except Exception as exc:  # persisted as a standard notebook error output
            failed = True
            outputs.append({
                "output_type": "error",
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "traceback": traceback.format_exc().splitlines(),
            })
        text = stdout.getvalue()
        if text:
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": text.splitlines(True)})
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs
    notebook.setdefault("metadata", {})["phase10_execution"] = {
        "executor": "scripts/execute_phase10_notebook.py",
        "mode": "SEQUENTIAL_STANDARD_LIBRARY_READ_ONLY",
        "code_cells_executed": execution_count,
        "failed": failed,
    }
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    refresh_reproducibility_notebook_index(execution_count)
    print(json.dumps(notebook["metadata"]["phase10_execution"], indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
