from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
NOTEBOOK = BASE / "Phase_10_Final_Synthesis_and_Demo_UI.ipynb"
MARKER = "CORE_FINAL_SYNTHESIS_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_sha256(cell: dict) -> str:
    payload = json.dumps(cell, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def append_cells(notebook: dict) -> tuple[int, list[str]]:
    original_count = len(notebook["cells"])
    original_hashes = [cell_sha256(cell) for cell in notebook["cells"]]
    if any(cell.get("metadata", {}).get("phase10_stage") == MARKER for cell in notebook["cells"]):
        return original_count, original_hashes
    sections = [
        ("Core Final Synthesis runtime", """from pathlib import Path\nimport csv, hashlib, json\nPHASE10 = Path.cwd()\ndef digest(path):\n    h=hashlib.sha256()\n    with path.open('rb') as f:\n        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)\n    return h.hexdigest()\ndef read_json(relative): return json.loads((PHASE10/relative).read_text(encoding='utf-8-sig'))\ndef count_csv(relative):\n    with (PHASE10/relative).open('r',encoding='utf-8-sig',newline='') as f: return sum(1 for _ in csv.DictReader(f))\nprint({'chapter':'Core Final Synthesis','phase10':str(PHASE10)})"""),
        ("Frozen Core Contract verification", """freeze=read_json('configs/phase10_core_contract_freeze.json')\nprint({'status':freeze['status'],'contracts':len(freeze['contracts']),'ready':freeze['ready_for_phase10_final_synthesis']})"""),
        ("Final prediction library index and source audit", """idx='results/final_prediction_library/final_prediction_library_index.csv'\nmanifest=read_json('results/final_prediction_library/final_prediction_library_manifest.json')\nprint({'rows':count_csv(idx),'sources_verified':manifest['source_count'],'predictions_generated':manifest['predictions_generated']})"""),
        ("Final statistics bundle index and source audit", """manifest=read_json('results/final_statistics_bundle/final_statistics_manifest.json')\nprint({'artifacts':manifest['artifact_count'],'effects':manifest['effect_size_artifacts'],'cis':manifest['confidence_interval_artifacts'],'tests':manifest['statistical_test_artifacts'],'recomputed':manifest['statistics_recomputed']})"""),
        ("Paper table registry and source map", """audit=read_json('audits/phase10_paper_table_audit.json')\nprint({'tables':count_csv('reports/paper_tables/paper_table_registry.csv'),'sources':audit['source_artifacts_verified'],'recomputed':audit['numeric_values_recomputed']})"""),
        ("Paper figure registry and source map", """audit=read_json('audits/phase10_paper_figure_audit.json')\nprint({'figures':count_csv('reports/paper_figures/paper_figure_registry.csv'),'redrawn':audit['figures_redrawn'],'format_copies':audit['format_copies_created']})"""),
        ("RQ—experiment—evidence—conclusion matrix", """audit=read_json('audits/phase10_rq_evidence_audit.json')\nprint({'rq_rows':count_csv('results/summaries/phase10_rq_experiment_evidence_conclusion_matrix.csv'),'boundary_checks':audit['status']})"""),
        ("Reproducibility checksum verification", """checks=read_json('reproducibility/checksum_verification.json')\nprint({'verified':checks['verified_count'],'artifacts':checks['artifact_count'],'failures':len(checks['failures'])})"""),
        ("Cross-phase numerical consistency", """audit=read_json('audits/phase10_cross_phase_numerical_consistency_audit.json')\nprint({'status':audit['status'],'scientific_conflicts':audit['scientific_source_conflicts'],'unresolved_differences':audit['unresolved_numerical_differences']})"""),
        ("Core Final Synthesis terminal state", """status=read_json('configs/phase10_final_synthesis_status.json')\nprint({'status':status['status'],'phase10_final_frozen':status['phase10_final_frozen'],'ui':status['ui_status'],'onlinehd':status['onlinehd_replay_status']})"""),
    ]
    for title, code in sections:
        metadata = {"phase10_stage": MARKER}
        notebook["cells"].extend([
            {"cell_type": "markdown", "metadata": metadata, "source": [f"## {title}\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": metadata, "outputs": [], "source": [code]},
        ])
    return original_count, original_hashes


def execute_new_cells(notebook: dict) -> tuple[int, bool]:
    namespace = {"__name__": "__phase10_final_synthesis_notebook__"}
    execution_count = max((cell.get("execution_count") or 0 for cell in notebook["cells"] if cell.get("cell_type") == "code"), default=0)
    executed = 0
    failed = False
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code" or cell.get("metadata", {}).get("phase10_stage") != MARKER:
            continue
        execution_count += 1
        executed += 1
        stdout = io.StringIO()
        outputs = []
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
                exec(compile("".join(cell.get("source", [])), f"<phase10-final-synthesis-cell-{executed}>", "exec"), namespace, namespace)
        except Exception as exc:
            failed = True
            outputs.append({"output_type": "error", "ename": type(exc).__name__, "evalue": str(exc), "traceback": traceback.format_exc().splitlines()})
        text = stdout.getvalue()
        if text:
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": text.splitlines(True)})
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs
    return executed, failed


def update_registry() -> None:
    path = BASE / "reproducibility/notebook_registry.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(handle.seek(0) or csv.DictReader(handle).fieldnames or [])
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8-sig"))
    code = [cell for cell in data["cells"] if cell.get("cell_type") == "code"]
    record = {"phase": "10", "path": str(NOTEBOOK.resolve()), "sha256": sha256(NOTEBOOK), "code_cells": str(len(code)), "executed_code_cells": str(sum(cell.get("execution_count") is not None for cell in code)), "error_outputs": str(sum(any(out.get("output_type") == "error" for out in cell.get("outputs", [])) for cell in code))}
    target = str(NOTEBOOK.resolve())
    rows = [row for row in rows if row["path"] != target] + [record]
    fields = ["phase", "path", "sha256", "code_cells", "executed_code_cells", "error_outputs"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8-sig"))
    marker_present = any(cell.get("metadata", {}).get("phase10_stage") == MARKER for cell in notebook["cells"])
    if marker_present:
        original_count = next((i for i, cell in enumerate(notebook["cells"]) if cell.get("metadata", {}).get("phase10_stage") == MARKER), len(notebook["cells"]))
        original_hashes = [cell_sha256(cell) for cell in notebook["cells"][:original_count]]
    else:
        original_count, original_hashes = append_cells(notebook)
    snapshot = {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "original_cell_count": original_count, "original_cell_hashes": original_hashes, "marker": MARKER}
    (BASE / "logs/phase10_notebook_pre_final_synthesis_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    executed, failed = execute_new_cells(notebook)
    notebook.setdefault("metadata", {})["phase10_final_synthesis_execution"] = {"marker": MARKER, "new_code_cells_executed": executed, "failed": failed, "historical_cells_reexecuted": False}
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    update_registry()
    print(json.dumps(notebook["metadata"]["phase10_final_synthesis_execution"], indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
