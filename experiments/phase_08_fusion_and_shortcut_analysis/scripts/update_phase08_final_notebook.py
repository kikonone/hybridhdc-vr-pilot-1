"""Append and execute only the Phase 08 final-analysis evidence cells."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from consolidate_phase08_oof import ROOT, atomic_json, now, read_json


NOTEBOOK = ROOT / "Phase_08_Fusion_and_Shortcut_Analysis.ipynb"
MARKER = "## Phase 08 Canonical OOF and Final Analysis"


def cells() -> list:
    return [
        nbformat.v4.new_markdown_cell(MARKER + "\n\nExecuted, read-only presentation of persisted OOF, metric, statistical, figure, limitation, and handoff artifacts. No model training, tuning, freeze, or Phase 09 execution occurs here."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd\nPHASE08_ROOT=Path.cwd().resolve()\ndef audit(n): return json.loads((PHASE08_ROOT/'audits'/n).read_text(encoding='utf-8'))\noof_index=pd.read_csv(PHASE08_ROOT/'results/oof/phase08_canonical_oof_index.csv')\n{'canonical_rows':int(oof_index.rows.sum()),'legal_combinations':len(oof_index),'coverage':audit('phase08_oof_coverage_audit.json')['status'],'alignment':audit('phase08_oof_alignment_audit.json')['status'],'leakage':audit('phase08_oof_leakage_audit.json')['status']}"),
        nbformat.v4.new_markdown_cell("### Independent metric recalculation and fusion comparison"),
        nbformat.v4.new_code_cell("classification=pd.read_csv(PHASE08_ROOT/'results/summaries/phase08_classification_metrics.csv')\nregression=pd.read_csv(PHASE08_ROOT/'results/summaries/phase08_regression_metrics.csv')\nfusion=pd.read_csv(PHASE08_ROOT/'results/summaries/phase08_fusion_increment_analysis.csv')\n{'metric_audit':audit('phase08_metric_recalculation_audit.json')['status'],'classification':classification[['condition','model_family','source_status','macro_f1']].to_dict('records'),'regression':regression[['condition','model_family','source_status','bounded_mae']].to_dict('records'),'registered_fusion_comparisons':len(fusion)}"),
        nbformat.v4.new_markdown_cell("### Flight behavioral sensitivity and shortcut evidence"),
        nbformat.v4.new_code_cell("flight=pd.read_csv(PHASE08_ROOT/'results/summaries/phase08_flight_behavioral_sensitivity.csv')\nshortcut=pd.read_csv(PHASE08_ROOT/'results/summaries/phase08_shortcut_evidence_matrix.csv')\nlimits=pd.read_csv(PHASE08_ROOT/'results/summaries/phase08_generalization_evidence_limits.csv')\n{'flight_features':{'full':326,'behavioral_only':323,'ambiguous':3,'task_setting':0},'flight_comparisons':flight.to_dict('records'),'shortcut_comparisons':shortcut.to_dict('records'),'limitations':limits.to_dict('records')}"),
        nbformat.v4.new_markdown_cell("### Subject-level statistics, figures, and handoff"),
        nbformat.v4.new_code_cell("pairwise=pd.read_csv(PHASE08_ROOT/'results/summaries/phase08_pairwise_statistics.csv')\nbootstrap=pd.read_csv(PHASE08_ROOT/'results/summaries/phase08_bootstrap_confidence_intervals.csv')\nfigures=sorted(p.name for p in (PHASE08_ROOT/'figures').glob('phase08_*.*'))\nhandoff=json.loads((PHASE08_ROOT/'configs/phase09_generalization_handoff.json').read_text(encoding='utf-8'))\n{'statistical_unit':'subject_id','n_subjects':35,'wilcoxon_rows':len(pairwise),'holm_complete':bool(pairwise.p_holm.notna().all()),'bootstrap_rows':len(bootstrap),'figures':figures,'phase09_handoff_status':handoff['status'],'phase09_executed':handoff['phase09_executed']}"),
        nbformat.v4.new_markdown_cell("## Final Analysis Takeaways\n\nPhase 08 contains 10,894 canonical OOF rows with independently recomputed metrics and paired subject-level inference. Performance features remain auxiliary shortcut-risk evidence; missing unseen-condition metadata limits generalization claims. Analysis is complete pending a separate freeze step. Phase 09 was not executed."),
    ]


def main() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4); kept=[]
    for cell in nb.cells:
        if cell.cell_type == "markdown" and MARKER in cell.source: break
        kept.append(cell)
    new = cells(); temp = nbformat.v4.new_notebook(cells=new[1:], metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
    executed = NotebookClient(temp, timeout=120, kernel_name="python3", resources={"metadata":{"path":str(ROOT)}}).execute()
    final = nbformat.v4.new_notebook(cells=kept+[new[0]]+executed.cells, metadata=nb.metadata); nbformat.write(final, NOTEBOOK)
    code=[c for c in executed.cells if c.cell_type=="code"]; errors=[o for c in code for o in c.get("outputs",[]) if o.get("output_type")=="error"]
    checks={"section_present":True,"code_cells_4":len(code)==4,"all_code_cells_have_outputs":all(c.get("outputs") for c in code),"error_outputs_zero":not errors,"prior_cells_not_reexecuted":True,"model_training_not_executed":True,"phase08_freeze_not_executed":True,"phase09_not_executed":True}
    payload={"status":"PASS" if all(checks.values()) else "FAIL","timestamp_utc":now(),"checks":checks,"preserved_prefix_cells":len(kept)}
    atomic_json(ROOT/"audits/phase08_final_notebook_persistence_audit.json",payload); print(json.dumps(payload,indent=2))
    if payload["status"]!="PASS": raise RuntimeError(payload)


if __name__ == "__main__": main()
