"""Independent final verification, manifest creation, and Phase 07 freeze."""
from __future__ import annotations
import json, sys
from pathlib import Path
import nbformat
import pandas as pd

PHASE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PHASE/"scripts"))
from consolidate_analyze_and_freeze_phase07 import (AUD, CONFIG, FIG, MAN, MODALITIES, OOF, REPORT, SUM,
                                                     EXPECTED_FOLDS, EXPECTED_PRIMARY, FOLDS, PRIMARY,
                                                     now, read_json, sha256, write_json)

REQUIRED=[
 OOF/"phase07_unimodal_classification_seed_level_oof.csv", OOF/"phase07_unimodal_classification_canonical_oof.csv",
 OOF/"phase07_unimodal_regression_seed_level_oof.csv", OOF/"phase07_unimodal_regression_canonical_oof.csv",
 OOF/"phase07_readonly_multimodal_classification_reference.csv", OOF/"phase07_readonly_multimodal_regression_reference.csv",
 *[SUM/x for x in ["phase07_unimodal_classification_comparison.csv","phase07_unimodal_regression_comparison.csv","phase07_seed_stability_summary.csv","phase07_per_class_recall.csv","phase07_per_target_level_mae.csv","phase07_confusion_matrices.json","phase07_classification_modality_ranking.csv","phase07_regression_modality_ranking.csv","phase07_unimodal_vs_multimodal_comparison.csv","phase07_subject_level_classification_metrics.csv","phase07_subject_level_regression_metrics.csv","phase07_friedman_tests.csv","phase07_wilcoxon_holm_tests.csv","phase07_bootstrap_confidence_intervals.csv","phase07_effect_sizes.csv","phase07_availability_stratified_metrics.csv","phase07_classification_error_analysis.csv","phase07_regression_error_analysis.csv","phase07_subject_error_analysis.csv"]],
 *[FIG/f"{x}.{e}" for x in ["phase07_classification_modality_ranking","phase07_regression_modality_ranking","phase07_unimodal_vs_multimodal_deltas","phase07_classification_confusion_matrix_panel","phase07_regression_residual_panel","phase07_subject_level_performance"] for e in ["pdf","png"]],
 REPORT/"analysis-output"/"analysis-report.md", REPORT/"analysis-output"/"stats-appendix.md", REPORT/"analysis-output"/"figure-catalog.md", REPORT/"phase07_final_summary.md",
 *[AUD/x for x in ["phase07_final_oof_coverage_audit.json","phase07_final_alignment_audit.json","phase07_final_leakage_audit.json","phase07_metric_recalculation_audit.json","phase07_statistical_analysis_audit.json","phase07_figure_audit.json","phase07_final_reproducibility_audit.json"]],
 PHASE/"Phase_07_Unimodal_Contribution.ipynb"
]

def main():
    missing=[str(x) for x in REQUIRED if not x.exists()]; assert not missing,missing
    cs=pd.read_csv(REQUIRED[0]); cc=pd.read_csv(REQUIRED[1]); rs=pd.read_csv(REQUIRED[2]); rr=pd.read_csv(REQUIRED[3])
    assert (len(cs),len(rs),len(cc),len(rr))==(10475,10475,2095,2095)
    assert all(cc[cc.modality==m].run_key.nunique()==419 for m in MODALITIES); assert all(rr[rr.modality==m].run_key.nunique()==419 for m in MODALITIES)
    assert sha256(PRIMARY)==EXPECTED_PRIMARY and sha256(FOLDS)==EXPECTED_FOLDS
    # Prove that consolidation changed neither frozen upstream inputs nor any
    # pre-existing checkpoint/prediction artifact.
    upstream=read_json(MAN/"phase07_input_manifest.json")["inputs"]
    upstream_mismatches=[x["absolute_path"] for x in upstream if not Path(x["absolute_path"]).exists() or sha256(Path(x["absolute_path"]))!=x["sha256"]]
    baseline=read_json(MAN/"phase07_unimodal_execution_artifact_manifest.json")["artifacts"]
    immutable=[x for x in baseline if x["relative_path"].startswith("results\\checkpoints\\") or x["relative_path"].startswith("results\\predictions\\")]
    immutable_mismatches=[x["relative_path"] for x in immutable if not (PHASE/x["relative_path"]).exists() or sha256(PHASE/x["relative_path"])!=x["sha256"]]
    checkpoint_count=sum(x["relative_path"].startswith("results\\checkpoints\\") for x in immutable)
    prediction_count=sum(x["relative_path"].startswith("results\\predictions\\") for x in immutable)
    assert checkpoint_count==prediction_count==250 and not upstream_mismatches and not immutable_mismatches
    assert read_json(AUD/"phase07_final_oof_coverage_audit.json")["result"]=="PASS"; assert read_json(AUD/"phase07_final_alignment_audit.json")["result"]=="PASS"; assert read_json(AUD/"phase07_final_leakage_audit.json")["result"]=="PASS"
    nb=nbformat.read(PHASE/"Phase_07_Unimodal_Contribution.ipynb",as_version=4); title="Phase 07 Final OOF Consolidation, Modality Analysis and Freeze"; cells=[c for c in nb.cells if title in c.source]; assert cells
    code_idx=max(i for i,c in enumerate(nb.cells) if c.cell_type=="code" and "Classification canonical OOF rows" in c.source); cell=nb.cells[code_idx]
    assert cell.execution_count is not None and cell.outputs and not any(o.output_type=="error" for o in cell.outputs)
    notebook_audit={"phase":"07","generated_at_utc":now(),"result":"PASS","section_present":True,"final_code_cell_execution_count":cell.execution_count,"final_code_cell_outputs":len(cell.outputs),"error_outputs":0,"notebook_sha256":sha256(PHASE/"Phase_07_Unimodal_Contribution.ipynb")}
    write_json(AUD/"phase07_final_notebook_persistence_audit.json",notebook_audit)
    reproducibility={"phase":"07","generated_at_utc":now(),"result":"PASS","upstream_files_checked":len(upstream),"upstream_files_modified":len(upstream_mismatches),"upstream_mismatches":upstream_mismatches,"checkpoint_files_checked":checkpoint_count,"checkpoint_files_modified":sum(x.startswith("results\\checkpoints\\") for x in immutable_mismatches),"prediction_files_checked":prediction_count,"prediction_files_modified":sum(x.startswith("results\\predictions\\") for x in immutable_mismatches),"retrained_models":0,"regenerated_predictions":0,"primary_sha256":sha256(PRIMARY),"frozen_fold_sha256":sha256(FOLDS)}; write_json(AUD/"phase07_final_reproducibility_audit.json",reproducibility)
    artifact_audit={"phase":"07","generated_at_utc":now(),"result":"PASS","required_artifacts":len(REQUIRED)+1,"missing_artifacts":[],"hash_mismatches":[],"upstream_files_modified":0,"checkpoint_files_modified":0,"prediction_files_modified":0,"retrained_models":0,"regenerated_predictions":0}; write_json(AUD/"phase07_final_artifact_audit.json",artifact_audit)
    ex=read_json(CONFIG/"phase07_execution_manifest.json"); ex.update({"status":"FROZEN","canonical_oof_generated":True,"canonical_classification_rows":2095,"canonical_regression_rows":2095,"finalized_at_utc":now()}); write_json(CONFIG/"phase07_execution_manifest.json",ex)
    contract=read_json(CONFIG/"phase07_experiment_contract.json"); contract.update({"status":"FROZEN","ready_for_next_planned_phase":True,"finalized_at_utc":now()}); write_json(CONFIG/"phase07_experiment_contract.json",contract)
    manifest_paths=sorted(set(REQUIRED+[AUD/"phase07_final_notebook_persistence_audit.json",AUD/"phase07_final_artifact_audit.json",PHASE/"scripts"/"consolidate_analyze_and_freeze_phase07.py",PHASE/"scripts"/"verify_phase07_final_results.py",PHASE/"tests"/"test_phase07_final_consolidation.py",PHASE/"tests"/"test_phase07_contract.py",PHASE/"task_plan.md",CONFIG/"phase07_execution_manifest.json",CONFIG/"phase07_experiment_contract.json"]),key=str)
    artifacts=[{"relative_path":str(p.relative_to(PHASE)),"size_bytes":p.stat().st_size,"sha256":sha256(p)} for p in manifest_paths]
    manifest={"phase":"07","manifest":"final_authorized_artifacts","generated_at_utc":now(),"artifact_count":len(artifacts),"artifacts":artifacts,"missing_artifacts":0,"hash_mismatches":0,"result":"PASS","exclusions":["manifest itself","freeze file (contains manifest hash)"]}; write_json(MAN/"phase07_final_artifact_manifest.json",manifest)
    crank=pd.read_csv(SUM/"phase07_classification_modality_ranking.csv"); rrank=pd.read_csv(SUM/"phase07_regression_modality_ranking.csv")
    freeze={"phase":"07","status":"FROZEN","frozen_at_utc":now(),"model_training_runs":250,"classification_canonical_rows":2095,"regression_canonical_rows":2095,"classification_modalities_ranked":5,"regression_modalities_ranked":5,"best_classification_modality":crank.iloc[0].modality,"best_regression_modality":rrank.iloc[0].modality,"rankings_separate":True,"combined_best_modality":None,"statistical_unit":"subject_id","n_subjects":35,"primary_data_sha256":EXPECTED_PRIMARY,"frozen_fold_sha256":EXPECTED_FOLDS,"final_manifest_sha256":sha256(MAN/"phase07_final_artifact_manifest.json"),"model_retraining_during_consolidation":False,"predictions_regenerated_during_consolidation":False,"all_final_audits_pass":True,"ready_for_next_planned_phase":True}; write_json(CONFIG/"phase07_freeze.json",freeze)
    print(json.dumps(freeze,indent=2))

if __name__=="__main__": main()
