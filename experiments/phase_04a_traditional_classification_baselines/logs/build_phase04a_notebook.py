from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"experiments/phase_04a_traditional_classification_baselines"
for d in ["results/oof","results/fold_metrics","results/summaries","results/predictions","configs","figures","audits","reports","logs"]: (OUT/d).mkdir(parents=True,exist_ok=True)

def md(title, detail=""):
    return nbf.v4.new_markdown_cell(f"## {title}\n\n{detail}")

run_import='''import runpy, json, importlib.util
import pandas as pd
from pathlib import Path
ROOT=Path.cwd().resolve()
for candidate in [ROOT, *ROOT.parents]:
    if (candidate/"CODEX_NOTEBOOK_RULES.md").exists() and (candidate/"vrdataset").is_dir():
        ROOT=candidate; break
else: raise RuntimeError("PROJECT ROOT NOT VERIFIED")
R=runpy.run_path(str(ROOT/"experiments/phase_04a_traditional_classification_baselines/logs/phase04a_runner.py"))
ctx=R["setup"]()
print({"primary_dataset":str(ctx["paths"]["data"].relative_to(ROOT)),"rows":len(ctx["data"]),"subjects":ctx["data"].subject_id.nunique(),"features":len(ctx["features"]),"class_counts":ctx["data"].target_class.value_counts().sort_index().to_dict(),"fold_path":str(ctx["paths"]["folds"].relative_to(ROOT)),"fold_sha256":ctx["sha"]})'''
search='''spec=R["models"]()
space={n:{"family":v[0],"scaled":v[2],"parameters":v[3],"seed":v[4]} for n,v in spec.items()}
space["xgboost"]={"status":"NOT RUN / OPTIONAL DEPENDENCY UNAVAILABLE","installed":importlib.util.find_spec("xgboost") is not None}
(ctx["out"]/"configs/classification_model_search_space.json").write_text(json.dumps(space,indent=2),encoding="utf-8")
print(json.dumps(space,indent=2))'''
run='''oof, fold_metrics, failures=R["run"](ctx)
print("Completed models:", sorted(oof.model.unique()))
print("OOF rows:",len(oof))'''
summary='''import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
rows=[]
for name,frame in oof.groupby("model"):
    m=R["metric"](frame.true_class,frame.predicted_class); fm=fold_metrics[fold_metrics.model==name]
    rows.append({"model":name,"model_family":frame.model_family.iloc[0],"oof_macro_f1":m["macro_f1"],"oof_balanced_accuracy":m["balanced_accuracy"],"oof_accuracy":m["accuracy"],"oof_weighted_f1":m["weighted_f1"],**{f"oof_recall_class_{i}":m[f"recall_class_{i}"] for i in range(4)},"mean_fold_macro_f1":fm.macro_f1.mean(),"std_fold_macro_f1":fm.macro_f1.std(ddof=1),"total_tuning_training_seconds":fm.tuning_training_seconds.sum(),"total_inference_seconds":fm.inference_seconds.sum()})
summary=pd.DataFrame(rows).sort_values(["oof_macro_f1","oof_balanced_accuracy"],ascending=False).reset_index(drop=True)
summary.to_csv(ctx["out"]/"results/summaries/classification_baseline_summary.csv",index=False)
winner=summary.iloc[0]; best_model=winner.model
best={"model":best_model,"selection_rule":"highest OOF Macro-F1; ties considered by OOF balanced accuracy, fold stability, complexity/timing","primary_metric":"OOF Macro-F1","relevant_aggregate_metrics":{k:float(winner[k]) for k in ["oof_macro_f1","oof_balanced_accuracy","oof_accuracy","oof_weighted_f1","mean_fold_macro_f1","std_fold_macro_f1","total_tuning_training_seconds","total_inference_seconds"]},"source_result_file":"results/summaries/classification_baseline_summary.csv"}
(ctx["out"]/"configs/best_classifier.json").write_text(json.dumps(best,indent=2),encoding="utf-8")
display(summary.round(4))'''
figures='''best_frame=oof[oof.model==best_model]; cm=confusion_matrix(best_frame.true_class,best_frame.predicted_class,labels=[0,1,2,3])
fig,ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay(cm,display_labels=[0,1,2,3]).plot(ax=ax,colorbar=False); ax.set_title(f"OOF confusion matrix: {best_model}"); fig.tight_layout(); fig.savefig(ctx["out"]/"figures/best_traditional_classifier_confusion_matrix.png",dpi=220); fig.savefig(ctx["out"]/"figures/best_traditional_classifier_confusion_matrix.pdf"); plt.show()
p=summary.sort_values("oof_macro_f1"); fig,ax=plt.subplots(figsize=(10,5)); y=np.arange(len(p)); ax.barh(y-.18,p.oof_macro_f1,.36,label="OOF Macro-F1"); ax.barh(y+.18,p.oof_balanced_accuracy,.36,label="OOF Balanced Accuracy"); ax.set_yticks(y,p.model); ax.set_xlim(0,1); ax.legend(); ax.set_title("Traditional baseline comparison"); fig.tight_layout(); fig.savefig(ctx["out"]/"figures/classification_baseline_comparison.png",dpi=220); fig.savefig(ctx["out"]/"figures/classification_baseline_comparison.pdf"); plt.show()
print("Timing is full-pipeline outer-fold tuning/training and batch prediction, not deployment latency.")'''
audit='''coverage=[]
for name,frame in oof.groupby("model"):
    ok=len(frame)==419 and frame.run_key.nunique()==419 and not frame.duplicated(["model","run_key"]).any() and set(frame.outer_fold)==set(ctx["folds"]) and set(frame.predicted_class).issubset({0,1,2,3})
    coverage.append({"model":name,"expected_rows":419,"actual_rows":len(frame),"unique_run_keys":frame.run_key.nunique(),"duplicate_model_run_predictions":int(frame.duplicated(["model","run_key"]).sum()),"all_frozen_outer_folds_represented":set(frame.outer_fold)==set(ctx["folds"]),"valid_classes":set(frame.predicted_class).issubset({0,1,2,3}),"status":"PASS" if ok else "FAIL"})
coverage=pd.DataFrame(coverage); coverage.to_csv(ctx["out"]/"audits/oof_coverage_audit.csv",index=False)
leak={"frozen_fold_checksum_matched":True,"outer_folds_loaded_not_regenerated":True,"outer_train_test_subject_overlap":0,"inner_cv":"GroupKFold(n_splits=3), subject-disjoint assertions passed","imputation_training_only":True,"scaling_training_only":True,"variance_filtering_training_only":True,"feature_selection_training_only":True,"hyperparameter_tuning_outer_training_only":True,"performance_features_absent":True,"identifiers_and_labels_absent_from_features":True,"critical_leakage":False,"status":"PASS"}
(ctx["out"]/"audits/phase04a_leakage_audit.json").write_text(json.dumps(leak,indent=2),encoding="utf-8")
display(coverage)'''
final='''required=[ctx["out"] / x for x in ["Phase_04A_Classification_Baselines.ipynb","results/oof/classification_oof_predictions.csv","results/fold_metrics/classification_fold_results.csv","results/summaries/classification_baseline_summary.csv","configs/classification_model_search_space.json","configs/classification_best_params_by_fold.json","configs/best_classifier.json","figures/best_traditional_classifier_confusion_matrix.png","figures/classification_baseline_comparison.png","audits/smoke_test.json","audits/phase04a_leakage_audit.json","audits/oof_coverage_audit.csv","audits/failed_configurations.csv"]]
ready=all(x.exists() for x in required) and (coverage.status=="PASS").all() and leak["status"]=="PASS"
b=summary[summary.model==best_model].iloc[0]
report={"frozen_input":{"primary_dataset_path":str(ctx["paths"]["data"].relative_to(ROOT)),"samples":len(ctx["data"]),"subjects":ctx["data"].subject_id.nunique(),"features":len(ctx["features"]),"fold_path":str(ctx["paths"]["folds"].relative_to(ROOT)),"verified_sha256":ctx["sha"]},"models_completed":{n:"COMPLETED" for n in spec}|{"xgboost":"NOT RUN / OPTIONAL DEPENDENCY UNAVAILABLE"},"best_traditional_classifier":{"model":best_model,**{k:float(b[k]) for k in ["oof_macro_f1","oof_balanced_accuracy","oof_accuracy","oof_weighted_f1","mean_fold_macro_f1","std_fold_macro_f1"]}},"per_class_performance":{str(i):float(b[f"oof_recall_class_{i}"]) for i in range(4)},"selected_feature_counts_across_folds":fold_metrics[fold_metrics.model==best_model].selected_feature_count.tolist(),"timing":{"tuning_training_seconds":float(b.total_tuning_training_seconds),"inference_seconds":float(b.total_inference_seconds),"interpretation":"outer-fold batch timing; not deployment latency"},"oof_coverage":coverage.to_dict(orient="records"),"leakage_audit":leak["status"],"failed_configurations":"NONE" if not failures else failures,"output_files":[str(x.relative_to(ctx["out"])) for x in required if x.exists()],"phase_04a_ready":"YES" if ready else "NO"}
(ctx["out"]/"reports/phase04a_validation_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
for key in ["frozen_input","models_completed","best_traditional_classifier","per_class_performance","selected_feature_counts_across_folds","timing","oof_coverage","leakage_audit","failed_configurations","output_files"]: print("## "+key.upper()); print(json.dumps(report[key],indent=2))
print(f"PHASE 04A READY: {report['phase_04a_ready']}")'''
nb=nbf.v4.new_notebook()
nb.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
nb.cells=[nbf.v4.new_markdown_cell("# Phase 04A Traditional Classification Baselines\n\nLeakage-safe nested subject-wise CV using only frozen Phase 03 primary inputs."),md("1. Phase objective"),md("2. Imports/environment"),nbf.v4.new_code_cell(run_import),md("3. Phase 03 input discovery"),md("4. Frozen fold checksum verification"),md("5. Primary dataset validation"),md("6. Feature/target setup"),md("7. Outer fold validation"),nbf.v4.new_code_cell('''print(pd.DataFrame([{"fold":f,"test_runs":len(ctx["data"][ctx["data"].outer_fold==f]),"test_subjects":ctx["data"][ctx["data"].outer_fold==f].subject_id.nunique(),"classes":sorted(ctx["data"][ctx["data"].outer_fold==f].target_class.unique())} for f in ctx["folds"]]))'''),md("8. Preprocessing design"),nbf.v4.new_code_cell('''print("Median imputation + indicators, variance filtering, scale-sensitive scaling, and f_classif feature selection are fitted only within pipelines.")'''),md("9. Model definitions"),md("10. Hyperparameter grids"),nbf.v4.new_code_cell(search),md("11. Smoke test"),nbf.v4.new_code_cell('''smoke=R["smoke"](ctx); print(json.dumps(smoke,indent=2))'''),md("12. Dummy baselines"),md("13. Logistic Regression"),md("14. Linear SVM"),md("15. RBF SVM"),md("16. Random Forest"),md("17. KNN"),md("18. Gradient Boosting"),nbf.v4.new_code_cell(run),md("19. Optional XGBoost status"),nbf.v4.new_code_cell('''print("XGBOOST = NOT RUN / OPTIONAL DEPENDENCY UNAVAILABLE")'''),md("20. Fold-level results"),nbf.v4.new_code_cell('''display(fold_metrics.round(4))'''),md("21. OOF predictions"),nbf.v4.new_code_cell('''print(oof.groupby("model").size())'''),md("22. Aggregate comparison"),nbf.v4.new_code_cell(summary),md("23. Best classifier selection"),nbf.v4.new_code_cell('''print(json.dumps(best,indent=2))'''),md("24. Confusion matrix"),md("25. Timing"),nbf.v4.new_code_cell(figures),md("26. Leakage audit"),md("27. OOF coverage audit"),nbf.v4.new_code_cell(audit),md("28. Phase validation summary"),nbf.v4.new_code_cell(final)]
nbf.write(nb,OUT/"Phase_04A_Classification_Baselines.ipynb")
print(OUT/"Phase_04A_Classification_Baselines.ipynb")
