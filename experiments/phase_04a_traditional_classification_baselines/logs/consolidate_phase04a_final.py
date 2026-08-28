from pathlib import Path
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'experiments/phase_04a_traditional_classification_baselines'
CP = BASE / 'results/checkpoints/gradient_boosting'
PRED = BASE / 'results/predictions'; SUM = BASE / 'results/summaries'; AUD = BASE / 'audits'; CFG = BASE / 'configs'; REP = BASE / 'reports'; FIG = BASE / 'figures'
EXPECTED = 'e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f'

def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', dir=path.parent, delete=False, encoding='utf-8', newline='') as handle:
        temp = Path(handle.name); frame.to_csv(handle, index=False); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)

def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=path.parent, delete=False, encoding='utf-8') as handle:
        temp = Path(handle.name); json.dump(value, handle, indent=2); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)

fold_path = ROOT / 'experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv'
sha = hashlib.sha256(fold_path.read_bytes()).hexdigest()
if sha != EXPECTED: raise RuntimeError(f'frozen checksum mismatch: {sha}')
assignments = pd.read_csv(fold_path)
data = pd.read_csv(ROOT / 'experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv')
if 'outer_fold' not in data: data = data.merge(assignments[['run_key','outer_fold']], on='run_key', validate='one_to_one')
features = json.loads((ROOT / 'experiments/phase_03_multimodal_dataset_labeling/manifests/primary_feature_manifest.json').read_text(encoding='utf-8'))['features']
if (len(data), data.subject_id.nunique(), len(features), data.target_class.value_counts().sort_index().tolist()) != (419,35,1176,[104,106,104,105]): raise RuntimeError('primary identity mismatch')

# Gradient Boosting OOF: concatenate only persisted official fold predictions.
gb_frames = []
fold_rows = []
for fold in range(1,6):
    pred_path = CP / f'gradient_boosting_fold_{fold}_predictions.csv'; met_path = CP / f'gradient_boosting_fold_{fold}_metrics.json'; par_path = CP / f'gradient_boosting_fold_{fold}_best_params.json'
    if not all(path.exists() for path in (pred_path, met_path, par_path)): raise FileNotFoundError(f'Gradient Boosting fold {fold} checkpoint missing')
    frame = pd.read_csv(pred_path); met = json.loads(met_path.read_text(encoding='utf-8')); par = json.loads(par_path.read_text(encoding='utf-8'))
    expected_runs = set(assignments.loc[assignments.outer_fold.eq(fold),'run_key'])
    if len(frame) != len(expected_runs) or frame.run_key.nunique()!=len(frame) or set(frame.run_key)!=expected_runs or set(frame.outer_fold.astype(int))!={fold}: raise RuntimeError(f'Gradient Boosting fold {fold} coverage invalid')
    if fold >= 4 and met.get('frozen_fold_sha256') != sha: raise RuntimeError(f'Gradient Boosting fold {fold} checksum metadata invalid')
    hp = met.get('selected_hyperparameters', {})
    if isinstance(hp, str): hp = json.loads(hp)
    def getv(new, legacy): return hp.get(new, hp.get(legacy, par.get(new, par.get(legacy))))
    saved_recalls = {f'recall_class_{i}': float(recall_score(frame.true_class, frame.predicted_class, labels=[i], average='macro', zero_division=0)) for i in range(4)}
    fold_rows.append({'outer_fold':fold,'best_candidate_id':getv('candidate_id','candidate_id'),'selected_k':getv('k','effective_selected_k') or getv('selector__k','selector__k'),'selected_n_estimators':getv('n_estimators','classifier__n_estimators'),'selected_learning_rate':getv('learning_rate','classifier__learning_rate'),'selected_max_depth':getv('max_depth','classifier__max_depth'),'best_inner_macro_f1':getv('mean_inner_macro_f1','mean_inner_macro_f1'),'best_inner_macro_f1_std':getv('std_inner_macro_f1','std_inner_macro_f1'),'outer_macro_f1':met['macro_f1'],'outer_balanced_accuracy':met['balanced_accuracy'],'outer_accuracy':met['accuracy'],'outer_weighted_f1':met['weighted_f1'],**saved_recalls,'refit_time':met.get('final_refit_seconds',met.get('tuning_training_seconds')),'inference_time':met['inference_seconds']})
    gb_frames.append(frame)
gb_oof = pd.concat(gb_frames, ignore_index=True).sort_values(['outer_fold','run_key']).reset_index(drop=True)
prob_cols=[f'probability_class_{i}' for i in range(4)]
if len(gb_oof)!=419 or gb_oof.run_key.nunique()!=419 or set(gb_oof.outer_fold.astype(int))!=set(range(1,6)) or set(gb_oof.run_key)!=set(assignments.run_key): raise RuntimeError('Gradient Boosting OOF coverage invalid')
if not np.isfinite(gb_oof[prob_cols].to_numpy(dtype=float)).all() or not np.allclose(gb_oof[prob_cols].sum(axis=1),1.0,atol=1e-8): raise RuntimeError('Gradient Boosting OOF probabilities invalid')
if not gb_oof.merge(data[['run_key','target_class']],on='run_key',validate='one_to_one').eval('true_class == target_class').all(): raise RuntimeError('Gradient Boosting OOF labels invalid')
atomic_csv(gb_oof, PRED/'gradient_boosting_oof.csv'); atomic_csv(pd.DataFrame(fold_rows), SUM/'gradient_boosting_fold_summary.csv')

models={'logistic_regression':'Logistic Regression','linear_svm':'Linear SVM','rbf_svm':'RBF SVM','random_forest':'Random Forest','knn':'KNN','gradient_boosting':'Gradient Boosting'}
coverage=[]; comparison=[]
for slug, name in models.items():
    path=PRED/f'{slug}_oof.csv'
    if not path.exists(): raise FileNotFoundError(path)
    frame=pd.read_csv(path)
    match=frame.merge(assignments[['run_key','outer_fold']],on='run_key',suffixes=('','_frozen'),validate='one_to_one')
    labels=frame.merge(data[['run_key','target_class']],on='run_key',validate='one_to_one')
    valid=(len(frame)==419 and frame.run_key.nunique()==419 and not frame.run_key.duplicated().any() and set(frame.outer_fold.astype(int))==set(range(1,6)) and (match.outer_fold.astype(int)==match.outer_fold_frozen.astype(int)).all() and (labels.true_class.astype(int)==labels.target_class.astype(int)).all())
    if not valid: raise RuntimeError(f'{name} OOF coverage/label mismatch')
    vals={'oof_macro_f1':f1_score(frame.true_class,frame.predicted_class,average='macro'),'oof_balanced_accuracy':balanced_accuracy_score(frame.true_class,frame.predicted_class),'oof_accuracy':accuracy_score(frame.true_class,frame.predicted_class),'oof_weighted_f1':f1_score(frame.true_class,frame.predicted_class,average='weighted'),**{f'recall_class_{i}':float(recall_score(frame.true_class,frame.predicted_class,labels=[i],average='macro',zero_division=0)) for i in range(4)}}
    fold_f1=[f1_score(part.true_class,part.predicted_class,average='macro') for _,part in frame.groupby('outer_fold')]
    comparison.append({'model':name,'model_slug':slug,**vals,'fold_macro_f1_mean':float(np.mean(fold_f1)),'fold_macro_f1_std':float(np.std(fold_f1))})
    coverage.append({'model':name,'expected_rows':419,'actual_rows':len(frame),'unique_runs':frame.run_key.nunique(),'duplicates':int(frame.run_key.duplicated().sum()),'folds_present':'[1,2,3,4,5]','frozen_fold_match':'PASS','label_match':'PASS','coverage_status':'PASS'})
comparison_df=pd.DataFrame(comparison).sort_values(['oof_macro_f1','model'],ascending=[False,True]).reset_index(drop=True)
atomic_csv(comparison_df, SUM/'phase04a_final_classifier_comparison.csv'); atomic_csv(comparison_df, SUM/'classification_baseline_summary.csv'); atomic_csv(comparison_df.assign(rank=np.arange(1,len(comparison_df)+1)), SUM/'phase04a_final_classifier_ranking.csv'); atomic_csv(pd.DataFrame(coverage), AUD/'phase04a_final_oof_coverage_audit.csv')
best=comparison_df.iloc[0].to_dict()

# Required final visuals are generated only from complete persisted OOF results.
FIG.mkdir(exist_ok=True)
best_frame=pd.read_csv(PRED/f"{best['model_slug']}_oof.csv")
cm=confusion_matrix(best_frame.true_class,best_frame.predicted_class,labels=[0,1,2,3]); fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap='Blues')
for i in range(4):
    for j in range(4): ax.text(j,i,int(cm[i,j]),ha='center',va='center')
ax.set(xlabel='Predicted class',ylabel='True class',xticks=range(4),yticks=range(4),title=f"Best traditional classifier: {best['model']}"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(FIG/'best_traditional_classifier_confusion_matrix.png',dpi=200); plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4)); ax.bar(comparison_df.model,comparison_df.oof_macro_f1); ax.set_ylabel('OOF Macro-F1'); ax.set_ylim(0,1); ax.tick_params(axis='x',rotation=25); fig.tight_layout(); fig.savefig(FIG/'classification_baseline_comparison.png',dpi=200); plt.close(fig)

config={'frozen_phase03_sha256':sha,'dataset':'PRIMARY_WITHOUT_PERFORMANCE','samples':419,'subjects':35,'primary_features':1176,'outer_cv':'frozen 5-fold subject-wise','inner_cv':'3-fold GroupKFold(subject_id)','primary_metric':'Macro-F1','gradient_boosting_grid_policy':'COMPUTE_CONSTRAINED_COMPACT_GRID_V2','gradient_boosting_grid':{'k':[100,200],'n_estimators':[100,200],'learning_rate':[0.05,0.1],'max_depth':[2]},'completed_models':list(models.values()),'xgboost':'OPTIONAL / NOT RUN','best_traditional_classifier':best['model'],'best_oof_macro_f1':float(best['oof_macro_f1'])}
atomic_json(config, CFG/'phase04a_final_configuration.json')
atomic_json({'model':best['model'],'selection_criterion':'highest complete 419-row OOF Macro-F1','oof_macro_f1':float(best['oof_macro_f1']),'oof_balanced_accuracy':float(best['oof_balanced_accuracy']),'oof_accuracy':float(best['oof_accuracy']),'oof_weighted_f1':float(best['oof_weighted_f1'])}, CFG/'best_classifier.json')
summary_lines=['# Phase 04A Final Summary','',f"- Frozen Phase 03 SHA-256: `{sha}`",'- Primary data: 419 runs, 35 subjects, 1,176 features.','- Outer CV: frozen subject-wise five-fold; inner CV: subject-wise GroupKFold(3).','- Primary selection metric: Macro-F1.','- Gradient Boosting used the V2 candidate-level checkpoint workflow; all five outer folds are complete.','', '## Complete OOF comparison','',comparison_df.to_markdown(index=False),'',f"## Best traditional classifier\n\n{best['model']} — OOF Macro-F1 {best['oof_macro_f1']:.6f}.",'','XGBoost: OPTIONAL / NOT RUN. No statistical-significance claim is made in Phase 04A.']
REP.mkdir(exist_ok=True); (REP/'phase04a_final_summary.md').write_text('\n'.join(summary_lines),encoding='utf-8')
freeze={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'frozen_phase03_sha256':sha,'completed_models':list(models.values()),'final_oof_paths':{slug:str((PRED/f'{slug}_oof.csv').relative_to(BASE)) for slug in models},'gradient_boosting_status':'COMPLETE','best_traditional_classifier':best['model'],'best_oof_macro_f1':float(best['oof_macro_f1']),'xgboost':'OPTIONAL / NOT RUN','checkpoint_integrity_result':'PASS','oof_coverage_result':'PASS','phase04a_status':'COMPLETE','phase04a_frozen':'YES'}
atomic_json(freeze, CFG/'phase04a_freeze.json')
progress_path=BASE/'logs/phase04a_progress.json'
progress=json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else {}
for slug in models: progress[slug]='COMPLETE'
progress['gradient_boosting']={'status':'COMPLETE','completed_folds':[1,2,3,4,5]}
progress['xgboost']='OPTIONAL / NOT RUN'; progress['phase04a_status']='COMPLETE'; progress['phase04a_frozen']='YES'
atomic_json(progress,progress_path)
print(json.dumps({'gradient_boosting_oof_metrics':comparison_df.loc[comparison_df.model_slug.eq('gradient_boosting')].iloc[0].to_dict(),'best':best,'coverage':'PASS','freeze':freeze},indent=2))
