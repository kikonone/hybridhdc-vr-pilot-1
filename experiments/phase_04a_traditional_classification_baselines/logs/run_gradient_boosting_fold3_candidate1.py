from pathlib import Path
import os
import json
import hashlib
import tempfile
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest,VarianceThreshold,f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from gradient_boosting_candidate_checkpoints import candidate_manifest,atomic_csv,load_or_initialize,validate_results
FOLD=int(os.environ.get('GB_OUTER_FOLD','3')); CANDIDATE=int(os.environ.get('GB_CANDIDATE','8'))
ROOT=Path(__file__).resolve().parents[3]; base=ROOT/'experiments/phase_04a_traditional_classification_baselines'; cp=base/'results/checkpoints/gradient_boosting'; manifest_path=cp/f'fold_{FOLD}_search_candidates.csv'; manifest=candidate_manifest(FOLD)
if not manifest_path.exists(): atomic_csv(manifest,manifest_path)
manifest=pd.read_csv(manifest_path); results_path=cp/f'gradient_boosting_fold_{FOLD}_inner_search_results.csv'; results=load_or_initialize(results_path,manifest)
data=pd.read_csv(ROOT/'experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv'); folds=pd.read_csv(ROOT/'experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv')[['run_key','outer_fold']]
expected_sha='e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f'
actual_sha=hashlib.sha256((ROOT/'experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv').read_bytes()).hexdigest()
if actual_sha != expected_sha: raise RuntimeError(f'frozen fold checksum mismatch: {actual_sha}')
if 'outer_fold' not in data: data=data.merge(folds,on='run_key',validate='one_to_one')
features=json.loads((ROOT/'experiments/phase_03_multimodal_dataset_labeling/manifests/primary_feature_manifest.json').read_text())['features']; tr=data[data.outer_fold!=FOLD].reset_index(drop=True); te=data[data.outer_fold==FOLD].reset_index(drop=True)
if set(tr.subject_id) & set(te.subject_id): raise RuntimeError('outer subject overlap')
outer_summary={'outer_fold':FOLD,'outer_train_rows':len(tr),'outer_test_rows':len(te),'outer_train_subjects':int(tr.subject_id.nunique()),'outer_test_subjects':int(te.subject_id.nunique()),'outer_test_used_for_evaluation':False}
split_path=cp/f'fold_{FOLD}_inner_splits.json'
if split_path.exists(): splits=json.loads(split_path.read_text())
else:
 splits=[]
 for i,(a,b) in enumerate(GroupKFold(3).split(tr,tr.target_class,tr.subject_id),1):
  if set(tr.iloc[a].subject_id)&set(tr.iloc[b].subject_id): raise RuntimeError('overlap')
  splits.append({'inner_fold':i,'train_run_keys':tr.iloc[a].run_key.tolist(),'validation_run_keys':tr.iloc[b].run_key.tolist()})
 split_path.write_text(json.dumps(splits,indent=2),encoding='utf-8')
if (results.candidate_id.eq(CANDIDATE)&results.status.eq('COMPLETE')).any(): print('already complete'); raise SystemExit
row=manifest[manifest.candidate_id.eq(CANDIDATE)].iloc[0]; lookup={k:i for i,k in enumerate(tr.run_key)}; scores=[]
if CANDIDATE not in set(manifest.candidate_id): raise RuntimeError('candidate absent from manifest')
in_progress={**row.to_dict(),'inner_fold_1_macro_f1':np.nan,'inner_fold_2_macro_f1':np.nan,'inner_fold_3_macro_f1':np.nan,'mean_inner_macro_f1':np.nan,'std_inner_macro_f1':np.nan,'status':'IN_PROGRESS'}
results=results.loc[results.candidate_id.ne(CANDIDATE)].copy()
results=pd.concat([results,pd.DataFrame([in_progress])],ignore_index=True)[results.columns]
validate_results(results,manifest); atomic_csv(results,results_path)
if pd.read_csv(results_path).query('candidate_id == @CANDIDATE').iloc[0].status != 'IN_PROGRESS': raise RuntimeError('IN_PROGRESS readback failed')
for s in splits:
 a=[lookup[k] for k in s['train_run_keys']]; b=[lookup[k] for k in s['validation_run_keys']]
 p=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('variance',VarianceThreshold()),('selector',SelectKBest(f_classif,k=int(row.k))),('classifier',GradientBoostingClassifier(n_estimators=int(row.n_estimators),learning_rate=float(row.learning_rate),max_depth=int(row.max_depth),random_state=42))])
 p.fit(tr.iloc[a][features],tr.iloc[a].target_class); scores.append(f1_score(tr.iloc[b].target_class,p.predict(tr.iloc[b][features]),average='macro',zero_division=0))
rec={**row.to_dict(),'inner_fold_1_macro_f1':scores[0],'inner_fold_2_macro_f1':scores[1],'inner_fold_3_macro_f1':scores[2],'mean_inner_macro_f1':float(np.mean(scores)),'std_inner_macro_f1':float(np.std(scores)),'status':'COMPLETE'}; results=results.loc[results.candidate_id.ne(CANDIDATE)].copy(); results=pd.concat([results,pd.DataFrame([rec])],ignore_index=True)[results.columns]; validate_results(results,manifest); atomic_csv(results,results_path)
readback=pd.read_csv(results_path); validate_results(readback,manifest)
saved=readback.loc[readback.candidate_id.eq(CANDIDATE)].iloc[0].to_dict()
if saved['status']!='COMPLETE': raise RuntimeError('COMPLETE readback failed')
progress={'outer_fold':FOLD,'grid_policy':'COMPUTE_CONSTRAINED_COMPACT_GRID_V2','full_grid_size':8,'completed_candidate_ids':sorted(readback.loc[readback.status.eq('COMPLETE'),'candidate_id'].astype(int).tolist()),'remaining_candidate_ids':sorted(manifest.loc[~manifest.candidate_id.isin(readback.loc[readback.status.eq('COMPLETE'),'candidate_id']),'candidate_id'].astype(int).tolist()),'outer_test_used':False,'fold_status':'INCOMPLETE','frozen_fold_sha256':actual_sha}
progress_path=cp/f'gradient_boosting_fold_{FOLD}_search_progress.json'
with tempfile.NamedTemporaryFile(mode='w',suffix='.json',dir=cp,delete=False,encoding='utf-8') as handle:
 temp=Path(handle.name); json.dump(progress,handle,indent=2); handle.flush(); os.fsync(handle.fileno())
os.replace(temp,progress_path)
if json.loads(progress_path.read_text(encoding='utf-8')) != progress: raise RuntimeError('progress readback failed')
print(json.dumps({'outer_summary':outer_summary,'candidate':saved,'progress':progress}))
