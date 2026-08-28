from pathlib import Path
import json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from gradient_boosting_candidate_checkpoints import load_or_initialize, validate_results, atomic_csv

ROOT=Path(__file__).resolve().parents[3]; base=ROOT/'experiments/phase_04a_traditional_classification_baselines'; cp=base/'results/checkpoints/gradient_boosting'
manifest=pd.read_csv(cp/'fold_2_search_candidates.csv'); results_path=cp/'gradient_boosting_fold_2_inner_search_results.csv'; results=load_or_initialize(results_path,manifest)
data=pd.read_csv(ROOT/'experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv'); folds=pd.read_csv(ROOT/'experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv')[['run_key','outer_fold']]
if 'outer_fold' not in data: data=data.merge(folds,on='run_key',validate='one_to_one')
features=json.loads((ROOT/'experiments/phase_03_multimodal_dataset_labeling/manifests/primary_feature_manifest.json').read_text())['features']; train=data[data.outer_fold!=2].reset_index(drop=True); lookup={k:i for i,k in enumerate(train.run_key)}; splits=json.loads((cp/'fold_2_inner_splits.json').read_text())
for cid in [1,2,3,4,5,6,7,8]:
    if (results.candidate_id.eq(cid)&results.status.eq('COMPLETE')).any(): continue
    row=manifest.loc[manifest.candidate_id.eq(cid)].iloc[0]; scores=[]
    for split in splits:
        a=[lookup[k] for k in split['train_run_keys']]; b=[lookup[k] for k in split['validation_run_keys']]
        pipe=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('variance',VarianceThreshold()),('selector',SelectKBest(f_classif,k=int(row.k))),('classifier',GradientBoostingClassifier(n_estimators=int(row.n_estimators),learning_rate=float(row.learning_rate),max_depth=int(row.max_depth),random_state=42))])
        pipe.fit(train.iloc[a][features],train.iloc[a].target_class); scores.append(f1_score(train.iloc[b].target_class,pipe.predict(train.iloc[b][features]),average='macro',zero_division=0))
    record={**row.to_dict(),'inner_fold_1_macro_f1':scores[0],'inner_fold_2_macro_f1':scores[1],'inner_fold_3_macro_f1':scores[2],'mean_inner_macro_f1':float(np.mean(scores)),'std_inner_macro_f1':float(np.std(scores)),'status':'COMPLETE'}
    results=results[results.candidate_id.ne(cid)]; results=pd.concat([results,pd.DataFrame([record])],ignore_index=True)[results.columns]; validate_results(results,manifest); atomic_csv(results,results_path); print(cid,record['mean_inner_macro_f1'],flush=True)
