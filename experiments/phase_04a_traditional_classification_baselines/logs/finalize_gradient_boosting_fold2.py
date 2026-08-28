from pathlib import Path
import json, time
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline

ROOT=Path(__file__).resolve().parents[3]; base=ROOT/'experiments/phase_04a_traditional_classification_baselines'; cp=base/'results/checkpoints/gradient_boosting'
results=pd.read_csv(cp/'gradient_boosting_fold_2_inner_search_results.csv'); winner=results[results.status=='COMPLETE'].sort_values(['mean_inner_macro_f1','candidate_id'],ascending=[False,True]).iloc[0]
data=pd.read_csv(ROOT/'experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv'); folds=pd.read_csv(ROOT/'experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv')[['run_key','outer_fold']]
if 'outer_fold' not in data: data=data.merge(folds,on='run_key',validate='one_to_one')
features=json.loads((ROOT/'experiments/phase_03_multimodal_dataset_labeling/manifests/primary_feature_manifest.json').read_text())['features']; tr=data[data.outer_fold!=2]; te=data[data.outer_fold==2]
if set(tr.subject_id)&set(te.subject_id): raise RuntimeError('subject leakage')
pipe=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('variance',VarianceThreshold()),('selector',SelectKBest(f_classif,k=int(winner.k))),('classifier',GradientBoostingClassifier(n_estimators=int(winner.n_estimators),learning_rate=float(winner.learning_rate),max_depth=int(winner.max_depth),random_state=42))])
t=time.perf_counter(); pipe.fit(tr[features],tr.target_class); train_s=time.perf_counter()-t; t=time.perf_counter(); pred=pipe.predict(te[features]); proba=pipe.predict_proba(te[features]); infer_s=time.perf_counter()-t
r=recall_score(te.target_class,pred,labels=[0,1,2,3],average=None,zero_division=0); frame=te[['subject_id','session_id','run_id','run_key','outer_fold','target_class']].rename(columns={'target_class':'true_class'}).copy(); frame['predicted_class']=pred; frame['model']='gradient_boosting'; frame['model_family']='Gradient Boosting'; frame['selected_k']=int(winner.k); frame['seed']=42
for i in range(4): frame[f'probability_class_{i}']=proba[:,i]; frame[f'decision_score_class_{i}']=float('nan')
frame.to_csv(cp/'gradient_boosting_fold_2_predictions.csv',index=False)
params={k: (int(winner[k]) if k in ['candidate_id','k','n_estimators','max_depth'] else float(winner[k])) for k in ['candidate_id','k','n_estimators','learning_rate','max_depth','mean_inner_macro_f1','std_inner_macro_f1']}
metrics={'model':'gradient_boosting','outer_fold':2,'test_run_count':len(te),'test_subject_count':te.subject_id.nunique(),'selected_hyperparameters':params,'selected_feature_count':int(winner.k),'tuning_training_seconds':train_s,'inference_seconds':infer_s,'macro_f1':f1_score(te.target_class,pred,average='macro'),'balanced_accuracy':balanced_accuracy_score(te.target_class,pred),'accuracy':accuracy_score(te.target_class,pred),'weighted_f1':f1_score(te.target_class,pred,average='weighted'),**{f'recall_class_{i}':float(r[i]) for i in range(4)}}
(cp/'gradient_boosting_fold_2_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); (cp/'gradient_boosting_fold_2_best_params.json').write_text(json.dumps(params,indent=2),encoding='utf-8'); print(json.dumps(metrics,indent=2))
