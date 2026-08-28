from pathlib import Path
import json,time
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest,VarianceThreshold,f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score,recall_score
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parents[3]; base=ROOT/'experiments/phase_04a_traditional_classification_baselines'; cp=base/'results/checkpoints/gradient_boosting'; r=pd.read_csv(cp/'gradient_boosting_fold_3_inner_search_results.csv'); w=r[r.status=='COMPLETE'].sort_values(['mean_inner_macro_f1','candidate_id'],ascending=[False,True]).iloc[0]
d=pd.read_csv(ROOT/'experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv'); f=pd.read_csv(ROOT/'experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv')[['run_key','outer_fold']];
if 'outer_fold' not in d:d=d.merge(f,on='run_key',validate='one_to_one')
x=json.loads((ROOT/'experiments/phase_03_multimodal_dataset_labeling/manifests/primary_feature_manifest.json').read_text())['features']; tr=d[d.outer_fold!=3]; te=d[d.outer_fold==3]
p=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('variance',VarianceThreshold()),('selector',SelectKBest(f_classif,k=int(w.k))),('classifier',GradientBoostingClassifier(n_estimators=int(w.n_estimators),learning_rate=float(w.learning_rate),max_depth=int(w.max_depth),random_state=42))]); t=time.perf_counter();p.fit(tr[x],tr.target_class);ts=time.perf_counter()-t;t=time.perf_counter();pred=p.predict(te[x]);pro=p.predict_proba(te[x]);ins=time.perf_counter()-t
out=te[['subject_id','session_id','run_id','run_key','outer_fold','target_class']].rename(columns={'target_class':'true_class'}).copy();out['predicted_class']=pred;out['model']='gradient_boosting';out['model_family']='Gradient Boosting';out['selected_k']=int(w.k);out['seed']=42
for i in range(4):out[f'probability_class_{i}']=pro[:,i];out[f'decision_score_class_{i}']=float('nan')
out.to_csv(cp/'gradient_boosting_fold_3_predictions.csv',index=False); rec={k:(int(w[k]) if k in ['candidate_id','k','n_estimators','max_depth'] else float(w[k])) for k in ['candidate_id','k','n_estimators','learning_rate','max_depth','mean_inner_macro_f1','std_inner_macro_f1']}; q=rec|{'macro_f1':f1_score(te.target_class,pred,average='macro'),'balanced_accuracy':balanced_accuracy_score(te.target_class,pred),'accuracy':accuracy_score(te.target_class,pred),'weighted_f1':f1_score(te.target_class,pred,average='weighted'),'tuning_training_seconds':ts,'inference_seconds':ins};(cp/'gradient_boosting_fold_3_best_params.json').write_text(json.dumps(rec,indent=2));(cp/'gradient_boosting_fold_3_metrics.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
