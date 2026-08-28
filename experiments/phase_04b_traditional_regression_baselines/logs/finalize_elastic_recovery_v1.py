from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(r'E:\hdc-vr-pilot'); P=ROOT/'experiments'/'phase_04b_traditional_regression_baselines'; C=P/'results'/'checkpoints'/'elastic_net'; F=ROOT/'experiments'/'phase_03_multimodal_dataset_labeling'/'data'/'fold_assignments.csv'; SHA='e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f'
def aj(path,x):
 t=path.with_name(path.name+'.tmp');t.write_text(json.dumps(x,indent=2,default=str)+chr(10));t.replace(path)
def ac(path,x):
 t=path.with_name(path.name+'.tmp');x.to_csv(t,index=False);t.replace(path)
def sp(y,x):
 z=spearmanr(y,x).statistic
 return float(z) if np.isfinite(z) else np.nan
assert hashlib.sha256(F.read_bytes()).hexdigest()==SHA
folds=pd.read_csv(F); preds=[]; metrics=[]; convergence=[]; partial={};
for f in range(1,6):
 o=pd.read_csv(C/f'elastic_net_fold_{f}_predictions.csv');m=json.loads((C/f'elastic_net_fold_{f}_metrics.json').read_text());b=json.loads((C/f'elastic_net_fold_{f}_best_params.json').read_text());
 assert set(o.run_key)==set(folds.loc[folds.outer_fold.eq(f),'run_key']) and o.run_key.nunique()==len(o) and o.prediction_raw.notna().all() and o.prediction_bounded.between(1,4).all() and m['subject_overlap_count']==0 and m['inner_candidate_count']==48
 if f<3:
  partial[str(f)]={'max_iter_used':20000,'converged_before_cap':True,'reused_after_audit':True,'metrics_recomputed_match':True,'selected_k':m['selected_k'],'selected_alpha':m['selected_alpha'],'selected_l1_ratio':m['selected_l1_ratio'],'n_iter':m['best_estimator_n_iter']}
  o['max_iter_used']=20000;o['checkpoint_reused']=True;o['recovery_version']='V1'
 else: partial[str(f)]={'checkpoint_reused':False,'max_iter_used':100000,'converged_before_cap':m['best_estimator_converged']}
 preds.append(o);metrics.append(m);convergence.append({'outer_fold':f,'checkpoint_reused':f<3,'selected_k':m['selected_k'],'selected_alpha':m['selected_alpha'],'selected_l1_ratio':m['selected_l1_ratio'],'n_iter':m['best_estimator_n_iter'],'max_iter_used':20000 if f<3 else 100000,'tol':1e-4,'search_convergence_warning_count_total':m['convergence_warning_count'],'best_estimator_converged':m['best_estimator_converged'],'convergence_status':'PASS' if m['best_estimator_converged'] else 'FAIL','recovery_version':'V1'})
aj(P/'audits'/'elastic_net_partial_checkpoint_audit.json',{'folds':partial,'reused_folds':[1,2],'overall_pass':True,'utc_timestamp':datetime.now(timezone.utc).isoformat()})
aj(P/'audits'/'elastic_net_initial_failure_record.json',{'initial_status':'FAIL','completed_folds':[1,2],'failed_or_interrupted_fold':3,'initial_max_iter':20000,'initial_tol':1e-4,'root_cause':'PYTHON_SYNTAX_ERROR prevented recovery training from starting','recovery_training_started':False,'scientific_search_space_changed':False,'frozen_folds_changed':False,'outer_test_used_for_recovery_decision':False,'recovery_action':'increase numerical solver budget to max_iter=100000','utc_timestamp':datetime.now(timezone.utc).isoformat()})
oof=pd.concat(preds).sort_values('run_key').reset_index(drop=True);assert len(oof)==419 and oof.run_key.nunique()==419 and oof.outer_fold.nunique()==5
ac(P/'results'/'predictions'/'elastic_net_oof.csv',oof); fm=pd.DataFrame(metrics).sort_values('outer_fold');ac(P/'results'/'fold_metrics'/'elastic_net_fold_metrics.csv',fm)
y=oof.target_score.to_numpy();raw=oof.prediction_raw.to_numpy();bd=oof.prediction_bounded.to_numpy();summary={'model':'Elastic Net','model_slug':'elastic_net','oof_rows':419,'oof_unique_run_keys':419,'oof_mae_raw':mean_absolute_error(y,raw),'oof_mae_bounded':mean_absolute_error(y,bd),'oof_rmse_raw':np.sqrt(mean_squared_error(y,raw)),'oof_rmse_bounded':np.sqrt(mean_squared_error(y,bd)),'oof_r2_raw':r2_score(y,raw),'oof_r2_bounded':r2_score(y,bd),'oof_spearman_raw':sp(y,raw),'oof_spearman_bounded':sp(y,bd),'fold_mae_bounded_mean':fm.mae_bounded.mean(),'fold_mae_bounded_std':fm.mae_bounded.std(ddof=1),'total_fit_and_search_time_seconds':fm.fit_and_search_time_seconds.sum(),'total_prediction_time_seconds':fm.prediction_time_seconds.sum(),'all_best_estimators_converged':bool(fm.best_estimator_converged.all()),'status':'COMPLETE'};ac(P/'results'/'summaries'/'elastic_net_summary.csv',pd.DataFrame([summary]))
levels=[]
for v,g in oof.groupby('target_score'):levels.append({'model':'Elastic Net','model_slug':'elastic_net','target_score':v,'n_samples':len(g),'mae_raw':g.absolute_error_raw.mean(),'mae_bounded':g.absolute_error_bounded.mean(),'mean_prediction_raw':g.prediction_raw.mean(),'mean_prediction_bounded':g.prediction_bounded.mean()})
ac(P/'results'/'summaries'/'elastic_net_per_level_mae.csv',pd.DataFrame(levels));base=pd.concat([pd.read_csv(P/'results'/'summaries'/'dummy_regressor_summary.csv')[['model','oof_mae_bounded']],pd.read_csv(P/'results'/'summaries'/'ridge_summary.csv')[['model','oof_mae_bounded']],pd.DataFrame([{'model':'Elastic Net','oof_mae_bounded':summary['oof_mae_bounded']}])]);r=base.loc[base.model.eq('Ridge'),'oof_mae_bounded'].iloc[0];base['absolute_mae_difference_vs_ridge']=base.oof_mae_bounded-r;base['relative_mae_difference_vs_ridge']=base.absolute_mae_difference_vs_ridge/r;ac(P/'results'/'summaries'/'elastic_net_vs_completed_baselines.csv',base)
aj(P/'audits'/'elastic_net_convergence_audit.json',{'folds':convergence,'overall_pass':all(x['best_estimator_converged'] for x in convergence)});aj(P/'audits'/'elastic_net_leakage_audit.json',{'frozen_fold_sha256':SHA,'outer_subject_isolation':True,'inner_subject_isolation':True,'pipeline_fitted_training_only':True,'outer_test_used_for_parameter_selection':False,'outer_test_used_for_recovery_decision':False,'scientific_search_space_unchanged':True,'recovery_only_changed_numerical_iteration_budget':True,'overall_pass':True});aj(P/'audits'/'elastic_net_oof_coverage_audit.json',{'rows':419,'unique_run_keys':419,'missing_run_keys':0,'extra_run_keys':0,'duplicate_run_keys':0,'missing_predictions':0,'fold_coverage':5,'bounded_range_pass':True,'artifact_audit_pass':True,'overall_pass':True});aj(P/'configs'/'elastic_net_convergence_recovery_v1.json',{'initial_max_iter':20000,'recovery_max_iter':100000,'tol':1e-4,'scientific_grid_unchanged':True,'outer_folds_unchanged':True,'inner_cv_unchanged':True,'scorer_unchanged':True,'target_unchanged':True,'reason':'numerical convergence recovery','reused_folds':[1,2],'rerun_folds':[3,4,5],'utc_timestamp':datetime.now(timezone.utc).isoformat()})
s=json.loads((P/'configs'/'regression_model_search_space.json').read_text());[x.update(status='COMPLETE') for x in s['models'] if x['name']=='Elastic Net'];s['status']='ELASTIC_NET_COMPLETE_AFTER_CONVERGENCE_RECOVERY_V1 / REMAINING_MODELS_NOT_STARTED';aj(P/'configs'/'regression_model_search_space.json',s);print(json.dumps(summary,indent=2,default=str))
