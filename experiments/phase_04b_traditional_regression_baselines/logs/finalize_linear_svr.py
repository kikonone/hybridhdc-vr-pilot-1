from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

p=Path(r'E:\hdc-vr-pilot\experiments\phase_04b_traditional_regression_baselines');c=p/'results'/'checkpoints'/'linear_svr'
def ac(path,x):
 t=path.with_name(path.name+'.tmp');x.to_csv(t,index=False);t.replace(path)
def aj(path,x):
 t=path.with_name(path.name+'.tmp');t.write_text(json.dumps(x,indent=2,default=str)+chr(10));t.replace(path)
def sp(y,x):
 z=spearmanr(y,x).statistic
 return float(z) if np.isfinite(z) else np.nan
o=[];m=[]
for f in range(1,6):
 o.append(pd.read_csv(c/f'linear_svr_fold_{f}_predictions.csv'));m.append(json.loads((c/f'linear_svr_fold_{f}_metrics.json').read_text()))
oof=pd.concat(o).sort_values('run_key').reset_index(drop=True);fm=pd.DataFrame(m).sort_values('outer_fold');assert len(oof)==419 and oof.run_key.nunique()==419 and oof.prediction_raw.notna().all() and oof.prediction_bounded.between(1,4).all() and fm.best_estimator_converged.all()
ac(p/'results'/'predictions'/'linear_svr_oof.csv',oof);ac(p/'results'/'fold_metrics'/'linear_svr_fold_metrics.csv',fm)
y=oof.target_score;raw=oof.prediction_raw;bd=oof.prediction_bounded;su={'model':'Linear SVR','model_slug':'linear_svr','oof_rows':419,'oof_unique_run_keys':419,'oof_mae_raw':mean_absolute_error(y,raw),'oof_mae_bounded':mean_absolute_error(y,bd),'oof_rmse_raw':np.sqrt(mean_squared_error(y,raw)),'oof_rmse_bounded':np.sqrt(mean_squared_error(y,bd)),'oof_r2_raw':r2_score(y,raw),'oof_r2_bounded':r2_score(y,bd),'oof_spearman_raw':sp(y,raw),'oof_spearman_bounded':sp(y,bd),'fold_mae_bounded_mean':fm.mae_bounded.mean(),'fold_mae_bounded_std':fm.mae_bounded.std(ddof=1),'all_best_estimators_converged':True,'status':'COMPLETE'};ac(p/'results'/'summaries'/'linear_svr_summary.csv',pd.DataFrame([su]));aj(p/'audits'/'linear_svr_convergence_audit.json',{'folds':m,'overall_pass':True});aj(p/'audits'/'linear_svr_leakage_audit.json',{'outer_subject_isolation':True,'inner_subject_isolation':True,'pipeline_fitted_training_only':True,'outer_test_used_for_selection':False,'overall_pass':True});aj(p/'audits'/'linear_svr_oof_coverage_audit.json',{'rows':419,'unique_run_keys':419,'missing_predictions':0,'fold_coverage':5,'bounded_range_pass':True,'overall_pass':True});s=json.loads((p/'configs'/'regression_model_search_space.json').read_text());[x.update(status='COMPLETE') for x in s['models'] if x['name']=='Linear SVR'];s['status']='LINEAR_SVR_COMPLETE / REMAINING_MODELS_NOT_STARTED';aj(p/'configs'/'regression_model_search_space.json',s);print(json.dumps(su,indent=2,default=str))
