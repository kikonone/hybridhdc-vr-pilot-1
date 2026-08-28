from pathlib import Path
import json,time,warnings,hashlib
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold,SelectKBest,f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold,GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,make_scorer
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import spearmanr
P=Path(r'E:\hdc-vr-pilot\experiments\phase_04b_traditional_regression_baselines');D=Path(r'E:\hdc-vr-pilot\experiments\phase_03_multimodal_dataset_labeling\data\primary_without_performance.csv');F=D.with_name('fold_assignments.csv');d=pd.read_csv(D);fo=pd.read_csv(F);sha=hashlib.sha256(F.read_bytes()).hexdigest();C=P/'results'/'checkpoints'/'elastic_net';
def wj(x,p):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,default=str)+chr(10));t.replace(p)
def wc(x,p):
 t=p.with_name(p.name+'.tmp');x.to_csv(t,index=False);t.replace(p)
def sc(y,p):return -mean_absolute_error(y,np.clip(p,1,4))
def sp(y,p):
 with warnings.catch_warnings():warnings.simplefilter('ignore');z=spearmanr(y,p).statistic
 return float(z) if np.isfinite(z) else np.nan
N=['subject_id','session_id','run_id','difficulty_level_raw','difficulty_level','run_key','target_class','target_score','outer_fold'];X=[x for x in d if x not in N];grid={'feature_selection__k':[50,100,200,'all'],'regressor__alpha':[.001,.01,.1,1.],'regressor__l1_ratio':[.1,.5,.9]};recs=[]
for f in [3,4,5]:
 mask=fo.outer_fold.eq(f).to_numpy();tr=d.loc[~mask];te=d.loc[mask];assert not(set(tr.subject_id)&set(te.subject_id));spl=list(GroupKFold(3).split(tr[X],tr.target_score,tr.subject_id));pipe=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('variance_filter',VarianceThreshold(0)),('scaler',StandardScaler()),('feature_selection',SelectKBest(f_regression)),('regressor',ElasticNet(fit_intercept=True,max_iter=100000,tol=1e-4,selection='cyclic'))])
 with warnings.catch_warnings(record=True) as ww:
  warnings.simplefilter('always',ConvergenceWarning);st=time.perf_counter();gs=GridSearchCV(pipe,grid,cv=spl,scoring=make_scorer(sc),n_jobs=1,return_train_score=False,error_score='raise').fit(tr[X],tr.target_score);ft=time.perf_counter()-st
 reg=gs.best_estimator_.named_steps['regressor'];ni=int(np.max(np.atleast_1d(reg.n_iter_)));conv=ni<100000;cv=pd.DataFrame(gs.cv_results_);ir=pd.DataFrame({'candidate_index':range(48),'feature_selection__k':cv.param_feature_selection__k.astype(str),'regressor__alpha':cv.param_regressor__alpha.astype(float),'regressor__l1_ratio':cv.param_regressor__l1_ratio.astype(float),'mean_validation_bounded_mae':-cv.mean_test_score,'std_validation_bounded_mae':cv.std_test_score,'rank':cv.rank_test_score,'search_convergence_warning_count_total':sum(issubclass(q.category,ConvergenceWarning) for q in ww),'candidate_level_warning_mapping':'NOT_AVAILABLE_FROM_GRIDSEARCHCV','status':'COMPLETE'});[ir.__setitem__(f'split_{i+1}_validation_bounded_mae',-cv[f'split{i}_test_score']) for i in range(3)];wc(ir,C/f'elastic_net_fold_{f}_inner_search_results.csv');bp=gs.best_params_;wj({'outer_fold':f,'best_params':bp,'candidate_count':48,'best_inner_bounded_mae':-gs.best_score_,'n_iter':ni,'max_iter_used':100000,'tol':1e-4,'converged':conv,'frozen_fold_sha256':sha,'recovery_version':'V1'},C/f'elastic_net_fold_{f}_best_params.json');wj({'outer_fold':f,'n_iter':ni,'max_iter':100000,'warning_count_total':sum(issubclass(q.category,ConvergenceWarning) for q in ww),'converged':conv},C/f'elastic_net_fold_{f}_convergence_diagnostics.json');assert conv
 st=time.perf_counter();raw=gs.predict(te[X]);pt=time.perf_counter()-st;bd=np.clip(raw,1,4);o=te[['run_key','subject_id','session_id','run_id','outer_fold','target_score']].copy();o.insert(0,'model_slug','elastic_net');o.insert(0,'model','Elastic Net');o['prediction_raw']=raw;o['prediction_bounded']=bd;o['absolute_error_raw']=abs(te.target_score-raw);o['absolute_error_bounded']=abs(te.target_score-bd);o['selected_k']=bp['feature_selection__k'];o['selected_alpha']=bp['regressor__alpha'];o['selected_l1_ratio']=bp['regressor__l1_ratio'];o['nonzero_coefficient_count']=np.count_nonzero(reg.coef_);o['max_iter_used']=100000;o['checkpoint_reused']=False;o['recovery_version']='V1';wc(o,C/f'elastic_net_fold_{f}_predictions.csv');m={'model':'Elastic Net','model_slug':'elastic_net','outer_fold':f,'train_rows':len(tr),'test_rows':len(te),'train_subjects':tr.subject_id.nunique(),'test_subjects':te.subject_id.nunique(),'subject_overlap_count':0,'inner_candidate_count':48,'selected_k':bp['feature_selection__k'],'selected_alpha':bp['regressor__alpha'],'selected_l1_ratio':bp['regressor__l1_ratio'],'nonzero_coefficient_count':int(np.count_nonzero(reg.coef_)),'best_inner_bounded_mae':-gs.best_score_,'mae_raw':mean_absolute_error(te.target_score,raw),'mae_bounded':mean_absolute_error(te.target_score,bd),'rmse_raw':np.sqrt(mean_squared_error(te.target_score,raw)),'rmse_bounded':np.sqrt(mean_squared_error(te.target_score,bd)),'r2_raw':r2_score(te.target_score,raw),'r2_bounded':r2_score(te.target_score,bd),'spearman_raw':sp(te.target_score,raw),'spearman_bounded':sp(te.target_score,bd),'convergence_warning_count':sum(issubclass(q.category,ConvergenceWarning) for q in ww),'best_estimator_n_iter':ni,'best_estimator_converged':conv,'fit_and_search_time_seconds':ft,'prediction_time_seconds':pt,'max_iter_used':100000,'checkpoint_reused':False,'recovery_version':'V1'};wj(m,C/f'elastic_net_fold_{f}_metrics.json');recs.append(m)
print('RECOVERY FOLDS COMPLETE',recs)
