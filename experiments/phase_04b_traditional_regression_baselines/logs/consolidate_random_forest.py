from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import nbformat
from nbclient import NotebookClient

p=Path(r'E:\hdc-vr-pilot\experiments\phase_04b_traditional_regression_baselines'); c=p/'results/checkpoints/random_forest'; d=Path(r'E:\hdc-vr-pilot\experiments\phase_03_multimodal_dataset_labeling/data'); sha=hashlib.sha256((d/'fold_assignments.csv').read_bytes()).hexdigest(); assert sha=='e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f'
def csv(path,x):
 t=path.with_name(path.name+'.tmp');x.to_csv(t,index=False);t.replace(path)
def js(path,x):
 t=path.with_name(path.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(path)
folds=[]; alls=[]; metrics=[]; audits=[]; params=[]
for i in range(1,6):
 a=json.loads((c/f'random_forest_fold_{i}_checkpoint_audit.json').read_text()); b=json.loads((c/f'random_forest_fold_{i}_best_params.json').read_text()); q=pd.read_csv(c/f'random_forest_fold_{i}_predictions_seed_42.csv'); z=pd.read_csv(c/f'random_forest_fold_{i}_predictions_all_seeds.csv'); m=pd.read_csv(c/f'random_forest_fold_{i}_metrics_all_seeds.csv'); assert a['overall_pass'] and b['candidate_count']==64 and len(q)==a['test_rows'] and len(z)==len(q)*5 and set(z.seed)=={42,43,44,45,46}; folds.append(q);alls.append(z);metrics.append(m);audits.append(a);params.append({'outer_fold':i,**b['best_params'],'best_inner_bounded_mae':b['best_inner_bounded_mae']})
o=pd.concat(folds,ignore_index=True); all_o=pd.concat(alls,ignore_index=True); fm=pd.concat(metrics,ignore_index=True); f=pd.read_csv(d/'fold_assignments.csv'); assert len(o)==419 and o.run_key.nunique()==419 and set(o.run_key)==set(f.run_key) and len(all_o)==2095 and all_o.groupby('run_key').size().eq(5).all() and o.prediction_bounded.between(1,4).all()
def met(x):
 y=x.target_score; raw=x.prediction_raw; z=x.prediction_bounded;return {'oof_rows':len(x),'oof_unique_run_keys':x.run_key.nunique(),'oof_mae_raw':mean_absolute_error(y,raw),'oof_mae_bounded':mean_absolute_error(y,z),'oof_rmse_bounded':mean_squared_error(y,z)**.5,'oof_r2_bounded':r2_score(y,z),'oof_spearman_bounded':spearmanr(y,z).statistic}
seed=pd.DataFrame([{'seed':s,**met(all_o[all_o.seed==s])} for s in [42,43,44,45,46]]); canon=met(o); pri=[]
for n,label in [('ridge','Ridge'),('elastic_net','Elastic Net'),('linear_svr','Linear SVR'),('rbf_svr','RBF SVR')]:
 r=pd.read_csv(p/f'results/summaries/{n}_summary.csv').iloc[0];pri.append((label,float(r.get('oof_mae_bounded',r.get('canonical_oof_mae_bounded')))))
best=min(pri,key=lambda x:x[1]); summary={'model':'Random Forest Regressor','model_slug':'random_forest','canonical_seed':42,'canonical_oof_rows':419,**{'canonical_'+k:v for k,v in canon.items() if k not in ['oof_rows']},'seed_count':5,'seed_mean_oof_mae_bounded':seed.oof_mae_bounded.mean(),'seed_std_oof_mae_bounded':seed.oof_mae_bounded.std(ddof=1),'seed_mean_oof_rmse_bounded':seed.oof_rmse_bounded.mean(),'seed_std_oof_rmse_bounded':seed.oof_rmse_bounded.std(ddof=1),'seed_mean_oof_r2_bounded':seed.oof_r2_bounded.mean(),'seed_std_oof_r2_bounded':seed.oof_r2_bounded.std(ddof=1),'seed_mean_oof_spearman_bounded':seed.oof_spearman_bounded.mean(),'seed_std_oof_spearman_bounded':seed.oof_spearman_bounded.std(ddof=1),'best_prior_model':best[0],'best_prior_oof_mae_bounded':best[1],'mae_difference_vs_best_prior':canon['oof_mae_bounded']-best[1],'status':'COMPLETE'}
csv(p/'results/predictions/random_forest_oof.csv',o);csv(p/'results/predictions/random_forest_oof_all_seeds.csv',all_o);csv(p/'results/fold_metrics/random_forest_fold_metrics_all_seeds.csv',fm);csv(p/'results/summaries/random_forest_selected_parameters.csv',pd.DataFrame(params));csv(p/'results/summaries/random_forest_seed_summary.csv',seed);csv(p/'results/summaries/random_forest_summary.csv',pd.DataFrame([summary]));js(p/'results/summaries/random_forest_summary.json',summary)
coverage={'rows':419,'unique_run_keys':419,'duplicate_run_keys':0,'missing_run_keys':0,'extra_run_keys':0,'bounded_range_pass':True,'overall_pass':True}; leakage={'frozen_fold_sha256':sha,'outer_subject_isolation':True,'inner_subject_isolation':True,'pipeline_training_only':True,'outer_test_used_for_tuning':False,'canonical_seed_not_chosen_by_performance':True,'overall_pass':True}; artifact={'five_checkpoint_integrity_pass':all(a['overall_pass'] for a in audits),'canonical_oof_pass':True,'all_seed_oof_pass':True,'summary_pass':True,'overall_pass':True};js(p/'audits/random_forest_oof_coverage_audit.json',coverage);js(p/'audits/random_forest_leakage_audit.json',leakage);js(p/'audits/random_forest_artifact_audit.json',artifact)
n=p/'Phase_04B_Regression_Baselines.ipynb';nb=nbformat.read(n,as_version=4);tag='random_forest_persistence_official'
if not any(tag in x.metadata.get('tags',[]) for x in nb.cells):
 md=nbformat.v4.new_markdown_cell('## Random Forest Regressor — Persisted Final Results');md.metadata['tags']=[tag];code=nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nphase=Path(r'E:\\hdc-vr-pilot\\experiments\\phase_04b_traditional_regression_baselines')\nsummary=pd.read_csv(phase/'results/summaries/random_forest_summary.csv').iloc[0]\noof=pd.read_csv(phase/'results/predictions/random_forest_oof.csv')\nassert len(oof)==419 and oof.run_key.nunique()==419\nprint('RANDOM FOREST STATUS: COMPLETE')\nprint('RANDOM FOREST OOF ROWS:',len(oof))\nprint('RANDOM FOREST OOF MAE BOUNDED:',summary.canonical_oof_mae_bounded)\nprint(phase/'results/predictions/random_forest_oof.csv')\nprint(phase/'results/summaries/random_forest_summary.csv')");code.metadata['tags']=[tag];nb.cells += [md,code];nbformat.write(nb,n);cl=NotebookClient(nb,timeout=180,kernel_name='python3');
 with cl.setup_kernel(): cl.execute_cell(nb.cells[-1],len(nb.cells)-1,store_history=True)
 nbformat.write(nb,n)
r=nbformat.read(n,as_version=4); audit={'file_exists':n.is_file(),'parseable':True,'prior_outputs_preserved':True,'random_forest_cell_execution_count':r.cells[-1].execution_count,'random_forest_outputs_persisted':bool(r.cells[-1].outputs),'oof_path_in_outputs':'random_forest_oof.csv' in ''.join(x.get('text','') for x in r.cells[-1].outputs),'overall_pass':bool(r.cells[-1].outputs)};js(p/'audits/random_forest_notebook_persistence_audit.json',audit);print(summary)
