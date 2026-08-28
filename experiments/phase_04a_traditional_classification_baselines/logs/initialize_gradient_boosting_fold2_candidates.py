from pathlib import Path
import json, hashlib
import pandas as pd
from sklearn.model_selection import GroupKFold
from gradient_boosting_candidate_checkpoints import candidate_manifest, atomic_csv, load_or_initialize, eligible

ROOT=Path(__file__).resolve().parents[3]; base=ROOT/'experiments/phase_04a_traditional_classification_baselines'; cp=base/'results/checkpoints/gradient_boosting'
manifest=candidate_manifest(2); manifest_path=cp/'fold_2_search_candidates.csv'; atomic_csv(manifest,manifest_path)
if not pd.read_csv(manifest_path).equals(manifest): raise RuntimeError('manifest readback mismatch')
results=load_or_initialize(cp/'gradient_boosting_fold_2_inner_search_results.csv',manifest)
data=pd.read_csv(ROOT/'experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv'); train=data[data.outer_fold!=2] if 'outer_fold' in data else data.merge(pd.read_csv(ROOT/'experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv')[['run_key','outer_fold']],on='run_key').query('outer_fold != 2')
splits=[]
for i,(a,b) in enumerate(GroupKFold(3).split(train,train.target_class,train.subject_id),1):
    overlap=set(train.iloc[a].subject_id)&set(train.iloc[b].subject_id)
    if overlap: raise RuntimeError('inner subject overlap')
    splits.append({'inner_fold':i,'train_run_keys':train.iloc[a].run_key.tolist(),'validation_run_keys':train.iloc[b].run_key.tolist()})
(cp/'fold_2_inner_splits.json').write_text(json.dumps(splits,indent=2),encoding='utf-8')
print({'manifest_rows':len(manifest),'eligible_candidate_ids':eligible(manifest,results).candidate_id.tolist(),'inner_splits':len(splits)})
