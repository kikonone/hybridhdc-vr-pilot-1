from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
base=ROOT/'experiments/phase_04a_traditional_classification_baselines'
directory=base/'results/checkpoints/rbf_svm'
files=[directory/f'rbf_svm_fold_{fold}_predictions.csv' for fold in [1,2,3,4,5]]
if not all(path.is_file() for path in files): raise RuntimeError('Incomplete RBF SVM checkpoints')
oof=pd.concat([pd.read_csv(path) for path in files],ignore_index=True).sort_values(['outer_fold','run_key']).reset_index(drop=True)
if len(oof)!=419 or oof.run_key.nunique()!=419 or oof.duplicated(['model','run_key']).any(): raise RuntimeError('OOF coverage invalid')
oof.to_csv(base/'results/predictions/rbf_svm_oof.csv',index=False)
print(len(oof),oof.run_key.nunique())
