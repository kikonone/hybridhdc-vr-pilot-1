from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
base=ROOT/'experiments/phase_04a_traditional_classification_baselines'
checkpoints=base/'results/checkpoints/logistic_regression'
files=[checkpoints/f'logistic_regression_fold_{fold}_predictions.csv' for fold in [1,2,3,4,5]]
if not all(path.is_file() for path in files): raise RuntimeError('Incomplete Logistic Regression checkpoints')
oof=pd.concat([pd.read_csv(path) for path in files],ignore_index=True).sort_values(['outer_fold','run_key']).reset_index(drop=True)
if len(oof)!=419 or oof.run_key.nunique()!=419 or oof.duplicated(['model','run_key']).any(): raise RuntimeError('OOF coverage invalid')
oof.to_csv(base/'results/predictions/logistic_regression_oof.csv',index=False)
oof.to_csv(base/'results/oof/classification_oof_predictions.csv',index=False)
print(len(oof),oof.run_key.nunique())
