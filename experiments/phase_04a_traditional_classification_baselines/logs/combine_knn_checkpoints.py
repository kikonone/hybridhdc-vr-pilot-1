from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; base=ROOT/'experiments/phase_04a_traditional_classification_baselines'; d=base/'results/checkpoints/knn'
files=[d/f'knn_fold_{i}_predictions.csv' for i in [1,2,3,4,5]]
oof=pd.concat([pd.read_csv(f) for f in files],ignore_index=True).sort_values(['outer_fold','run_key'])
if len(oof)!=419 or oof.run_key.nunique()!=419: raise RuntimeError('Invalid KNN OOF coverage')
oof.to_csv(base/'results/predictions/knn_oof.csv',index=False); print(len(oof),oof.run_key.nunique())
