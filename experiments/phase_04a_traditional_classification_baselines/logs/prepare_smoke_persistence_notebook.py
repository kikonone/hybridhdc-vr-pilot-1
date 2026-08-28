from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[3]
source=ROOT/'experiments/phase_04a_traditional_classification_baselines/Phase_04A_Classification_Baselines.ipynb'
target=ROOT/'experiments/phase_04a_traditional_classification_baselines/Phase_04A_Classification_Baselines.executed.ipynb'
nb=nbformat.read(source,as_version=4)
for cell in nb.cells:
    if cell.cell_type=='code' and cell.source.startswith('oof, fold_metrics, failures=R["run"]'):
        cell.source='''# Persistence-only run: full nested CV intentionally disabled.
RUN_FULL_EXPERIMENT = False
print("RUN_FULL_EXPERIMENT:", RUN_FULL_EXPERIMENT)
print("Full nested CV skipped; smoke-test evidence is the persistence target.")'''
    elif cell.cell_type=='code' and any(token in cell.source for token in ['fold_metrics.round', 'oof.groupby', 'import numpy as np, pandas as pd, matplotlib', 'json.dumps(best', 'best_frame=oof', 'coverage=[]', 'required=[ctx["out"]']):
        cell.source='''print("Skipped: requires full nested CV outputs; persistence-only smoke execution.")'''
nbformat.write(nb,target)
print(target)
