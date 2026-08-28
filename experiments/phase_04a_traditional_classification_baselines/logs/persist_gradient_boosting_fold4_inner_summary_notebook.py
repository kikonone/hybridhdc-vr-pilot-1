from pathlib import Path
import os
import json
import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'experiments/phase_04a_traditional_classification_baselines'
CP = BASE / 'results/checkpoints/gradient_boosting'
NB_PATH = BASE / 'Phase_04A_Classification_Baselines.ipynb'
FOLD = int(os.environ.get('GB_OUTER_FOLD', '4'))
results = pd.read_csv(CP / f'gradient_boosting_fold_{FOLD}_inner_search_results.csv')
if len(results) != 8 or set(results.status) != {'COMPLETE'}:
    raise RuntimeError('all eight candidates must be COMPLETE before summary persistence')
progress = json.loads((CP / f'gradient_boosting_fold_{FOLD}_search_progress.json').read_text(encoding='utf-8'))
if progress['outer_test_used'] or progress['fold_status'] != 'INCOMPLETE':
    raise RuntimeError('outer-test boundary or fold state invalid')
view = results.sort_values(['mean_inner_macro_f1', 'candidate_id'], ascending=[False, True])[
    ['candidate_id', 'k', 'n_estimators', 'learning_rate', 'max_depth', 'mean_inner_macro_f1', 'std_inner_macro_f1']
].copy()
for col in ('candidate_id', 'k', 'n_estimators', 'max_depth'):
    view[col] = view[col].astype(int)
source = f"# Gradient Boosting Fold {FOLD} — complete inner-CV candidate summary\nprint('Eight-candidate inner-CV checkpoint summary; outer test has not been used.')"
output = f'Gradient Boosting Fold {FOLD} — all 8 inner-CV candidates\n' + view.to_string(index=False) + '\nouter test used for evaluation: NO\nfold status: INCOMPLETE\n'
nb = nbformat.read(NB_PATH, as_version=4)
matching = [cell for cell in nb.cells if cell.cell_type == 'code' and cell.source == source]
if matching:
    cell = matching[-1]
    cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text=output)]
else:
    cell = nbformat.v4.new_code_cell(source=source)
    cell.execution_count = None
    cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text=output)]
    nb.cells.append(cell)
nbformat.write(nb, NB_PATH)
reloaded = nbformat.read(NB_PATH, as_version=4)
if not any(cell.cell_type == 'code' and cell.source == source and cell.outputs and cell.outputs[0]['text'] == output for cell in reloaded.cells):
    raise RuntimeError('summary notebook persistence readback failed')
print(output)
print('NOTEBOOK_SUMMARY_PERSISTENCE=PASS')
