from pathlib import Path
import os
import json
import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'experiments/phase_04a_traditional_classification_baselines'
CP = BASE / 'results/checkpoints/gradient_boosting'
NB_PATH = BASE / 'Phase_04A_Classification_Baselines.ipynb'
candidate_id = int(os.environ.get('GB_CANDIDATE', '1'))
fold = int(os.environ.get('GB_OUTER_FOLD', '4'))
result = pd.read_csv(CP / f'gradient_boosting_fold_{fold}_inner_search_results.csv').query('candidate_id == @candidate_id').iloc[0].to_dict()
progress = json.loads((CP / f'gradient_boosting_fold_{fold}_search_progress.json').read_text(encoding='utf-8'))
if result['status'] != 'COMPLETE' or progress['outer_test_used']:
    raise RuntimeError('cannot persist invalid Candidate 1 evidence')
source = f"# Gradient Boosting Fold {fold} — Candidate {candidate_id} persisted checkpoint\nprint('Gradient Boosting Fold {fold} Candidate {candidate_id} checkpoint readback: PASS')"
output = (f"Gradient Boosting Fold {fold} — Candidate {candidate_id}\nstatus: COMPLETE\n"
          f"parameters: k={int(result['k'])}, n_estimators={int(result['n_estimators'])}, learning_rate={result['learning_rate']}, max_depth={int(result['max_depth'])}\n"
          f"inner Macro-F1: {result['inner_fold_1_macro_f1']:.10f}, {result['inner_fold_2_macro_f1']:.10f}, {result['inner_fold_3_macro_f1']:.10f}\n"
          f"mean ± std: {result['mean_inner_macro_f1']:.10f} ± {result['std_inner_macro_f1']:.10f}\n"
          "outer test used for evaluation: NO\ncheckpoint readback: PASS\n")
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
    raise RuntimeError('notebook persistence readback failed')
print('NOTEBOOK_PERSISTENCE=PASS')
