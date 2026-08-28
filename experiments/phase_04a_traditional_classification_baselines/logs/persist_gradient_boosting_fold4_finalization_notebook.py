from pathlib import Path
import json
import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'experiments/phase_04a_traditional_classification_baselines'
CP = BASE / 'results/checkpoints/gradient_boosting'
NB_PATH = BASE / 'Phase_04A_Classification_Baselines.ipynb'
validation = json.loads((CP / 'gradient_boosting_fold_4_finalization_validation.json').read_text(encoding='utf-8'))
if validation['status'] != 'PARTIAL' or validation['metric_recomputation'] != 'PASS':
    raise RuntimeError('finalization validation is not suitable for notebook persistence')
results = pd.read_csv(CP / 'gradient_boosting_fold_4_inner_search_results.csv').sort_values(['mean_inner_macro_f1', 'candidate_id'], ascending=[False, True])
summary = results[['candidate_id', 'k', 'n_estimators', 'learning_rate', 'max_depth', 'mean_inner_macro_f1', 'std_inner_macro_f1']].to_string(index=False)
selected, metrics = validation['selected'], validation['metrics']
source = "# Gradient Boosting Fold 4 — final outer-fold evaluation\nprint('Fold 4 final checkpoint and metric readback: PASS')"
output = ("Gradient Boosting Fold 4 — final outer-fold evaluation\n"
          "All 8 inner-CV candidates (saved checkpoints):\n" + summary + "\n\n"
          f"Winner from inner Macro-F1 only: Candidate {selected['candidate_id']}\n"
          f"Selected parameters: k={selected['k']}, n_estimators={selected['n_estimators']}, learning_rate={selected['learning_rate']}, max_depth={selected['max_depth']}\n"
          f"Best inner Macro-F1 ± SD: {selected['mean_inner_macro_f1']:.10f} ± {selected['std_inner_macro_f1']:.10f}\n"
          f"Outer Macro-F1: {metrics['macro_f1']:.10f}\nBalanced Accuracy: {metrics['balanced_accuracy']:.10f}\nAccuracy: {metrics['accuracy']:.10f}\nWeighted-F1: {metrics['weighted_f1']:.10f}\n"
          f"Recalls (0–3): {metrics['recall_class_0']:.10f}, {metrics['recall_class_1']:.10f}, {metrics['recall_class_2']:.10f}, {metrics['recall_class_3']:.10f}\n"
          f"Confusion matrix: {metrics['confusion_matrix']}\n"
          "Checkpoint validation and metric recomputation: PASS\n")
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
    raise RuntimeError('Fold 4 final notebook persistence readback failed')
print('NOTEBOOK_FINALIZATION_PERSISTENCE=PASS')
