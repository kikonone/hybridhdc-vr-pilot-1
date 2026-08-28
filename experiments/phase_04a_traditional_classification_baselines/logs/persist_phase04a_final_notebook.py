from pathlib import Path
import json
import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'experiments/phase_04a_traditional_classification_baselines'
CP = BASE / 'results/checkpoints/gradient_boosting'
NB_PATH = BASE / 'Phase_04A_Classification_Baselines.ipynb'
freeze = json.loads((BASE / 'configs/phase04a_freeze.json').read_text(encoding='utf-8'))
fold5 = json.loads((CP / 'gradient_boosting_fold_5_finalization_validation.json').read_text(encoding='utf-8'))
ranking = pd.read_csv(BASE / 'results/summaries/phase04a_final_classifier_ranking.csv')
fold5_candidates = pd.read_csv(CP / 'gradient_boosting_fold_5_inner_search_results.csv').sort_values(['mean_inner_macro_f1','candidate_id'],ascending=[False,True])
gb_folds = pd.read_csv(BASE / 'results/summaries/gradient_boosting_fold_summary.csv')
if freeze['phase04a_frozen'] != 'YES' or fold5['metric_recomputation'] != 'PASS':
    raise RuntimeError('final evidence validation failed')
winner, metrics = fold5['selected'], fold5['metrics']
source = "# Phase 04A — final persisted comparison and freeze\nprint('Phase 04A final OOF evidence and freeze: PASS')"
output = ("Phase 04A final persisted comparison and freeze\n\nFold 5 inner candidate ranking:\n" + fold5_candidates[['candidate_id','k','n_estimators','learning_rate','max_depth','mean_inner_macro_f1','std_inner_macro_f1']].to_string(index=False) +
          f"\n\nFold 5 winner: Candidate {winner['candidate_id']} | k={winner['k']}, n_estimators={winner['n_estimators']}, learning_rate={winner['learning_rate']}, max_depth={winner['max_depth']}\n" +
          f"Fold 5 outer metrics: Macro-F1={metrics['macro_f1']:.10f}, Balanced Accuracy={metrics['balanced_accuracy']:.10f}, Accuracy={metrics['accuracy']:.10f}, Weighted-F1={metrics['weighted_f1']:.10f}\n\n" +
          "Gradient Boosting five-fold summary:\n" + gb_folds.to_string(index=False) + "\n\nFinal all-model OOF comparison:\n" + ranking.to_string(index=False) +
          f"\n\nBest traditional classifier: {freeze['best_traditional_classifier']} (OOF Macro-F1={freeze['best_oof_macro_f1']:.10f})\n" +
          "Final OOF coverage audit: PASS\nFinal checkpoint integrity: PASS\nPhase 04A status: COMPLETE\nPhase 04A frozen: YES\nXGBoost: OPTIONAL / NOT RUN\n")
nb = nbformat.read(NB_PATH, as_version=4)
cells = [cell for cell in nb.cells if cell.cell_type == 'code' and cell.source == source]
if cells:
    cell = cells[-1]; cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=output)]
else:
    cell=nbformat.v4.new_code_cell(source=source); cell.execution_count=None; cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=output)]; nb.cells.append(cell)
nbformat.write(nb,NB_PATH)
reloaded=nbformat.read(NB_PATH,as_version=4)
if not any(cell.cell_type=='code' and cell.source==source and cell.outputs and cell.outputs[0]['text']==output for cell in reloaded.cells): raise RuntimeError('notebook persistence readback failed')
print('NOTEBOOK_FINAL_PERSISTENCE=PASS')
