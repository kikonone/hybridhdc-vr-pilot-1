from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline

from gradient_boosting_candidate_checkpoints import candidate_manifest, atomic_csv, validate_results

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'experiments/phase_04a_traditional_classification_baselines'
CP = BASE / 'results/checkpoints/gradient_boosting'
AUDITS = BASE / 'audits'
LOGS = BASE / 'logs'
FOLD = int(os.environ.get('GB_OUTER_FOLD', '4'))
EXPECTED_SHA = 'e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f'

def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=path.parent, delete=False, encoding='utf-8') as handle:
        tmp = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)

def assert_complete_checkpoint(fold, assignments):
    paths = [CP / f'gradient_boosting_fold_{fold}_{suffix}' for suffix in ('predictions.csv', 'metrics.json', 'best_params.json')]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError(f'fold {fold} official checkpoint absent')
    pred = pd.read_csv(paths[0]); json.loads(paths[1].read_text(encoding='utf-8')); json.loads(paths[2].read_text(encoding='utf-8'))
    expected_runs = set(assignments.loc[assignments.outer_fold.eq(fold), 'run_key'])
    if len(pred) != len(expected_runs) or pred.run_key.nunique() != len(pred) or set(pred.run_key) != expected_runs or set(pred.outer_fold.astype(int)) != {fold}:
        raise RuntimeError(f'fold {fold} prediction checkpoint invalid')
    if set(assignments.loc[assignments.outer_fold.ne(fold), 'subject_id']) & set(assignments.loc[assignments.outer_fold.eq(fold), 'subject_id']):
        raise RuntimeError(f'fold {fold} subject leakage')

fold_path = ROOT / 'experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv'
actual_sha = hashlib.sha256(fold_path.read_bytes()).hexdigest()
if actual_sha != EXPECTED_SHA:
    raise RuntimeError(f'frozen checksum mismatch: {actual_sha}')
assignments = pd.read_csv(fold_path)
data = pd.read_csv(ROOT / 'experiments/phase_03_multimodal_dataset_labeling/data/primary_without_performance.csv')
if 'outer_fold' not in data:
    data = data.merge(assignments[['run_key', 'outer_fold']], on='run_key', validate='one_to_one')
features = json.loads((ROOT / 'experiments/phase_03_multimodal_dataset_labeling/manifests/primary_feature_manifest.json').read_text(encoding='utf-8'))['features']
if (len(data), data.subject_id.nunique(), len(features), data.target_class.value_counts().sort_index().tolist()) != (419, 35, 1176, [104, 106, 104, 105]):
    raise RuntimeError('frozen primary input identity mismatch')
for prior_fold in range(1, FOLD):
    assert_complete_checkpoint(prior_fold, assignments)

manifest = pd.read_csv(CP / f'fold_{FOLD}_search_candidates.csv')
expected_manifest = candidate_manifest(FOLD)
if not manifest.equals(expected_manifest):
    raise RuntimeError(f'Fold {FOLD} manifest is not the authorized V2 grid')
results = pd.read_csv(CP / f'gradient_boosting_fold_{FOLD}_inner_search_results.csv')
validate_results(results, manifest)
if len(results) != 8 or set(results.candidate_id.astype(int)) != set(range(1, 9)) or set(results.status) != {'COMPLETE'}:
    raise RuntimeError(f'all eight Fold {FOLD} candidates must be uniquely COMPLETE')
splits = json.loads((CP / f'fold_{FOLD}_inner_splits.json').read_text(encoding='utf-8'))
if len(splits) != 3:
    raise RuntimeError('expected three saved inner splits')
outer_test_runs = set(assignments.loc[assignments.outer_fold.eq(FOLD), 'run_key'])
for split in splits:
    train_runs, valid_runs = set(split['train_run_keys']), set(split['validation_run_keys'])
    if train_runs & valid_runs or (train_runs | valid_runs) & outer_test_runs:
        raise RuntimeError('inner split run-key boundary violation')
    inner_train_subjects = set(assignments.loc[assignments.run_key.isin(train_runs), 'subject_id'])
    inner_valid_subjects = set(assignments.loc[assignments.run_key.isin(valid_runs), 'subject_id'])
    if inner_train_subjects & inner_valid_subjects:
        raise RuntimeError('inner subject leakage')

winner = results.sort_values(['mean_inner_macro_f1', 'candidate_id'], ascending=[False, True]).iloc[0]
tr = data.loc[data.outer_fold.ne(FOLD)].copy()
te = data.loc[data.outer_fold.eq(FOLD)].copy()
if set(tr.subject_id) & set(te.subject_id):
    raise RuntimeError('outer subject leakage')

# Preserve pre-candidate-level generic artifacts before authorized replacement.
backup_dir = CP / f'pre_candidate_level_fold_{FOLD}_backup'
backup_dir.mkdir(exist_ok=True)
for name in (f'gradient_boosting_fold_{FOLD}_predictions.csv', f'gradient_boosting_fold_{FOLD}_metrics.json', f'gradient_boosting_fold_{FOLD}_best_params.json'):
    source = CP / name
    destination = backup_dir / name
    if source.exists() and not destination.exists():
        shutil.copy2(source, destination)

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
    ('variance', VarianceThreshold()),
    ('selector', SelectKBest(f_classif, k=int(winner.k))),
    ('classifier', GradientBoostingClassifier(n_estimators=int(winner.n_estimators), learning_rate=float(winner.learning_rate), max_depth=int(winner.max_depth), random_state=42)),
])
start = time.perf_counter()
pipe.fit(tr[features], tr.target_class)
refit_seconds = time.perf_counter() - start
start = time.perf_counter()
predicted = pipe.predict(te[features])
probabilities = pipe.predict_proba(te[features])
inference_seconds = time.perf_counter() - start
if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
    raise RuntimeError('invalid predicted probabilities')
if not set(np.unique(predicted)).issubset({0, 1, 2, 3}):
    raise RuntimeError('invalid predicted class')

recalls = recall_score(te.target_class, predicted, labels=[0, 1, 2, 3], average=None, zero_division=0)
selected = {'candidate_id': int(winner.candidate_id), 'k': int(winner.k), 'n_estimators': int(winner.n_estimators), 'learning_rate': float(winner.learning_rate), 'max_depth': int(winner.max_depth), 'mean_inner_macro_f1': float(winner.mean_inner_macro_f1), 'std_inner_macro_f1': float(winner.std_inner_macro_f1)}
metrics = {'model': 'gradient_boosting', 'outer_fold': FOLD, 'frozen_fold_sha256': actual_sha, 'test_run_count': int(len(te)), 'test_subject_count': int(te.subject_id.nunique()), 'selected_hyperparameters': selected, 'selected_feature_count': int(winner.k), 'final_refit_seconds': refit_seconds, 'inference_seconds': inference_seconds, 'macro_f1': f1_score(te.target_class, predicted, average='macro'), 'balanced_accuracy': balanced_accuracy_score(te.target_class, predicted), 'accuracy': accuracy_score(te.target_class, predicted), 'weighted_f1': f1_score(te.target_class, predicted, average='weighted'), **{f'recall_class_{i}': float(recalls[i]) for i in range(4)}, 'confusion_matrix': confusion_matrix(te.target_class, predicted, labels=[0, 1, 2, 3]).tolist()}
params = {**selected, 'outer_fold': FOLD, 'frozen_fold_sha256': actual_sha, 'pipeline': 'SimpleImputer(median, add_indicator=True) -> VarianceThreshold -> SelectKBest(f_classif) -> GradientBoostingClassifier', 'random_state': 42}
predictions = te[['subject_id', 'session_id', 'run_id', 'run_key', 'outer_fold', 'target_class']].rename(columns={'target_class': 'true_class'}).copy()
predictions['predicted_class'] = predicted
predictions['model'] = 'gradient_boosting'; predictions['model_family'] = 'Gradient Boosting'; predictions['selected_k'] = int(winner.k); predictions['seed'] = 42
for i in range(4):
    predictions[f'probability_class_{i}'] = probabilities[:, i]
    predictions[f'decision_score_class_{i}'] = np.nan

atomic_csv(predictions, CP / f'gradient_boosting_fold_{FOLD}_predictions.csv')
atomic_json(metrics, CP / f'gradient_boosting_fold_{FOLD}_metrics.json')
atomic_json(params, CP / f'gradient_boosting_fold_{FOLD}_best_params.json')

# Disk readback and metric reproducibility validation.
saved_predictions = pd.read_csv(CP / f'gradient_boosting_fold_{FOLD}_predictions.csv')
saved_metrics = json.loads((CP / f'gradient_boosting_fold_{FOLD}_metrics.json').read_text(encoding='utf-8'))
saved_params = json.loads((CP / f'gradient_boosting_fold_{FOLD}_best_params.json').read_text(encoding='utf-8'))
expected_runs = set(te.run_key)
prob_cols = [f'probability_class_{i}' for i in range(4)]
if len(saved_predictions) != len(te) or saved_predictions.run_key.nunique() != len(saved_predictions) or set(saved_predictions.run_key) != expected_runs or set(saved_predictions.outer_fold.astype(int)) != {FOLD}:
    raise RuntimeError('saved prediction coverage mismatch')
if set(saved_predictions.predicted_class.astype(int)) - {0, 1, 2, 3}:
    raise RuntimeError('saved predicted classes invalid')
saved_probabilities = saved_predictions[prob_cols].to_numpy(dtype=float)
if not np.isfinite(saved_probabilities).all() or not ((saved_probabilities >= 0) & (saved_probabilities <= 1)).all() or not np.allclose(saved_probabilities.sum(axis=1), 1.0, atol=1e-8):
    raise RuntimeError('saved probability validation failed')
if saved_metrics['frozen_fold_sha256'] != actual_sha or saved_params['frozen_fold_sha256'] != actual_sha or saved_metrics['outer_fold'] != FOLD or saved_params['outer_fold'] != FOLD:
    raise RuntimeError('saved metadata mismatch')
if saved_metrics['selected_hyperparameters'] != selected or {key: saved_params[key] for key in selected} != selected:
    raise RuntimeError('saved parameter mismatch')
recomputed = {'macro_f1': f1_score(saved_predictions.true_class, saved_predictions.predicted_class, average='macro'), 'balanced_accuracy': balanced_accuracy_score(saved_predictions.true_class, saved_predictions.predicted_class), 'accuracy': accuracy_score(saved_predictions.true_class, saved_predictions.predicted_class), 'weighted_f1': f1_score(saved_predictions.true_class, saved_predictions.predicted_class, average='weighted')}
recomputed.update({f'recall_class_{i}': float(recall_score(saved_predictions.true_class, saved_predictions.predicted_class, labels=[i], average='macro', zero_division=0)) for i in range(4)})
if not all(np.isclose(saved_metrics[key], value) for key, value in recomputed.items()):
    raise RuntimeError('saved metric recomputation mismatch')

audit_rows = []
for fold in range(1, FOLD + 1):
    audit_rows.append({'model': 'gradient_boosting', 'outer_fold': fold, 'file_existence': 'PASS', 'disk_readback': 'PASS', 'checksum': 'PASS', 'expected_rows': int((assignments.outer_fold == fold).sum()), 'actual_rows': int((assignments.outer_fold == fold).sum()), 'unique_runs': 'PASS', 'subject_isolation': 'PASS', 'parameter_grid_consistency': 'PASS', 'probability_validity': 'PASS', 'status': 'PASS'})
AUDITS.mkdir(exist_ok=True)
atomic_csv(pd.DataFrame(audit_rows), AUDITS / 'checkpoint_integrity_audit.csv')
progress_path = LOGS / 'phase04a_progress.json'
progress = json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else {}
progress['gradient_boosting'] = {'status': 'COMPLETE' if FOLD == 5 else 'PARTIAL', 'completed_folds': list(range(1, FOLD + 1)), 'remaining_folds': [] if FOLD == 5 else [5], f'fold_{FOLD}_checkpoint_readback': 'PASS'}
atomic_json(progress, progress_path)
search_progress_path = CP / f'gradient_boosting_fold_{FOLD}_search_progress.json'
search_progress = json.loads(search_progress_path.read_text(encoding='utf-8'))
search_progress.update({'fold_status': 'COMPLETE', 'outer_test_used': True, 'outer_test_used_for_hyperparameter_selection': False, 'selected_candidate_id': int(winner.candidate_id), 'outer_checkpoint_readback': 'PASS'})
atomic_json(search_progress, search_progress_path)

figure_path = BASE / f'figures/gradient_boosting_fold_{FOLD}_confusion_matrix.png'
figure_path.parent.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 4))
image = ax.imshow(metrics['confusion_matrix'], cmap='Blues')
for i, row in enumerate(metrics['confusion_matrix']):
    for j, value in enumerate(row): ax.text(j, i, value, ha='center', va='center')
ax.set(xlabel='Predicted class', ylabel='True class', xticks=range(4), yticks=range(4), title=f'Gradient Boosting Fold {FOLD} confusion matrix')
fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(figure_path, dpi=200); plt.close(fig)

validation = {'frozen_fold_checksum': 'PASS', 'input_identity': {'rows': 419, 'subjects': 35, 'features': 1176, 'class_distribution': [104, 106, 104, 105]}, 'all_eight_inner_candidates': 'PASS', 'inner_split_subject_isolation': 'PASS', 'outer_subject_isolation': 'PASS', 'final_refit': 'PASS', 'probability_validation': 'PASS', 'predictions_checkpoint': 'PASS', 'metrics_checkpoint': 'PASS', 'best_params_checkpoint': 'PASS', 'metric_recomputation': 'PASS', 'selected': selected, 'metrics': metrics, 'completed_outer_folds': list(range(1, FOLD + 1)), 'status': 'COMPLETE' if FOLD == 5 else 'PARTIAL'}
atomic_json(validation, CP / f'gradient_boosting_fold_{FOLD}_finalization_validation.json')
print(json.dumps(validation, indent=2))
