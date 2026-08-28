from pathlib import Path
import os
import hashlib
import json
import numpy as np
import pandas as pd
from gradient_boosting_candidate_checkpoints import candidate_manifest, validate_results

ROOT = Path(__file__).resolve().parents[3]
CP = ROOT / 'experiments/phase_04a_traditional_classification_baselines/results/checkpoints/gradient_boosting'
EXPECTED = 'e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f'
FOLD = int(os.environ.get('GB_OUTER_FOLD', '4'))
CANDIDATE = int(os.environ.get('GB_CANDIDATE', '1'))
fold_path = ROOT / 'experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv'
actual_sha = hashlib.sha256(fold_path.read_bytes()).hexdigest()
if actual_sha != EXPECTED:
    raise RuntimeError(f'checksum mismatch: {actual_sha}')
folds = pd.read_csv(fold_path)
manifest = pd.read_csv(CP / f'fold_{FOLD}_search_candidates.csv')
expected_manifest = candidate_manifest(FOLD)
if not manifest.equals(expected_manifest):
    raise RuntimeError('candidate manifest differs from the authorized V2 grid')
results = pd.read_csv(CP / f'gradient_boosting_fold_{FOLD}_inner_search_results.csv')
validate_results(results, manifest)
candidate = results.loc[results.candidate_id.eq(CANDIDATE)]
if len(candidate) != 1 or candidate.iloc[0].status != 'COMPLETE':
    raise RuntimeError(f'Candidate {CANDIDATE} is not uniquely COMPLETE')
record = candidate.iloc[0]
scores = np.array([record[f'inner_fold_{i}_macro_f1'] for i in range(1, 4)], dtype=float)
if not np.isclose(record.mean_inner_macro_f1, scores.mean()) or not np.isclose(record.std_inner_macro_f1, scores.std()):
    raise RuntimeError(f'Candidate {CANDIDATE} mean/std readback mismatch')
splits = json.loads((CP / f'fold_{FOLD}_inner_splits.json').read_text(encoding='utf-8'))
if len(splits) != 3:
    raise RuntimeError('expected exactly 3 persisted inner splits')
outer_test_runs = set(folds.loc[folds.outer_fold.eq(FOLD), 'run_key'])
inner_isolation = []
for split in splits:
    train_runs = set(split['train_run_keys'])
    valid_runs = set(split['validation_run_keys'])
    if train_runs & valid_runs or (train_runs | valid_runs) & outer_test_runs:
        raise RuntimeError(f"inner split {split['inner_fold']} mixes run keys incorrectly")
    train_subjects = set(folds.loc[folds.run_key.isin(train_runs), 'subject_id'])
    valid_subjects = set(folds.loc[folds.run_key.isin(valid_runs), 'subject_id'])
    if train_subjects & valid_subjects:
        raise RuntimeError(f"inner split {split['inner_fold']} has subject overlap")
    inner_isolation.append(True)
progress = json.loads((CP / f'gradient_boosting_fold_{FOLD}_search_progress.json').read_text(encoding='utf-8'))
expected_completed = list(range(1, CANDIDATE + 1))
expected_remaining = list(range(CANDIDATE + 1, 9))
if progress['completed_candidate_ids'] != expected_completed or progress['remaining_candidate_ids'] != expected_remaining or progress['outer_test_used'] is not False or progress['fold_status'] != 'INCOMPLETE':
    raise RuntimeError('invalid search progress readback')
print(json.dumps({'checksum': actual_sha, 'manifest_valid': True, 'inner_subject_isolation': inner_isolation, f'candidate_{CANDIDATE}': record.to_dict(), 'progress': progress}, indent=2))
