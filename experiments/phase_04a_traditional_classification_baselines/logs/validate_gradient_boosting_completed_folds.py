from pathlib import Path
import os
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CP = ROOT / 'experiments/phase_04a_traditional_classification_baselines/results/checkpoints/gradient_boosting'
FOLDS = pd.read_csv(ROOT / 'experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv')
EXPECTED = 'e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f'
actual = hashlib.sha256((ROOT / 'experiments/phase_03_multimodal_dataset_labeling/data/fold_assignments.csv').read_bytes()).hexdigest()
if actual != EXPECTED:
    raise RuntimeError(f'frozen checksum mismatch: {actual}')

folds_to_check = tuple(int(value) for value in os.environ.get('GB_COMPLETED_FOLDS', '1,2,3').split(','))
report = {'frozen_checksum': actual, 'folds': {}}
for fold in folds_to_check:
    pred_path = CP / f'gradient_boosting_fold_{fold}_predictions.csv'
    metrics_path = CP / f'gradient_boosting_fold_{fold}_metrics.json'
    params_path = CP / f'gradient_boosting_fold_{fold}_best_params.json'
    for path in (pred_path, metrics_path, params_path):
        if not path.exists():
            raise FileNotFoundError(path)
    pred = pd.read_csv(pred_path)
    json.loads(metrics_path.read_text(encoding='utf-8'))
    json.loads(params_path.read_text(encoding='utf-8'))
    expected = FOLDS.loc[FOLDS.outer_fold.eq(fold), 'run_key']
    expected_runs = set(expected)
    observed_runs = set(pred.run_key)
    if len(pred) != len(expected) or pred.run_key.nunique() != len(pred) or observed_runs != expected_runs:
        raise RuntimeError(f'fold {fold}: prediction run-key coverage mismatch')
    if set(pred.outer_fold.astype(int)) != {fold}:
        raise RuntimeError(f'fold {fold}: wrong outer fold ID')
    train_subjects = set(FOLDS.loc[~FOLDS.outer_fold.eq(fold), 'subject_id'])
    test_subjects = set(FOLDS.loc[FOLDS.outer_fold.eq(fold), 'subject_id'])
    if train_subjects & test_subjects:
        raise RuntimeError(f'fold {fold}: subject isolation failure')
    report['folds'][str(fold)] = {
        'prediction_readback': True,
        'metrics_readback': True,
        'best_params_readback': True,
        'rows': int(len(pred)),
        'unique_run_keys': int(pred.run_key.nunique()),
        'subject_isolation': True,
    }
print(json.dumps(report, indent=2))
