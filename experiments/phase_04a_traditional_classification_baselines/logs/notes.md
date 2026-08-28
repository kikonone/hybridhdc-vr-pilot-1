# Notes: Gradient Boosting Fold 4 Candidate 1

## Verified constraints
- Frozen fold SHA-256 must be `e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f`.
- Evaluate Candidate 1 only; Fold 4 outer test must not be used.
- Inner validation uses the persisted GroupKFold subject splits.

## Findings
- Frozen fold checksum matches the required value.
- Gradient Boosting Folds 1–3 prediction, metric, and parameter checkpoints reload successfully. Each has 84 expected unique run keys and subject isolation passes.
- Fold 4 Candidate 1: k=100, n_estimators=100, learning_rate=0.05, max_depth=2; inner Macro-F1 values are 0.9154257874, 0.8888620098, and 0.9535162950; mean 0.9192680308 and std 0.0265344590.
- Fold 4 Candidate 1 CSV/progress readback and notebook persistence passed. The outer test was not used for evaluation.
- Fold 4 Candidate 2: k=100, n_estimators=100, learning_rate=0.1, max_depth=2; inner Macro-F1 values are 0.9147261424, 0.8977632508, and 0.9348326028; mean 0.9157739987 and std 0.0151516273. CSV/progress readback and notebook persistence passed; outer test remains unused.
- Fold 4 Candidate 3: k=100, n_estimators=200, learning_rate=0.05, max_depth=2; inner Macro-F1 values are 0.9154257874, 0.8977632508, and 0.9442665650; mean 0.9191518678 and std 0.0191668518. CSV/progress readback and notebook persistence passed; outer test remains unused.
- Fold 4 Candidate 4: k=100, n_estimators=200, learning_rate=0.1, max_depth=2; inner Macro-F1 values are 0.9154095831, 0.8981332101, and 0.9348326028; mean 0.9161251320 and std 0.0149910054. CSV/progress readback and notebook persistence passed; outer test remains unused.
- Fold 4 Candidate 5: k=200, n_estimators=100, learning_rate=0.05, max_depth=2; inner Macro-F1 values are 0.9414334205, 0.8820365458, and 0.9535162950; mean 0.9256620871 and std 0.0312398234. CSV/progress readback and notebook persistence passed; outer test remains unused.
- Fold 4 Candidate 6: k=200, n_estimators=100, learning_rate=0.1, max_depth=2; inner Macro-F1 values are 0.9409665080, 0.8635929817, and 0.9348326028; mean 0.9131306975 and std 0.0351178507. CSV/progress readback and notebook persistence passed; outer test remains unused.
- Fold 4 Candidate 7: k=200, n_estimators=200, learning_rate=0.05, max_depth=2; inner Macro-F1 values are 0.9409665080, 0.8723067369, and 0.9348326028; mean 0.9160352825 and std 0.0310219866. CSV/progress readback and notebook persistence passed; outer test remains unused.
- Fold 4 Candidate 8: k=200, n_estimators=200, learning_rate=0.1, max_depth=2; inner Macro-F1 values are 0.9321655569, 0.8720334154, and 0.9163140842; mean 0.9068376855 and std 0.0254469410. CSV/progress readback and notebook persistence passed; outer test remains unused.
- All eight inner-CV checkpoints are COMPLETE. Candidate 5 has the highest saved mean inner Macro-F1 (0.9256620871); this is an inner-CV ranking only and Fold 4 has not been finalized.
- Fold 4 was finalized from Candidate 5 only after all eight candidate records validated. Final outer metrics: Macro-F1 0.9301582122, balanced accuracy 0.9285714286, accuracy 0.9285714286, weighted-F1 0.9301582122. Recalls classes 0–3: 0.8571428571, 0.9523809524, 0.9523809524, 0.9523809524.
- Official Fold 4 CSV/JSON checkpoints, metric recomputation, checkpoint integrity audit, confusion-matrix PNG, and notebook persistence all passed. The pre-candidate-level Fold 4 checkpoint artifacts were retained under `results/checkpoints/gradient_boosting/pre_candidate_level_fold_4_backup/` before replacement.
- Fold 5 Candidate 1: k=100, n_estimators=100, learning_rate=0.05, max_depth=2; inner Macro-F1 values are 0.9669169448, 0.8995098039, and 0.8760822511; mean 0.9141696666 and std 0.0385047103. CSV/progress readback and notebook persistence passed; outer test remains unused.
- Fold 5 Candidate 2: k=100, n_estimators=100, learning_rate=0.1, max_depth=2; inner Macro-F1 values are 0.9669218152, 0.8894871795, and 0.8760822511; mean 0.9108304153 and std 0.0400383723. CSV/progress readback and notebook persistence passed; outer test remains unused.
- Fold 5 Candidate 3: k=100, n_estimators=200, learning_rate=0.05, max_depth=2; inner Macro-F1 values are 0.9507004310, 0.8894871795, and 0.8760822511; mean 0.9054232872 and std 0.0324801256. CSV/progress readback and notebook persistence passed; outer test remains unused.
- Fold 5 Candidate 4: k=100, n_estimators=200, learning_rate=0.1, max_depth=2; inner Macro-F1 values are 0.9425264089, 0.8894871795, and 0.8761501211; mean 0.9027212365 and std 0.0286683111. CSV/progress readback and notebook persistence passed; outer test remains unused.
- Fold 5 Candidate 5: k=200, n_estimators=100, learning_rate=0.05, max_depth=2; inner Macro-F1 values are 0.9666643512, 0.9091025641, and 0.8691520468; mean 0.9149729874 and std 0.0400250653. CSV/progress readback and notebook persistence passed; outer test remains unused.
- Fold 5 Candidate 6: k=200, n_estimators=100, learning_rate=0.1, max_depth=2; inner Macro-F1 values are 0.9666643512, 0.9091025641, and 0.8989173789; mean 0.9248947647 and std 0.0298268142. CSV/progress readback and notebook persistence passed; outer test remains unused.
- Fold 5 Candidate 7: k=200, n_estimators=200, learning_rate=0.05, max_depth=2; inner Macro-F1 values are 0.9666666667, 0.9091025641, and 0.9081365305; mean 0.9279685871 and std 0.0273665164. CSV/progress readback and notebook persistence passed; outer test remains unused.
- Fold 5 Candidate 8: k=200, n_estimators=200, learning_rate=0.1, max_depth=2; inner Macro-F1 values are 0.9751342966, 0.9181238336, and 0.9090123569; mean 0.9340901624 and std 0.0292599894. CSV/progress readback and notebook persistence passed; outer test remains unused.
- All eight Fold 5 inner-CV checkpoints are COMPLETE. Candidate 8 has the highest saved mean inner Macro-F1 (0.9340901624); this is an inner-CV ranking only and Fold 5 has not been finalized.
- Fold 5 was finalized from Candidate 8 only after all eight candidate records validated. Final outer metrics: Macro-F1 0.9636837236, balanced accuracy 0.9636904762, accuracy 0.9638554217, weighted-F1 0.9638485878. Recalls classes 0–3: 0.9523809524, 0.9523809524, 0.9500000000, 1.0000000000.
- Final all-model OOF coverage passed: each of Logistic Regression, Linear SVM, RBF SVM, Random Forest, KNN, and Gradient Boosting has exactly 419 unique frozen runs. Gradient Boosting is the selected best traditional classifier by complete OOF Macro-F1 (0.9356075023).
- Consolidation note: one legacy Gradient Boosting fold metrics JSON omitted per-class recall fields; the final fold-summary recalls were recomputed directly from that fold's persisted official predictions. No model training or predictions were rerun.
