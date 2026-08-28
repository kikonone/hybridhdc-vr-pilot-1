# Phase 06 Final Consolidation Notes

## Frozen inputs

- Primary SHA-256 expected: `0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44`.
- Fold SHA-256 expected: `e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f`.
- New-variant Final Confirmation: 300 fold-config checkpoints, already audited PASS.
- Phase 05 must remain frozen and byte-identical.

## Analysis boundaries

- Prediction unit: run (419 aligned run keys per configuration).
- Inferential unit: subject (35 subjects).
- Partition unit: outer fold (5 folds), not an independent sample.
- Repetition unit: seed (5 seeds), not an independent subject.
- Regression wording: bounded difficulty-induced workload proxy regression.

## Findings

- Preflight, OOF coverage/alignment, metric recalculation, stability, efficiency, Pareto, reports, figures, and Notebook persistence passed.
- Final model selection is blocked because the frozen rules do not specify final outer-OOF classification/regression ranking or tie-breaking.
- No best-model configs or phase06_freeze.json were created; status is NOT_FROZEN.
