# Phase 08 — Fusion and Shortcut Analysis

Phase 08 studies whether manifest-defined Early Fusion adds reliable value over the best frozen single modality and assesses the shortcut-learning risk associated with performance features.

## Evidence Roles

- `primary_without_performance.csv` is the only main-results dataset.
- `auxiliary_with_performance.csv` is an auxiliary upper-bound and shortcut-risk comparison.
- `performance_only.csv` is an auxiliary shortcut-risk diagnostic.

Phase 07 is complete and frozen. Its best single-modality results are read-only references in this phase.

## Execution Status

Phase 08 is now `FROZEN`. All 370 model-runs completed: 150 HDC classification, 150 HDC regression, 35 traditional classification, and 35 traditional regression. Independent coverage, checkpoint-integrity, leakage, artifact, reproducibility, manifest, freeze, upstream-integrity, and Notebook-persistence audits pass for 31,006 raw prediction rows.

The execution contains 300 core runs, 10 traditional flight-full runs, and 60 behavioral-flight sensitivity runs. Canonical consolidation produced exactly 10,894 OOF rows (5,447 per task), with five-seed HDC averaging and frozen-reference indexing verified. No outer-test tuning or Phase 09 execution occurred.

Flight provenance is frozen before modeling: 323 features are `BEHAVIORAL_RESPONSE`, none have sufficient evidence for `TASK_SETTING_OR_SCENARIO`, and 3 acquisition-metadata features remain `AMBIGUOUS`. This classification is provenance-based and does not itself establish leakage or causal interpretation.

Phase 09 scenario/task-template/route holdouts are not currently feasible because explicit identifiers are absent. Session is perfectly nested within subject, so the current metadata cannot isolate session generalization from subject generalization. No Phase 09 directory or holdout was created.

The next permitted lifecycle entry is **Phase 09**, subject to separate authorization and the saved metadata-feasibility guardrails. Phase 09 has not been initialized or executed. Optional Late Fusion and HDC modality-aware binding remain unauthorized.

## Final OOF Analysis

- Canonical OOF rows: 10,894/10,894; run-key coverage, alignment, leakage, and upstream reference audits pass.
- Independent metric recalculation, Wilcoxon tests, Holm correction, rank-biserial effects, and 2,000-repetition paired-subject bootstrap confidence intervals pass at `subject_id` level (`n=35`).
- Fusion, performance-shortcut, and flight behavioral-sensitivity tables, figures, reports, and the safely appended executed Notebook section are persisted.
- Unseen-session, unseen-scenario, and task-template holdouts remain `NOT_FEASIBLE_DUE_TO_METADATA`; the Phase 09 handoff records this limitation without executing Phase 09.
- Final artifact, reproducibility, manifest, freeze, upstream-integrity, and Notebook persistence audits pass. The final manifest and `phase08_freeze.json` are saved and independently hash-verified.
- Current results do not prove cross-session, cross-scenario, or cross-task-template generalization; the corresponding holdouts remain infeasible with available metadata.
