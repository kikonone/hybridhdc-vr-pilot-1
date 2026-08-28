# Frozen execution order

Phase 00 → 01 → 02 → 03 → 04A/04B → 05 → 06 → 07 → 08 → 09 → 10.

Phase 03 supplies `primary_without_performance.csv` and the immutable subject-grouped fold assignments. Phase 04A/04B establish traditional baselines; Phase 05 establishes Vanilla HDC; Phase 06 freezes HDC variants and the inner-only selected dual-task interface; Phase 07 evaluates unimodal contribution; Phase 08 evaluates fusion and shortcut sensitivity; Phase 09 evaluates missing-modality robustness and 35-subject LOSO; Phase 10 indexes and synthesizes only.

For this package, execute only `python scripts/verify_phase10_final_synthesis.py`. Do not rerun upstream training, prediction, tuning, model selection, or statistics.
