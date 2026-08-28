# Read-only verification plan

1. Parse every indexed JSON/CSV/notebook without modifying it.
2. Recompute SHA-256 for the frozen Primary dataset and fold assignment; compare with registered values.
3. Verify each Phase 04A-09 freeze interface and available final manifest.
4. Verify every prediction/statistics/paper inventory path, byte size, and SHA-256.
5. Verify the Phase 06 best classification/regression interfaces and metric directions.
6. Verify notebook persisted outputs and claim guardrails.
7. Fail closed on any mismatch; record it as `REQUIRES_RECONCILIATION`; never rewrite upstream artifacts.

No training, prediction generation, statistical recomputation, UI execution, or network access is part of this plan.
