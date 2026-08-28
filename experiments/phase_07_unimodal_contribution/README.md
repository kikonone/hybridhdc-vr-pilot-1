# Phase 07 — Unimodal Contribution Analysis

Phase 07 studies the contribution of each single modality to the frozen dual-task design. The five experimentally eligible modalities are physiological, eye tracking, head movement, flight parameter, and verified body movement features.

Control input is unavailable and is not an experimental modality. Performance features are excluded from the Primary thesis dataset and from Phase 07. Classification reuses the frozen Phase 06 HDC+OnlineHD Hybrid interface; regression reuses the frozen Phase 06 Common Encoder Ridge readout interface. The regression estimand is a **bounded difficulty-induced workload proxy**, not directly measured continuous cognitive workload.

Initialization reads and audits the frozen Phase 03 data, manifests, outer folds, and Phase 06 selection interface. The subsequent Contract Freeze fixes the full-cohort missingness policy, fold-local preprocessing, frozen Phase 06 model interfaces, seed aggregation, separate task rankings, subject-level statistical analysis, and error-analysis requirements. Neither step performs training, hypervector generation, prediction, OOF generation, row deletion, or global preprocessing. Upstream data and configuration files remain in place and are never copied into `data/`.

Current status: `UNIMODAL_EXECUTION_COMPLETE_PENDING_OOF_CONSOLIDATION`.

The frozen unimodal batch is complete: 125 Hybrid classification runs and 125 Common Encoder Ridge regression runs produced 20,950 audited seed-level outer-test prediction rows. Checkpoint integrity, leakage, subject isolation, and modality/task/seed coverage passed. Canonical seed-aggregated OOF, modality ranking, multimodal comparison, bootstrap, Friedman, Wilcoxon, and Holm analyses have not been executed.

The next eligible step is the separately authorized **Phase 07 OOF consolidation and statistical analysis**.
