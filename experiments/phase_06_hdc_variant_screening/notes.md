# Notes: Phase 06 Initialization

## Authoritative sources
- `最新完整实验计划_分类回归双任务.md`
- `CODEX_NOTEBOOK_RULES.md`
- User-supplied Phase 06 initialization contract

## Verified boundaries
- This step is preflight-only; HDC variant training is prohibited.
- Vanilla Prototype HDC must be referenced read-only from frozen Phase 05.
- `target_score` is a bounded difficulty-induced workload proxy regression target, not directly measured continuous cognitive workload.
- Project conclusions remain limited to workload-proxy classification and regression.

## Findings
- Phase 03 actual Primary SHA-256 is `0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44`; frozen fold SHA-256 is `e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f`.
- Primary data has 419 rows, 35 subjects, 1,176 manifest-backed predictive features, 419 unique run keys, complete class values 0–3, and complete score values 1.0–4.0.
- All five outer folds have zero train/test subject overlap. Every outer-training set supports subject-disjoint three-fold GroupKFold.
- Phase 04A and Phase 04B frozen comparison interfaces pass read-only checks.
- Phase 05 is `FROZEN`; the final confirmation has 5/5 folds and 100/100 executions with dimensions 1,000/2,000/5,000/10,000, seeds 42–46, levels 51, and feature_k 50.
- Phase 05 classification, similarity-regression, and Ridge-regression OOF artifacts exist. Final leakage, artifact, reproducibility, Notebook, manifest, and upstream-checksum gates pass.
- Canonical Phase 05 selection remains `NOT_PERFORMED_OUTER_TEST_SELECTION_PROHIBITED`.
- The Phase 06 Notebook executed without error and persisted explicit zero-training and pending-contract-freeze markers.

## Contract-freeze and quick-screen execution
- User authorized one continuous run covering the Phase 05 amendment gate, Phase 06 contract freeze, implementation/tests, and all three five-fold quick screens.
- Outer-test feature access, outer-test predictions, Final Confirmation, regression heads, and Phase 07 remain prohibited.
- Existing Phase 05 no-retraining amendment is valid and more complete than the requested gate: all amendment, diagnostic, and efficiency audits are PASS, so no duplicate Phase 05 files were created.
- Phase 06 contracts froze 24 OnlineHD, 6 Multi-centroid, and 32 Hybrid candidates per outer fold; the immutable upstream snapshot contains 1,026 Phase 03–05 files.
- The initial implementation/guard suite passes 15 tests.
- OnlineHD completed 5/5 folds with 24/24 candidates per fold; Multi-centroid completed 5/5 with 6/6; Hybrid completed 5/5 with 32/32.
- All 15 fold-specific best configurations were independently reproduced from candidate CSVs under the frozen lexicographic rule.
- All fold leakage, coverage, checkpoint, and artifact audits pass. No outer-test feature/label access, outer-test prediction, OOF, regression head, or Final Confirmation occurred.
- The executed Notebook preserves all 13 initialization cells and adds 5 executed quick-screen cells; persistence audit is PASS.
