---
type: results-report
date: 2026-08-20
experiment_line: basic-dual-output-hdc
round: 00
purpose: final-phase-freeze
status: frozen
source_artifacts:
  - analysis-output/analysis-report.md
  - analysis-output/stats-appendix.md
  - analysis-output/figure-catalog.md
linked_experiments: []
linked_results: []
---

# Phase 05 / Round 00 / Final Phase Freeze / 2026-08-20

## 1. Executive Summary
Phase 05 completed an audited Vanilla Prototype HDC benchmark with four preregistered dimensions, five seeds, a prototype cosine classification head, and two regression heads. Every configuration covers 419 subject-wise OOF runs. The strongest observed mean Macro-F1 occurred at D=5,000 (0.779764 ± 0.011622 sample SD), while the lowest observed similarity and Ridge bounded MAE occurred at D=2,000 (0.699930 ± 0.008131) and D=10,000 (0.276390 ± 0.006419), respectively. These observations do not select a canonical configuration.

## 2. Experiment Identity and Decision Context
The experiment tests whether a shared bipolar HDC representation can support four-class difficulty-induced workload proxy classification and bounded difficulty-induced workload proxy regression. It follows frozen traditional baselines from Phase 04A/04B and stops before Phase 06 variants.

## 3. Frozen Data and Evaluation Protocol
- Primary interface: 419 runs, 35 subjects, 1,176 without-performance predictive features.
- Primary SHA-256: `0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44`.
- Frozen subject-wise five-fold SHA-256: `e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f`.
- Inner selection: three-fold `GroupKFold(groups=subject_id)` on each outer-training set.
- Outer-test data were never used for hyperparameter tuning.
- Compatibility with Phase 04A/04B: `PASS` for run keys, folds, subject-wise protocol, and feature interface.

## 4. HDC Representation
The frozen representation uses bipolar item, level, and sample hypervectors. Feature identity and ordered level hypervectors are bound by elementwise bipolar multiplication, then bundled by integer accumulation and deterministic sign/tie resolution. Quantization uses equal-width training-fold minimum/maximum boundaries after training-fitted imputation, variance filtering, standardization, and feature selection. Classification uses cosine similarity to four class prototypes.

## 5. Quick Screen and Final Confirmation
Quick screening evaluated the frozen 16-candidate space independently in each outer fold and selected `levels=51`, `feature_k=50` for all five folds. Final Confirmation then evaluated every preregistered dimension `[1000, 2000, 5000, 10000]` and seed `[42, 43, 44, 45, 46]`, totaling 100 fold-runs and 20 complete OOF configurations. Temperature and Ridge alpha were selected only through outer-training inner CV.

## 6. Four-Class OOF Results
The classification tables report accuracy, balanced accuracy, Macro-F1, weighted F1, severe error rate, and complete confusion matrices for all 20 configurations. Across five seeds per dimension, the highest observed mean Macro-F1 was 0.779764 at D=5,000. This is a descriptive property of the preregistered matrix, not a post-hoc choice.

## 7. Similarity Regression OOF Results
The similarity decoder maps prototype similarities through an inner-selected temperature and produces a prediction bounded to `[1,4]`. The lowest observed dimension-level mean bounded MAE was 0.699930 at D=2,000. The target is the difficulty-level-derived bounded proxy, not directly measured continuous cognitive workload.

## 8. Ridge Readout OOF Results
The Ridge readout uses normalized sample hypervectors and an alpha selected only in inner CV. The lowest observed dimension-level mean bounded MAE was 0.276390 at D=10,000. Predictions were clipped only by the frozen `[1,4]` contract.

## 9. Dimension and Seed Stability
Each dimension is summarized over five preregistered seeds by mean, sample SD, minimum, and maximum. Seed predictions were not averaged into a new ensemble and no seed was selected from outer-test performance. Full configuration tables remain the authoritative benchmark.

## 10. Efficiency Analysis
Efficiency summaries use only recorded inner-selection, outer-training encoding/fit, and outer-test inference times. Model artifact bytes are measured from saved model files. Peak memory and encoding throughput were not recorded and are left missing rather than estimated. Dimension-level efficiency summaries use the same five-seed descriptive aggregation.

## 11. Fair Comparison with Phase 04A and Phase 04B
Phase 04A classification and Phase 04B regression artifacts were read from their frozen comparisons, reports, manifests, freeze files, and canonical OOF predictions. HDC is displayed as mean ± sample SD over five seeds per dimension; traditional rows retain their canonical OOF values. No single highest HDC seed is promoted as the formal result. No unregistered significance test or “significantly better” claim is made.

## 12. Leakage Protection
All preprocessing, quantization, codebooks, prototypes, temperature selection, and Ridge alpha selection were confined to training scopes. Final OOF consolidation only copied and aligned already-saved predictions; it did not fit or predict. Fold/run/subject/target alignment and source hashes were audited.

## 13. Limitations and Negative Results
- The regression target has only four difficulty-derived values and is not a direct continuous cognitive-workload measurement.
- Five seeds provide a useful stability description but do not justify an unregistered inferential claim.
- Similarity regression and Ridge readout differ materially; neither should be conflated with the classification head.
- Peak memory and encoding throughput were not recorded, so no such efficiency values are reported.
- Traditional baselines and HDC have compatible OOF interfaces, but seed-repetition structures differ; comparison is descriptive.

## 14. Canonical Configuration
No preregistered rule authorizes selection of a canonical dimension or seed from outer-test performance. **No post-hoc canonical configuration was selected from outer-test performance.** Freeze status: `NOT_PERFORMED_OUTER_TEST_SELECTION_PROHIBITED`.

## 15. Figure-by-Figure Interpretation
The dimension figures show mean trends with sample SD across seeds; the stability figures expose individual seed trajectories; the efficiency plot displays the measured accuracy/time tradeoff; and the baseline plots place the complete HDC dimension summaries beside frozen traditional results. None is annotated with a significance claim.

## 16. What Changed Our Belief
Vanilla HDC now has complete, reproducible dual-output OOF evidence rather than quick-screen-only evidence. The full dimension/seed matrix, not an observed best row, is the stable Phase 05 result.

## 17. Phase Boundary and Next Actions
Phase 05 may be frozen only after final artifact, reproducibility, upstream-integrity, and Notebook audits pass. The next planned phase may read this frozen matrix; it must not treat an observed best outer-test configuration as an unbiased selected model. Phase 06 was not executed here.

## 18. Artifact and Reproducibility Index
- OOF predictions: `results/oof/vanilla_hdc_*_oof.csv`
- Configuration and seed summaries: `results/summaries/`
- Figures: `figures/phase05_*.png` with PDF companions
- Strict analysis bundle: `analysis-output/`
- Final audits: `audits/phase05_final_*`
- Final manifest: `manifests/phase05_final_artifact_manifest.json`
- Notebook: `Phase_05_Basic_Dual_Output_HDC.ipynb`
- Freeze record: `configs/phase05_freeze.json`

## 19. No-Retraining Compliance Amendment

After the original freeze, Phase 05 was audited against the complete regression-head and efficiency contracts. Missing descriptive diagnostics were derived from the unchanged frozen OOF table, and complete-batch inference was measured from saved fitted artifacts with five warm-ups, thirty repetitions, and `time.perf_counter_ns`. Reconstructed predictions matched the frozen artifacts (maximum absolute difference `8.88e-16`), and no fit, refit, prediction replacement, or Phase 06 execution occurred.

The added diagnostics show that similarity-regression predictions round only to levels 2 and 3 across the complete matrix, whereas Ridge predictions cover levels 1–4. This limitation is now explicit and does not change the preregistered result matrix or select a configuration. Protocol-compliant training time was not remeasured because doing so would require prohibited retraining; its amendment status is `NOT_PERFORMED_RETRAINING_PROHIBITED`. Full details are in `reports/phase05_no_retraining_completion_report.md`.
