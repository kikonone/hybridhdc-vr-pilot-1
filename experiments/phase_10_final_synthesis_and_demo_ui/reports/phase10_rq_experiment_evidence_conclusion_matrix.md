# Phase 10 RQ—Experiment—Evidence—Conclusion Matrix

All rows reference frozen artifacts. Statistics were not recomputed.

## RQ1: How well does the frozen dual-task HDC system perform?

- Phases: 05;06
- Protocol: Frozen subject-grouped outer evaluation; selection uses inner evidence only where applicable
- Metrics: Macro-F1; bounded MAE
- Evidence: Referenced; not recomputed
- Main finding: The frozen dual-task interface supports separate HDC classification and bounded proxy-regression outputs under the registered five-fold subject-grouped protocol; it does not establish deployment validity.
- Limitation: Four-level proxy target
- Prohibited overclaim: Deployment-ready, diagnostic, clinical, or universally generalizable system.
- Sources: `E:\hdc-vr-pilot\最新完整实验计划_分类回归双任务.md;E:\hdc-vr-pilot\experiments\phase_05_basic_dual_output_hdc\reports\phase05_final_summary.md;E:\hdc-vr-pilot\experiments\phase_06_hdc_variant_screening\reports\phase06_final_summary.md;E:\hdc-vr-pilot\experiments\phase_06_hdc_variant_screening\configs\phase06_freeze.json`

## RQ2: How does HDC compare with traditional models?

- Phases: 04A;04B;05;06
- Protocol: Frozen subject-grouped outer evaluation; selection uses inner evidence only where applicable
- Metrics: Macro-F1; bounded MAE
- Evidence: Referenced comparisons
- Main finding: Frozen HDC and traditional OOF artifacts permit descriptive task-specific comparison. Classification and regression/readout conclusions remain separate, and a numerical advantage is not called significant without the registered subject-level test.
- Limitation: Small cohort
- Prohibited overclaim: Statistically significant superiority based only on a better point estimate; combined classification/regression superiority claim.
- Sources: `E:\hdc-vr-pilot\最新完整实验计划_分类回归双任务.md;E:\hdc-vr-pilot\experiments\phase_04a_traditional_classification_baselines\reports\phase04a_final_summary.md;E:\hdc-vr-pilot\experiments\phase_04b_traditional_regression_baselines\reports\phase04b_final_summary.md;E:\hdc-vr-pilot\experiments\phase_05_basic_dual_output_hdc\reports\phase05_final_summary.md;E:\hdc-vr-pilot\experiments\phase_06_hdc_variant_screening\reports\phase06_final_summary.md`

## RQ3: How do HDC variants compare?

- Phases: 05;06
- Protocol: Frozen subject-grouped outer evaluation; selection uses inner evidence only where applicable
- Metrics: Macro-F1; bounded MAE
- Evidence: Inner-CV selection and stability
- Main finding: Phase 06 selected HDC+OnlineHD Hybrid (5,000 dimensions) for classification and COMMON_ENCODER_READOUT_BASELINE (10,000 dimensions) for regression using INNER_CV_ONLY evidence; no single seed or outer-test result was selected.
- Limitation: Frozen candidate grid
- Prohibited overclaim: Outer-test-selected configuration or best single seed.
- Sources: `E:\hdc-vr-pilot\最新完整实验计划_分类回归双任务.md;E:\hdc-vr-pilot\experiments\phase_05_basic_dual_output_hdc\reports\phase05_final_summary.md;E:\hdc-vr-pilot\experiments\phase_06_hdc_variant_screening\reports\phase06_final_summary.md;E:\hdc-vr-pilot\experiments\phase_06_hdc_variant_screening\configs\phase06_freeze.json`

## RQ4: What is the incremental value of bounded regression?

- Phases: 04B;05;06
- Protocol: Frozen subject-grouped outer evaluation; selection uses inner evidence only where applicable
- Metrics: bounded MAE
- Evidence: Referenced; not recomputed
- Main finding: The regression branch predicts a bounded difficulty-induced workload proxy with four target values. It must not be interpreted as directly measured continuous cognitive workload.
- Limitation: Four discrete target values
- Prohibited overclaim: Directly measured continuous cognitive workload.
- Sources: `E:\hdc-vr-pilot\最新完整实验计划_分类回归双任务.md;E:\hdc-vr-pilot\experiments\phase_04b_traditional_regression_baselines\reports\phase04b_final_summary.md;E:\hdc-vr-pilot\experiments\phase_05_basic_dual_output_hdc\reports\phase05_final_summary.md;E:\hdc-vr-pilot\experiments\phase_06_hdc_variant_screening\reports\phase06_final_summary.md;E:\hdc-vr-pilot\experiments\phase_06_hdc_variant_screening\configs\phase06_best_regression_hdc.json`

## RQ5: What are modality contribution and fusion effects?

- Phases: 07;08
- Protocol: Frozen subject-grouped outer evaluation; selection uses inner evidence only where applicable
- Metrics: Macro-F1; bounded MAE
- Evidence: Subject-level corrected comparisons
- Main finding: Flight-parameter features rank highest in the frozen Phase 07 unimodal analyses, while Phase 08 quantifies fusion and shortcut sensitivity. These results support predictive contribution in this flight-task setting, not causal sensor claims.
- Limitation: Flight-task setting
- Prohibited overclaim: Causal physiological/sensor mechanism or proven cross-domain flight behavior.
- Sources: `E:\hdc-vr-pilot\最新完整实验计划_分类回归双任务.md;E:\hdc-vr-pilot\experiments\phase_07_unimodal_contribution\reports\phase07_final_summary.md;E:\hdc-vr-pilot\experiments\phase_08_fusion_and_shortcut_analysis\reports\phase08_final_analysis.md;E:\hdc-vr-pilot\experiments\phase_08_fusion_and_shortcut_analysis\configs\phase08_freeze.json`

## RQ6: What do shortcut, missing-modality and LOSO evidence support?

- Phases: 08;09
- Protocol: Frozen subject-grouped outer evaluation; selection uses inner evidence only where applicable
- Metrics: Macro-F1; bounded MAE
- Evidence: Subject-level Wilcoxon/Holm/bootstrap
- Main finding: Phase 09 evaluates missing-modality retraining and 35-subject LOSO generalization. Cross-session, cross-scenario, task-template and route generalization remain unverified because required metadata are unavailable; flight generalizable-behavior claims remain inconclusive.
- Limitation: Required metadata unavailable
- Prohibited overclaim: Proven cross-session, cross-scenario, cross-route, or cross-task-template generalization.
- Sources: `E:\hdc-vr-pilot\最新完整实验计划_分类回归双任务.md;E:\hdc-vr-pilot\experiments\phase_08_fusion_and_shortcut_analysis\reports\phase08_final_analysis.md;E:\hdc-vr-pilot\experiments\phase_09_robustness_and_generalization\reports\phase09_final_analysis.md;E:\hdc-vr-pilot\experiments\phase_09_robustness_and_generalization\configs\phase09_freeze.json`
