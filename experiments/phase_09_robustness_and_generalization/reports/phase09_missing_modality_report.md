# Phase 09 Missing-Modality Robustness Report

All five modality-removal conditions were compared with frozen Full Primary references using 35 paired subject summaries. Positive degradation always means worse performance, after respecting that Macro-F1 is maximized and bounded MAE is minimized.

## Largest degradation by model-task
- hdc_classification: MISSING_FLIGHT_PARAMETER (mean degradation 0.518092)
- hdc_regression: MISSING_FLIGHT_PARAMETER (mean degradation 0.590627)
- traditional_classification: MISSING_FLIGHT_PARAMETER (mean degradation 0.510262)
- traditional_regression: MISSING_FLIGHT_PARAMETER (mean degradation 0.718438)

## Flight parameters
- hdc_classification: degradation=0.518092, Holm p=2.91038e-10, effect=1.000
- traditional_classification: degradation=0.510262, Holm p=2.91038e-10, effect=1.000
- hdc_regression: degradation=0.590627, Holm p=2.91038e-10, effect=1.000
- traditional_regression: degradation=0.718438, Holm p=2.91038e-10, effect=1.000

HDC-versus-traditional robustness is reported descriptively in `phase09_model_robustness_comparison.csv`; no extra unregistered inferential family was invented. Conditions with negative mean degradation are retained as possible improvements, not discarded. Smaller changes for small modalities may reflect fewer removed features and cannot be interpreted as causal irrelevance.
