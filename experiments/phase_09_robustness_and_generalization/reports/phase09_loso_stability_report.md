# Phase 09 LOSO Subject Stability Report

LOSO includes all 35 subjects exactly once per model-task canonical OOF. No high-error subject was deleted or used for retraining.

- hdc_classification: mean=0.845750, 95% CI [0.784148, 0.904544], worst subject=sub-cp039 (diagnostic only)
- hdc_regression: mean=0.251182, 95% CI [0.203644, 0.306645], worst subject=sub-cp039 (diagnostic only)
- traditional_classification: mean=0.956139, 95% CI [0.929098, 0.980544], worst subject=sub-cp039 (diagnostic only)
- traditional_regression: mean=0.105292, 95% CI [0.070598, 0.147651], worst subject=sub-cp039 (diagnostic only)

This supports estimation of held-out-subject behavior for the frozen interfaces. It does not test unseen sessions, scenarios, task templates, or route configurations.
