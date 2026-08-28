# Phase 07 final summary

Status target: FROZEN after independent verification.

- Best classification modality: flight_parameter_features (Macro-F1 0.863457813)
- Best regression modality: flight_parameter_features (bounded MAE 0.261109976)
- Rankings are separate; no combined best modality was created.
- Statistical unit: subject_id, n=35.
- Model retraining during consolidation: 0.
- Prediction regeneration during consolidation: 0.
- Multimodal provenance: {"result": "PASS", "classification": {"source_path": "E:\\hdc-vr-pilot\\experiments\\phase_06_hdc_variant_screening\\results\\oof\\phase06_hybrid_final_oof.csv", "source_sha256": "ff619baf4be600279482c9e1f4f4139000fc05c1dfaf41555d644674b45d875a", "filter": {"variant": "hybrid", "dimension": 5000, "seeds": [42, 43, 44, 45, 46]}}, "regression": {"source_path": "E:\\hdc-vr-pilot\\experiments\\phase_05_basic_dual_output_hdc\\results\\oof\\vanilla_hdc_ridge_regression_oof.csv", "source_sha256": "a449d8f43a0935f0a3fcf8cf901894e426a83e552807dcef9551bc983ba22758", "filter": {"head": "COMMON_ENCODER_READOUT_BASELINE", "dimension": 10000, "ridge_alpha": 0.01, "seeds": [42, 43, 44, 45, 46]}}, "phase06_selection_trace_sha256": "228b55379460fcb6b2dafa1d392316c77d71671fc1ad086932fc71a6989f1339"}
