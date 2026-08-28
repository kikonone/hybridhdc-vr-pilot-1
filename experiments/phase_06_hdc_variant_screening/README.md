# Phase 06 — HDC Variant Screening

Status: `QUICK_SCREEN_COMPLETE`

This directory is initialized for read-only screening preparation of four P1 HDC variants:

1. Vanilla Prototype HDC
2. OnlineHD-style HDC
3. Multi-centroid HDC
4. HDC+OnlineHD Hybrid

The current step performs only upstream integrity checks and input preflight. It does not train, tune, select, or evaluate any HDC variant. Vanilla Prototype HDC evidence is referenced read-only from frozen Phase 05; no observed outer-test dimension or seed is promoted to a Phase 06 configuration.

The planned Phase 06 deliverables, after a separate contract freeze and later authorized modeling, are:

- a comparison table for all four P1 HDC variants;
- a comparison table for HDC classification and regression heads;
- a performance–time–memory Pareto plot;
- best-classification-HDC and best-regression-HDC configurations selected under a preregistered rule.

None of those results or best configurations is claimed during initialization.

The Phase 06 variant contract is now frozen, and classification-only quick screening is complete for all five outer folds: 24 OnlineHD-style candidates, 6 Multi-centroid candidates, and 32 Hybrid candidates per fold. Selection used only three-fold subject-wise inner CV within each outer-training set. These are fold-specific screening selections, not a final-best-HDC decision.

Outer-test features and labels remained sealed. No outer-test prediction, OOF output, similarity regression, Ridge readout, or Final Confirmation was executed. Final Confirmation requires a separate authorization.

Primary data and frozen fold assignments remain in Phase 03 and are not copied here. Phase 04A/04B are read-only traditional-baseline comparison interfaces, and Phase 05 is the read-only Vanilla HDC baseline interface.

The regression target is interpreted only as **bounded difficulty-induced workload proxy regression**. Project claims are limited to **workload-proxy classification and regression**; the target is not directly measured continuous cognitive workload.

Before any modeling, a separate Phase 06 contract-freeze step must define the OnlineHD update rule, low-confidence criterion, update counts, multi-centroid construction and empty-center handling, Hybrid order, screening spaces, elimination/Pareto rules, and canonical-configuration selection rule.

Run the preflight from this directory with:

```powershell
python scripts/initialize_phase06.py
```

The Notebook is the authoritative executed initialization record. Generated JSON files contain machine-readable PASS/FAIL results and hashed evidence paths.


## Final OOF consolidation status

- OOF consolidation, independent metric recalculation, stability analysis, Pareto analysis, figures, reports, and final manifest: complete.
- Historical model-selection gate: `MODEL_SELECTION_BLOCKED` under the original frozen rules; the original FAIL audit is preserved.
- Current Phase 06 status: `FROZEN` after the audited inner-CV-only post-freeze amendment v2 resolved the gate. Phase 07 is ready but has not been started.
- Key artifacts: `results/oof/`, `results/summaries/phase06_*`, `reports/analysis-output/`, `reports/phase06_final_summary.md`, and `manifests/phase06_final_artifact_manifest.json`.

<!-- PHASE06_SELECTION_RESOLUTION_V2 -->
## Model-selection resolution (post-freeze amendment v2)

The original `MODEL_SELECTION_BLOCKED` result is retained as provenance. Its cause was that the frozen rule covered only per-outer-fold, new-variant Quick Screen classification (`classification_only=true`, `regression_heads_executed=false`); it did not define the final four-family classification comparison, regression-head comparison, or efficiency/Pareto handling. The original rule and FAIL audit were not modified.

Amendment v2 was necessary to define an executable final selection without consulting outer-test performance. It evaluated 8 classification families and 20 regression families from saved inner-CV evidence, using equal weighting of the five outer-training tasks and no selection of a single seed. The unique selections are **HDC+OnlineHD Hybrid at d=5000** for classification and **COMMON_ENCODER_READOUT_BASELINE at d=10000** for regression. The preselection outer-evidence seal contains 72 artifacts and has SHA-256 `7f78fe0e66f6c020142cc7fe099b7844e84cd34af5c39df5dbf5f48e528a42e9`; its post-selection integrity audit passed.

The Phase 06 canonical model-selection amendment was defined after final-confirmation artifacts existed, but its executable selector was restricted to previously saved inner-CV and unlabeled efficiency evidence. Outer-OOF artifacts were hash-sealed before selection and were not read by the selector.

After selection was fixed, descriptive outer-OOF summaries were read. Across the five frozen seeds, the selected classifier had Macro-F1 0.822309 (sample SD 0.032325), balanced accuracy 0.821894, and severe-error rate 0.020048. The selected regression head had bounded MAE 0.276390 (sample SD 0.006419), bounded RMSE 0.407517, R² 0.866858, and Spearman correlation 0.924752. These are descriptive results, not evidence of statistically significant superiority.

Because the amendment was defined after final-confirmation artifacts existed, selection-induced optimism cannot be ruled out even with executable isolation. Phase 07 must use this now-fixed procedure and the frozen selected families without revisiting outer-OOF ranking. A later final LOSO evaluation is responsible for the more independent confirmation.

Phase 06 is eligible for freezing only after all resolution gates pass; the final gate audit records that outcome.
