# Notes: Frozen HDC Dual Task Demonstration UI

## Inventory
- The prechange Streamlit application contained six Chinese sidebar pages and only a classification record explorer.
- A 23-file timestamped backup is stored at `backups/pre_dual_task_20260826_131701`.
- The existing full upstream baseline covers 8,525 Phase 00-10 files outside `phase10/ui` and was captured before UI development.

## Frozen source resolution
- Classification canonical reference: `phase07_readonly_multimodal_classification_reference.csv`, 419 rows, SHA-256 `5933f705875e205a31c487384bbc6bb0460fe36076a1dafa01825859759c42a8`.
- Classification source chain: Phase 07 reference -> Phase 06 `phase06_hybrid_final_oof.csv`, SHA-256 `ff619baf4be600279482c9e1f4f4139000fc05c1dfaf41555d644674b45d875a`.
- Regression canonical reference: `phase07_readonly_multimodal_regression_reference.csv`, 419 rows, SHA-256 `6af63df19a14c652b991230833a5d9bdf264cb1c5c387823f42d6e5cd30e0c7a`.
- Regression source chain: Phase 07 reference -> Phase 05 `vanilla_hdc_ridge_regression_oof.csv`, SHA-256 `a449d8f43a0935f0a3fcf8cf901894e426a83e552807dcef9551bc983ba22758`; Phase 06 preselection seal and final OOF alignment audit independently record the same hash.
- Phase 10 final prediction-library index marks both Phase 07 references as `CANONICAL_OOF`, `CANONICAL`, `NO_SINGLE_SEED`, 419 unique run keys, 35 subjects, and folds 1-5.
- Frozen regression interface: `COMMON_ENCODER_READOUT_BASELINE`, `common_ridge`, dimension 10000, feature_k 50, levels 51, ridge alpha 0.01 for folds 1-5 and seeds 42-46.
- Frozen regression aggregation: arithmetic mean of five raw seed predictions, followed upstream by clipping to [1.0, 4.0]. The UI must copy both canonical raw and bounded values and derive absolute error only for presentation.
- Frozen classification interface: `HDC+OnlineHD Hybrid`, dimension 5000, feature_k 50, levels 51. Canonical scores are arithmetic means across five seeds followed by argmax.
- Frozen fold assignment SHA-256: `e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f`.

## Final verification
- Compile/import/package validation: PASS.
- Pytest: 26 passed, 0 failed, including 10 fail-closed negative cases.
- Playwright E2E: PASS at 1366x768 and 1920x1080 with 0 console errors, 0 page errors, and 0 tracebacks.
- Bounded stress: PASS; 25 sessions, 1,675 HTTP/page interactions, 500 rapid switches, 100 reloads, 603.032 seconds.
- Latency: p50 14.016 ms, p95 18.119 ms, maximum 139.760 ms; throughput 2.778 interactions/s; error rate 0.
- Memory: PASS; peak server-process-tree RSS 104.684 MB and final-minus-initial growth -1.773 MB.
- Phase 00-10 upstream immutability: PASS; 0 modified, 0 added, 0 removed files relative to the 8,525-file baseline.

## Audience-facing simplification request
- Remove the top `PHASE 10 · OFFLINE DEFENSE CONSOLE` and `HDC // OOF` identifiers.
- Remove `Frozen`, `canonical`, and `OOF` language from the normal demonstration view.
- Render internal `DEMO-####` identifiers as `Record ####` without changing data alignment or source files.
- Preserve the existing dark aviation console design and scientific integrity controls behind the UI.

## Audience-facing simplification verification
- Visible top provenance strip: removed.
- Visible record format: `Record ####`; internal `DEMO-####` keys remain unchanged.
- Visible occurrences of `Frozen`, `OOF`, `canonical`, `PHASE 10`, and `DEMO-`: 0 in the tested page body.
- Pytest: 27 passed, 0 failed.
- Playwright E2E: PASS at 1366x768 and 1920x1080, including classification, regression, record selection, refresh, and rapid task switching.
- Browser diagnostics: 0 console errors, 0 page errors, and 0 tracebacks.
- Updated screenshots: four required viewport/task captures regenerated.
