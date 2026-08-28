# Phase 06 Final OOF Consolidation and Freeze Plan

## Goal

Consolidate existing frozen predictions, independently recalculate metrics, apply the frozen model-selection rules, produce the strict analysis bundle, persist the notebook, and freeze Phase 06 without retraining or repredicting.

## Phases

- [x] Phase 1: Preflight Phase 05/06 frozen interfaces, checksums, manifests, and all Final Confirmation artifacts.
- [x] Phase 2: Consolidate three new-variant OOF files and build the aligned four-variant comparison library.
- [x] Phase 3: Independently recalculate metrics and cross-check saved results at tolerance 1e-12.
- [x] Phase 4: Aggregate seed/dimension stability and actual efficiency; compute valid Pareto fronts.
- [x] Phase 5: Apply the frozen selection rules and save complete selection traces.
- [x] Phase 6: Generate scientific figures, strict analysis bundle, comparison tables, and final report.
- [x] Phase 7: Append and execute the final notebook section; audit persistence.
- [x] Phase 8: Build and independently audit the final manifest, verify upstream integrity, and create phase06_freeze.json.
- [x] Phase 9: Stop before Phase 07.

## Decisions Made

- Existing Quick Screen and Final Confirmation artifacts are read-only.
- Phase 05 Vanilla and common Ridge predictions are referenced from their frozen source artifacts.
- Subject is the inferential unit; run is the prediction unit; folds and seeds are not independent subjects.
- No inferential test or confidence interval will be added unless explicitly preregistered in the frozen selection rules.

## Errors Encountered

- None.

## Status

**Completed with MODEL_SELECTION_BLOCKED** — all permitted consolidation and analysis work finished; stopped before Phase 07.
