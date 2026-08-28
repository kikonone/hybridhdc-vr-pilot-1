# Phase 06 Inner-Only Model Selection Resolution Plan

## Goal

Resolve the preserved final-selection blocker using the explicitly authorized post-freeze inner-CV-only amendment, seal outer evidence before selection, and conditionally freeze Phase 06 without retraining or repredicting.

## Phases

- [x] Phase 1: Confirm the preserved root cause and snapshot the original rule/audit hashes.
- [x] Phase 2: Create amendment v2 and seal every outer OOF/prediction/outer-metric artifact.
- [x] Phase 3: Implement the allowlisted, deterministic inner-evidence selector and tests.
- [x] Phase 4: Run classification selection across 8 families and regression selection across 20 families.
- [x] Phase 5: Re-run the selector and verify deterministic best-JSON SHA-256 values.
- [x] Phase 6: Verify the outer seal is unchanged, then read selected outer OOF results for descriptive reporting only.
- [x] Phase 7: Update reports, README, and append/execute the Notebook while preserving historical FAIL provenance.
- [x] Phase 8: Regenerate audits/manifest, create and validate phase06_freeze.json, and stop before Phase 07.

## Decisions Made

- The original `phase06_model_selection_rules.json` and original FAIL audit are immutable provenance.
- Amendment status is `INNER_CV_ONLY_POST_FREEZE_AMENDMENT` and is not described as originally preregistered.
- The selector uses a hard input allowlist and rejects OOF/prediction/outer-metric paths.
- Seeds are aggregated and never selected individually.

## Errors Encountered

- The first selector run exposed a pandas column-access collision (`selected.head` resolved to the DataFrame method); corrected to `selected["head"]` before any best-model artifact was produced.

## Status

**Complete** — all resolution audits pass, Phase 06 is frozen, and Phase 07 was not started.
