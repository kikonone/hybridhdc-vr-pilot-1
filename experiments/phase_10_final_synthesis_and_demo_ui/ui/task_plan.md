# Task Plan: HDC Dual Task System Demonstration UI

## Goal
Convert the existing Streamlit application into a single page, English only, offline classification and proxy regression demonstration backed exclusively by verified frozen canonical records, then complete functional, browser, stress, soak, and immutability audits.

## Phases
- [x] Phase 1: Inventory UI files, instructions, dependencies, manifests, tests, and launch scripts
- [x] Phase 2: Back up the prechange UI and establish the upstream frozen source integrity ledger
- [x] Phase 3: Resolve canonical classification and regression sources and build aligned anonymous demo data
- [x] Phase 4: Refactor the application into one English demonstration page with two task tabs
- [x] Phase 5: Expand static, schema, negative, language, alignment, and contract tests
- [x] Phase 6: Run compile, import, unit, contract, data, and frozen hash verification
- [x] Phase 7: Start the local server and complete Playwright E2E at both required viewports
- [x] Phase 8: Run bounded concurrency, interaction, reload, and ten minute soak tests
- [x] Phase 9: Stop all test processes, verify immutability, write final audits, and deliver the exact result summary
- [x] Phase 10: Simplify the audience-facing system demo, remove marked provenance labels, and hide the DEMO prefix
- [x] Phase 11: Update UI selectors/tests and verify both required viewports

## Key Questions
1. Which Phase 06/10 manifest chain authoritatively identifies the selected common ridge canonical regression OOF?
2. Can classification and regression records be aligned by real run key before creating one stable DEMO ID mapping?
3. Does the final UI remain read only and stable under rapid task/record interaction and concurrent local sessions?

## Decisions Made
- All writes are confined to the requested `ui` directory.
- Existing aviation and mission control styling will be retained and refined rather than replaced.
- Scientific source files remain read only; the UI receives only minimized anonymous presentation copies.
- The Phase 07 read-only multimodal canonical references are the presentation sources because the Phase 10 prediction-library index marks both as canonical 419-row OOF artifacts.
- The regression reference is accepted only because it points to the Phase 05 ridge OOF hash sealed by Phase 06 and implements the frozen five-seed aggregation rule before bounded clipping.
- Scientific provenance remains intact in internal data and audits, while the normal audience-facing page uses neutral system-demonstration language.
- Stable internal DEMO identifiers remain unchanged; only their rendered label changes to `Record 0001` through `Record 0419`.

## Errors Encountered
- Audience-copy update changed the fold validation message capitalization; one negative test used a case-sensitive old regex. Updated the assertion and reran the suite.
- The first final test command used the bundled base interpreter, which does not include pytest. Re-ran with the project Python environment; all 27 tests passed.

## Status
**COMPLETE** - the audience-facing simplification is implemented, 27 tests pass, and Playwright verification passes at 1366x768 and 1920x1080 with no browser errors.
