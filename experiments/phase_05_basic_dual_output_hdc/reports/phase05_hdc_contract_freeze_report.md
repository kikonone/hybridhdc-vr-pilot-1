# Phase 05 HDC contract freeze report

## Status

`CONTRACT_FROZEN_NOT_TRAINED`. This is a static contract freeze only. No HDC model fit, hypervector generation, encoding, prototype construction, parameter search, prediction, or OOF artifact was executed.

## Frozen design

- Representation: bipolar `[-1, +1]`; item/level/sample arrays are `int8`, accumulators `int32`, and normalized prototype/similarity arrays `float32`.
- Shared encoder candidates: 16 combinations of dimensions `[2000, 5000]`, levels `[21, 51]`, k `[50, 100, 200, all]`, and seed `42`.
- Final-confirmation plan: dimensions `[1000, 2000, 5000, 10000]`; seeds `[42, 43, 44, 45, 46]`; not executed.
- Similarity temperature grid: `[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]`. Ridge alpha grid: `[0.01, 0.1, 1.0, 10.0, 100.0]`.

## Static gate

All JSON contracts parsed, selection-space checks passed, outer-test selection is prohibited, Phase 06 variants are excluded, and every results subdirectory remains file-empty. The next authorized operation is the Vanilla HDC quick screen under this frozen contract.
