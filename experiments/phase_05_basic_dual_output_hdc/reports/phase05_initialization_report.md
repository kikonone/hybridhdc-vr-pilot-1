# Phase 05 initialization report

## Status

Initialized / not trained. This report records preflight evidence only; no HDC model, encoder, random hypervector, preprocessing fit, prediction, OOF generation, or parameter search was executed.

## Frozen inputs

- Primary SHA-256: `0a2aef89c01b43c3a4e5afe40b96797784627665490982731b2541f96b45fc44` (pass: True)
- Frozen-fold SHA-256: `e4dc943af21851bace49345f6336f9c88f82613ca3a26b3c433efe7dfb041f6f` (pass: True)
- Modeling rows / subjects / predictive features: 419 / 35 / 1176.

## Upstream comparison references

- Phase 04A: Gradient Boosting, OOF Macro-F1=0.9356075023489924.
- Phase 04B: Gradient Boosting Regressor, bounded OOF MAE=0.1074862234692518.

These values are read-only comparison references; they did not select any HDC configuration.

## Gate

The detailed HDC implementation contract remains `PENDING_CONTRACT_FREEZE`. No Vanilla HDC training may begin until it is frozen.
