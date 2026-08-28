# Reproduction scope and limits

The package reproduces provenance checks, file existence, SHA-256 checksums, row/run-key/subject/fold/seed coverage, artifact registries, and report-source alignment. It does not reproduce historical model training or statistical computation.

Scientific boundaries: regression is a **bounded difficulty-induced workload proxy regression**, not directly measured continuous cognitive workload. LOSO supports held-out-subject generalization only. Cross-session, cross-scenario, task-template, and route generalization remain unevaluated because required metadata are unavailable. Flight-feature evidence is predictive and setting-specific, not causal.

Historical engineering/provenance caveats remain: the Phase 06 original final-manifest hash is verified, six non-scientific embedded metadata records differ, and the historical frozen-artifact immutability audit remains FAIL for two non-scientific files. Scientific artifact changes remain zero.

Two Phase 09 hashes recorded by the earlier Phase 10 initialization manifest are stale relative to the stable current direct freeze chain. The current `phase09_freeze.json` embeds the current final-manifest SHA-256, both current files predate this synthesis, and neither changed during it. The registry retains both recorded and current hashes as a non-scientific initialization-reference alignment caveat.

Frozen artifacts must never be overwritten. Verification is read-only outside Phase 10.
