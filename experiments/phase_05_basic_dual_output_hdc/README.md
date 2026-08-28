# Phase 05 — Basic Dual-Output HDC

## Goal

Define the audited, initialization-only contract for Vanilla Prototype HDC with a shared encoder, four-class prototype head, similarity-weighted regression decoder, and regularized Ridge hypervector readout. Its tasks are **four-class task-difficulty-induced workload proxy classification** and **bounded difficulty-induced workload proxy regression**.

## Position in the experiment plan

Phase 04A and 04B provide frozen traditional classification and regression comparison baselines; their files remain read-only. Phase 05 does not enter Phase 06: no OnlineHD-style, multi-centroid, or hybrid HDC variants are included.

## Status and gate

**contract frozen / not trained.** The detailed HDC implementation contract is frozen in `configs/`. Vanilla HDC quick screening is now authorized only under that contract; all other HDC variants and later-phase experiments remain prohibited.

## Layout

- `data/`: policy only; no input copies.
- `configs/`: experiment, environment, frozen encoding, model-selection, regression-head, efficiency, and search-space contracts.
- `manifests/` and `audits/`: input provenance and executed preflight evidence.
- `logs/`, `reports/`, `figures/`, and `results/`: reserved for later authorized work; all result subdirectories are empty at initialization.
- `Phase_05_Basic_Dual_Output_HDC.ipynb`: executed initialization preflight plus executed JSON-only contract validation.


## Final Phase 05 status

Phase 05 completed 5/5 Final Confirmation folds and 100/100 fold-level runs, consolidated 20 aligned 419-run OOF configurations, and produced audited classification plus two bounded regression-head result matrices. The final frozen interface preserves all four dimensions and five seeds; no post-hoc canonical dimension or seed was selected from outer-test performance.

### Main artifacts

- Final report: `reports/phase05_final_summary.md`
- OOF outputs: `results/oof/vanilla_hdc_*_oof.csv`
- Seed/dimension and baseline summaries: `results/summaries/`
- Publication figures: `figures/phase05_*.png`
- Final manifest: `manifests/phase05_final_artifact_manifest.json`
- Final audits: `audits/phase05_final_*`
- Freeze: `configs/phase05_freeze.json`

## No-retraining compliance completion

The post-freeze compliance amendment adds diagnostics derived from the unchanged frozen OOF table and protocol-complete read-only inference measurements from saved fitted artifacts. It does not retrain models or alter historical checkpoints, predictions, OOF tables, or the preregistered result matrix. Training timing is explicitly `NOT_PERFORMED_RETRAINING_PROHIBITED`; it was not fabricated or remeasured by refitting. See `reports/phase05_no_retraining_completion_report.md`.
