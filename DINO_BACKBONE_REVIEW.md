# DINO_BACKBONE_REVIEW — frozen foundation backbone for multi-view LeWM
**Date:** 2026-10-11 · Backbone: frozen **DINOv2 ViT-S/14** (documented fallback —
DINOv3 ViT-S/16 code loads but LVD weights are license-gated, HTTP 403).
Total A100 budget used: **zero** (D0 probes, D1 pilot, D2 full training on cached
fp16 patch tokens, and closed-loop eval all ran on the single L4 evaluator).

## D0 — frozen feature probes (PASS)
| features | tactile ΔR² | mode F1 | geometry (plug/socket/ee/rel) |
|---|---|---|---|
| wrist mean | **+0.075** | **0.642** | 0.42/—/0.44/0.77 |
| wrist 2×2 grid | +0.076 | 0.639 | 0.47/—/0.48/0.87 |
| front 2×2 grid | −0.026 | 0.555 | 0.93/**0.996**/0.93/0.90 |
| both grid | +0.033 | 0.629 | 0.96/1.00/0.96/0.94 |
Frozen DINO wrist beats every trained encoder to date (legacy +0.067, scratch ViT
+0.051) at zero training; views are cleanly complementary (front = global geometry,
wrist = contact). Naive concat dilutes the wrist tactile signal — fusion must
protect the wrist channel.

## D1 — pilot + four-stage information audit (PASS; projector erasure FIXED)
Trainable: linear adapters 384→192 + existing cross-attention / 8 queries /
projection / projector MLP / original ARPredictor (14.3M params). 15k-step pilot,
23 min on L4, stable, beats persistence, eff-rank ≈ 17.
| stage | old B1 (scratch ViT) | DINO-D1 |
|---|---|---|
| wrist raw | +0.051 | +0.075 |
| after cross-attn | +0.041 | +0.061 |
| after pooling | +0.021 | +0.041 |
| **final joint latent** | **+0.002** | **+0.032** (mode F1 0.576 vs 0.537) |

## D2 — full 100k training + unchanged MSE-CEM closed-loop
Frozen protocol (same splits/optimizer/steps/CEM/joint-success).
| arm | 12 | 24 | 36 | 48 | pooled |
|---|---|---|---|---|---|
| **DINO-B1** | 42 | 19 | 3 | 0 | **16.0%** |
| old B1 (scratch ViT) | 48 | 18 | 4 | 0 | 17.5% |
| official front reference | 55 | 37 | 16 | 9 | 29.3% |
Paired per-offset differences are within binomial noise (n=100/cell): the backbone
swap did **not** move closed-loop success despite clearly better representation
probes at every stage.

## Interpretation
1. The backbone WAS an information bottleneck (D0/D1 prove it) — but removing it
   did not lift SR. The binding constraint for planning sits elsewhere:
   most plausibly (a) the action-conditioned predictor trained on 201 demos /
   13.5k transitions, and (b) the discriminative geometry of a single latent-MSE
   goal cost at millimeter tolerances (pair-check: contact-different states
   remain barely separated in any of our joint spaces).
2. Dual-target structure (earlier Exp C, 8.3%) and backbone swap (16.0%) both fail
   to beat the plain joint latent — architecture-level edits are exhausted at this
   data scale.

## Recommendation — **REVISE (data/objective axis), HOLD B3/Ours**
Per the staged directive, DINO-B1 did not show credible planning improvement, so
tactile Gate+Residual retraining stays on hold. Highest-value next probes, both
cheap on the existing stack:
- R-A **action-ranking diagnostic** (memo §9.6 style): matched-state candidate sets,
  realized-outcome ranking vs model-cost ranking for old-B1 and DINO-B1 — directly
  tests whether the latent-MSE cost, not the representation, is what fails.
- R-B **data scaling on cached tokens**: pool the 5 other downloaded ContactWorld
  task datasets through the frozen backbone (token caches make multi-task training
  nearly free) and re-measure USB prediction + SR.
Decision on R-A/R-B (or stopping the line with the current negative-result dossier)
is left to the owner.

## Artifacts
GCS: gs://prism-training-us/contactworld/outputs/{d2,closedloop_all/dinofull}/ ·
branch `tactile`: d0_dino_probe.py, d1_pilot.py, d1_audit.py, d2_train.py,
dino_mv.py, dino_eval_adapter.py (JIT-traced backbone for the legacy-python
evaluator), results/preflight/{d0_dino_probe,d1_audit}.json.
