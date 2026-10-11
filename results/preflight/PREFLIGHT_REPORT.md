# PREFLIGHT REPORT — Tactile-Guided Multi-View LeWM (insertion_usb)
**Date:** 2026-10-09 · **Recommendation: GO** (with two riders, see end)

## 1. Code provenance
- Official modules reused: ContactWorld dataset/simulator/eval (`eval_planner.py`,
  official USB checkpoints 100000.ckpt), LeWM blocks copied verbatim into
  `touchwm/lewm_modules.py` (ARPredictor, SIGReg, Embedder, MLP).
- New modules: `touchwm/multiview_encoder.py`, `touchwm/lewm_adapter.py`,
  `touchwm/data_adapter.py`, `preflight/*` (this branch).
- Logged deviations: D1 ViT-Tiny reimplemented at identical spec (no HF dep);
  D2 projector on fused latent instead of CLS token.

## 2. Stage results
| Stage | Verdict | Key numbers |
|---|---|---|
| P0 data audit | **PASS** | 201 eps / 13,671 frames / 13,470 transitions; 18 keys; 0 NaN/Inf; canonical split 141/30/30 (seed 3072); alignment invariants verified |
| P1 multi-view smoke | **PASS 6/6** | shapes/finiteness OK; gradients flow everywhere (action path confirmed post-AdaLN-zero warmup); 24-clip overfit ×82; both views verified load-bearing; 6-step rollout bounded & action-sensitive; **Isaac evaluator end-to-end OK on L4 (8-env smoke: joint 3/8 at offset 12, within noise of the official 55%)** |
| P2 mode feasibility | **PASS** | modes {free, contact, transition, ambiguous 18.8%}; every mode in ≥185/201 eps; **matched-command motion ratio free/contact = 1.7–2.0× (geo-mean 1.8×)**, consistent across command quartiles |
| P3 visual gate probe | **PASS** | wrist+action macro-F1 **0.64** [0.58,0.68] vs action+phase 0.49 [0.44,0.54] (non-overlapping CIs); transition AUPRC 0.51 vs 0.43; front features *hurt* (0.38) |
| P4 mode-conditioned dynamics | **PASS (offline core)** | per-mode ridge beats single model by **22.7%** held-out (transition −28.7%); random-mode control ≈0%; deployable gate (69% agreement) retains **18.0%**; phase & action magnitude fed to both models |
| P4 §9.4 WM errors | partial | transition mode is hardest for the smoke-scale WM (pred/persistence 3.1× vs 2.5×/1.8×); full-model version deferred |
| Test F wrist retention | **DEFERRED** | fused-latent tactile R² ≈ 0 at smoke scale — but pre-pool wrist tokens also ≈ 0 ⇒ **not a pooling bottleneck; insufficient training**. Re-test after full B1 training; B2's tactile auxiliary head is the planned injection mechanism |
| §9.6 action ranking | deferred | needs matched-state sim harness; scheduled with full training |

## 3. Hypotheses (proposal §14)
1. Complementary front/wrist info preservable in one latent — *engineering verified; information-level verdict deferred with Test F*.
2. Tactile identifies meaningful transition regimes — **supported** (P2).
3. Regimes partially predictable from RGB+action — **supported** (P3, wrist-driven).
4. Separate dynamics modeling improves contact-consequence prediction — **supported** (P4, controls held).
5. Path to better action selection under unchanged latent-MSE CEM — **untested**; this is what full training + closed-loop evaluation decides.

## 4. Known risks
- Modes correlate with episode phase (0.31/0.64/0.73 mean progress); controlled in
  P3/P4 but must be monitored in closed-loop (gate must not become a progress clock).
- Effect sizes are consistent-but-moderate (1.8× motion, 18–23% dynamics MSE).
- Front view dilutes gate prediction; fusion design should keep wrist primary for gating.
- Legacy frozen features have encoder-provenance caveat (possible train overlap);
  probe-level splits were episode-held-out.

## 5. Recommendation
**GO** — proceed to P5 protocol freeze and full training (B0/B1/B2/B3/Ours), with riders:
R1: re-run Test F + M5 on the fully trained B1 encoder before training the gate;
if tactile signal is still absent after B2's auxiliary supervision, REVISE the fusion.
R2: evaluate gating on wrist-primary features as an ablation of the fused gate input.

## 6. Artifacts
All outputs mirror to `gs://prism-training-us/contactworld/outputs/preflight/usb/`
(manifest, splits, sync report, mode features/labels/examples, probe metrics,
dynamics comparison, smoke logs, figures, fused features, trained smoke ckpt).
Results board: Tactile Hard-Negative Board V11.
