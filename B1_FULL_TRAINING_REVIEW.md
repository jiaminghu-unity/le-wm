# B1 FULL TRAINING REVIEW — and three-arm closed-loop verdict
**Date:** 2026-10-10 · ckpts: B1 `ckpt_100k` (clean re-selection), B3/Ours `ckpt_best`
**Protocol:** configs/p5_protocol_freeze.md (100k steps each; identical data/split/optimizer;
official ContactWorld Isaac evaluation, unchanged latent-MSE CEM, offsets 12/24/36/48 × 100 envs).

## Q1 — Did Multi-view LeWM train successfully? **YES**
- No collapse: latent var mean 0.98 / min 0.37, covariance effective rank 18.3.
- Beats persistence at every rollout horizon on held-out episodes:
  h1 0.0017 vs 0.0025 · h2 0.0053 vs 0.0093 · h4 0.0177 vs 0.0325 (−35…−45%).
- Protocol correction logged: the in-training "best-val" metric evaluated teacher-forced
  prediction at T=6 while positions were trained at T=3 — artifact made step-17.5k look
  best. Checkpoint selection redone at the trained regime (T=3, one-step val MSE):
  monotone improvement, best = ckpt_100k (0.00166 < persistence 0.0021).

## Q2 — Does the joint latent preserve wrist tactile-predictive information? **NO**
Matched ridge probes (fixed α=1000, episode-held-out, target = PCA-32 tactile
descriptor r_tau; incremental over action+phase nuisance):
| features | R² | ΔR² over nuisance | episodes with positive gain |
|---|---|---|---|
| A legacy wrist encoder | 0.109 | +0.067 | 83% [70, 93] |
| B B1 wrist tokens (pre-pool) | 0.083 | **+0.041** | **90% [77, 100]** |
| C B1 joint latent | 0.044 | +0.002 | 63% [47, 80] |
The information survives the ViT+cross-attention (B) and is lost at the 8-query
pooling + projection (C). Earlier smoke-era Test F numbers are void (wrong target
array and misalignment in the descriptor npz; corrected protocol above).

## Q3 — Are contact modes predictable from current RGB + action? **YES, but fusion degrades it**
macro-F1 (held-out): action+phase 0.50 · legacy front 0.38 · legacy wrist 0.64 ·
B1 pre-pool wrist 0.59 · **B1 joint latent 0.54**. Fusion suppresses rather than
helps — reported explicitly per directive §4.

## Q4 — Are contact transitions a systematic predictive weakness after full training? **NO**
Error/persistence ratios (held-out): contact 0.38–0.59 (easiest), free 0.48–0.79,
transition 0.48–0.74 ≈ free, all < 1 at every horizon. The smoke-era "transition
hardest" signal does not survive full training in persistence-normalized terms.

## Q5 — Closed-loop success (official protocol, joint success /100)
| arm | 12 | 24 | 36 | 48 | pooled |
|---|---|---|---|---|---|
| **B1** pure-vision multi-view | 48 | 18 | 4 | 0 | **17.5%** |
| **B3** multi-branch, no tactile (capacity control) | 30 | 5 | 3 | 0 | **9.5%** |
| **Ours** tactile-supervised gate + residuals | 48 | 14 | 2 | 0 | **16.0%** |
| reference: official front-only | 55 | 37 | 16 | 9 | 29.3% |
| reference: official wrist-only | 41 | 28 | 1 | 2 | 18.0% |
(References are architectural references, not matched controls.)

## Interpretation
1. **No net gain from tactile supervision:** Ours ≈ B1 (16.0 vs 17.5, within noise).
2. **The supervision is real:** the same architecture without it (B3) collapses to
   9.5%; tactile supervision restores +6.5pp. The gate (92–99% train accuracy,
   P3-validated modes) works — it just cannot help planning when its input lost the
   contact signal (Q2/Q3).
3. **All multi-view arms trail the official front-only reference**, with the gap
   widening at long offsets — consistent with the fusion bottleneck and an
   unmatched architecture comparison.

## Recommendation — **REVISE** (not STOP)
The offline causal chain (real modes → visually predictable → dynamics-distinct)
held at every step; the failure is localized to information routing:
- **R1.** Feed the gate (and residuals) from pre-pool wrist tokens, where the
  contact signal demonstrably survives (ΔR² +0.041, 90% episodes).
- **R2.** Widen/retune the fusion bottleneck (more queries or larger latent) and
  re-run Test F before any dynamics retraining; require C ≥ ~0.7×B retention.
- **R3.** Train a matched front-only LeWM control to separate "multi-view cost"
  from "tactile value" before further claims.
One arm (revised Ours) suffices to test R1+R2; no new baseline sweep needed.

## Artifacts
GCS: gs://prism-training-us/contactworld/outputs/{b1,b3_cluster,ours_cluster,closedloop_all}/
Board: Tactile Hard-Negative Board V12 · Branch: `tactile` (this commit).
