# P5 Protocol Freeze — B1 full training (insertion_usb)
Frozen: 2026-10-09. Changes after this point require explicit approval.

## Lineup (per Next-Steps directive)
B0 = official ContactWorld single-view checkpoints (done) · **B1 = pure-vision
multi-view LeWM (train now)** · B3/Ours = HOLD (no auto-launch) · B2 = CANCELLED.

## Code provenance
- Branch: `tactile` @ jiaminghu-unity/le-wm (this commit).
- ContactWorld repo: PokuangZhou/ContactWorld @ 8d0d0ed ("arXiv").
- LeWM blocks: copied verbatim from le-wm reproduce branch (module.py) into
  touchwm/lewm_modules.py; deviations D1 (ViT reimpl, same spec), D2 (projector on
  fused latent).

## Data
- Task: insertion_usb; Zarr store data/demo_data/insertion_usb (201 eps, 13,671 rows).
- Split: episode-level seed 3072 → 141 train / 30 val / 30 test
  (outputs/preflight/usb/splits.json; never regenerated).
- Preprocessing: both views 256→224 bilinear+antialias, float [0,1], identical
  transforms; frameskip 1; clips = 4 obs + 3 actions (terminal action dropped);
  no transition crosses an episode boundary.

## Architecture (B1, no tactile anywhere)
- Shared ViT-Tiny: dim 192, patch 14, depth 12, heads 3, 256 patch tokens/view,
  learned view embeddings.
- Fusion: 1 bidirectional cross-attention block per direction (4 heads, MLP 4×,
  pre-update features both directions) → 8 shared state queries → flatten →
  Linear → latent 192.
- Projector / pred-projector: MLP 192→2048→192, BatchNorm1d (official LeWM config).
- Predictor: original ARPredictor — depth 6, heads 16, dim_head 64, mlp 2048,
  dropout 0.1, AdaLN-zero action conditioning; action encoder = original Embedder.
- History size 3.

## Training
- Loss: L = ||pred − z_next||² (mean) + 0.09·SIGReg(knots 17, proj 1024);
  end-to-end, no EMA, no stop-grad.
- Optimizer: AdamW lr 5e-5 (constant), weight decay 1e-3, grad-clip 1.0.
- Batch 32 × clip 4 obs; seed 3072; 100,000 steps; bf16 off (fp32, matches smoke).
- Checkpoints: every 10k steps + best-val; resume-capable (optimizer state saved).

## Diagnostics (recorded during training)
Every 500 steps: train pred/SIGReg. Every 2,500 steps on a FIXED set of 64 val
clips (length 7): one-step and multi-step (h=1,2,4,6) latent MSE vs persistence
(ẑ=z_t), latent per-dim variance, covariance effective rank, per-module gradient
norms (front/wrist shared ViT, cross-attn f/w, query pool, projector, predictor,
action encoder).

## Checkpoint selection (predeclared)
Best held-out VAL one-step prediction MSE (not test SR, not train loss).

## Evaluation (after training; unchanged ContactWorld protocol)
- Isaac Gym evaluator on cw-l4-01; goal offsets 12/24/36/48; CEM: horizon 6,
  100 candidates, top-k 8, 4 iterations, num-envs 100, max-steps 15,
  stop-on-success; official joint-success definition; official action penalties.
- Cost: C(A)=||ẑ_{t+H}(A) − z_g||² only. No tuning of CEM on B1 results.
- References (architectural, not matched controls): front-only 29.3%, wrist-only
  18.0% pooled.

## Post-B1 analysis set (prepared during training, run after)
Test F (3 feature locations A/B/C, matched ridge protocol) · M5 four-input mode
probe · mode-conditioned 1/多步误差 vs persistence(匹配动作幅度/相位/几何) ·
closed-loop SR。Deliverable: B1_FULL_TRAINING_REVIEW.md + summary JSON →
GO/REVISE/STOP for B3/Ours(不自动开训)。
