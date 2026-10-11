# tactile-ckpts — trained checkpoints for the tactile multi-view LeWM study
- b1_ckpt_100k.pt   — B1 pure-vision multi-view LeWM (100k steps, best by clean val)
- b3_ckpt_best.pt   — B3 capacity-matched multi-branch, no tactile supervision
- ours_ckpt_best.pt — Ours tactile-supervised gate + residual dynamics
Protocol: tactile branch configs/p5_protocol_freeze.md · results: tactile branch results/
Closed-loop (pooled): B1 17.5% / B3 9.5% / Ours 16.0% — see B1_FULL_TRAINING_REVIEW.md
