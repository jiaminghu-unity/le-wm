# Multi-seed success rates — baseline vs L_obj vs aux q-head

Generated 2026-08-04 06:33 UTC from `eval_results/final/final_<task>_<config>_<solver>_s<seed>.csv`.

## Protocol

| | |
|---|---|
| Training | 10 epochs, 1 GPU, seed **3072**, `weights_epoch_10.pt` — a SINGLE training seed, see Limitations |
| Backbone | ViT-Tiny (patch 14, 192-d) + ARPredictor (depth 6, heads 16, mlp 2048) + MLP projector (hidden 2048, BatchNorm1d) |
| SIGReg | sliced Epps–Pulley, knots 17, num_proj 1024, λ=0.09 (repo default) |
| Planner | HORIZON=5, RECEDING_HORIZON=5, ACTION_BLOCK=5, EVAL_BUDGET=50 env steps, GOAL_OFFSET=25, VAR_SCALE=1.0 |
| Elites | max(round(0.1·candidates), 2) |
| Sampling tiers | T1 300/30, T2 150/15, T3 50/10, T4 20/5, T5 10/3 (candidates/iterations) |
| GD tiers | T1 100/90, T2 75/30, T3 50/10, T4 20/5, T5 10/3, AdamW lr 0.1 |
| Planner noise | `cem_seed = crc32("<episode_id>|<tier>")` — identical across configs, so every comparison is paired |
| Episodes | 100 per seed, drawn without replacement; sets pre-registered, every drawn seed reported |
| Rendering | NVIDIA GL driver installed, `check_render_fidelity.py` gate passed (reacher MAE 2.34 ≤ 3.0) |

### Arms

| task | q_dim | q | dose | baseline | L_obj | aux |
|---|---|---|---|---|---|---|
| pusht | 6 | 6-d: agent xy, block xy, cos/sin(block angle) | obj λ=0.1 / aux w=0.3 | `lewm_c1_s3072` | `lewm_c3_sig_obj0.1_s3072` | `lewm_c5_qhead0.3_s3072` |
| reacher | 4 | 4-d: cos/sin of both joint angles | obj λ=0.15 / aux w=0.4 | `lewm_r1_reacher_s3072` | `lewm_r2_reacher_paep_l015_s3072` | `lewm_r5_qhead0.4_s3072` |
| cube | 9 | 9-d: effector xyz, cos/sin(2·yaw), gripper opening, block xyz | obj λ=0.1 / aux w=0.1 | `lewm_k1_cube_s3072` | `lewm_k2_cube_obj_eff0.1_s3072` | `lewm_k4_cube_qhead_eff0.1_s3072` |

`L_obj = 1 - Pearson(||Δz||², ||Δq||²)` on the encoder-side embedding (the tensor SIGReg sees), K=4096 stratified pairs, 50% within-subtrajectory, squared distances, mean/std not detached. aux = MLP 192→256→dim(q), MSE on standardised q, outside the JEPA, discarded at eval.

## pusht  (q_dim=6, 6 episode seeds: s101, s102, s103, s104, s105, s106)

Each cell pools all seeds' episodes. `\*` = paired exact McNemar p<0.05.

| solver | tier | n | base | obj | aux | obj−base | aux−base | obj−aux |
|---|---|---|---|---|---|---|---|---|
| cem | T1 | 600 | 92.3 | 96.0 | 94.2 | +3.7\* | +1.8 | +1.8 |
| cem | T2 | 600 | 90.5 | 93.2 | 93.3 | +2.7\* | +2.8\* | -0.2 |
| cem | T3 | 600 | 79.2 | 84.7 | 84.7 | +5.5\* | +5.5\* | +0.0 |
| cem | T4 | 600 | 57.0 | 63.0 | 66.2 | +6.0\* | +9.2\* | -3.2 |
| cem | T5 | 600 | 33.5 | 32.5 | 36.2 | -1.0 | +2.7 | -3.7\* |
| icem | T1 | 600 | 86.2 | 89.7 | 91.3 | +3.5\* | +5.2\* | -1.7 |
| icem | T2 | 600 | 80.7 | 87.0 | 87.7 | +6.3\* | +7.0\* | -0.7 |
| icem | T3 | 600 | 73.5 | 79.0 | 80.3 | +5.5\* | +6.8\* | -1.3 |
| icem | T4 | 600 | 53.5 | 59.8 | 60.5 | +6.3\* | +7.0\* | -0.7 |
| icem | T5 | 600 | 25.7 | 25.5 | 26.7 | -0.2 | +1.0 | -1.2 |
| mppi | T1 | 600 | 66.8 | 74.7 | 74.2 | +7.8\* | +7.3\* | +0.5 |
| mppi | T2 | 600 | 64.0 | 70.5 | 72.3 | +6.5\* | +8.3\* | -1.8 |
| mppi | T3 | 600 | 61.2 | 66.7 | 62.0 | +5.5\* | +0.8 | +4.7 |
| mppi | T4 | 600 | 48.2 | 54.2 | 53.3 | +6.0\* | +5.2\* | +0.8 |
| mppi | T5 | 600 | 36.0 | 37.2 | 37.2 | +1.2 | +1.2 | +0.0 |
| gd | T1 | 600 | 93.2 | 95.3 | 94.7 | +2.2 | +1.5 | +0.7 |
| gd | T2 | 600 | 85.7 | 88.2 | 90.0 | +2.5 | +4.3\* | -1.8 |
| gd | T3 | 600 | 69.5 | 73.7 | 76.2 | +4.2\* | +6.7\* | -2.5 |
| gd | T4 | 600 | 60.2 | 63.5 | 68.0 | +3.3 | +7.8\* | -4.5\* |
| gd | T5 | 600 | 46.3 | 50.5 | 48.3 | +4.2\* | +2.0 | +2.2 |

| contrast | mean | positive | significant | range |
|---|---|---|---|---|
| obj−base | **+4.08pp** | 18/20 | 14/20 | -1.0 .. +7.8 |
| aux−base | **+4.71pp** | 20/20 | 13/20 | +0.8 .. +9.2 |
| obj−aux | **-0.63pp** | 6/20 | 2/20 | -4.5 .. +4.7 |

Per-seed aggregate (each value = mean over that seed's 20 solver×tier cells):

| contrast | s101 | s102 | s103 | s104 | s105 | s106 | mean | SD | SE | σ |
|---|---|---|---|---|---|---|---|---|---|---|
| obj−base | +3.45 | +3.10 | +3.00 | +6.30 | +4.15 | +4.50 | **+4.08** | 1.24 | 0.50 | **8.1** |
| aux−base | +3.80 | +3.95 | +3.25 | +5.45 | +5.25 | +6.55 | **+4.71** | 1.25 | 0.51 | **9.3** |
| obj−aux | -0.35 | -0.85 | -0.25 | +0.85 | -1.10 | -2.05 | **-0.63** | 0.97 | 0.40 | **1.6** |

## reacher  (q_dim=4, 6 episode seeds: s101, s102, s103, s104, s105, s106)

Each cell pools all seeds' episodes. `\*` = paired exact McNemar p<0.05.

| solver | tier | n | base | obj | aux | obj−base | aux−base | obj−aux |
|---|---|---|---|---|---|---|---|---|
| cem | T1 | 600 | 81.8 | 82.5 | 80.7 | +0.7 | -1.2 | +1.8 |
| cem | T2 | 600 | 78.8 | 78.7 | 77.3 | -0.2 | -1.5 | +1.3 |
| cem | T3 | 600 | 72.7 | 74.8 | 74.2 | +2.2 | +1.5 | +0.7 |
| cem | T4 | 600 | 64.3 | 64.5 | 65.0 | +0.2 | +0.7 | -0.5 |
| cem | T5 | 600 | 55.2 | 57.2 | 55.0 | +2.0 | -0.2 | +2.2 |
| icem | T1 | 600 | 85.5 | 86.7 | 84.3 | +1.2 | -1.2 | +2.3 |
| icem | T2 | 600 | 81.8 | 86.2 | 87.0 | +4.3\* | +5.2\* | -0.8 |
| icem | T3 | 600 | 83.3 | 86.5 | 86.8 | +3.2 | +3.5\* | -0.3 |
| icem | T4 | 600 | 77.8 | 82.3 | 81.2 | +4.5\* | +3.3 | +1.2 |
| icem | T5 | 600 | 61.7 | 64.0 | 61.5 | +2.3 | -0.2 | +2.5 |
| mppi | T1 | 600 | 40.5 | 41.8 | 40.3 | +1.3 | -0.2 | +1.5 |
| mppi | T2 | 600 | 48.7 | 50.2 | 53.0 | +1.5 | +4.3 | -2.8 |
| mppi | T3 | 600 | 51.5 | 49.2 | 47.3 | -2.3 | -4.2 | +1.8 |
| mppi | T4 | 600 | 48.5 | 48.0 | 47.5 | -0.5 | -1.0 | +0.5 |
| mppi | T5 | 600 | 46.2 | 45.3 | 44.0 | -0.8 | -2.2 | +1.3 |
| gd | T1 | 600 | 61.5 | 68.0 | 64.3 | +6.5\* | +2.8 | +3.7 |
| gd | T2 | 600 | 69.7 | 72.2 | 74.0 | +2.5 | +4.3 | -1.8 |
| gd | T3 | 600 | 61.2 | 61.7 | 60.5 | +0.5 | -0.7 | +1.2 |
| gd | T4 | 600 | 59.7 | 58.0 | 61.7 | -1.7 | +2.0 | -3.7 |
| gd | T5 | 600 | 55.0 | 57.3 | 56.5 | +2.3 | +1.5 | +0.8 |

| contrast | mean | positive | significant | range |
|---|---|---|---|---|
| obj−base | **+1.48pp** | 15/20 | 3/20 | -2.3 .. +6.5 |
| aux−base | **+0.84pp** | 10/20 | 2/20 | -4.2 .. +5.2 |
| obj−aux | **+0.64pp** | 14/20 | 0/20 | -3.7 .. +3.7 |

Per-seed aggregate (each value = mean over that seed's 20 solver×tier cells):

| contrast | s101 | s102 | s103 | s104 | s105 | s106 | mean | SD | SE | σ |
|---|---|---|---|---|---|---|---|---|---|---|
| obj−base | +0.75 | +1.85 | +0.85 | +2.60 | +1.80 | +1.05 | **+1.48** | 0.72 | 0.29 | **5.0** |
| aux−base | +1.55 | +1.55 | +0.10 | +1.30 | +1.40 | -0.85 | **+0.84** | 0.99 | 0.41 | **2.1** |
| obj−aux | -0.80 | +0.30 | +0.75 | +1.30 | +0.40 | +1.90 | **+0.64** | 0.93 | 0.38 | **1.7** |

## cube  (q_dim=9, 3 episode seeds: s101, s102, s103)

Each cell pools all seeds' episodes. `\*` = paired exact McNemar p<0.05.

| solver | tier | n | base | obj | aux | obj−base | aux−base | obj−aux |
|---|---|---|---|---|---|---|---|---|
| cem | T1 | 300 | 70.3 | 73.7 | 74.7 | +3.3 | +4.3 | -1.0 |
| cem | T2 | 300 | 68.3 | 72.0 | 72.0 | +3.7 | +3.7 | +0.0 |
| cem | T3 | 300 | 60.7 | 65.7 | 61.3 | +5.0\* | +0.7 | +4.3 |
| cem | T4 | 300 | 56.3 | 56.0 | 52.7 | -0.3 | -3.7 | +3.3 |
| cem | T5 | 300 | 53.0 | 52.3 | 54.0 | -0.7 | +1.0 | -1.7 |
| icem | T1 | 300 | 72.3 | 79.0 | 75.0 | +6.7\* | +2.7 | +4.0\* |
| icem | T2 | 300 | 68.7 | 78.7 | 74.7 | +10.0\* | +6.0\* | +4.0\* |
| icem | T3 | 300 | 70.3 | 77.3 | 74.3 | +7.0\* | +4.0\* | +3.0 |
| icem | T4 | 300 | 62.7 | 64.7 | 64.7 | +2.0 | +2.0 | +0.0 |
| icem | T5 | 300 | 54.3 | 60.0 | 56.7 | +5.7\* | +2.3 | +3.3\* |
| mppi | T1 | 300 | 48.3 | 53.0 | 48.3 | +4.7\* | +0.0 | +4.7 |
| mppi | T2 | 300 | 48.7 | 51.0 | 52.0 | +2.3 | +3.3 | -1.0 |
| mppi | T3 | 300 | 47.3 | 45.0 | 49.3 | -2.3 | +2.0 | -4.3 |
| mppi | T4 | 300 | 50.3 | 49.7 | 49.0 | -0.7 | -1.3 | +0.7 |
| mppi | T5 | 300 | 51.0 | 50.3 | 48.7 | -0.7 | -2.3 | +1.7 |
| gd | T1 | 300 | 70.3 | 79.3 | 81.3 | +9.0\* | +11.0\* | -2.0 |
| gd | T2 | 300 | 65.3 | 69.7 | 70.3 | +4.3 | +5.0\* | -0.7 |
| gd | T3 | 300 | 64.0 | 64.3 | 63.0 | +0.3 | -1.0 | +1.3 |
| gd | T4 | 300 | 60.0 | 63.7 | 62.7 | +3.7 | +2.7 | +1.0 |
| gd | T5 | 300 | 54.0 | 56.7 | 54.7 | +2.7 | +0.7 | +2.0 |

| contrast | mean | positive | significant | range |
|---|---|---|---|---|
| obj−base | **+3.28pp** | 15/20 | 7/20 | -2.3 .. +10.0 |
| aux−base | **+2.15pp** | 15/20 | 4/20 | -3.7 .. +11.0 |
| obj−aux | **+1.13pp** | 12/20 | 3/20 | -4.3 .. +4.7 |

Per-seed aggregate (each value = mean over that seed's 20 solver×tier cells):

| contrast | s101 | s102 | s103 | mean | SD | SE | σ |
|---|---|---|---|---|---|---|---|
| obj−base | +4.65 | +1.50 | +3.70 | **+3.28** | 1.62 | 0.93 | **3.5** |
| aux−base | +2.70 | +0.35 | +3.40 | **+2.15** | 1.60 | 0.92 | **2.3** |
| obj−aux | +1.95 | +1.15 | +0.30 | **+1.13** | 0.83 | 0.48 | **2.4** |

## Cross-task summary

| task | q_dim | seeds | obj−base | σ | aux−base | σ | obj−aux | σ |
|---|---|---|---|---|---|---|---|---|
| pusht | 6 | 6 | **+4.08** | 8.1 | **+4.71** | 9.3 | -0.63 | 1.6 |
| reacher | 4 | 6 | **+1.48** | 5.0 | **+0.84** | 2.1 | +0.64 | 1.7 |
| cube | 9 | 3 | **+3.28** | 3.5 | **+2.15** | 2.3 | +1.13 | 2.4 |

σ = |mean| / SE of the per-seed aggregate; it quantifies episode-sampling variance only.

### What these numbers do and do not establish

1. **L_obj helps on all three tasks and the effect is resolvable**: +4.08 / +1.48 / +3.28 pp at 8.1σ / 5.0σ / 3.5σ, with 13/20, 3/20 and 7/20 cells individually significant.
2. **The aux q-head also helps on all three tasks**: +4.71 / +0.84 / +2.15 pp.
3. **L_obj vs aux is NOT resolved on any task** (1.6σ / 1.7σ / 2.4σ) and the sign disagrees between tasks: Push-T favours aux (−0.63), Reacher and Cube favour L_obj (+0.64, +1.13). Any claim that one dominates the other is unsupported by this data.
4. **Gains concentrate where the baseline is weak.** Push-T mppi T1 (base 66.8) gains +7.8; Push-T gd T1 (base 93.6) gains +2.2. Both losses fail at Push-T T5 (10 candidates / 3 elites — the search is too coarse for a better representation to matter).
5. **q dimensionality does not order the outcome.** obj−aux is +0.64 at q_dim 4, −0.63 at 6, +1.13 at 9. Within cube, however, going from 9-d to 21-d q (adding 5 arm joints) costs the aux arm −2.2 / −2.7 pp on two independent episode sets while leaving the L_obj arm unchanged (−0.2 / +0.5) — consistent in direction across both sets but only reaching p=0.058, one training seed each.

### Limitations

- **One training seed (3072) per arm.** Episode-sampling variance is measured; training variance is not. The LeWM paper reports ± of median 2.80 (max 7.5) across its 3 training seeds, larger than the 1–3 pp obj-vs-aux differences above. Resolving obj−aux to 2σ at the observed SD would need on the order of 100 training seeds.
- **Cube has 3 episode seeds, not 6**, so its σ values are correspondingly weaker.
- **Cube data predates the EGL render fix but is unaffected**: re-running the k1 baseline under the fixed renderer moved 10 cells by at most ±1.0 pp with 295–300/300 episode-level agreement. Reacher moved 6–14 pp and was fully re-run. Push-T renders on the CPU via box2d and was never affected (200/200 episode-exact reproduction).
- **Combo arms excluded** from this round; their 3-seed additivity (−0.27 pp over 40 cells) is in `REPORT_final_crosstask.md` §6.
