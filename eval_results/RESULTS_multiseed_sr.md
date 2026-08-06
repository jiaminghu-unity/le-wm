# Multi-seed success rates — baseline vs L_obj vs aux q-head vs combo

Generated 2026-08-06 01:44 UTC by `scripts/make_multiseed_report.py` from `eval_results/final/final_<task>_<config>_<solver>_s<seed>.csv`. Seeds are discovered from the files present, not hard-coded.

## Protocol

| | |
|---|---|
| Training | 10 epochs, 1 GPU, seed **3072**, `weights_epoch_10.pt` — a SINGLE training seed, see Limitations |
| Planner | HORIZON=5, RECEDING_HORIZON=5, ACTION_BLOCK=5, EVAL_BUDGET=50 env steps, GOAL_OFFSET=25 |
| Tiers | sampling: T1 300/30, T2 150/15, T3 50/10, T4 20/5, T5 10/3 (candidates/iterations), elites = max(round(0.1·cand), 2) |
| | gradient: T1 100/90, T2 75/30, T3 50/10, T4 20/5, T5 10/3, AdamW lr=0.1 |
| | rollout evaluations per replan are matched across families: 9000 / 2250 / 500 / 100 / 30 |
| Planner noise | `cem_seed = crc32("<episode_id>\|<tier>")` — identical across configs, so every comparison is paired |
| Episodes | 100 per seed, drawn without replacement by `gen_episodes.py` (unfiltered); sets pre-registered, every drawn seed reported |
| Statistics | exact paired McNemar per cell; per-seed aggregate for SD/SE/σ |

### Arms

| task | q_dim | q | dose | base | obj | aux | combo |
|---|---|---|---|---|---|---|---|
| pusht | 6 | 6-d: pusher xy, block xy, cos/sin(block angle) | obj λ=0.1 / aux w=0.3 / combo 0.1+0.3 | `lewm_c1_s3072` | `lewm_c3_sig_obj0.1_s3072` | `lewm_c5_qhead0.3_s3072` | `lewm_c6_o01a03_s3072` |
| reacher | 4 | 4-d: cos/sin of the two joint angles | obj λ=0.15 / aux w=0.4 (no combo arm) | `lewm_r1_reacher_s3072` | `lewm_r2_reacher_paep_l015_s3072` | `lewm_r5_qhead0.4_s3072` | — |
| cube | 9 | 9-d: effector xyz, cos/sin(2·yaw), gripper opening, block xyz | obj λ=0.1 / aux w=0.1 / combo 0.1+0.1 | `lewm_k1_cube_s3072` | `lewm_k2_cube_obj_eff0.1_s3072` | `lewm_k4_cube_qhead_eff0.1_s3072` | `lewm_k6_cube_combo_o0.1a0.1_s3072` |


## pusht  (q_dim=6, 6 episode seeds: s101, s102, s103, s104, s105, s106)

### Per-seed SR by arm

Absolute SR, each value averaged over that seed's 20 solver×tier cells (100 episodes each). This is the layer that makes a seed drawn under a different protocol visible instead of averaged away.

| seed | base | obj | aux | combo | obj−base | aux−base | combo−base | obj−aux |
|---|---|---|---|---|---|---|---|---|
| s101 | 62.5 | 66.0 | 66.3 | 68.0 | +3.45 | +3.80 | +5.40 | -0.35 |
| s102 | 68.8 | 72.0 | 72.8 | 73.7 | +3.10 | +3.95 | +4.85 | -0.85 |
| s103 | 64.5 | 67.5 | 67.7 | 69.2 | +3.00 | +3.25 | +4.80 | -0.25 |
| s104 | 62.2 | 68.5 | 67.7 | 71.2 | +6.30 | +5.45 | +9.00 | +0.85 |
| s105 | 68.1 | 72.2 | 73.3 | 73.3 | +4.15 | +5.25 | +5.20 | -1.10 |
| s106 | 64.7 | 69.2 | 71.2 | 71.2 | +4.50 | +6.55 | +6.55 | -2.05 |
| **mean** | **65.1** | **69.2** | **69.9** | **71.1** | **+4.08** | **+4.71** | **+5.97** | **-0.62** |
| SD | 2.8 | 2.5 | 3.0 | 2.2 |  |  |  |  |

### Per-seed SR by solver

The same absolute SR split by solver — each value is the mean over that solver's 5 tiers for one seed.

| solver | seed | base | obj | aux | combo |
|---|---|---|---|---|---|
| cem | s101 | 70.0 | 70.0 | 68.6 | 71.4 |
| cem | s102 | 76.2 | 76.8 | 79.4 | 79.8 |
| cem | s103 | 70.0 | 73.6 | 72.0 | 72.4 |
| cem | s104 | 66.8 | 73.8 | 74.6 | 74.4 |
| cem | s105 | 69.6 | 75.0 | 78.6 | 74.2 |
| cem | s106 | 70.4 | 74.0 | 76.2 | 74.4 |
| **cem mean** | — | **70.5** | **73.9** | **74.9** | **74.4** |
| icem | s101 | 63.4 | 66.2 | 69.4 | 66.8 |
| icem | s102 | 68.0 | 71.4 | 71.0 | 73.4 |
| icem | s103 | 63.0 | 67.0 | 67.6 | 71.0 |
| icem | s104 | 58.6 | 66.0 | 65.4 | 68.0 |
| icem | s105 | 67.0 | 70.4 | 72.0 | 73.4 |
| icem | s106 | 63.4 | 68.2 | 70.4 | 68.0 |
| **icem mean** | — | **63.9** | **68.2** | **69.3** | **70.1** |
| mppi | s101 | 47.6 | 53.6 | 54.2 | 57.2 |
| mppi | s102 | 56.2 | 64.6 | 63.8 | 64.4 |
| mppi | s103 | 52.6 | 56.0 | 54.6 | 57.8 |
| mppi | s104 | 57.8 | 61.8 | 57.8 | 66.0 |
| mppi | s105 | 62.4 | 67.4 | 65.2 | 68.6 |
| mppi | s106 | 54.8 | 60.4 | 63.2 | 66.0 |
| **mppi mean** | — | **55.2** | **60.6** | **59.8** | **63.3** |
| gd | s101 | 69.2 | 74.2 | 73.2 | 76.4 |
| gd | s102 | 75.0 | 75.0 | 77.0 | 77.2 |
| gd | s103 | 72.2 | 73.2 | 76.6 | 75.8 |
| gd | s104 | 65.8 | 72.6 | 73.0 | 76.6 |
| gd | s105 | 73.4 | 76.2 | 77.6 | 77.0 |
| gd | s106 | 70.2 | 74.2 | 75.2 | 76.6 |
| **gd mean** | — | **71.0** | **74.2** | **75.4** | **76.6** |

### Per cell (seed-average)

Every seed's episodes pooled — 600 per cell. `\*` = paired exact McNemar p<0.05. Pooling and averaging the per-seed SRs coincide here because every seed contributes the same 100 episodes.

| solver | tier | n | base | obj | aux | combo | obj−base | aux−base | combo−base | obj−aux |
|---|---|---|---|---|---|---|---|---|---|---|
| cem | T1 | 600 | 92.3 | 96.0 | 94.2 | 96.0 | +3.7\* | +1.8 | +3.7\* | +1.8 |
| cem | T2 | 600 | 90.5 | 93.2 | 93.3 | 94.2 | +2.7\* | +2.8\* | +3.7\* | -0.2 |
| cem | T3 | 600 | 79.2 | 84.7 | 84.7 | 85.2 | +5.5\* | +5.5\* | +6.0\* | +0.0 |
| cem | T4 | 600 | 57.0 | 63.0 | 66.2 | 64.0 | +6.0\* | +9.2\* | +7.0\* | -3.2 |
| cem | T5 | 600 | 33.5 | 32.5 | 36.2 | 32.8 | -1.0 | +2.7 | -0.7 | -3.7\* |
| icem | T1 | 600 | 86.2 | 89.7 | 91.3 | 92.2 | +3.5\* | +5.2\* | +6.0\* | -1.7 |
| icem | T2 | 600 | 80.7 | 87.0 | 87.7 | 88.3 | +6.3\* | +7.0\* | +7.7\* | -0.7 |
| icem | T3 | 600 | 73.5 | 79.0 | 80.3 | 82.2 | +5.5\* | +6.8\* | +8.7\* | -1.3 |
| icem | T4 | 600 | 53.5 | 59.8 | 60.5 | 61.8 | +6.3\* | +7.0\* | +8.3\* | -0.7 |
| icem | T5 | 600 | 25.7 | 25.5 | 26.7 | 26.0 | -0.2 | +1.0 | +0.3 | -1.2 |
| mppi | T1 | 600 | 66.8 | 74.7 | 74.2 | 76.8 | +7.8\* | +7.3\* | +10.0\* | +0.5 |
| mppi | T2 | 600 | 64.0 | 70.5 | 72.3 | 75.2 | +6.5\* | +8.3\* | +11.2\* | -1.8 |
| mppi | T3 | 600 | 61.2 | 66.7 | 62.0 | 67.8 | +5.5\* | +0.8 | +6.7\* | +4.7 |
| mppi | T4 | 600 | 48.2 | 54.2 | 53.3 | 58.8 | +6.0\* | +5.2\* | +10.7\* | +0.8 |
| mppi | T5 | 600 | 36.0 | 37.2 | 37.2 | 38.0 | +1.2 | +1.2 | +2.0 | +0.0 |
| gd | T1 | 600 | 93.2 | 95.3 | 94.7 | 96.7 | +2.2 | +1.5 | +3.5\* | +0.7 |
| gd | T2 | 600 | 85.7 | 88.2 | 90.0 | 90.8 | +2.5 | +4.3\* | +5.2\* | -1.8 |
| gd | T3 | 600 | 69.5 | 73.7 | 76.2 | 76.7 | +4.2\* | +6.7\* | +7.2\* | -2.5 |
| gd | T4 | 600 | 60.2 | 63.5 | 68.0 | 68.3 | +3.3 | +7.8\* | +8.2\* | -4.5\* |
| gd | T5 | 600 | 46.3 | 50.5 | 48.3 | 50.5 | +4.2\* | +2.0 | +4.2\* | +2.2 |

### By solver

SR is the mean over that solver's 5 tiers. σ comes from the per-seed aggregate restricted to the same solver, so it is episode-sampling variance within that solver.

| solver | base | obj | aux | combo | obj−base | σ | aux−base | σ | combo−base | σ | obj−aux | σ | obj−base sig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cem | 70.5 | 73.9 | 74.9 | 74.4 | **+3.37** | 3.1 | **+4.40** | 2.8 | **+3.93** | 4.5 | **-1.03** | 1.2 | 4/5 |
| icem | 63.9 | 68.2 | 69.3 | 70.1 | **+4.30** | 6.3 | **+5.40** | 8.7 | **+6.20** | 6.9 | **-1.10** | 1.8 | 4/5 |
| mppi | 55.2 | 60.6 | 59.8 | 63.3 | **+5.40** | 7.5 | **+4.57** | 3.3 | **+8.10** | 9.1 | **+0.83** | 0.9 | 4/5 |
| gd | 71.0 | 74.2 | 75.4 | 76.6 | **+3.27** | 3.2 | **+4.47** | 6.5 | **+5.63** | 4.4 | **-1.20** | 2.0 | 2/5 |

### Over all 20 solver×tier cells

| contrast | mean | cells >0 | cells sig | range |
|---|---|---|---|---|
| obj−base | **+4.08** | 18/20 | 14/20 | -1.0 … +7.8 |
| aux−base | **+4.71** | 20/20 | 13/20 | +0.8 … +9.2 |
| combo−base | **+5.97** | 19/20 | 17/20 | -0.7 … +11.2 |
| obj−aux | **-0.63** | 6/20 | 2/20 | -4.5 … +4.7 |

### Per-seed contrasts

Each value = mean over that seed's 20 solver×tier cells. σ = |mean| / SE, and it quantifies **episode-sampling variance only**.

| contrast | s101 | s102 | s103 | s104 | s105 | s106 | mean | SD | SE | σ |
|---|---|---|---|---|---|---|---|---|---|---|
| obj−base | +3.45 | +3.10 | +3.00 | +6.30 | +4.15 | +4.50 | **+4.08** | 1.24 | 0.50 | 8.1 |
| aux−base | +3.80 | +3.95 | +3.25 | +5.45 | +5.25 | +6.55 | **+4.71** | 1.25 | 0.51 | 9.3 |
| combo−base | +5.40 | +4.85 | +4.80 | +9.00 | +5.20 | +6.55 | **+5.97** | 1.62 | 0.66 | 9.0 |
| obj−aux | -0.35 | -0.85 | -0.25 | +0.85 | -1.10 | -2.05 | **-0.63** | 0.97 | 0.40 | 1.6 |

### Additivity  (6 seeds: s101, s102, s103, s104, s105, s106)

| quantity | value |
|---|---|
| obj−base | +4.08 |
| aux−base | +4.71 |
| sum, if the two losses stacked | +8.79 |
| combo−base, measured | **+5.97** |
| shortfall | **-2.82** pp |
| fraction of the expected gain realised | **68%** |
| combo − whichever single loss is better, per cell | +0.72 (15/20 positive, 0/20 significant) |

Combo lands near the better single loss rather than near their sum. Since obj and aux both act on the same channel — candidate ranking, with obj roughly twice the effect (see the P4/P5 diagnostics) — they compete for the same gain, so the absence of stacking is what that mechanism predicts.


## reacher  (q_dim=4, 6 episode seeds: s101, s102, s103, s104, s105, s106)

### Per-seed SR by arm

Absolute SR, each value averaged over that seed's 20 solver×tier cells (100 episodes each). This is the layer that makes a seed drawn under a different protocol visible instead of averaged away.

| seed | base | obj | aux | obj−base | aux−base | obj−aux |
|---|---|---|---|---|---|---|
| s101 | 65.4 | 66.2 | 67.0 | +0.75 | +1.55 | -0.80 |
| s102 | 62.9 | 64.7 | 64.4 | +1.85 | +1.55 | +0.30 |
| s103 | 63.8 | 64.7 | 63.9 | +0.85 | +0.10 | +0.75 |
| s104 | 64.3 | 67.0 | 65.7 | +2.60 | +1.30 | +1.30 |
| s105 | 65.2 | 67.0 | 66.7 | +1.80 | +1.40 | +0.40 |
| s106 | 64.0 | 65.0 | 63.1 | +1.05 | -0.85 | +1.90 |
| **mean** | **64.3** | **65.8** | **65.1** | **+1.48** | **+0.84** | **+0.64** |
| SD | 1.0 | 1.1 | 1.6 |  |  |  |

### Per-seed SR by solver

The same absolute SR split by solver — each value is the mean over that solver's 5 tiers for one seed.

| solver | seed | base | obj | aux |
|---|---|---|---|---|
| cem | s101 | 71.6 | 73.4 | 71.4 |
| cem | s102 | 67.2 | 65.6 | 67.4 |
| cem | s103 | 70.8 | 69.6 | 69.8 |
| cem | s104 | 67.6 | 71.0 | 70.8 |
| cem | s105 | 73.8 | 77.2 | 76.0 |
| cem | s106 | 72.4 | 72.4 | 67.2 |
| **cem mean** | — | **70.6** | **71.5** | **70.4** |
| icem | s101 | 78.0 | 81.8 | 83.0 |
| icem | s102 | 78.2 | 80.2 | 79.6 |
| icem | s103 | 78.4 | 80.0 | 81.0 |
| icem | s104 | 76.8 | 84.2 | 81.8 |
| icem | s105 | 80.0 | 82.2 | 79.6 |
| icem | s106 | 76.8 | 78.4 | 76.0 |
| **icem mean** | — | **78.0** | **81.1** | **80.2** |
| mppi | s101 | 50.8 | 46.2 | 49.8 |
| mppi | s102 | 45.8 | 48.8 | 45.8 |
| mppi | s103 | 45.6 | 43.4 | 43.8 |
| mppi | s104 | 45.6 | 48.4 | 45.0 |
| mppi | s105 | 49.2 | 46.6 | 48.6 |
| mppi | s106 | 45.4 | 48.0 | 45.6 |
| **mppi mean** | — | **47.1** | **46.9** | **46.4** |
| gd | s101 | 61.2 | 63.2 | 63.6 |
| gd | s102 | 60.2 | 64.2 | 64.8 |
| gd | s103 | 60.4 | 65.6 | 61.0 |
| gd | s104 | 67.4 | 64.2 | 65.0 |
| gd | s105 | 58.0 | 62.2 | 62.4 |
| gd | s106 | 61.2 | 61.2 | 63.6 |
| **gd mean** | — | **61.4** | **63.4** | **63.4** |

### Per cell (seed-average)

Every seed's episodes pooled — 600 per cell. `\*` = paired exact McNemar p<0.05. Pooling and averaging the per-seed SRs coincide here because every seed contributes the same 100 episodes.

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

### By solver

SR is the mean over that solver's 5 tiers. σ comes from the per-seed aggregate restricted to the same solver, so it is episode-sampling variance within that solver.

| solver | base | obj | aux | obj−base | σ | aux−base | σ | obj−aux | σ | obj−base sig |
|---|---|---|---|---|---|---|---|---|---|---|
| cem | 70.6 | 71.5 | 70.4 | **+0.97** | 1.1 | **-0.13** | 0.1 | **+1.10** | 1.1 | 0/5 |
| icem | 78.0 | 81.1 | 80.2 | **+3.10** | 3.4 | **+2.13** | 2.1 | **+0.97** | 1.3 | 2/5 |
| mppi | 47.1 | 46.9 | 46.4 | **-0.17** | 0.1 | **-0.63** | 2.2 | **+0.47** | 0.4 | 0/5 |
| gd | 61.4 | 63.4 | 63.4 | **+2.03** | 1.6 | **+2.00** | 1.9 | **+0.03** | 0.0 | 1/5 |

### Over all 20 solver×tier cells

| contrast | mean | cells >0 | cells sig | range |
|---|---|---|---|---|
| obj−base | **+1.48** | 15/20 | 3/20 | -2.3 … +6.5 |
| aux−base | **+0.84** | 10/20 | 2/20 | -4.2 … +5.2 |
| obj−aux | **+0.64** | 14/20 | 0/20 | -3.7 … +3.7 |

### Per-seed contrasts

Each value = mean over that seed's 20 solver×tier cells. σ = |mean| / SE, and it quantifies **episode-sampling variance only**.

| contrast | s101 | s102 | s103 | s104 | s105 | s106 | mean | SD | SE | σ |
|---|---|---|---|---|---|---|---|---|---|---|
| obj−base | +0.75 | +1.85 | +0.85 | +2.60 | +1.80 | +1.05 | **+1.48** | 0.72 | 0.29 | 5.0 |
| aux−base | +1.55 | +1.55 | +0.10 | +1.30 | +1.40 | -0.85 | **+0.84** | 0.99 | 0.41 | 2.1 |
| obj−aux | -0.80 | +0.30 | +0.75 | +1.30 | +0.40 | +1.90 | **+0.64** | 0.93 | 0.38 | 1.7 |

## cube  (q_dim=9, 6 episode seeds: s101, s102, s103, s104, s105, s106)

### Per-seed SR by arm

Absolute SR, each value averaged over that seed's 20 solver×tier cells (100 episodes each). This is the layer that makes a seed drawn under a different protocol visible instead of averaged away.

| seed | base | obj | aux | combo | obj−base | aux−base | combo−base | obj−aux |
|---|---|---|---|---|---|---|---|---|
| s101 | 60.6 | 65.3 | 63.4 | 64.0 | +4.65 | +2.70 | +3.40 | +1.95 |
| s102 | 62.6 | 64.2 | 63.0 | 64.7 | +1.50 | +0.35 | +2.00 | +1.15 |
| s103 | 56.1 | 59.9 | 59.5 | 59.6 | +3.70 | +3.40 | +3.50 | +0.30 |
| s104 | 64.2 | 65.0 | 64.2 | 64.7 | +0.85 | +0.10 | +0.55 | +0.75 |
| s105 | 53.4 | 56.6 | 57.0 | 59.0 | +3.30 | +3.65 | +5.70 | -0.35 |
| s106 | 60.9 | 64.3 | 64.8 | 64.0 | +3.45 | +3.90 | +3.10 | -0.45 |
| **mean** | **59.6** | **62.5** | **62.0** | **62.7** | **+2.91** | **+2.35** | **+3.04** | **+0.56** |
| SD | 4.1 | 3.5 | 3.1 | 2.6 |  |  |  |  |

### Per-seed SR by solver

The same absolute SR split by solver — each value is the mean over that solver's 5 tiers for one seed.

| solver | seed | base | obj | aux | combo |
|---|---|---|---|---|---|
| cem | s101 | 61.8 | 66.6 | 65.2 | 66.8 |
| cem | s102 | 64.8 | 65.0 | 64.2 | 65.6 |
| cem | s103 | 58.6 | 60.2 | 59.4 | 62.4 |
| cem | s104 | 66.0 | 67.6 | 66.6 | 66.6 |
| cem | s105 | 54.6 | 58.2 | 58.0 | 59.8 |
| cem | s106 | 62.4 | 67.4 | 67.6 | 66.2 |
| **cem mean** | — | **61.4** | **64.2** | **63.5** | **64.6** |
| icem | s101 | 67.0 | 74.6 | 71.2 | 74.2 |
| icem | s102 | 68.6 | 72.2 | 68.6 | 73.0 |
| icem | s103 | 61.4 | 69.0 | 67.4 | 68.8 |
| icem | s104 | 69.0 | 72.0 | 71.2 | 71.2 |
| icem | s105 | 58.0 | 65.0 | 63.0 | 65.6 |
| icem | s106 | 67.4 | 70.2 | 70.2 | 71.4 |
| **icem mean** | — | **65.2** | **70.5** | **68.6** | **70.7** |
| mppi | s101 | 50.4 | 50.0 | 49.6 | 50.4 |
| mppi | s102 | 51.6 | 52.2 | 52.6 | 51.6 |
| mppi | s103 | 45.4 | 47.2 | 46.2 | 45.4 |
| mppi | s104 | 54.6 | 53.2 | 54.0 | 54.4 |
| mppi | s105 | 44.2 | 43.6 | 47.2 | 47.4 |
| mppi | s106 | 50.6 | 54.0 | 52.6 | 52.8 |
| **mppi mean** | — | **49.5** | **50.0** | **50.4** | **50.3** |
| gd | s101 | 63.4 | 70.0 | 67.4 | 64.8 |
| gd | s102 | 65.6 | 67.2 | 66.6 | 68.4 |
| gd | s103 | 59.2 | 63.0 | 65.2 | 62.0 |
| gd | s104 | 67.0 | 67.2 | 65.2 | 66.6 |
| gd | s105 | 56.6 | 59.8 | 59.8 | 63.4 |
| gd | s106 | 63.2 | 65.8 | 68.8 | 65.6 |
| **gd mean** | — | **62.5** | **65.5** | **65.5** | **65.1** |

### Per cell (seed-average)

Every seed's episodes pooled — 600 per cell. `\*` = paired exact McNemar p<0.05. Pooling and averaging the per-seed SRs coincide here because every seed contributes the same 100 episodes.

| solver | tier | n | base | obj | aux | combo | obj−base | aux−base | combo−base | obj−aux |
|---|---|---|---|---|---|---|---|---|---|---|
| cem | T1 | 600 | 67.5 | 73.5 | 73.5 | 76.7 | +6.0\* | +6.0\* | +9.2\* | +0.0 |
| cem | T2 | 600 | 67.0 | 71.3 | 70.3 | 71.0 | +4.3\* | +3.3\* | +4.0\* | +1.0 |
| cem | T3 | 600 | 61.2 | 65.5 | 62.8 | 64.0 | +4.3\* | +1.7 | +2.8 | +2.7 |
| cem | T4 | 600 | 57.3 | 56.3 | 55.5 | 55.5 | -1.0 | -1.8 | -1.8 | +0.8 |
| cem | T5 | 600 | 53.8 | 54.2 | 55.3 | 55.7 | +0.3 | +1.5 | +1.8 | -1.2 |
| icem | T1 | 600 | 70.7 | 76.7 | 74.5 | 77.0 | +6.0\* | +3.8\* | +6.3\* | +2.2\* |
| icem | T2 | 600 | 68.0 | 76.5 | 73.3 | 77.8 | +8.5\* | +5.3\* | +9.8\* | +3.2\* |
| icem | T3 | 600 | 69.7 | 74.8 | 73.0 | 75.3 | +5.2\* | +3.3\* | +5.7\* | +1.8 |
| icem | T4 | 600 | 62.7 | 64.8 | 64.5 | 64.2 | +2.2 | +1.8 | +1.5 | +0.3 |
| icem | T5 | 600 | 55.2 | 59.7 | 57.7 | 59.2 | +4.5\* | +2.5\* | +4.0\* | +2.0 |
| mppi | T1 | 600 | 48.3 | 53.0 | 49.8 | 52.3 | +4.7\* | +1.5 | +4.0\* | +3.2\* |
| mppi | T2 | 600 | 48.7 | 51.5 | 50.0 | 50.5 | +2.8 | +1.3 | +1.8 | +1.5 |
| mppi | T3 | 600 | 47.8 | 46.5 | 50.7 | 48.8 | -1.3 | +2.8 | +1.0 | -4.2\* |
| mppi | T4 | 600 | 50.8 | 49.5 | 50.8 | 48.8 | -1.3 | +0.0 | -2.0 | -1.3 |
| mppi | T5 | 600 | 51.7 | 49.7 | 50.5 | 51.2 | -2.0 | -1.2 | -0.5 | -0.8 |
| gd | T1 | 600 | 69.5 | 76.2 | 78.2 | 77.5 | +6.7\* | +8.7\* | +8.0\* | -2.0 |
| gd | T2 | 600 | 65.0 | 67.7 | 68.7 | 69.3 | +2.7 | +3.7\* | +4.3\* | -1.0 |
| gd | T3 | 600 | 61.8 | 63.5 | 62.2 | 61.7 | +1.7 | +0.3 | -0.2 | +1.3 |
| gd | T4 | 600 | 60.3 | 63.5 | 62.2 | 62.7 | +3.2\* | +1.8 | +2.3 | +1.3 |
| gd | T5 | 600 | 55.8 | 56.7 | 56.3 | 54.5 | +0.8 | +0.5 | -1.3 | +0.3 |

### By solver

SR is the mean over that solver's 5 tiers. σ comes from the per-seed aggregate restricted to the same solver, so it is episode-sampling variance within that solver.

| solver | base | obj | aux | combo | obj−base | σ | aux−base | σ | combo−base | σ | obj−aux | σ | obj−base sig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cem | 61.4 | 64.2 | 63.5 | 64.6 | **+2.80** | 3.5 | **+2.13** | 2.4 | **+3.20** | 3.9 | **+0.67** | 2.8 | 3/5 |
| icem | 65.2 | 70.5 | 68.6 | 70.7 | **+5.27** | 5.5 | **+3.37** | 3.8 | **+5.47** | 6.0 | **+1.90** | 3.3 | 4/5 |
| mppi | 49.5 | 50.0 | 50.4 | 50.3 | **+0.57** | 0.8 | **+0.90** | 1.5 | **+0.87** | 1.5 | **-0.33** | 0.5 | 1/5 |
| gd | 62.5 | 65.5 | 65.5 | 65.1 | **+3.00** | 3.4 | **+3.00** | 2.5 | **+2.63** | 2.7 | **-0.00** | 0.0 | 2/5 |

### Over all 20 solver×tier cells

| contrast | mean | cells >0 | cells sig | range |
|---|---|---|---|---|
| obj−base | **+2.91** | 16/20 | 10/20 | -2.0 … +8.5 |
| aux−base | **+2.35** | 17/20 | 8/20 | -1.8 … +8.7 |
| combo−base | **+3.04** | 15/20 | 9/20 | -2.0 … +9.8 |
| obj−aux | **+0.56** | 13/20 | 4/20 | -4.2 … +3.2 |

### Per-seed contrasts

Each value = mean over that seed's 20 solver×tier cells. σ = |mean| / SE, and it quantifies **episode-sampling variance only**.

| contrast | s101 | s102 | s103 | s104 | s105 | s106 | mean | SD | SE | σ |
|---|---|---|---|---|---|---|---|---|---|---|
| obj−base | +4.65 | +1.50 | +3.70 | +0.85 | +3.30 | +3.45 | **+2.91** | 1.44 | 0.59 | 5.0 |
| aux−base | +2.70 | +0.35 | +3.40 | +0.10 | +3.65 | +3.90 | **+2.35** | 1.70 | 0.69 | 3.4 |
| combo−base | +3.40 | +2.00 | +3.50 | +0.55 | +5.70 | +3.10 | **+3.04** | 1.71 | 0.70 | 4.3 |
| obj−aux | +1.95 | +1.15 | +0.30 | +0.75 | -0.35 | -0.45 | **+0.56** | 0.92 | 0.38 | 1.5 |

### Additivity  (6 seeds: s101, s102, s103, s104, s105, s106)

| quantity | value |
|---|---|
| obj−base | +2.91 |
| aux−base | +2.35 |
| sum, if the two losses stacked | +5.26 |
| combo−base, measured | **+3.04** |
| shortfall | **-2.22** pp |
| fraction of the expected gain realised | **58%** |
| combo − whichever single loss is better, per cell | -0.39 (7/20 positive, 1/20 significant) |

Combo lands near the better single loss rather than near their sum. Since obj and aux both act on the same channel — candidate ranking, with obj roughly twice the effect (see the P4/P5 diagnostics) — they compete for the same gain, so the absence of stacking is what that mechanism predicts.


## Cross-task summary

| task | q_dim | seeds | obj−base | σ | aux−base | σ | obj−aux | σ |
|---|---|---|---|---|---|---|---|---|
| pusht | 6 | 6 | **+4.08** | 8.1 | **+4.71** | 9.3 | **-0.63** | 1.6 |
| reacher | 4 | 6 | **+1.48** | 5.0 | **+0.84** | 2.1 | **+0.64** | 1.7 |
| cube | 9 | 6 | **+2.91** | 5.0 | **+2.35** | 3.4 | **+0.56** | 1.5 |

σ = |mean| / SE of the per-seed aggregate; **episode-sampling variance only**.

### Cross-task absolute SR, by arm

Mean over all seeds and all 20 cells, with the seed-to-seed SD.

| task | base | obj | aux | combo |
|---|---|---|---|---|
| pusht | 65.1 ± 2.8 | 69.2 ± 2.5 | 69.9 ± 3.0 | 71.1 ± 2.2 |
| reacher | 64.3 ± 1.0 | 65.8 ± 1.1 | 65.1 ± 1.6 | — |
| cube | 59.6 ± 4.1 | 62.5 ± 3.5 | 62.0 ± 3.1 | 62.7 ± 2.6 |

### Cross-task, by solver

The three core contrasts split by solver, mean over that solver's 5 tiers, σ in brackets.

| solver | pusht obj−base | pusht aux−base | pusht obj−aux | reacher obj−base | reacher aux−base | reacher obj−aux | cube obj−base | cube aux−base | cube obj−aux |
|---|---|---|---|---|---|---|---|---|---|
| cem | +3.37 (3.1) | +4.40 (2.8) | -1.03 (1.2) | +0.97 (1.1) | -0.13 (0.1) | +1.10 (1.1) | +2.80 (3.5) | +2.13 (2.4) | +0.67 (2.8) |
| icem | +4.30 (6.3) | +5.40 (8.7) | -1.10 (1.8) | +3.10 (3.4) | +2.13 (2.1) | +0.97 (1.3) | +5.27 (5.5) | +3.37 (3.8) | +1.90 (3.3) |
| mppi | +5.40 (7.5) | +4.57 (3.3) | +0.83 (0.9) | -0.17 (0.1) | -0.63 (2.2) | +0.47 (0.4) | +0.57 (0.8) | +0.90 (1.5) | -0.33 (0.5) |
| gd | +3.27 (3.2) | +4.47 (6.5) | -1.20 (2.0) | +2.03 (1.6) | +2.00 (1.9) | +0.03 (0.0) | +3.00 (3.4) | +3.00 (2.5) | -0.00 (0.0) |

## Limitations

- **One training seed (3072) per arm.** Episode-sampling variance is measured; training variance is not. The LeWM paper reports ± of median 2.80 (max 7.5) across its 3 training seeds — larger than the sub-1 pp obj-vs-aux differences here. Resolving obj−aux to 2σ at the observed SD would need on the order of 100 training seeds.
- **Reacher has no combo arm** — it was never trained.
- **Cube's SR data predates the EGL render fix but is unaffected**: re-running the k1 baseline under the fixed renderer moved 10 cells by at most ±1.0 pp with 295–300/300 episode-level agreement. Reacher moved 6–14 pp and was fully re-run. Push-T renders on the CPU via box2d and was never involved (200/200 episode-exact reproduction). The render-fidelity gate itself was also wrong for a while — it read `world.infos["pixels"]`, a snapshot refreshed only on reset/step, so straight after `set_state` it returned the post-reset frame and reported cube at MAE 9.04; rendering explicitly gives 0.17, and reacher 0.0001.
- **q dimensionality does not order obj−aux across tasks.** Within cube, however, going from 9-d to 21-d q (adding 5 live arm joints) costs the aux arm −2.2 / −2.7 pp on two independent episode sets while leaving the L_obj arm unchanged (−0.2 / +0.5) — consistent in direction on both sets, but only reaching p=0.058, one training seed each.
- **mppi's budget ladder is not monotone** on Reacher and Cube (T5 sometimes beats T1), so it is not searching effectively there and its contrasts carry little information regardless of arm.
