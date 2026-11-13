**Table E. SCALE vs. constrained-regression baselines: planning success rate (%) by planning-budget tier.** Aux is the paper's auxiliary state-regression baseline (a linear readout of the state from the latent, trained with MSE). The constrained variants add an L2 penalty on the readout weights, λ_aux · (MSE + β · ‖W‖²), with β ∈ {0.01, 0.1, 1.0}. All numbers average two training seeds (1024, 1025). Each entry is the mean over 600 test episodes (6 sets × 100). Budgets (candidates × iterations): T1 300×30, T2 150×15, T3 50×10, T4 20×5, T5 10×3; Avg. is the mean over the five tiers.

**CEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 91.3 | 90.7 | 80.9 | 58.7 | 33.3 | 71.0 |
| Push-T | Aux | 94.6 | 92.6 | 85.3 | 64.0 | 33.8 | 74.1 |
| Push-T | Aux + L2 (β = 0.01) | 95.2 | 92.8 | 86.1 | 67.5 | 34.8 | 75.3 |
| Push-T | Aux + L2 (β = 0.1) | 95.3 | 91.4 | 84.8 | 64.3 | 34.1 | 74.0 |
| Push-T | Aux + L2 (β = 1.0) | 93.0 | 91.6 | 82.2 | 60.3 | 33.1 | 72.0 |
| Push-T | SCALE | 95.3 | 92.3 | 84.5 | 63.8 | 34.6 | 74.1 |
| Reacher | LeWM | 81.2 | 78.8 | 73.5 | 64.8 | 57.0 | 71.1 |
| Reacher | Aux | 83.3 | 80.8 | 73.3 | 68.8 | 57.1 | 72.7 |
| Reacher | Aux + L2 (β = 0.01) | 81.4 | 78.7 | 75.8 | 68.4 | 55.9 | 72.0 |
| Reacher | Aux + L2 (β = 0.1) | 80.3 | 80.3 | 73.6 | 66.7 | 58.1 | 71.8 |
| Reacher | Aux + L2 (β = 1.0) | 82.1 | 78.8 | 73.5 | 64.7 | 56.6 | 71.1 |
| Reacher | SCALE | 79.9 | 80.6 | 73.3 | 66.5 | 56.1 | 71.3 |
| Cube | LeWM | 68.0 | 67.0 | 60.0 | 56.2 | 55.3 | 61.3 |
| Cube | Aux | 71.9 | 68.5 | 65.0 | 56.6 | 54.6 | 63.3 |
| Cube | Aux + L2 (β = 0.01) | 73.7 | 69.9 | 62.8 | 56.8 | 54.8 | 63.6 |
| Cube | Aux + L2 (β = 0.1) | 73.8 | 71.2 | 63.8 | 56.1 | 53.6 | 63.7 |
| Cube | Aux + L2 (β = 1.0) | 67.1 | 65.8 | 62.3 | 55.9 | 55.0 | 61.2 |
| Cube | SCALE | 75.5 | 72.2 | 64.8 | 54.9 | 55.7 | 64.6 |
| Two-Room | LeWM | 86.8 | 85.4 | 81.7 | 71.8 | 58.2 | 76.8 |
| Two-Room | Aux | 88.1 | 86.8 | 83.2 | 71.9 | 59.3 | 77.9 |
| Two-Room | Aux + L2 (β = 0.01) | 90.9 | 90.4 | 84.3 | 74.8 | 59.9 | 80.1 |
| Two-Room | Aux + L2 (β = 0.1) | 94.6 | 94.8 | 87.8 | 77.1 | 61.3 | 83.1 |
| Two-Room | Aux + L2 (β = 1.0) | 97.8 | 96.4 | 91.2 | 77.8 | 65.6 | 85.8 |
| Two-Room | SCALE | 99.3 | 98.9 | 95.8 | 82.8 | 71.8 | 89.8 |
| PointMaze | LeWM | 83.4 | 81.9 | 80.7 | 79.3 | 76.3 | 80.3 |
| PointMaze | Aux | 90.5 | 90.9 | 89.8 | 86.4 | 77.9 | 87.1 |
| PointMaze | Aux + L2 (β = 0.01) | 86.8 | 86.6 | 87.0 | 83.2 | 77.1 | 84.1 |
| PointMaze | Aux + L2 (β = 0.1) | 90.0 | 87.8 | 88.4 | 81.8 | 80.9 | 85.8 |
| PointMaze | Aux + L2 (β = 1.0) | 80.2 | 82.3 | 84.3 | 80.2 | 78.5 | 81.1 |
| PointMaze | SCALE | 79.7 | 80.2 | 79.2 | 77.0 | 73.3 | 77.9 |

**iCEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 86.2 | 82.6 | 74.3 | 56.7 | 26.4 | 65.2 |
| Push-T | Aux | 89.5 | 86.3 | 80.4 | 61.5 | 26.3 | 68.8 |
| Push-T | Aux + L2 (β = 0.01) | 90.6 | 88.4 | 81.3 | 60.6 | 27.1 | 69.6 |
| Push-T | Aux + L2 (β = 0.1) | 89.6 | 86.9 | 81.4 | 59.1 | 26.7 | 68.7 |
| Push-T | Aux + L2 (β = 1.0) | 87.8 | 83.7 | 75.1 | 57.5 | 26.4 | 66.1 |
| Push-T | SCALE | 89.8 | 84.7 | 79.2 | 58.8 | 27.4 | 68.0 |
| Reacher | LeWM | 85.4 | 84.3 | 86.3 | 79.8 | 61.2 | 79.4 |
| Reacher | Aux | 84.8 | 85.8 | 85.9 | 80.2 | 62.0 | 79.7 |
| Reacher | Aux + L2 (β = 0.01) | 84.5 | 86.5 | 85.8 | 81.3 | 61.8 | 80.0 |
| Reacher | Aux + L2 (β = 0.1) | 86.4 | 85.6 | 85.9 | 81.7 | 62.4 | 80.4 |
| Reacher | Aux + L2 (β = 1.0) | 85.4 | 85.3 | 86.3 | 79.1 | 61.7 | 79.6 |
| Reacher | SCALE | 84.8 | 82.6 | 84.2 | 78.8 | 61.7 | 78.4 |
| Cube | LeWM | 68.9 | 67.8 | 69.3 | 61.8 | 56.8 | 64.9 |
| Cube | Aux | 74.2 | 72.9 | 73.5 | 64.6 | 58.1 | 68.7 |
| Cube | Aux + L2 (β = 0.01) | 74.9 | 74.5 | 73.0 | 63.5 | 58.1 | 68.8 |
| Cube | Aux + L2 (β = 0.1) | 76.5 | 76.7 | 74.0 | 63.8 | 57.8 | 69.7 |
| Cube | Aux + L2 (β = 1.0) | 68.8 | 68.4 | 68.3 | 62.6 | 55.4 | 64.7 |
| Cube | SCALE | 76.8 | 76.7 | 75.0 | 65.6 | 58.8 | 70.6 |
| Two-Room | LeWM | 90.3 | 92.8 | 88.2 | 77.8 | 60.8 | 82.0 |
| Two-Room | Aux | 92.5 | 93.5 | 87.6 | 78.5 | 62.1 | 82.8 |
| Two-Room | Aux + L2 (β = 0.01) | 95.1 | 96.0 | 90.2 | 80.2 | 63.5 | 85.0 |
| Two-Room | Aux + L2 (β = 0.1) | 97.3 | 97.9 | 93.7 | 81.6 | 66.2 | 87.3 |
| Two-Room | Aux + L2 (β = 1.0) | 98.8 | 98.8 | 95.3 | 84.1 | 70.7 | 89.6 |
| Two-Room | SCALE | 99.8 | 99.7 | 98.2 | 90.5 | 77.3 | 93.1 |
| PointMaze | LeWM | 74.3 | 72.8 | 73.2 | 73.4 | 73.1 | 73.4 |
| PointMaze | Aux | 89.3 | 87.3 | 88.3 | 83.6 | 77.8 | 85.2 |
| PointMaze | Aux + L2 (β = 0.01) | 87.3 | 86.8 | 84.8 | 80.9 | 72.7 | 82.5 |
| PointMaze | Aux + L2 (β = 0.1) | 86.6 | 83.8 | 84.5 | 81.3 | 75.7 | 82.4 |
| PointMaze | Aux + L2 (β = 1.0) | 81.2 | 79.8 | 77.6 | 78.1 | 77.3 | 78.8 |
| PointMaze | SCALE | 81.3 | 79.8 | 75.4 | 73.5 | 69.9 | 76.0 |

**Table F. SCALE against each regression baseline: success rate (%) averaged over the five budget tiers, with paired significance tests.** All models are evaluated on the same 600 test episodes with identical planner noise, so outcomes are paired per episode. Δ = SCALE − baseline in percentage points (positive = SCALE better); the 95% CI is a paired t-interval over episodes; p is from an exact two-sided paired sign-flip permutation test. All numbers average two training seeds (1024, 1025).

| Task | Planner | Baseline | Baseline success | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | Aux | 74.1 | 74.1 | +0.0 | [−1.1, +1.2] | 1.000 |
| Push-T | CEM | Aux + L2 (β = 0.01) | 75.3 | 74.1 | −1.2 | [−2.3, −0.1] | 0.030 |
| Push-T | CEM | Aux + L2 (β = 0.1) | 74.0 | 74.1 | +0.1 | [−1.0, +1.2] | 0.883 |
| Push-T | CEM | Aux + L2 (β = 1.0) | 72.0 | 74.1 | +2.1 | [+0.9, +3.2] | 0.0004 |
| Push-T | iCEM | Aux | 68.8 | 68.0 | −0.9 | [−2.0, +0.3] | 0.166 |
| Push-T | iCEM | Aux + L2 (β = 0.01) | 69.6 | 68.0 | −1.7 | [−2.8, −0.5] | 0.0052 |
| Push-T | iCEM | Aux + L2 (β = 0.1) | 68.7 | 68.0 | −0.8 | [−1.8, +0.3] | 0.158 |
| Push-T | iCEM | Aux + L2 (β = 1.0) | 66.1 | 68.0 | +1.9 | [+0.7, +3.0] | 0.0027 |
| Reacher | CEM | Aux | 72.7 | 71.3 | −1.4 | [−2.7, −0.0] | 0.046 |
| Reacher | CEM | Aux + L2 (β = 0.01) | 72.0 | 71.3 | −0.8 | [−2.2, +0.6] | 0.291 |
| Reacher | CEM | Aux + L2 (β = 0.1) | 71.8 | 71.3 | −0.5 | [−1.9, +0.9] | 0.483 |
| Reacher | CEM | Aux + L2 (β = 1.0) | 71.1 | 71.3 | +0.2 | [−1.2, +1.5] | 0.850 |
| Reacher | iCEM | Aux | 79.7 | 78.4 | −1.3 | [−2.5, −0.1] | 0.029 |
| Reacher | iCEM | Aux + L2 (β = 0.01) | 80.0 | 78.4 | −1.6 | [−2.7, −0.4] | 0.010 |
| Reacher | iCEM | Aux + L2 (β = 0.1) | 80.4 | 78.4 | −2.0 | [−3.2, −0.8] | 0.0012 |
| Reacher | iCEM | Aux + L2 (β = 1.0) | 79.6 | 78.4 | −1.2 | [−2.2, −0.0] | 0.052 |
| Cube | CEM | Aux | 63.3 | 64.6 | +1.3 | [+0.3, +2.3] | 0.014 |
| Cube | CEM | Aux + L2 (β = 0.01) | 63.6 | 64.6 | +1.0 | [+0.1, +2.0] | 0.039 |
| Cube | CEM | Aux + L2 (β = 0.1) | 63.7 | 64.6 | +0.9 | [+0.0, +1.8] | 0.046 |
| Cube | CEM | Aux + L2 (β = 1.0) | 61.2 | 64.6 | +3.4 | [+2.3, +4.5] | <0.0001 |
| Cube | iCEM | Aux | 68.7 | 70.6 | +1.9 | [+1.1, +2.8] | <0.0001 |
| Cube | iCEM | Aux + L2 (β = 0.01) | 68.8 | 70.6 | +1.8 | [+1.0, +2.6] | <0.0001 |
| Cube | iCEM | Aux + L2 (β = 0.1) | 69.7 | 70.6 | +0.8 | [+0.3, +1.4] | 0.0055 |
| Cube | iCEM | Aux + L2 (β = 1.0) | 64.7 | 70.6 | +5.9 | [+4.6, +7.2] | <0.0001 |
| Two-Room | CEM | Aux | 77.9 | 89.8 | +11.9 | [+10.0, +13.8] | <0.0001 |
| Two-Room | CEM | Aux + L2 (β = 0.01) | 80.1 | 89.8 | +9.7 | [+8.0, +11.4] | <0.0001 |
| Two-Room | CEM | Aux + L2 (β = 0.1) | 83.1 | 89.8 | +6.7 | [+5.3, +8.0] | <0.0001 |
| Two-Room | CEM | Aux + L2 (β = 1.0) | 85.8 | 89.8 | +4.0 | [+3.0, +5.0] | <0.0001 |
| Two-Room | iCEM | Aux | 82.8 | 93.1 | +10.3 | [+8.5, +12.0] | <0.0001 |
| Two-Room | iCEM | Aux + L2 (β = 0.01) | 85.0 | 93.1 | +8.1 | [+6.7, +9.6] | <0.0001 |
| Two-Room | iCEM | Aux + L2 (β = 0.1) | 87.3 | 93.1 | +5.8 | [+4.6, +6.9] | <0.0001 |
| Two-Room | iCEM | Aux + L2 (β = 1.0) | 89.6 | 93.1 | +3.5 | [+2.6, +4.5] | <0.0001 |
| PointMaze | CEM | Aux | 87.1 | 77.9 | −9.3 | [−10.7, −7.8] | <0.0001 |
| PointMaze | CEM | Aux + L2 (β = 0.01) | 84.1 | 77.9 | −6.3 | [−7.8, −4.7] | <0.0001 |
| PointMaze | CEM | Aux + L2 (β = 0.1) | 85.8 | 77.9 | −7.9 | [−9.5, −6.3] | <0.0001 |
| PointMaze | CEM | Aux + L2 (β = 1.0) | 81.1 | 77.9 | −3.2 | [−4.9, −1.5] | 0.0002 |
| PointMaze | iCEM | Aux | 85.2 | 76.0 | −9.3 | [−10.8, −7.7] | <0.0001 |
| PointMaze | iCEM | Aux + L2 (β = 0.01) | 82.5 | 76.0 | −6.5 | [−8.2, −4.8] | <0.0001 |
| PointMaze | iCEM | Aux + L2 (β = 0.1) | 82.4 | 76.0 | −6.4 | [−8.1, −4.7] | <0.0001 |
| PointMaze | iCEM | Aux + L2 (β = 1.0) | 78.8 | 76.0 | −2.8 | [−4.5, −1.1] | 0.0017 |

**Table F (short). SCALE and each regression baseline: success rate (%) averaged over the five budget tiers.** All numbers average two training seeds (1024, 1025).

| Task | Planner | Baseline | Baseline success | SCALE |
|---|---|---|---|---|
| Push-T | CEM | Aux | 74.1 | 74.1 |
| Push-T | CEM | Aux + L2 (β = 0.01) | 75.3 | 74.1 |
| Push-T | CEM | Aux + L2 (β = 0.1) | 74.0 | 74.1 |
| Push-T | CEM | Aux + L2 (β = 1.0) | 72.0 | 74.1 |
| Push-T | iCEM | Aux | 68.8 | 68.0 |
| Push-T | iCEM | Aux + L2 (β = 0.01) | 69.6 | 68.0 |
| Push-T | iCEM | Aux + L2 (β = 0.1) | 68.7 | 68.0 |
| Push-T | iCEM | Aux + L2 (β = 1.0) | 66.1 | 68.0 |
| Reacher | CEM | Aux | 72.7 | 71.3 |
| Reacher | CEM | Aux + L2 (β = 0.01) | 72.0 | 71.3 |
| Reacher | CEM | Aux + L2 (β = 0.1) | 71.8 | 71.3 |
| Reacher | CEM | Aux + L2 (β = 1.0) | 71.1 | 71.3 |
| Reacher | iCEM | Aux | 79.7 | 78.4 |
| Reacher | iCEM | Aux + L2 (β = 0.01) | 80.0 | 78.4 |
| Reacher | iCEM | Aux + L2 (β = 0.1) | 80.4 | 78.4 |
| Reacher | iCEM | Aux + L2 (β = 1.0) | 79.6 | 78.4 |
| Cube | CEM | Aux | 63.3 | 64.6 |
| Cube | CEM | Aux + L2 (β = 0.01) | 63.6 | 64.6 |
| Cube | CEM | Aux + L2 (β = 0.1) | 63.7 | 64.6 |
| Cube | CEM | Aux + L2 (β = 1.0) | 61.2 | 64.6 |
| Cube | iCEM | Aux | 68.7 | 70.6 |
| Cube | iCEM | Aux + L2 (β = 0.01) | 68.8 | 70.6 |
| Cube | iCEM | Aux + L2 (β = 0.1) | 69.7 | 70.6 |
| Cube | iCEM | Aux + L2 (β = 1.0) | 64.7 | 70.6 |
| Two-Room | CEM | Aux | 77.9 | 89.8 |
| Two-Room | CEM | Aux + L2 (β = 0.01) | 80.1 | 89.8 |
| Two-Room | CEM | Aux + L2 (β = 0.1) | 83.1 | 89.8 |
| Two-Room | CEM | Aux + L2 (β = 1.0) | 85.8 | 89.8 |
| Two-Room | iCEM | Aux | 82.8 | 93.1 |
| Two-Room | iCEM | Aux + L2 (β = 0.01) | 85.0 | 93.1 |
| Two-Room | iCEM | Aux + L2 (β = 0.1) | 87.3 | 93.1 |
| Two-Room | iCEM | Aux + L2 (β = 1.0) | 89.6 | 93.1 |
| PointMaze | CEM | Aux | 87.1 | 77.9 |
| PointMaze | CEM | Aux + L2 (β = 0.01) | 84.1 | 77.9 |
| PointMaze | CEM | Aux + L2 (β = 0.1) | 85.8 | 77.9 |
| PointMaze | CEM | Aux + L2 (β = 1.0) | 81.1 | 77.9 |
| PointMaze | iCEM | Aux | 85.2 | 76.0 |
| PointMaze | iCEM | Aux + L2 (β = 0.01) | 82.5 | 76.0 |
| PointMaze | iCEM | Aux + L2 (β = 0.1) | 82.4 | 76.0 |
| PointMaze | iCEM | Aux + L2 (β = 1.0) | 78.8 | 76.0 |

**Table G. Distance alignment of the learned representation.** Spearman correlation between latent distance and ground-truth state distance on held-out data (1,500 samples); higher means latent distances track state distances more closely. Mean of two training seeds (1024, 1025). At β = 1.0 on Push-T, Reacher and Cube the penalty drives the readout weights to zero, so the regression term no longer shapes the representation.

| Task | LeWM | Aux | Aux + L2 (β = 0.01) | Aux + L2 (β = 0.1) | Aux + L2 (β = 1.0) | SCALE |
|---|---|---|---|---|---|---|
| Push-T | 0.14 | 0.19 | 0.55 | 0.82 | 0.14 | 0.77 |
| Reacher | 0.18 | 0.18 | 0.42 | 0.74 | 0.18 | 0.52 |
| Cube | 0.02 | 0.04 | 0.29 | 0.67 | 0.02 | 0.56 |
| Two-Room | 0.39 | 0.41 | 0.43 | 0.56 | 0.72 | 0.77 |
| PointMaze | 0.54 | 0.53 | 0.55 | 0.58 | 0.73 | 0.80 |
