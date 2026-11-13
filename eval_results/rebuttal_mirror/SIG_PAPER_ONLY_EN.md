**Table L. Planning success rate (%) by planning-budget tier.** Every method averages the two independently trained runs pooled in the paper. Each entry is the mean over 600 test episodes (6 sets × 100) per run. Avg. is the mean over the five tiers.

**CEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 92.3 | 89.3 | 80.5 | 57.8 | 33.8 | 70.7 |
| Push-T | SCALE | 94.8 | 92.1 | 84.3 | 61.6 | 32.8 | 73.1 |
| Push-T | Aux | 94.5 | 92.6 | 84.4 | 65.1 | 34.9 | 74.3 |
| Push-T | DINO-WM | 74.8 | 72.1 | 65.3 | 46.1 | 24.3 | 56.5 |
| Reacher | LeWM | 82.3 | 79.4 | 73.3 | 66.2 | 55.8 | 71.4 |
| Reacher | SCALE | 82.3 | 79.1 | 74.5 | 65.2 | 57.4 | 71.7 |
| Reacher | Aux | 81.6 | 78.8 | 74.9 | 65.8 | 55.9 | 71.4 |
| Reacher | DINO-WM | 77.2 | 74.8 | 70.8 | 61.8 | 50.3 | 67.0 |
| Cube | LeWM | 67.6 | 65.8 | 60.8 | 56.8 | 53.8 | 60.9 |
| Cube | SCALE | 73.9 | 71.1 | 64.7 | 55.9 | 54.8 | 64.1 |
| Cube | Aux | 71.6 | 70.3 | 62.8 | 56.6 | 55.3 | 63.3 |
| Cube | DINO-WM | 73.4 | 66.9 | 58.4 | 51.6 | 53.9 | 60.9 |
| Two-Room | LeWM | 87.8 | 87.1 | 83.0 | 71.8 | 57.8 | 77.5 |
| Two-Room | SCALE | 99.7 | 99.3 | 94.0 | 83.5 | 69.7 | 89.2 |
| Two-Room | Aux | 88.3 | 87.7 | 82.4 | 70.6 | 59.7 | 77.7 |
| Two-Room | DINO-WM | 100.0 | 99.8 | 98.3 | 94.1 | 85.8 | 95.6 |
| PointMaze | LeWM | 85.7 | 86.0 | 86.9 | 83.5 | 75.3 | 83.5 |
| PointMaze | SCALE | 87.6 | 87.4 | 86.8 | 84.8 | 78.1 | 84.9 |
| PointMaze | Aux | 85.0 | 83.9 | 82.7 | 79.8 | 78.1 | 81.9 |
| PointMaze | DINO-WM | 98.2 | 98.3 | 97.2 | 92.4 | 87.2 | 94.7 |

**iCEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 86.0 | 79.8 | 73.4 | 55.3 | 25.1 | 63.9 |
| Push-T | SCALE | 89.1 | 86.3 | 79.7 | 61.3 | 25.2 | 68.3 |
| Push-T | Aux | 89.9 | 86.2 | 79.6 | 62.2 | 27.6 | 69.1 |
| Push-T | DINO-WM | 70.0 | 65.3 | 58.8 | 41.5 | 23.1 | 51.7 |
| Reacher | LeWM | 84.9 | 84.3 | 83.8 | 79.0 | 62.3 | 78.9 |
| Reacher | SCALE | 86.7 | 87.0 | 87.3 | 81.7 | 64.4 | 81.4 |
| Reacher | Aux | 85.0 | 86.1 | 85.8 | 80.1 | 62.9 | 80.0 |
| Reacher | DINO-WM | 78.7 | 78.2 | 80.8 | 76.0 | 53.0 | 73.3 |
| Cube | LeWM | 68.9 | 67.1 | 68.7 | 61.9 | 54.9 | 64.3 |
| Cube | SCALE | 76.8 | 76.6 | 75.3 | 65.8 | 59.8 | 70.8 |
| Cube | Aux | 73.8 | 73.0 | 72.5 | 64.8 | 57.8 | 68.4 |
| Cube | DINO-WM | 69.9 | 67.4 | 66.0 | 57.7 | 51.4 | 62.5 |
| Two-Room | LeWM | 91.1 | 92.9 | 88.5 | 77.3 | 62.4 | 82.5 |
| Two-Room | SCALE | 99.9 | 99.8 | 97.7 | 90.2 | 76.2 | 92.7 |
| Two-Room | Aux | 92.3 | 93.3 | 89.3 | 78.7 | 62.9 | 83.3 |
| Two-Room | DINO-WM | 100.0 | 100.0 | 99.8 | 98.6 | 92.5 | 98.2 |
| PointMaze | LeWM | 84.0 | 83.6 | 82.4 | 81.2 | 73.3 | 80.9 |
| PointMaze | SCALE | 87.6 | 85.8 | 84.6 | 81.9 | 77.0 | 83.4 |
| PointMaze | Aux | 80.8 | 79.5 | 78.7 | 79.4 | 75.0 | 78.7 |
| PointMaze | DINO-WM | 93.4 | 92.2 | 90.8 | 90.3 | 85.3 | 90.4 |

**Table M. SCALE vs. LeWM: success rate (%) averaged over the five budget tiers, with paired significance tests (two runs per method).** Δ = SCALE − LeWM in percentage points. All models are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode. The per-episode difference is averaged over training runs and the five budget tiers; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test. The last column counts the training seeds at which SCALE is ahead.

| Task | Planner | LeWM | SCALE | Δ | 95% CI | p-value | Runs with Δ > 0 |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | 70.7 | 73.1 | +2.4 | [+1.2, +3.6] | 0.0001 | 2/2 |
| Push-T | iCEM | 63.9 | 68.3 | +4.4 | [+3.2, +5.6] | <0.0001 | 2/2 |
| Reacher | CEM | 71.4 | 71.7 | +0.3 | [−1.0, +1.6] | 0.664 | 1/2 |
| Reacher | iCEM | 78.9 | 81.4 | +2.6 | [+1.5, +3.7] | <0.0001 | 2/2 |
| Cube | CEM | 60.9 | 64.1 | +3.1 | [+1.9, +4.3] | <0.0001 | 2/2 |
| Cube | iCEM | 64.3 | 70.8 | +6.5 | [+5.3, +7.8] | <0.0001 | 2/2 |
| Two-Room | CEM | 77.5 | 89.2 | +11.8 | [+9.9, +13.7] | <0.0001 | 2/2 |
| Two-Room | iCEM | 82.5 | 92.7 | +10.3 | [+8.6, +12.0] | <0.0001 | 2/2 |
| PointMaze | CEM | 83.5 | 84.9 | +1.5 | [+0.1, +2.8] | 0.038 | 1/2 |
| PointMaze | iCEM | 80.9 | 83.4 | +2.5 | [+1.1, +3.9] | 0.0006 | 2/2 |

**Table N. SCALE vs. Aux: success rate (%) averaged over the five budget tiers, with paired significance tests (two runs per method).** Δ = SCALE − Aux in percentage points. All models are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode. The per-episode difference is averaged over training runs and the five budget tiers; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test. The last column counts the training seeds at which SCALE is ahead.

| Task | Planner | Aux | SCALE | Δ | 95% CI | p-value | Runs with Δ > 0 |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | 74.3 | 73.1 | −1.2 | [−2.3, −0.1] | 0.036 | 0/2 |
| Push-T | iCEM | 69.1 | 68.3 | −0.8 | [−1.8, +0.3] | 0.169 | 0/2 |
| Reacher | CEM | 71.4 | 71.7 | +0.3 | [−0.9, +1.5] | 0.652 | 1/2 |
| Reacher | iCEM | 80.0 | 81.4 | +1.5 | [+0.4, +2.5] | 0.0071 | 2/2 |
| Cube | CEM | 63.3 | 64.1 | +0.8 | [−0.2, +1.7] | 0.118 | 2/2 |
| Cube | iCEM | 68.4 | 70.8 | +2.4 | [+1.6, +3.3] | <0.0001 | 2/2 |
| Two-Room | CEM | 77.7 | 89.2 | +11.5 | [+9.6, +13.5] | <0.0001 | 2/2 |
| Two-Room | iCEM | 83.3 | 92.7 | +9.5 | [+7.8, +11.2] | <0.0001 | 2/2 |
| PointMaze | CEM | 81.9 | 84.9 | +3.1 | [+1.5, +4.6] | 0.0002 | 2/2 |
| PointMaze | iCEM | 78.7 | 83.4 | +4.7 | [+3.0, +6.4] | <0.0001 | 2/2 |

**Table O. SCALE vs. DINO-WM: success rate (%) averaged over the five budget tiers, with paired significance tests (two runs per method).** Δ = SCALE − DINO-WM in percentage points. All models are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode. The per-episode difference is averaged over training runs and the five budget tiers; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | DINO-WM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 56.5 | 73.1 | +16.6 | [+14.3, +18.9] | <0.0001 |
| Push-T | iCEM | 51.7 | 68.3 | +16.6 | [+14.3, +18.9] | <0.0001 |
| Reacher | CEM | 67.0 | 71.7 | +4.7 | [+3.0, +6.5] | <0.0001 |
| Reacher | iCEM | 73.3 | 81.4 | +8.1 | [+6.3, +9.9] | <0.0001 |
| Cube | CEM | 60.9 | 64.1 | +3.2 | [+2.0, +4.4] | <0.0001 |
| Cube | iCEM | 62.5 | 70.8 | +8.3 | [+6.2, +10.5] | <0.0001 |
| Two-Room | CEM | 95.6 | 89.2 | −6.4 | [−7.3, −5.4] | <0.0001 |
| Two-Room | iCEM | 98.2 | 92.7 | −5.4 | [−6.4, −4.5] | <0.0001 |
| PointMaze | CEM | 94.7 | 84.9 | −9.7 | [−11.4, −8.1] | <0.0001 |
| PointMaze | iCEM | 90.4 | 83.4 | −7.0 | [−8.8, −5.2] | <0.0001 |
