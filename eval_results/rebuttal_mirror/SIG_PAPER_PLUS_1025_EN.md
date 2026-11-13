**Table L. Planning success rate (%) by planning-budget tier.** LeWM, SCALE and Aux average three independently trained runs per task: the two runs pooled in the paper and one new run (seed 1025). DINO-WM averages the two runs from the paper. Each entry is the mean over 600 test episodes (6 sets × 100) per run. Avg. is the mean over the five tiers.

**CEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 91.6 | 89.8 | 80.3 | 58.0 | 34.4 | 70.8 |
| Push-T | SCALE | 95.0 | 92.2 | 83.9 | 62.9 | 33.0 | 73.4 |
| Push-T | Aux | 94.3 | 92.8 | 84.9 | 64.7 | 34.6 | 74.2 |
| Push-T | DINO-WM | 74.8 | 72.1 | 65.3 | 46.1 | 24.3 | 56.5 |
| Reacher | LeWM | 82.5 | 79.2 | 73.4 | 65.6 | 56.1 | 71.3 |
| Reacher | SCALE | 80.9 | 79.8 | 74.1 | 66.1 | 57.2 | 71.6 |
| Reacher | Aux | 82.7 | 79.7 | 74.8 | 66.3 | 56.6 | 72.0 |
| Reacher | DINO-WM | 77.2 | 74.8 | 70.8 | 61.8 | 50.3 | 67.0 |
| Cube | LeWM | 68.2 | 66.2 | 60.6 | 56.4 | 54.1 | 61.1 |
| Cube | SCALE | 74.7 | 71.3 | 64.4 | 55.6 | 55.0 | 64.2 |
| Cube | Aux | 71.7 | 69.6 | 63.1 | 56.7 | 55.0 | 63.2 |
| Cube | DINO-WM | 73.4 | 66.9 | 58.4 | 51.6 | 53.9 | 60.9 |
| Two-Room | LeWM | 87.7 | 87.2 | 82.6 | 71.9 | 58.0 | 77.5 |
| Two-Room | SCALE | 99.6 | 99.3 | 95.0 | 83.2 | 70.5 | 89.5 |
| Two-Room | Aux | 88.4 | 87.5 | 83.2 | 71.3 | 59.1 | 77.9 |
| Two-Room | DINO-WM | 100.0 | 99.8 | 98.3 | 94.1 | 85.8 | 95.6 |
| PointMaze | LeWM | 85.0 | 85.3 | 85.3 | 83.2 | 74.8 | 82.7 |
| PointMaze | SCALE | 86.9 | 87.4 | 85.9 | 83.9 | 78.1 | 84.4 |
| PointMaze | Aux | 87.3 | 86.7 | 85.7 | 82.2 | 78.8 | 84.1 |
| PointMaze | DINO-WM | 98.2 | 98.3 | 97.2 | 92.4 | 87.2 | 94.7 |

**iCEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 85.8 | 80.4 | 73.7 | 55.2 | 25.6 | 64.1 |
| Push-T | SCALE | 89.3 | 86.0 | 79.7 | 60.2 | 25.7 | 68.2 |
| Push-T | Aux | 89.7 | 86.2 | 79.5 | 61.6 | 26.9 | 68.8 |
| Push-T | DINO-WM | 70.0 | 65.3 | 58.8 | 41.5 | 23.1 | 51.7 |
| Reacher | LeWM | 85.3 | 84.9 | 84.8 | 79.3 | 61.9 | 79.2 |
| Reacher | SCALE | 86.2 | 85.5 | 86.6 | 80.4 | 63.2 | 80.4 |
| Reacher | Aux | 84.9 | 86.4 | 85.7 | 80.2 | 63.0 | 80.0 |
| Reacher | DINO-WM | 78.7 | 78.2 | 80.8 | 76.0 | 53.0 | 73.3 |
| Cube | LeWM | 69.3 | 67.6 | 69.1 | 62.1 | 55.7 | 64.8 |
| Cube | SCALE | 76.8 | 76.6 | 75.6 | 65.9 | 59.5 | 70.9 |
| Cube | Aux | 74.1 | 72.4 | 72.7 | 64.9 | 57.6 | 68.3 |
| Cube | DINO-WM | 69.9 | 67.4 | 66.0 | 57.7 | 51.4 | 62.5 |
| Two-Room | LeWM | 90.9 | 93.0 | 88.7 | 77.8 | 62.0 | 82.5 |
| Two-Room | SCALE | 99.9 | 99.7 | 97.8 | 90.0 | 76.6 | 92.8 |
| Two-Room | Aux | 92.5 | 93.4 | 88.5 | 78.7 | 62.9 | 83.2 |
| Two-Room | DINO-WM | 100.0 | 100.0 | 99.8 | 98.6 | 92.5 | 98.2 |
| PointMaze | LeWM | 83.2 | 82.2 | 80.6 | 78.8 | 73.0 | 79.6 |
| PointMaze | SCALE | 87.4 | 85.8 | 83.9 | 80.7 | 76.1 | 82.8 |
| PointMaze | Aux | 83.5 | 82.6 | 82.9 | 82.0 | 77.3 | 81.6 |
| PointMaze | DINO-WM | 93.4 | 92.2 | 90.8 | 90.3 | 85.3 | 90.4 |

**Table M. SCALE vs. LeWM: success rate (%) averaged over the five budget tiers, with paired significance tests (three runs per method).** Δ = SCALE − LeWM in percentage points. All models are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode. The per-episode difference is averaged over training runs and the five budget tiers; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test. The last column counts the training seeds at which SCALE is ahead.

| Task | Planner | LeWM | SCALE | Δ | 95% CI | p-value | Runs with Δ > 0 |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | 70.8 | 73.4 | +2.6 | [+1.6, +3.6] | <0.0001 | 3/3 |
| Push-T | iCEM | 64.1 | 68.2 | +4.0 | [+3.0, +5.1] | <0.0001 | 3/3 |
| Reacher | CEM | 71.3 | 71.6 | +0.3 | [−0.7, +1.4] | 0.541 | 2/3 |
| Reacher | iCEM | 79.2 | 80.4 | +1.1 | [+0.2, +2.1] | 0.014 | 2/3 |
| Cube | CEM | 61.1 | 64.2 | +3.1 | [+2.1, +4.2] | <0.0001 | 3/3 |
| Cube | iCEM | 64.8 | 70.9 | +6.1 | [+4.9, +7.4] | <0.0001 | 3/3 |
| Two-Room | CEM | 77.5 | 89.5 | +12.0 | [+10.2, +13.9] | <0.0001 | 3/3 |
| Two-Room | iCEM | 82.5 | 92.8 | +10.3 | [+8.6, +12.0] | <0.0001 | 3/3 |
| PointMaze | CEM | 82.7 | 84.4 | +1.7 | [+0.6, +2.8] | 0.0027 | 2/3 |
| PointMaze | iCEM | 79.6 | 82.8 | +3.2 | [+2.0, +4.4] | <0.0001 | 3/3 |

**Table N. SCALE vs. Aux: success rate (%) averaged over the five budget tiers, with paired significance tests (three runs per method).** Δ = SCALE − Aux in percentage points. All models are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode. The per-episode difference is averaged over training runs and the five budget tiers; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test. The last column counts the training seeds at which SCALE is ahead.

| Task | Planner | Aux | SCALE | Δ | 95% CI | p-value | Runs with Δ > 0 |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | 74.2 | 73.4 | −0.8 | [−1.8, +0.1] | 0.093 | 0/3 |
| Push-T | iCEM | 68.8 | 68.2 | −0.6 | [−1.5, +0.3] | 0.220 | 0/3 |
| Reacher | CEM | 72.0 | 71.6 | −0.4 | [−1.4, +0.7] | 0.513 | 1/3 |
| Reacher | iCEM | 80.0 | 80.4 | +0.4 | [−0.5, +1.2] | 0.408 | 2/3 |
| Cube | CEM | 63.2 | 64.2 | +1.0 | [+0.2, +1.8] | 0.020 | 3/3 |
| Cube | iCEM | 68.3 | 70.9 | +2.5 | [+1.8, +3.3] | <0.0001 | 3/3 |
| Two-Room | CEM | 77.9 | 89.5 | +11.6 | [+9.8, +13.4] | <0.0001 | 3/3 |
| Two-Room | iCEM | 83.2 | 92.8 | +9.6 | [+7.9, +11.2] | <0.0001 | 3/3 |
| PointMaze | CEM | 84.1 | 84.4 | +0.3 | [−0.7, +1.3] | 0.588 | 2/3 |
| PointMaze | iCEM | 81.6 | 82.8 | +1.1 | [−0.0, +2.3] | 0.061 | 2/3 |

**Table O. SCALE vs. DINO-WM: success rate (%) averaged over the five budget tiers, with paired significance tests (SCALE: three runs; DINO-WM: the two runs from the paper).** Δ = SCALE − DINO-WM in percentage points. All models are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode. The per-episode difference is averaged over training runs and the five budget tiers; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | DINO-WM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 56.5 | 73.4 | +16.9 | [+14.7, +19.1] | <0.0001 |
| Push-T | iCEM | 51.7 | 68.2 | +16.4 | [+14.2, +18.7] | <0.0001 |
| Reacher | CEM | 67.0 | 71.6 | +4.7 | [+3.0, +6.4] | <0.0001 |
| Reacher | iCEM | 73.3 | 80.4 | +7.1 | [+5.3, +8.8] | <0.0001 |
| Cube | CEM | 60.9 | 64.2 | +3.3 | [+2.2, +4.5] | <0.0001 |
| Cube | iCEM | 62.5 | 70.9 | +8.4 | [+6.2, +10.6] | <0.0001 |
| Two-Room | CEM | 95.6 | 89.5 | −6.1 | [−7.0, −5.2] | <0.0001 |
| Two-Room | iCEM | 98.2 | 92.8 | −5.4 | [−6.3, −4.5] | <0.0001 |
| PointMaze | CEM | 94.7 | 84.4 | −10.2 | [−11.7, −8.7] | <0.0001 |
| PointMaze | iCEM | 90.4 | 82.8 | −7.6 | [−9.3, −5.9] | <0.0001 |
