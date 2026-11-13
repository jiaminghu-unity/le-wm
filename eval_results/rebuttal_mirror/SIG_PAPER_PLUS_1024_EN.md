**Table L. Planning success rate (%) by planning-budget tier.** LeWM, SCALE and Aux average three independently trained runs per task: the two runs pooled in the paper and one new run (seed 1024). DINO-WM averages the two runs from the paper. Each entry is the mean over 600 test episodes (6 sets × 100) per run. Avg. is the mean over the five tiers.

**CEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 92.4 | 89.7 | 80.9 | 58.1 | 32.8 | 70.8 |
| Push-T | SCALE | 95.0 | 92.1 | 84.8 | 61.7 | 33.8 | 73.5 |
| Push-T | Aux | 94.8 | 92.4 | 84.6 | 64.7 | 34.6 | 74.2 |
| Push-T | DINO-WM | 74.8 | 72.1 | 65.3 | 46.1 | 24.3 | 56.5 |
| Reacher | LeWM | 81.4 | 79.3 | 73.4 | 65.8 | 56.4 | 71.3 |
| Reacher | SCALE | 82.0 | 79.3 | 74.1 | 65.1 | 56.7 | 71.5 |
| Reacher | Aux | 81.6 | 79.2 | 73.9 | 67.3 | 56.1 | 71.6 |
| Reacher | DINO-WM | 77.2 | 74.8 | 70.8 | 61.8 | 50.3 | 67.0 |
| Cube | LeWM | 67.3 | 66.2 | 60.4 | 56.8 | 54.4 | 61.0 |
| Cube | SCALE | 74.2 | 71.6 | 65.0 | 55.6 | 55.1 | 64.3 |
| Cube | Aux | 71.7 | 69.8 | 64.1 | 56.4 | 55.1 | 63.4 |
| Cube | DINO-WM | 73.4 | 66.9 | 58.4 | 51.6 | 53.9 | 60.9 |
| Two-Room | LeWM | 87.1 | 85.9 | 82.6 | 71.6 | 57.8 | 77.0 |
| Two-Room | SCALE | 99.6 | 99.1 | 94.2 | 83.3 | 70.3 | 89.3 |
| Two-Room | Aux | 88.1 | 87.2 | 82.2 | 70.7 | 60.0 | 77.6 |
| Two-Room | DINO-WM | 100.0 | 99.8 | 98.3 | 94.1 | 85.8 | 95.6 |
| PointMaze | LeWM | 84.8 | 84.0 | 84.3 | 81.1 | 76.5 | 82.1 |
| PointMaze | SCALE | 83.0 | 82.6 | 82.6 | 80.6 | 74.9 | 80.7 |
| PointMaze | Aux | 86.3 | 85.8 | 84.4 | 81.8 | 77.2 | 83.1 |
| PointMaze | DINO-WM | 98.2 | 98.3 | 97.2 | 92.4 | 87.2 | 94.7 |

**iCEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 86.3 | 81.1 | 73.8 | 56.2 | 25.4 | 64.6 |
| Push-T | SCALE | 89.3 | 85.6 | 79.3 | 60.7 | 26.2 | 68.2 |
| Push-T | Aux | 89.9 | 86.2 | 80.2 | 62.3 | 27.4 | 69.2 |
| Push-T | DINO-WM | 70.0 | 65.3 | 58.8 | 41.5 | 23.1 | 51.7 |
| Reacher | LeWM | 84.8 | 83.7 | 84.5 | 79.2 | 61.9 | 78.8 |
| Reacher | SCALE | 85.9 | 85.6 | 86.0 | 80.9 | 63.8 | 80.4 |
| Reacher | Aux | 84.9 | 85.6 | 85.9 | 80.1 | 62.2 | 79.7 |
| Reacher | DINO-WM | 78.7 | 78.2 | 80.8 | 76.0 | 53.0 | 73.3 |
| Cube | LeWM | 68.5 | 67.1 | 68.6 | 61.6 | 55.4 | 64.2 |
| Cube | SCALE | 76.7 | 76.6 | 74.8 | 65.5 | 59.3 | 70.6 |
| Cube | Aux | 73.8 | 73.6 | 72.9 | 64.6 | 58.1 | 68.6 |
| Cube | DINO-WM | 69.9 | 67.4 | 66.0 | 57.7 | 51.4 | 62.5 |
| Two-Room | LeWM | 90.8 | 92.7 | 88.1 | 77.1 | 61.8 | 82.1 |
| Two-Room | SCALE | 99.9 | 99.8 | 97.9 | 90.6 | 76.5 | 92.9 |
| Two-Room | Aux | 92.3 | 93.2 | 88.9 | 78.5 | 62.3 | 83.0 |
| Two-Room | DINO-WM | 100.0 | 100.0 | 99.8 | 98.6 | 92.5 | 98.2 |
| PointMaze | LeWM | 78.3 | 77.8 | 78.1 | 78.4 | 73.4 | 77.2 |
| PointMaze | SCALE | 83.6 | 81.7 | 79.2 | 77.5 | 73.2 | 79.0 |
| PointMaze | Aux | 83.7 | 81.6 | 80.8 | 79.6 | 74.6 | 80.1 |
| PointMaze | DINO-WM | 93.4 | 92.2 | 90.8 | 90.3 | 85.3 | 90.4 |

**Table M. SCALE vs. LeWM: success rate (%) averaged over the five budget tiers, with paired significance tests (three runs per method).** Δ = SCALE − LeWM in percentage points. All models are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode. The per-episode difference is averaged over training runs and the five budget tiers; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test. The last column counts the training seeds at which SCALE is ahead.

| Task | Planner | LeWM | SCALE | Δ | 95% CI | p-value | Runs with Δ > 0 |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | 70.8 | 73.5 | +2.7 | [+1.6, +3.7] | <0.0001 | 3/3 |
| Push-T | iCEM | 64.6 | 68.2 | +3.6 | [+2.6, +4.7] | <0.0001 | 3/3 |
| Reacher | CEM | 71.3 | 71.5 | +0.2 | [−0.9, +1.3] | 0.714 | 2/3 |
| Reacher | iCEM | 78.8 | 80.4 | +1.6 | [+0.7, +2.5] | 0.0006 | 2/3 |
| Cube | CEM | 61.0 | 64.3 | +3.3 | [+2.2, +4.4] | <0.0001 | 3/3 |
| Cube | iCEM | 64.2 | 70.6 | +6.4 | [+5.1, +7.6] | <0.0001 | 3/3 |
| Two-Room | CEM | 77.0 | 89.3 | +12.3 | [+10.4, +14.2] | <0.0001 | 3/3 |
| Two-Room | iCEM | 82.1 | 92.9 | +10.8 | [+9.1, +12.6] | <0.0001 | 3/3 |
| PointMaze | CEM | 82.1 | 80.7 | −1.4 | [−2.7, −0.2] | 0.028 | 1/3 |
| PointMaze | iCEM | 77.2 | 79.0 | +1.8 | [+0.5, +3.2] | 0.0067 | 3/3 |

**Table N. SCALE vs. Aux: success rate (%) averaged over the five budget tiers, with paired significance tests (three runs per method).** Δ = SCALE − Aux in percentage points. All models are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode. The per-episode difference is averaged over training runs and the five budget tiers; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test. The last column counts the training seeds at which SCALE is ahead.

| Task | Planner | Aux | SCALE | Δ | 95% CI | p-value | Runs with Δ > 0 |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | 74.2 | 73.5 | −0.7 | [−1.7, +0.2] | 0.136 | 1/3 |
| Push-T | iCEM | 69.2 | 68.2 | −1.0 | [−2.0, −0.0] | 0.047 | 0/3 |
| Reacher | CEM | 71.6 | 71.5 | −0.2 | [−1.2, +0.9] | 0.768 | 1/3 |
| Reacher | iCEM | 79.7 | 80.4 | +0.7 | [−0.2, +1.6] | 0.131 | 2/3 |
| Cube | CEM | 63.4 | 64.3 | +0.9 | [+0.1, +1.7] | 0.040 | 3/3 |
| Cube | iCEM | 68.6 | 70.6 | +2.0 | [+1.2, +2.8] | <0.0001 | 3/3 |
| Two-Room | CEM | 77.6 | 89.3 | +11.7 | [+9.8, +13.6] | <0.0001 | 3/3 |
| Two-Room | iCEM | 83.0 | 92.9 | +9.9 | [+8.2, +11.6] | <0.0001 | 3/3 |
| PointMaze | CEM | 83.1 | 80.7 | −2.4 | [−3.7, −1.0] | 0.0006 | 2/3 |
| PointMaze | iCEM | 80.1 | 79.0 | −1.0 | [−2.5, +0.4] | 0.162 | 2/3 |

**Table O. SCALE vs. DINO-WM: success rate (%) averaged over the five budget tiers, with paired significance tests (SCALE: three runs; DINO-WM: the two runs from the paper).** Δ = SCALE − DINO-WM in percentage points. All models are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode. The per-episode difference is averaged over training runs and the five budget tiers; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | DINO-WM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 56.5 | 73.5 | +17.0 | [+14.7, +19.2] | <0.0001 |
| Push-T | iCEM | 51.7 | 68.2 | +16.5 | [+14.2, +18.7] | <0.0001 |
| Reacher | CEM | 67.0 | 71.5 | +4.5 | [+2.8, +6.2] | <0.0001 |
| Reacher | iCEM | 73.3 | 80.4 | +7.1 | [+5.4, +8.9] | <0.0001 |
| Cube | CEM | 60.9 | 64.3 | +3.5 | [+2.3, +4.6] | <0.0001 |
| Cube | iCEM | 62.5 | 70.6 | +8.1 | [+6.0, +10.3] | <0.0001 |
| Two-Room | CEM | 95.6 | 89.3 | −6.3 | [−7.2, −5.4] | <0.0001 |
| Two-Room | iCEM | 98.2 | 92.9 | −5.2 | [−6.1, −4.4] | <0.0001 |
| PointMaze | CEM | 94.7 | 80.7 | −13.9 | [−15.7, −12.1] | <0.0001 |
| PointMaze | iCEM | 90.4 | 79.0 | −11.4 | [−13.2, −9.5] | <0.0001 |
