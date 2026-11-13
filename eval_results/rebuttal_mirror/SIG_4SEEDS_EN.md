**Table L. Planning success rate (%) by planning-budget tier.** LeWM, SCALE and Aux average four independently trained runs per task: the two runs pooled in the paper and two new runs (seeds 1024 and 1025). DINO-WM averages the two runs from the paper. Each entry is the mean over 600 test episodes (6 sets × 100) per run. Avg. is the mean over the five tiers.

**CEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 91.8 | 90.0 | 80.7 | 58.2 | 33.5 | 70.9 |
| Push-T | SCALE | 95.1 | 92.2 | 84.4 | 62.7 | 33.7 | 73.6 |
| Push-T | Aux | 94.5 | 92.6 | 84.9 | 64.5 | 34.4 | 74.2 |
| Push-T | DINO-WM | 74.8 | 72.1 | 65.3 | 46.1 | 24.3 | 56.5 |
| Reacher | LeWM | 81.8 | 79.1 | 73.4 | 65.5 | 56.4 | 71.2 |
| Reacher | SCALE | 81.1 | 79.8 | 73.9 | 65.8 | 56.8 | 71.5 |
| Reacher | Aux | 82.4 | 79.8 | 74.1 | 67.3 | 56.5 | 72.0 |
| Reacher | DINO-WM | 77.2 | 74.8 | 70.8 | 61.8 | 50.3 | 67.0 |
| Cube | LeWM | 67.8 | 66.4 | 60.4 | 56.5 | 54.5 | 61.1 |
| Cube | SCALE | 74.7 | 71.6 | 64.7 | 55.4 | 55.2 | 64.3 |
| Cube | Aux | 71.8 | 69.4 | 63.9 | 56.6 | 54.9 | 63.3 |
| Cube | DINO-WM | 73.4 | 66.9 | 58.4 | 51.6 | 53.9 | 60.9 |
| Two-Room | LeWM | 87.3 | 86.3 | 82.3 | 71.8 | 58.0 | 77.1 |
| Two-Room | SCALE | 99.5 | 99.1 | 94.9 | 83.2 | 70.8 | 89.5 |
| Two-Room | Aux | 88.2 | 87.2 | 82.8 | 71.3 | 59.5 | 77.8 |
| Two-Room | DINO-WM | 100.0 | 99.8 | 98.3 | 94.1 | 85.8 | 95.6 |
| PointMaze | LeWM | 84.5 | 84.0 | 83.8 | 81.4 | 75.8 | 81.9 |
| PointMaze | SCALE | 83.6 | 83.8 | 83.0 | 80.9 | 75.7 | 81.4 |
| PointMaze | Aux | 87.8 | 87.4 | 86.3 | 83.1 | 78.0 | 84.5 |
| PointMaze | DINO-WM | 98.2 | 98.3 | 97.2 | 92.4 | 87.2 | 94.7 |

**iCEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 86.1 | 81.2 | 73.9 | 56.0 | 25.8 | 64.6 |
| Push-T | SCALE | 89.4 | 85.5 | 79.4 | 60.0 | 26.3 | 68.1 |
| Push-T | Aux | 89.7 | 86.2 | 80.0 | 61.8 | 27.0 | 68.9 |
| Push-T | DINO-WM | 70.0 | 65.3 | 58.8 | 41.5 | 23.1 | 51.7 |
| Reacher | LeWM | 85.2 | 84.3 | 85.0 | 79.4 | 61.7 | 79.1 |
| Reacher | SCALE | 85.8 | 84.8 | 85.8 | 80.2 | 63.0 | 79.9 |
| Reacher | Aux | 84.9 | 85.9 | 85.8 | 80.1 | 62.5 | 79.8 |
| Reacher | DINO-WM | 78.7 | 78.2 | 80.8 | 76.0 | 53.0 | 73.3 |
| Cube | LeWM | 68.9 | 67.4 | 69.0 | 61.8 | 55.8 | 64.6 |
| Cube | SCALE | 76.8 | 76.6 | 75.1 | 65.7 | 59.3 | 70.7 |
| Cube | Aux | 74.0 | 73.0 | 73.0 | 64.7 | 57.9 | 68.5 |
| Cube | DINO-WM | 69.9 | 67.4 | 66.0 | 57.7 | 51.4 | 62.5 |
| Two-Room | LeWM | 90.7 | 92.8 | 88.3 | 77.5 | 61.6 | 82.2 |
| Two-Room | SCALE | 99.9 | 99.7 | 97.9 | 90.3 | 76.7 | 92.9 |
| Two-Room | Aux | 92.4 | 93.4 | 88.4 | 78.6 | 62.5 | 83.1 |
| Two-Room | DINO-WM | 100.0 | 100.0 | 99.8 | 98.6 | 92.5 | 98.2 |
| PointMaze | LeWM | 79.2 | 78.2 | 77.8 | 77.3 | 73.2 | 77.1 |
| PointMaze | SCALE | 84.4 | 82.8 | 80.0 | 77.7 | 73.5 | 79.7 |
| PointMaze | Aux | 85.0 | 83.4 | 83.5 | 81.5 | 76.4 | 82.0 |
| PointMaze | DINO-WM | 93.4 | 92.2 | 90.8 | 90.3 | 85.3 | 90.4 |

**Table M. SCALE vs. LeWM: success rate (%) averaged over the five budget tiers, with paired significance tests (four runs per method).** Δ = SCALE − LeWM in percentage points. All models are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode. The per-episode difference is averaged over training runs and the five budget tiers; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test. The last column counts the training seeds at which SCALE is ahead.

| Task | Planner | LeWM | SCALE | Δ | 95% CI | p-value | Runs with Δ > 0 |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | 70.9 | 73.6 | +2.8 | [+1.8, +3.7] | <0.0001 | 4/4 |
| Push-T | iCEM | 64.6 | 68.1 | +3.6 | [+2.5, +4.6] | <0.0001 | 4/4 |
| Reacher | CEM | 71.2 | 71.5 | +0.3 | [−0.7, +1.2] | 0.598 | 3/4 |
| Reacher | iCEM | 79.1 | 79.9 | +0.8 | [−0.0, +1.6] | 0.054 | 2/4 |
| Cube | CEM | 61.1 | 64.3 | +3.2 | [+2.2, +4.3] | <0.0001 | 4/4 |
| Cube | iCEM | 64.6 | 70.7 | +6.1 | [+4.9, +7.3] | <0.0001 | 4/4 |
| Two-Room | CEM | 77.1 | 89.5 | +12.4 | [+10.5, +14.3] | <0.0001 | 4/4 |
| Two-Room | iCEM | 82.2 | 92.9 | +10.7 | [+9.0, +12.4] | <0.0001 | 4/4 |
| PointMaze | CEM | 81.9 | 81.4 | −0.5 | [−1.5, +0.5] | 0.340 | 2/4 |
| PointMaze | iCEM | 77.1 | 79.7 | +2.6 | [+1.4, +3.7] | <0.0001 | 4/4 |

**Table N. SCALE vs. Aux: success rate (%) averaged over the five budget tiers, with paired significance tests (four runs per method).** Δ = SCALE − Aux in percentage points. All models are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode. The per-episode difference is averaged over training runs and the five budget tiers; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test. The last column counts the training seeds at which SCALE is ahead.

| Task | Planner | Aux | SCALE | Δ | 95% CI | p-value | Runs with Δ > 0 |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | 74.2 | 73.6 | −0.6 | [−1.5, +0.3] | 0.199 | 1/4 |
| Push-T | iCEM | 68.9 | 68.1 | −0.8 | [−1.7, +0.1] | 0.080 | 0/4 |
| Reacher | CEM | 72.0 | 71.5 | −0.5 | [−1.5, +0.4] | 0.252 | 1/4 |
| Reacher | iCEM | 79.8 | 79.9 | +0.1 | [−0.7, +0.9] | 0.856 | 2/4 |
| Cube | CEM | 63.3 | 64.3 | +1.0 | [+0.2, +1.8] | 0.010 | 4/4 |
| Cube | iCEM | 68.5 | 70.7 | +2.2 | [+1.4, +2.9] | <0.0001 | 4/4 |
| Two-Room | CEM | 77.8 | 89.5 | +11.7 | [+9.9, +13.5] | <0.0001 | 4/4 |
| Two-Room | iCEM | 83.1 | 92.9 | +9.9 | [+8.2, +11.5] | <0.0001 | 4/4 |
| PointMaze | CEM | 84.5 | 81.4 | −3.1 | [−4.1, −2.1] | <0.0001 | 2/4 |
| PointMaze | iCEM | 82.0 | 79.7 | −2.3 | [−3.4, −1.2] | <0.0001 | 2/4 |

**Table O. SCALE vs. DINO-WM: success rate (%) averaged over the five budget tiers, with paired significance tests (SCALE: four runs; DINO-WM: the two runs from the paper).** Δ = SCALE − DINO-WM in percentage points. All models are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode. The per-episode difference is averaged over training runs and the five budget tiers; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | DINO-WM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 56.5 | 73.6 | +17.1 | [+14.9, +19.3] | <0.0001 |
| Push-T | iCEM | 51.7 | 68.1 | +16.4 | [+14.2, +18.6] | <0.0001 |
| Reacher | CEM | 67.0 | 71.5 | +4.5 | [+2.9, +6.2] | <0.0001 |
| Reacher | iCEM | 73.3 | 79.9 | +6.6 | [+4.8, +8.3] | <0.0001 |
| Cube | CEM | 60.9 | 64.3 | +3.5 | [+2.4, +4.6] | <0.0001 |
| Cube | iCEM | 62.5 | 70.7 | +8.2 | [+6.1, +10.4] | <0.0001 |
| Two-Room | CEM | 95.6 | 89.5 | −6.1 | [−7.0, −5.2] | <0.0001 |
| Two-Room | iCEM | 98.2 | 92.9 | −5.3 | [−6.1, −4.4] | <0.0001 |
| PointMaze | CEM | 94.7 | 81.4 | −13.3 | [−14.9, −11.6] | <0.0001 |
| PointMaze | iCEM | 90.4 | 79.7 | −10.7 | [−12.5, −8.9] | <0.0001 |
