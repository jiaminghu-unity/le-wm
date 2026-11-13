**SCALE vs. LeWM by planning-budget tier (T1 = largest budget, T5 = smallest): success rate (%) with paired significance tests (three runs per method).** Δ = SCALE − LeWM in percentage points. Both methods are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | Tier | LeWM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | T1 | 91.6 | 95.0 | +3.4 | [+2.0, +4.9] | <0.05 |
| Push-T | CEM | T2 | 89.8 | 92.2 | +2.4 | [+0.9, +3.8] | <0.05 |
| Push-T | CEM | T3 | 80.3 | 83.9 | +3.6 | [+1.4, +5.7] | <0.05 |
| Push-T | CEM | T4 | 58.0 | 62.9 | +4.9 | [+2.4, +7.5] | <0.05 |
| Push-T | CEM | T5 | 34.4 | 33.0 | −1.4 | [−3.6, +0.8] | 0.229 |
| Push-T | iCEM | T1 | 85.8 | 89.3 | +3.5 | [+1.8, +5.3] | <0.05 |
| Push-T | iCEM | T2 | 80.4 | 86.0 | +5.6 | [+3.7, +7.5] | <0.05 |
| Push-T | iCEM | T3 | 73.7 | 79.7 | +6.0 | [+3.9, +8.1] | <0.05 |
| Push-T | iCEM | T4 | 55.2 | 60.2 | +5.0 | [+2.8, +7.2] | <0.05 |
| Push-T | iCEM | T5 | 25.6 | 25.7 | +0.1 | [−1.7, +1.8] | 1.000 |
| Reacher | CEM | T1 | 82.5 | 80.9 | −1.6 | [−3.6, +0.5] | 0.147 |
| Reacher | CEM | T2 | 79.2 | 79.8 | +0.6 | [−1.5, +3.0] | 0.563 |
| Reacher | CEM | T3 | 73.4 | 74.1 | +0.7 | [−1.9, +3.3] | 0.613 |
| Reacher | CEM | T4 | 65.6 | 66.1 | +0.5 | [−1.9, +3.3] | 0.644 |
| Reacher | CEM | T5 | 56.1 | 57.2 | +1.1 | [−0.9, +3.2] | 0.290 |
| Reacher | iCEM | T1 | 85.3 | 86.2 | +0.9 | [−0.9, +2.7] | 0.324 |
| Reacher | iCEM | T2 | 84.9 | 85.5 | +0.6 | [−1.3, +2.5] | 0.563 |
| Reacher | iCEM | T3 | 84.8 | 86.6 | +1.8 | [−0.1, +3.6] | 0.072 |
| Reacher | iCEM | T4 | 79.3 | 80.4 | +1.1 | [−1.0, +3.1] | 0.330 |
| Reacher | iCEM | T5 | 61.9 | 63.2 | +1.3 | [−0.7, +3.5] | 0.207 |
| Cube | CEM | T1 | 68.2 | 74.7 | +6.5 | [+4.2, +8.8] | <0.05 |
| Cube | CEM | T2 | 66.2 | 71.3 | +5.1 | [+3.0, +7.3] | <0.05 |
| Cube | CEM | T3 | 60.6 | 64.4 | +3.8 | [+2.0, +5.7] | <0.05 |
| Cube | CEM | T4 | 56.4 | 55.6 | −0.8 | [−2.6, +0.8] | 0.340 |
| Cube | CEM | T5 | 54.1 | 55.0 | +0.9 | [−0.6, +2.5] | 0.267 |
| Cube | iCEM | T1 | 69.3 | 76.8 | +7.5 | [+5.7, +9.4] | <0.05 |
| Cube | iCEM | T2 | 67.6 | 76.6 | +9.0 | [+7.0, +11.1] | <0.05 |
| Cube | iCEM | T3 | 69.1 | 75.6 | +6.5 | [+4.6, +8.3] | <0.05 |
| Cube | iCEM | T4 | 62.1 | 65.9 | +3.8 | [+2.2, +5.3] | <0.05 |
| Cube | iCEM | T5 | 55.7 | 59.5 | +3.8 | [+2.2, +5.5] | <0.05 |
| Two-Room | CEM | T1 | 87.7 | 99.6 | +11.9 | [+9.5, +14.2] | <0.05 |
| Two-Room | CEM | T2 | 87.2 | 99.3 | +12.1 | [+9.8, +14.4] | <0.05 |
| Two-Room | CEM | T3 | 82.6 | 95.0 | +12.4 | [+10.0, +15.0] | <0.05 |
| Two-Room | CEM | T4 | 71.9 | 83.2 | +11.3 | [+8.7, +13.9] | <0.05 |
| Two-Room | CEM | T5 | 58.0 | 70.5 | +12.5 | [+9.5, +15.6] | <0.05 |
| Two-Room | iCEM | T1 | 90.9 | 99.9 | +9.0 | [+7.0, +11.1] | <0.05 |
| Two-Room | iCEM | T2 | 93.0 | 99.7 | +6.7 | [+4.9, +8.5] | <0.05 |
| Two-Room | iCEM | T3 | 88.7 | 97.8 | +9.1 | [+6.9, +11.2] | <0.05 |
| Two-Room | iCEM | T4 | 77.8 | 90.0 | +12.2 | [+9.7, +14.7] | <0.05 |
| Two-Room | iCEM | T5 | 62.0 | 76.6 | +14.6 | [+11.6, +17.5] | <0.05 |
| PointMaze | CEM | T1 | 85.0 | 86.9 | +1.9 | [+0.1, +3.7] | <0.05 |
| PointMaze | CEM | T2 | 85.3 | 87.4 | +2.1 | [+0.3, +3.9] | <0.05 |
| PointMaze | CEM | T3 | 85.3 | 85.9 | +0.6 | [−1.2, +2.4] | 0.586 |
| PointMaze | CEM | T4 | 83.2 | 83.9 | +0.7 | [−1.3, +2.8] | 0.522 |
| PointMaze | CEM | T5 | 74.8 | 78.1 | +3.3 | [+1.1, +5.5] | <0.05 |
| PointMaze | iCEM | T1 | 83.2 | 87.4 | +4.2 | [+2.4, +5.9] | <0.05 |
| PointMaze | iCEM | T2 | 82.2 | 85.8 | +3.6 | [+1.9, +5.4] | <0.05 |
| PointMaze | iCEM | T3 | 80.6 | 83.9 | +3.3 | [+1.3, +5.3] | <0.05 |
| PointMaze | iCEM | T4 | 78.8 | 80.7 | +1.9 | [−0.2, +4.1] | 0.086 |
| PointMaze | iCEM | T5 | 73.0 | 76.1 | +3.1 | [+0.8, +5.4] | <0.05 |

**Δ = SCALE − LeWM by tier (percentage points); \* marks p < 0.05.** The last rows average the five tasks (the test pools the episodes of all tasks).

| Task | Planner | T1 | T2 | T3 | T4 | T5 |
|---|---|---|---|---|---|---|
| Push-T | CEM | +3.4\* | +2.4\* | +3.6\* | +4.9\* | −1.4 |
| Push-T | iCEM | +3.5\* | +5.6\* | +6.0\* | +5.0\* | +0.1 |
| Reacher | CEM | −1.6 | +0.6 | +0.7 | +0.5 | +1.1 |
| Reacher | iCEM | +0.9 | +0.6 | +1.8 | +1.1 | +1.3 |
| Cube | CEM | +6.5\* | +5.1\* | +3.8\* | −0.8 | +0.9 |
| Cube | iCEM | +7.5\* | +9.0\* | +6.5\* | +3.8\* | +3.8\* |
| Two-Room | CEM | +11.9\* | +12.1\* | +12.4\* | +11.3\* | +12.5\* |
| Two-Room | iCEM | +9.0\* | +6.7\* | +9.1\* | +12.2\* | +14.6\* |
| PointMaze | CEM | +1.9\* | +2.1\* | +0.6 | +0.7 | +3.3\* |
| PointMaze | iCEM | +4.2\* | +3.6\* | +3.3\* | +1.9 | +3.1\* |
| Mean of five tasks | CEM | +4.4\* | +4.5\* | +4.2\* | +3.3\* | +3.3\* |
| Mean of five tasks | iCEM | +5.0\* | +5.1\* | +5.3\* | +4.8\* | +4.6\* |
| Tasks with significant gain | CEM | 4/5 | 4/5 | 3/5 | 2/5 | 2/5 |
| Tasks with significant gain | iCEM | 4/5 | 4/5 | 4/5 | 3/5 | 3/5 |
