**Table A. LpWM vs. LpWM+SCALE: planning success rate (%) by task, planning-budget tier and planner.** Each entry is the mean over 600 test episodes (6 sets × 100). Budgets (candidates × iterations): T1 300×30, T2 150×15, T3 50×10, T4 20×5, T5 10×3. Reacher, Cube, Two-Room and PointMaze average two training seeds (1024, 1025); Push-T uses seed 1024 only, because the LpWM+SCALE run at seed 1025 collapsed during training and that seed is excluded for both models.

| Task | Tier | Planner | LpWM | LpWM+SCALE |
|---|---|---|---|---|
| Push-T | T1 | CEM | 75.2 | 77.7 |
| Push-T | T1 | iCEM | 72.5 | 73.3 |
| Push-T | T2 | CEM | 71.5 | 78.3 |
| Push-T | T2 | iCEM | 67.8 | 69.0 |
| Push-T | T3 | CEM | 61.0 | 69.2 |
| Push-T | T3 | iCEM | 60.2 | 60.3 |
| Push-T | T4 | CEM | 43.2 | 52.3 |
| Push-T | T4 | iCEM | 42.8 | 42.2 |
| Push-T | T5 | CEM | 33.5 | 31.3 |
| Push-T | T5 | iCEM | 28.7 | 26.5 |
| Reacher | T1 | CEM | 78.3 | 77.4 |
| Reacher | T1 | iCEM | 81.8 | 81.8 |
| Reacher | T2 | CEM | 76.0 | 76.5 |
| Reacher | T2 | iCEM | 79.8 | 80.8 |
| Reacher | T3 | CEM | 72.8 | 72.5 |
| Reacher | T3 | iCEM | 80.8 | 80.0 |
| Reacher | T4 | CEM | 63.9 | 64.0 |
| Reacher | T4 | iCEM | 76.2 | 76.8 |
| Reacher | T5 | CEM | 55.4 | 55.5 |
| Reacher | T5 | iCEM | 60.2 | 62.2 |
| Cube | T1 | CEM | 66.8 | 71.8 |
| Cube | T1 | iCEM | 72.7 | 77.0 |
| Cube | T2 | CEM | 65.8 | 70.8 |
| Cube | T2 | iCEM | 72.4 | 76.4 |
| Cube | T3 | CEM | 60.3 | 63.8 |
| Cube | T3 | iCEM | 70.9 | 75.6 |
| Cube | T4 | CEM | 56.0 | 55.9 |
| Cube | T4 | iCEM | 60.0 | 63.1 |
| Cube | T5 | CEM | 55.8 | 54.7 |
| Cube | T5 | iCEM | 56.2 | 56.9 |
| Two-Room | T1 | CEM | 95.7 | 99.8 |
| Two-Room | T1 | iCEM | 96.9 | 99.6 |
| Two-Room | T2 | CEM | 94.3 | 99.5 |
| Two-Room | T2 | iCEM | 96.3 | 99.5 |
| Two-Room | T3 | CEM | 90.2 | 96.3 |
| Two-Room | T3 | iCEM | 93.2 | 98.2 |
| Two-Room | T4 | CEM | 79.8 | 88.4 |
| Two-Room | T4 | iCEM | 84.8 | 92.0 |
| Two-Room | T5 | CEM | 66.8 | 77.6 |
| Two-Room | T5 | iCEM | 71.1 | 83.1 |
| PointMaze | T1 | CEM | 72.8 | 84.3 |
| PointMaze | T1 | iCEM | 64.3 | 77.3 |
| PointMaze | T2 | CEM | 72.4 | 83.6 |
| PointMaze | T2 | iCEM | 62.3 | 75.7 |
| PointMaze | T3 | CEM | 71.3 | 82.5 |
| PointMaze | T3 | iCEM | 62.3 | 74.9 |
| PointMaze | T4 | CEM | 68.3 | 77.1 |
| PointMaze | T4 | iCEM | 63.2 | 72.8 |
| PointMaze | T5 | CEM | 68.7 | 71.1 |
| PointMaze | T5 | iCEM | 64.3 | 67.4 |

**Table A (wide layout): one table per planner; columns are the budget tiers, Avg. is the mean over the five tiers.**

**CEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LpWM | 75.2 | 71.5 | 61.0 | 43.2 | 33.5 | 56.9 |
| Push-T | LpWM+SCALE | 77.7 | 78.3 | 69.2 | 52.3 | 31.3 | 61.8 |
| Reacher | LpWM | 78.3 | 76.0 | 72.8 | 63.9 | 55.4 | 69.3 |
| Reacher | LpWM+SCALE | 77.4 | 76.5 | 72.5 | 64.0 | 55.5 | 69.2 |
| Cube | LpWM | 66.8 | 65.8 | 60.3 | 56.0 | 55.8 | 61.0 |
| Cube | LpWM+SCALE | 71.8 | 70.8 | 63.8 | 55.9 | 54.7 | 63.4 |
| Two-Room | LpWM | 95.7 | 94.3 | 90.2 | 79.8 | 66.8 | 85.4 |
| Two-Room | LpWM+SCALE | 99.8 | 99.5 | 96.3 | 88.4 | 77.6 | 92.3 |
| PointMaze | LpWM | 72.8 | 72.4 | 71.3 | 68.3 | 68.7 | 70.7 |
| PointMaze | LpWM+SCALE | 84.3 | 83.6 | 82.5 | 77.1 | 71.1 | 79.7 |

**iCEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LpWM | 72.5 | 67.8 | 60.2 | 42.8 | 28.7 | 54.4 |
| Push-T | LpWM+SCALE | 73.3 | 69.0 | 60.3 | 42.2 | 26.5 | 54.3 |
| Reacher | LpWM | 81.8 | 79.8 | 80.8 | 76.2 | 60.2 | 75.7 |
| Reacher | LpWM+SCALE | 81.8 | 80.8 | 80.0 | 76.8 | 62.2 | 76.3 |
| Cube | LpWM | 72.7 | 72.4 | 70.9 | 60.0 | 56.2 | 66.4 |
| Cube | LpWM+SCALE | 77.0 | 76.4 | 75.6 | 63.1 | 56.9 | 69.8 |
| Two-Room | LpWM | 96.9 | 96.3 | 93.2 | 84.8 | 71.1 | 88.5 |
| Two-Room | LpWM+SCALE | 99.6 | 99.5 | 98.2 | 92.0 | 83.1 | 94.5 |
| PointMaze | LpWM | 64.3 | 62.3 | 62.3 | 63.2 | 64.3 | 63.3 |
| PointMaze | LpWM+SCALE | 77.3 | 75.7 | 74.9 | 72.8 | 67.4 | 73.6 |


**Table B. LpWM vs. LpWM+SCALE: success rate (%) averaged over the five budget tiers, with paired significance tests.** Both models are evaluated on the same 600 test episodes with identical planner noise, so outcomes are paired per episode. Δ = LpWM+SCALE − LpWM in percentage points; the 95% CI is a paired t-interval over episodes; p is from an exact two-sided paired sign-flip permutation test. Reacher, Cube, Two-Room and PointMaze average two training seeds (1024, 1025); Push-T uses seed 1024 only, because the LpWM+SCALE run at seed 1025 collapsed during training and that seed is excluded for both models.

| Task | Planner | LpWM | LpWM+SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 56.9 | 61.8 | +4.9 | [+2.5, +7.3] | <0.0001 |
| Push-T | iCEM | 54.4 | 54.3 | −0.1 | [−2.4, +2.1] | 0.931 |
| Reacher | CEM | 69.3 | 69.2 | −0.1 | [−1.5, +1.3] | 0.926 |
| Reacher | iCEM | 75.7 | 76.3 | +0.6 | [−0.6, +1.9] | 0.331 |
| Cube | CEM | 61.0 | 63.4 | +2.5 | [+1.4, +3.5] | <0.0001 |
| Cube | iCEM | 66.4 | 69.8 | +3.4 | [+2.5, +4.3] | <0.0001 |
| Two-Room | CEM | 85.4 | 92.3 | +7.0 | [+5.7, +8.3] | <0.0001 |
| Two-Room | iCEM | 88.5 | 94.5 | +6.0 | [+4.8, +7.2] | <0.0001 |
| PointMaze | CEM | 70.7 | 79.7 | +9.0 | [+7.3, +10.7] | <0.0001 |
| PointMaze | iCEM | 63.3 | 73.6 | +10.4 | [+8.3, +12.5] | <0.0001 |
