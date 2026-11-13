**Table S1. Seed-level results, SCALE vs. LeWM: success rate (%) averaged over the five budget tiers, separately for each independently trained run.** Runs A and B are the two runs of the submission; run C is new. Within a run, SCALE and LeWM share the training seed and are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise. Δ = SCALE − LeWM in percentage points; the 95% CI is a paired t-interval over test episodes within that run and p is from an exact two-sided paired sign-flip permutation test, so each row reflects evaluation variability only, and the spread across rows of a task reflects training-seed variability.

| Task | Planner | Run | LeWM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | A (submission) | 70.5 | 73.9 | +3.4 | [+1.8, +5.0] | <0.0001 |
| Push-T | CEM | B (submission) | 71.0 | 72.4 | +1.4 | [−0.2, +3.0] | 0.089 |
| Push-T | CEM | C (new) | 71.0 | 74.0 | +3.0 | [+1.4, +4.6] | 0.0003 |
| Push-T | iCEM | A (submission) | 63.9 | 68.2 | +4.3 | [+2.7, +5.9] | <0.0001 |
| Push-T | iCEM | B (submission) | 63.9 | 68.4 | +4.5 | [+3.0, +6.0] | <0.0001 |
| Push-T | iCEM | C (new) | 64.6 | 67.9 | +3.3 | [+1.7, +4.9] | <0.0001 |
| Reacher | CEM | A (submission) | 70.6 | 71.5 | +1.0 | [−0.7, +2.8] | 0.269 |
| Reacher | CEM | B (submission) | 72.3 | 71.8 | −0.4 | [−2.2, +1.3] | 0.651 |
| Reacher | CEM | C (new) | 71.2 | 71.6 | +0.4 | [−1.4, +2.2] | 0.697 |
| Reacher | iCEM | A (submission) | 78.0 | 81.1 | +3.1 | [+1.6, +4.6] | <0.0001 |
| Reacher | iCEM | B (submission) | 79.7 | 81.7 | +2.0 | [+0.6, +3.4] | 0.0058 |
| Reacher | iCEM | C (new) | 80.0 | 78.3 | −1.7 | [−3.2, −0.1] | 0.036 |
| Cube | CEM | A (submission) | 61.3 | 64.1 | +2.8 | [+1.3, +4.3] | 0.0003 |
| Cube | CEM | B (submission) | 60.5 | 64.0 | +3.5 | [+2.0, +5.0] | <0.0001 |
| Cube | CEM | C (new) | 61.4 | 64.4 | +3.1 | [+1.6, +4.5] | <0.0001 |
| Cube | iCEM | A (submission) | 65.3 | 70.5 | +5.1 | [+3.8, +6.5] | <0.0001 |
| Cube | iCEM | B (submission) | 63.3 | 71.2 | +7.9 | [+6.3, +9.5] | <0.0001 |
| Cube | iCEM | C (new) | 65.7 | 71.0 | +5.3 | [+3.9, +6.8] | <0.0001 |
| Two-Room | CEM | A (submission) | 77.9 | 89.2 | +11.3 | [+9.2, +13.4] | <0.0001 |
| Two-Room | CEM | B (submission) | 77.0 | 89.2 | +12.2 | [+10.2, +14.3] | <0.0001 |
| Two-Room | CEM | C (new) | 77.5 | 90.1 | +12.6 | [+10.5, +14.7] | <0.0001 |
| Two-Room | iCEM | A (submission) | 82.9 | 92.9 | +10.0 | [+8.1, +11.8] | <0.0001 |
| Two-Room | iCEM | B (submission) | 82.0 | 92.6 | +10.6 | [+8.8, +12.5] | <0.0001 |
| Two-Room | iCEM | C (new) | 82.6 | 92.9 | +10.3 | [+8.4, +12.2] | <0.0001 |
| PointMaze | CEM | A (submission) | 82.1 | 86.2 | +4.1 | [+2.2, +5.9] | <0.0001 |
| PointMaze | CEM | B (submission) | 84.8 | 83.7 | −1.2 | [−3.2, +0.8] | 0.269 |
| PointMaze | CEM | C (new) | 81.2 | 83.4 | +2.3 | [+0.4, +4.1] | 0.016 |
| PointMaze | iCEM | A (submission) | 80.2 | 83.4 | +3.2 | [+1.2, +5.1] | 0.0015 |
| PointMaze | iCEM | B (submission) | 81.6 | 83.4 | +1.8 | [−0.2, +3.8] | 0.084 |
| PointMaze | iCEM | C (new) | 76.9 | 81.6 | +4.7 | [+2.8, +6.7] | <0.0001 |

**Table S2. Seed-level results, SCALE vs. Aux: success rate (%) averaged over the five budget tiers, separately for each independently trained run.** Runs A and B are the two runs of the submission; run C is new. Within a run, SCALE and Aux share the training seed and are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise. Δ = SCALE − Aux in percentage points; the 95% CI is a paired t-interval over test episodes within that run and p is from an exact two-sided paired sign-flip permutation test, so each row reflects evaluation variability only, and the spread across rows of a task reflects training-seed variability.

| Task | Planner | Run | Aux | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | A (submission) | 74.9 | 73.9 | −1.0 | [−2.6, +0.5] | 0.197 |
| Push-T | CEM | B (submission) | 73.7 | 72.4 | −1.3 | [−2.8, +0.1] | 0.082 |
| Push-T | CEM | C (new) | 74.1 | 74.0 | −0.2 | [−1.7, +1.4] | 0.867 |
| Push-T | iCEM | A (submission) | 69.3 | 68.2 | −1.1 | [−2.6, +0.4] | 0.163 |
| Push-T | iCEM | B (submission) | 68.9 | 68.4 | −0.4 | [−1.7, +0.9] | 0.551 |
| Push-T | iCEM | C (new) | 68.2 | 67.9 | −0.3 | [−1.7, +1.2] | 0.751 |
| Reacher | CEM | A (submission) | 70.4 | 71.5 | +1.1 | [−0.5, +2.8] | 0.182 |
| Reacher | CEM | B (submission) | 72.3 | 71.8 | −0.5 | [−2.3, +1.2] | 0.552 |
| Reacher | CEM | C (new) | 73.3 | 71.6 | −1.7 | [−3.6, +0.3] | 0.096 |
| Reacher | iCEM | A (submission) | 80.2 | 81.1 | +1.0 | [−0.4, +2.4] | 0.189 |
| Reacher | iCEM | B (submission) | 79.8 | 81.7 | +1.9 | [+0.5, +3.3] | 0.0080 |
| Reacher | iCEM | C (new) | 80.2 | 78.3 | −1.8 | [−3.2, −0.4] | 0.015 |
| Cube | CEM | A (submission) | 63.5 | 64.1 | +0.6 | [−0.6, +1.9] | 0.333 |
| Cube | CEM | B (submission) | 63.1 | 64.0 | +0.9 | [−0.4, +2.2] | 0.192 |
| Cube | CEM | C (new) | 63.0 | 64.4 | +1.4 | [+0.2, +2.7] | 0.029 |
| Cube | iCEM | A (submission) | 68.6 | 70.5 | +1.9 | [+0.9, +2.9] | 0.0003 |
| Cube | iCEM | B (submission) | 68.2 | 71.2 | +3.0 | [+2.0, +4.0] | <0.0001 |
| Cube | iCEM | C (new) | 68.2 | 71.0 | +2.8 | [+1.8, +3.8] | <0.0001 |
| Two-Room | CEM | A (submission) | 78.1 | 89.2 | +11.1 | [+9.0, +13.3] | <0.0001 |
| Two-Room | CEM | B (submission) | 77.4 | 89.2 | +11.9 | [+9.8, +13.9] | <0.0001 |
| Two-Room | CEM | C (new) | 78.3 | 90.1 | +11.8 | [+9.9, +13.8] | <0.0001 |
| Two-Room | iCEM | A (submission) | 83.1 | 92.9 | +9.8 | [+7.9, +11.6] | <0.0001 |
| Two-Room | iCEM | B (submission) | 83.5 | 92.6 | +9.1 | [+7.3, +11.0] | <0.0001 |
| Two-Room | iCEM | C (new) | 83.1 | 92.9 | +9.8 | [+7.9, +11.6] | <0.0001 |
| PointMaze | CEM | A (submission) | 84.3 | 86.2 | +1.9 | [+0.2, +3.7] | 0.036 |
| PointMaze | CEM | B (submission) | 79.5 | 83.7 | +4.2 | [+1.7, +6.7] | 0.0012 |
| PointMaze | CEM | C (new) | 88.7 | 83.4 | −5.2 | [−7.1, −3.4] | <0.0001 |
| PointMaze | iCEM | A (submission) | 81.3 | 83.4 | +2.0 | [−0.1, +4.1] | 0.062 |
| PointMaze | iCEM | B (submission) | 76.0 | 83.4 | +7.4 | [+4.5, +10.2] | <0.0001 |
| PointMaze | iCEM | C (new) | 87.6 | 81.6 | −6.0 | [−7.9, −4.1] | <0.0001 |

**Table S3. SCALE − LeWM per training run, with training-seed and evaluation uncertainty reported separately.** Success rate (%) averaged over the five budget tiers; Δ in percentage points. Runs A and B are the two runs of the submission; run C is new. "Across runs": mean of the three within-run differences, 95% t-interval and two-sided paired t-test over the three training runs (df = 2) — training-seed variability. "Across episodes": the per-episode difference averaged over the three runs, 95% paired t-interval and exact two-sided sign-flip permutation test over the 600 test episodes — evaluation variability.

| Task | Planner | Δ run A | Δ run B | Δ run C | Mean Δ | 95% CI across runs | p across runs | 95% CI across episodes | p across episodes |
|---|---|---|---|---|---|---|---|---|---|
| Push-T | CEM | +3.4 | +1.4 | +3.0 | +2.6 | [−0.0, +5.2] | 0.050 | [+1.6, +3.6] | <0.0001 |
| Push-T | iCEM | +4.3 | +4.5 | +3.3 | +4.0 | [+2.5, +5.6] | 0.0078 | [+3.0, +5.1] | <0.0001 |
| Reacher | CEM | +1.0 | −0.4 | +0.4 | +0.3 | [−1.4, +2.1] | 0.524 | [−0.7, +1.4] | 0.541 |
| Reacher | iCEM | +3.1 | +2.0 | −1.7 | +1.1 | [−5.1, +7.4] | 0.514 | [+0.2, +2.1] | 0.014 |
| Cube | CEM | +2.8 | +3.5 | +3.1 | +3.1 | [+2.3, +3.9] | 0.0039 | [+2.1, +4.2] | <0.0001 |
| Cube | iCEM | +5.1 | +7.9 | +5.3 | +6.1 | [+2.3, +10.0] | 0.021 | [+4.9, +7.4] | <0.0001 |
| Two-Room | CEM | +11.3 | +12.2 | +12.6 | +12.0 | [+10.4, +13.7] | 0.0010 | [+10.2, +13.9] | <0.0001 |
| Two-Room | iCEM | +10.0 | +10.6 | +10.3 | +10.3 | [+9.5, +11.1] | 0.0003 | [+8.6, +12.0] | <0.0001 |
| PointMaze | CEM | +4.1 | −1.2 | +2.3 | +1.7 | [−4.9, +8.3] | 0.379 | [+0.6, +2.8] | 0.0027 |
| PointMaze | iCEM | +3.2 | +1.8 | +4.7 | +3.2 | [−0.4, +6.9] | 0.062 | [+2.0, +4.4] | <0.0001 |

**Table S4. SCALE − Aux per training run, with training-seed and evaluation uncertainty reported separately.** Success rate (%) averaged over the five budget tiers; Δ in percentage points. Runs A and B are the two runs of the submission; run C is new. "Across runs": mean of the three within-run differences, 95% t-interval and two-sided paired t-test over the three training runs (df = 2) — training-seed variability. "Across episodes": the per-episode difference averaged over the three runs, 95% paired t-interval and exact two-sided sign-flip permutation test over the 600 test episodes — evaluation variability.

| Task | Planner | Δ run A | Δ run B | Δ run C | Mean Δ | 95% CI across runs | p across runs | 95% CI across episodes | p across episodes |
|---|---|---|---|---|---|---|---|---|---|
| Push-T | CEM | −1.0 | −1.3 | −0.2 | −0.8 | [−2.3, +0.7] | 0.137 | [−1.8, +0.1] | 0.093 |
| Push-T | iCEM | −1.1 | −0.4 | −0.3 | −0.6 | [−1.7, +0.5] | 0.143 | [−1.5, +0.3] | 0.220 |
| Reacher | CEM | +1.1 | −0.5 | −1.7 | −0.4 | [−3.9, +3.1] | 0.695 | [−1.4, +0.7] | 0.513 |
| Reacher | iCEM | +1.0 | +1.9 | −1.8 | +0.4 | [−4.5, +5.2] | 0.783 | [−0.5, +1.2] | 0.408 |
| Cube | CEM | +0.6 | +0.9 | +1.4 | +1.0 | [−0.0, +2.0] | 0.052 | [+0.2, +1.8] | 0.020 |
| Cube | iCEM | +1.9 | +3.0 | +2.8 | +2.5 | [+1.1, +4.0] | 0.018 | [+1.8, +3.3] | <0.0001 |
| Two-Room | CEM | +11.1 | +11.9 | +11.8 | +11.6 | [+10.6, +12.6] | 0.0004 | [+9.8, +13.4] | <0.0001 |
| Two-Room | iCEM | +9.8 | +9.1 | +9.8 | +9.6 | [+8.6, +10.5] | 0.0005 | [+7.9, +11.2] | <0.0001 |
| PointMaze | CEM | +1.9 | +4.2 | −5.2 | +0.3 | [−11.9, +12.5] | 0.928 | [−0.7, +1.3] | 0.588 |
| PointMaze | iCEM | +2.0 | +7.4 | −6.0 | +1.1 | [−15.5, +17.8] | 0.796 | [−0.0, +2.3] | 0.061 |
