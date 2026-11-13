**Table S1. Seed-level results, SCALE vs. LeWM: success rate (%) averaged over the five budget tiers, separately for each independently trained run.** Runs A and B are the two runs of the submission; runs C and D are new. Within a run, SCALE and LeWM share the training seed and are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise. Δ = SCALE − LeWM in percentage points; the 95% CI is a paired t-interval over test episodes within that run and p is from an exact two-sided paired sign-flip permutation test, so each row reflects evaluation variability only, and the spread across rows of a task reflects training-seed variability.

| Task | Planner | Run | LeWM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | A (submission) | 70.5 | 73.9 | +3.4 | [+1.8, +5.0] | <0.0001 |
| Push-T | CEM | B (submission) | 71.0 | 72.4 | +1.4 | [−0.2, +3.0] | 0.089 |
| Push-T | CEM | C (new) | 70.9 | 74.2 | +3.3 | [+1.7, +4.8] | <0.0001 |
| Push-T | CEM | D (new) | 71.0 | 74.0 | +3.0 | [+1.4, +4.6] | 0.0003 |
| Push-T | iCEM | A (submission) | 63.9 | 68.2 | +4.3 | [+2.7, +5.9] | <0.0001 |
| Push-T | iCEM | B (submission) | 63.9 | 68.4 | +4.5 | [+3.0, +6.0] | <0.0001 |
| Push-T | iCEM | C (new) | 65.9 | 68.0 | +2.1 | [+0.5, +3.7] | 0.0095 |
| Push-T | iCEM | D (new) | 64.6 | 67.9 | +3.3 | [+1.7, +4.9] | <0.0001 |
| Reacher | CEM | A (submission) | 70.6 | 71.5 | +1.0 | [−0.7, +2.8] | 0.269 |
| Reacher | CEM | B (submission) | 72.3 | 71.8 | −0.4 | [−2.2, +1.3] | 0.651 |
| Reacher | CEM | C (new) | 70.9 | 71.0 | +0.1 | [−1.8, +1.8] | 1.000 |
| Reacher | CEM | D (new) | 71.2 | 71.6 | +0.4 | [−1.4, +2.2] | 0.697 |
| Reacher | iCEM | A (submission) | 78.0 | 81.1 | +3.1 | [+1.6, +4.6] | <0.0001 |
| Reacher | iCEM | B (submission) | 79.7 | 81.7 | +2.0 | [+0.6, +3.4] | 0.0058 |
| Reacher | iCEM | C (new) | 78.7 | 78.5 | −0.3 | [−1.7, +1.2] | 0.750 |
| Reacher | iCEM | D (new) | 80.0 | 78.3 | −1.7 | [−3.2, −0.1] | 0.036 |
| Cube | CEM | A (submission) | 61.3 | 64.1 | +2.8 | [+1.3, +4.3] | 0.0003 |
| Cube | CEM | B (submission) | 60.5 | 64.0 | +3.5 | [+2.0, +5.0] | <0.0001 |
| Cube | CEM | C (new) | 61.2 | 64.8 | +3.6 | [+2.0, +5.1] | <0.0001 |
| Cube | CEM | D (new) | 61.4 | 64.4 | +3.1 | [+1.6, +4.5] | <0.0001 |
| Cube | iCEM | A (submission) | 65.3 | 70.5 | +5.1 | [+3.8, +6.5] | <0.0001 |
| Cube | iCEM | B (submission) | 63.3 | 71.2 | +7.9 | [+6.3, +9.5] | <0.0001 |
| Cube | iCEM | C (new) | 64.1 | 70.1 | +6.0 | [+4.6, +7.5] | <0.0001 |
| Cube | iCEM | D (new) | 65.7 | 71.0 | +5.3 | [+3.9, +6.8] | <0.0001 |
| Two-Room | CEM | A (submission) | 77.9 | 89.2 | +11.3 | [+9.2, +13.4] | <0.0001 |
| Two-Room | CEM | B (submission) | 77.0 | 89.2 | +12.2 | [+10.2, +14.3] | <0.0001 |
| Two-Room | CEM | C (new) | 76.0 | 89.4 | +13.4 | [+11.2, +15.7] | <0.0001 |
| Two-Room | CEM | D (new) | 77.5 | 90.1 | +12.6 | [+10.5, +14.7] | <0.0001 |
| Two-Room | iCEM | A (submission) | 82.9 | 92.9 | +10.0 | [+8.1, +11.8] | <0.0001 |
| Two-Room | iCEM | B (submission) | 82.0 | 92.6 | +10.6 | [+8.8, +12.5] | <0.0001 |
| Two-Room | iCEM | C (new) | 81.4 | 93.3 | +11.9 | [+10.0, +13.9] | <0.0001 |
| Two-Room | iCEM | D (new) | 82.6 | 92.9 | +10.3 | [+8.4, +12.2] | <0.0001 |
| PointMaze | CEM | A (submission) | 82.1 | 86.2 | +4.1 | [+2.2, +5.9] | <0.0001 |
| PointMaze | CEM | B (submission) | 84.8 | 83.7 | −1.2 | [−3.2, +0.8] | 0.269 |
| PointMaze | CEM | C (new) | 79.5 | 72.3 | −7.2 | [−9.7, −4.7] | <0.0001 |
| PointMaze | CEM | D (new) | 81.2 | 83.4 | +2.3 | [+0.4, +4.1] | 0.016 |
| PointMaze | iCEM | A (submission) | 80.2 | 83.4 | +3.2 | [+1.2, +5.1] | 0.0015 |
| PointMaze | iCEM | B (submission) | 81.6 | 83.4 | +1.8 | [−0.2, +3.8] | 0.084 |
| PointMaze | iCEM | C (new) | 69.8 | 70.3 | +0.5 | [−2.1, +3.2] | 0.714 |
| PointMaze | iCEM | D (new) | 76.9 | 81.6 | +4.7 | [+2.8, +6.7] | <0.0001 |

**Table S2. Seed-level results, SCALE vs. Aux: success rate (%) averaged over the five budget tiers, separately for each independently trained run.** Runs A and B are the two runs of the submission; runs C and D are new. Within a run, SCALE and Aux share the training seed and are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise. Δ = SCALE − Aux in percentage points; the 95% CI is a paired t-interval over test episodes within that run and p is from an exact two-sided paired sign-flip permutation test, so each row reflects evaluation variability only, and the spread across rows of a task reflects training-seed variability.

| Task | Planner | Run | Aux | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | A (submission) | 74.9 | 73.9 | −1.0 | [−2.6, +0.5] | 0.197 |
| Push-T | CEM | B (submission) | 73.7 | 72.4 | −1.3 | [−2.8, +0.1] | 0.082 |
| Push-T | CEM | C (new) | 74.0 | 74.2 | +0.2 | [−1.3, +1.7] | 0.823 |
| Push-T | CEM | D (new) | 74.1 | 74.0 | −0.2 | [−1.7, +1.4] | 0.867 |
| Push-T | iCEM | A (submission) | 69.3 | 68.2 | −1.1 | [−2.6, +0.4] | 0.163 |
| Push-T | iCEM | B (submission) | 68.9 | 68.4 | −0.4 | [−1.7, +0.9] | 0.551 |
| Push-T | iCEM | C (new) | 69.4 | 68.0 | −1.4 | [−3.0, +0.1] | 0.074 |
| Push-T | iCEM | D (new) | 68.2 | 67.9 | −0.3 | [−1.7, +1.2] | 0.751 |
| Reacher | CEM | A (submission) | 70.4 | 71.5 | +1.1 | [−0.5, +2.8] | 0.182 |
| Reacher | CEM | B (submission) | 72.3 | 71.8 | −0.5 | [−2.3, +1.2] | 0.552 |
| Reacher | CEM | C (new) | 72.1 | 71.0 | −1.1 | [−2.9, +0.7] | 0.248 |
| Reacher | CEM | D (new) | 73.3 | 71.6 | −1.7 | [−3.6, +0.3] | 0.096 |
| Reacher | iCEM | A (submission) | 80.2 | 81.1 | +1.0 | [−0.4, +2.4] | 0.189 |
| Reacher | iCEM | B (submission) | 79.8 | 81.7 | +1.9 | [+0.5, +3.3] | 0.0080 |
| Reacher | iCEM | C (new) | 79.3 | 78.5 | −0.8 | [−2.4, +0.8] | 0.331 |
| Reacher | iCEM | D (new) | 80.2 | 78.3 | −1.8 | [−3.2, −0.4] | 0.015 |
| Cube | CEM | A (submission) | 63.5 | 64.1 | +0.6 | [−0.6, +1.9] | 0.333 |
| Cube | CEM | B (submission) | 63.1 | 64.0 | +0.9 | [−0.4, +2.2] | 0.192 |
| Cube | CEM | C (new) | 63.6 | 64.8 | +1.1 | [−0.2, +2.4] | 0.100 |
| Cube | CEM | D (new) | 63.0 | 64.4 | +1.4 | [+0.2, +2.7] | 0.029 |
| Cube | iCEM | A (submission) | 68.6 | 70.5 | +1.9 | [+0.9, +2.9] | 0.0003 |
| Cube | iCEM | B (submission) | 68.2 | 71.2 | +3.0 | [+2.0, +4.0] | <0.0001 |
| Cube | iCEM | C (new) | 69.1 | 70.1 | +1.1 | [−0.0, +2.2] | 0.065 |
| Cube | iCEM | D (new) | 68.2 | 71.0 | +2.8 | [+1.8, +3.8] | <0.0001 |
| Two-Room | CEM | A (submission) | 78.1 | 89.2 | +11.1 | [+9.0, +13.3] | <0.0001 |
| Two-Room | CEM | B (submission) | 77.4 | 89.2 | +11.9 | [+9.8, +13.9] | <0.0001 |
| Two-Room | CEM | C (new) | 77.4 | 89.4 | +12.0 | [+9.9, +14.2] | <0.0001 |
| Two-Room | CEM | D (new) | 78.3 | 90.1 | +11.8 | [+9.9, +13.8] | <0.0001 |
| Two-Room | iCEM | A (submission) | 83.1 | 92.9 | +9.8 | [+7.9, +11.6] | <0.0001 |
| Two-Room | iCEM | B (submission) | 83.5 | 92.6 | +9.1 | [+7.3, +11.0] | <0.0001 |
| Two-Room | iCEM | C (new) | 82.6 | 93.3 | +10.7 | [+8.9, +12.6] | <0.0001 |
| Two-Room | iCEM | D (new) | 83.1 | 92.9 | +9.8 | [+7.9, +11.6] | <0.0001 |
| PointMaze | CEM | A (submission) | 84.3 | 86.2 | +1.9 | [+0.2, +3.7] | 0.036 |
| PointMaze | CEM | B (submission) | 79.5 | 83.7 | +4.2 | [+1.7, +6.7] | 0.0012 |
| PointMaze | CEM | C (new) | 85.6 | 72.3 | −13.3 | [−15.6, −11.0] | <0.0001 |
| PointMaze | CEM | D (new) | 88.7 | 83.4 | −5.2 | [−7.1, −3.4] | <0.0001 |
| PointMaze | iCEM | A (submission) | 81.3 | 83.4 | +2.0 | [−0.1, +4.1] | 0.062 |
| PointMaze | iCEM | B (submission) | 76.0 | 83.4 | +7.4 | [+4.5, +10.2] | <0.0001 |
| PointMaze | iCEM | C (new) | 82.9 | 70.3 | −12.5 | [−14.8, −10.2] | <0.0001 |
| PointMaze | iCEM | D (new) | 87.6 | 81.6 | −6.0 | [−7.9, −4.1] | <0.0001 |

**Table S3. SCALE − LeWM per training run, with training-seed and evaluation uncertainty reported separately.** Success rate (%) averaged over the five budget tiers; Δ in percentage points. Runs A and B are the two runs of the submission, C and D are new. "Across runs": mean of the four within-run differences, 95% t-interval and two-sided paired t-test over the four training runs (df = 3) — training-seed variability. "Across episodes": the per-episode difference averaged over the four runs, 95% paired t-interval and exact two-sided sign-flip permutation test over the 600 test episodes — evaluation variability.

| Task | Planner | Δ run A | Δ run B | Δ run C | Δ run D | Mean Δ | 95% CI across runs | p across runs | 95% CI across episodes | p across episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| Push-T | CEM | +3.4 | +1.4 | +3.3 | +3.0 | +2.8 | [+1.3, +4.2] | 0.0093 | [+1.8, +3.7] | <0.0001 |
| Push-T | iCEM | +4.3 | +4.5 | +2.1 | +3.3 | +3.6 | [+1.8, +5.3] | 0.0074 | [+2.5, +4.6] | <0.0001 |
| Reacher | CEM | +1.0 | −0.4 | +0.1 | +0.4 | +0.3 | [−0.7, +1.2] | 0.458 | [−0.7, +1.2] | 0.598 |
| Reacher | iCEM | +3.1 | +2.0 | −0.3 | −1.7 | +0.8 | [−2.7, +4.3] | 0.519 | [−0.0, +1.6] | 0.054 |
| Cube | CEM | +2.8 | +3.5 | +3.6 | +3.1 | +3.2 | [+2.7, +3.8] | 0.0004 | [+2.2, +4.3] | <0.0001 |
| Cube | iCEM | +5.1 | +7.9 | +6.0 | +5.3 | +6.1 | [+4.1, +8.1] | 0.0023 | [+4.9, +7.3] | <0.0001 |
| Two-Room | CEM | +11.3 | +12.2 | +13.4 | +12.6 | +12.4 | [+11.0, +13.8] | <0.0001 | [+10.5, +14.3] | <0.0001 |
| Two-Room | iCEM | +10.0 | +10.6 | +11.9 | +10.3 | +10.7 | [+9.3, +12.1] | 0.0001 | [+9.0, +12.4] | <0.0001 |
| PointMaze | CEM | +4.1 | −1.2 | −7.2 | +2.3 | −0.5 | [−8.4, +7.4] | 0.853 | [−1.5, +0.5] | 0.340 |
| PointMaze | iCEM | +3.2 | +1.8 | +0.5 | +4.7 | +2.6 | [−0.3, +5.4] | 0.066 | [+1.4, +3.7] | <0.0001 |

**Table S4. SCALE − Aux per training run, with training-seed and evaluation uncertainty reported separately.** Success rate (%) averaged over the five budget tiers; Δ in percentage points. Runs A and B are the two runs of the submission, C and D are new. "Across runs": mean of the four within-run differences, 95% t-interval and two-sided paired t-test over the four training runs (df = 3) — training-seed variability. "Across episodes": the per-episode difference averaged over the four runs, 95% paired t-interval and exact two-sided sign-flip permutation test over the 600 test episodes — evaluation variability.

| Task | Planner | Δ run A | Δ run B | Δ run C | Δ run D | Mean Δ | 95% CI across runs | p across runs | 95% CI across episodes | p across episodes |
|---|---|---|---|---|---|---|---|---|---|---|
| Push-T | CEM | −1.0 | −1.3 | +0.2 | −0.2 | −0.6 | [−1.7, +0.6] | 0.203 | [−1.5, +0.3] | 0.199 |
| Push-T | iCEM | −1.1 | −0.4 | −1.4 | −0.3 | −0.8 | [−1.7, +0.1] | 0.061 | [−1.7, +0.1] | 0.080 |
| Reacher | CEM | +1.1 | −0.5 | −1.1 | −1.7 | −0.5 | [−2.4, +1.4] | 0.433 | [−1.5, +0.4] | 0.252 |
| Reacher | iCEM | +1.0 | +1.9 | −0.8 | −1.8 | +0.1 | [−2.6, +2.8] | 0.942 | [−0.7, +0.9] | 0.856 |
| Cube | CEM | +0.6 | +0.9 | +1.1 | +1.4 | +1.0 | [+0.5, +1.6] | 0.0092 | [+0.2, +1.8] | 0.010 |
| Cube | iCEM | +1.9 | +3.0 | +1.1 | +2.8 | +2.2 | [+0.8, +3.6] | 0.016 | [+1.4, +2.9] | <0.0001 |
| Two-Room | CEM | +11.1 | +11.9 | +12.0 | +11.8 | +11.7 | [+11.1, +12.3] | <0.0001 | [+9.9, +13.5] | <0.0001 |
| Two-Room | iCEM | +9.8 | +9.1 | +10.7 | +9.8 | +9.9 | [+8.8, +10.9] | <0.0001 | [+8.2, +11.5] | <0.0001 |
| PointMaze | CEM | +1.9 | +4.2 | −13.3 | −5.2 | −3.1 | [−15.6, +9.4] | 0.489 | [−4.1, −2.1] | <0.0001 |
| PointMaze | iCEM | +2.0 | +7.4 | −12.5 | −6.0 | −2.3 | [−16.2, +11.7] | 0.639 | [−3.4, −1.2] | <0.0001 |
