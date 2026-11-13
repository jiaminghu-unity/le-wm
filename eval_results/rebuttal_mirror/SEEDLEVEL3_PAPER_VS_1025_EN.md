**Seed-level results, SCALE vs. LeWM: success rate (%) averaged over the five budget tiers, separately for each independently trained run.** Runs 1 and 2 are the two training runs of the original submission; the new run is one additional independently trained run with a new seed. Δ = SCALE − LeWM in percentage points. Within each run both methods share the training seed and are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | Training runs | LeWM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | Submission, run 1 | 70.5 | 73.9 | +3.4 | [+1.8, +5.0] | <0.05 |
| Push-T | CEM | Submission, run 2 | 71.0 | 72.4 | +1.4 | [−0.2, +3.0] | 0.089 |
| Push-T | CEM | New run | 71.0 | 74.0 | +3.0 | [+1.4, +4.6] | <0.05 |
| Push-T | iCEM | Submission, run 1 | 63.9 | 68.2 | +4.3 | [+2.7, +5.9] | <0.05 |
| Push-T | iCEM | Submission, run 2 | 63.9 | 68.4 | +4.5 | [+3.0, +6.0] | <0.05 |
| Push-T | iCEM | New run | 64.6 | 67.9 | +3.3 | [+1.7, +4.9] | <0.05 |
| Reacher | CEM | Submission, run 1 | 70.6 | 71.5 | +0.9 | [−0.7, +2.8] | 0.269 |
| Reacher | CEM | Submission, run 2 | 72.3 | 71.8 | −0.5 | [−2.2, +1.3] | 0.651 |
| Reacher | CEM | New run | 71.2 | 71.6 | +0.4 | [−1.4, +2.2] | 0.697 |
| Reacher | iCEM | Submission, run 1 | 78.0 | 81.1 | +3.1 | [+1.6, +4.6] | <0.05 |
| Reacher | iCEM | Submission, run 2 | 79.7 | 81.7 | +2.0 | [+0.6, +3.4] | <0.05 |
| Reacher | iCEM | New run | 80.0 | 78.3 | −1.7 | [−3.2, −0.1] | <0.05 |
| Cube | CEM | Submission, run 1 | 61.3 | 64.1 | +2.8 | [+1.3, +4.3] | <0.05 |
| Cube | CEM | Submission, run 2 | 60.5 | 64.0 | +3.5 | [+2.0, +5.0] | <0.05 |
| Cube | CEM | New run | 61.4 | 64.4 | +3.0 | [+1.6, +4.5] | <0.05 |
| Cube | iCEM | Submission, run 1 | 65.3 | 70.5 | +5.2 | [+3.8, +6.5] | <0.05 |
| Cube | iCEM | Submission, run 2 | 63.3 | 71.2 | +7.9 | [+6.3, +9.5] | <0.05 |
| Cube | iCEM | New run | 65.7 | 71.0 | +5.3 | [+3.9, +6.8] | <0.05 |
| Two-Room | CEM | Submission, run 1 | 77.9 | 89.2 | +11.3 | [+9.2, +13.4] | <0.05 |
| Two-Room | CEM | Submission, run 2 | 77.0 | 89.2 | +12.2 | [+10.2, +14.3] | <0.05 |
| Two-Room | CEM | New run | 77.5 | 90.1 | +12.6 | [+10.5, +14.7] | <0.05 |
| Two-Room | iCEM | Submission, run 1 | 82.9 | 92.9 | +10.0 | [+8.1, +11.8] | <0.05 |
| Two-Room | iCEM | Submission, run 2 | 82.0 | 92.6 | +10.6 | [+8.8, +12.5] | <0.05 |
| Two-Room | iCEM | New run | 82.6 | 92.9 | +10.3 | [+8.4, +12.2] | <0.05 |
| PointMaze | CEM | Submission, run 1 | 82.1 | 86.2 | +4.1 | [+2.2, +5.9] | <0.05 |
| PointMaze | CEM | Submission, run 2 | 84.8 | 83.7 | −1.1 | [−3.2, +0.8] | 0.269 |
| PointMaze | CEM | New run | 81.2 | 83.4 | +2.2 | [+0.4, +4.1] | <0.05 |
| PointMaze | iCEM | Submission, run 1 | 80.2 | 83.4 | +3.2 | [+1.2, +5.1] | <0.05 |
| PointMaze | iCEM | Submission, run 2 | 81.6 | 83.4 | +1.8 | [−0.2, +3.8] | 0.084 |
| PointMaze | iCEM | New run | 76.9 | 81.6 | +4.7 | [+2.8, +6.7] | <0.05 |
