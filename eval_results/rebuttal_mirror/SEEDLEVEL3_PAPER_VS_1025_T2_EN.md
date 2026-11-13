**Seed-level results, SCALE vs. LeWM: success rate (%) at the second-largest planning budget (tier T2), separately for each independently trained run.** Runs 1 and 2 are the two training runs of the original submission; the new run is one additional independently trained run with a new seed. Δ = SCALE − LeWM in percentage points. Within each run both methods share the training seed and are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | Training runs | LeWM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | Submission, run 1 | 90.5 | 93.2 | +2.7 | [+0.5, +4.9] | <0.05 |
| Push-T | CEM | Submission, run 2 | 88.2 | 91.0 | +2.8 | [+0.6, +5.1] | <0.05 |
| Push-T | CEM | New run | 90.8 | 92.3 | +1.5 | [−1.0, +4.0] | 0.289 |
| Push-T | iCEM | Submission, run 1 | 80.7 | 87.0 | +6.3 | [+3.6, +9.1] | <0.05 |
| Push-T | iCEM | Submission, run 2 | 79.0 | 85.7 | +6.7 | [+3.9, +9.4] | <0.05 |
| Push-T | iCEM | New run | 81.5 | 85.3 | +3.8 | [+1.0, +6.7] | <0.05 |
| Reacher | CEM | Submission, run 1 | 78.8 | 78.7 | −0.1 | [−4.3, +4.3] | 1.000 |
| Reacher | CEM | Submission, run 2 | 80.0 | 79.5 | −0.5 | [−4.3, +3.3] | 0.864 |
| Reacher | CEM | New run | 78.7 | 81.3 | +2.6 | [−1.4, +6.8] | 0.233 |
| Reacher | iCEM | Submission, run 1 | 81.8 | 86.2 | +4.4 | [+1.1, +7.5] | <0.05 |
| Reacher | iCEM | Submission, run 2 | 86.7 | 87.8 | +1.1 | [−1.6, +3.9] | 0.470 |
| Reacher | iCEM | New run | 86.2 | 82.5 | −3.7 | [−6.6, −0.7] | <0.05 |
| Cube | CEM | Submission, run 1 | 66.8 | 71.3 | +4.5 | [+1.4, +7.7] | <0.05 |
| Cube | CEM | Submission, run 2 | 64.7 | 70.8 | +6.1 | [+3.1, +9.3] | <0.05 |
| Cube | CEM | New run | 67.0 | 71.8 | +4.8 | [+1.9, +7.8] | <0.05 |
| Cube | iCEM | Submission, run 1 | 68.5 | 76.5 | +8.0 | [+5.6, +10.4] | <0.05 |
| Cube | iCEM | Submission, run 2 | 65.7 | 76.7 | +11.0 | [+8.3, +13.8] | <0.05 |
| Cube | iCEM | New run | 68.5 | 76.7 | +8.2 | [+5.7, +10.6] | <0.05 |
| Two-Room | CEM | Submission, run 1 | 87.7 | 99.3 | +11.6 | [+9.0, +14.3] | <0.05 |
| Two-Room | CEM | Submission, run 2 | 86.5 | 99.3 | +12.8 | [+10.1, +15.6] | <0.05 |
| Two-Room | CEM | New run | 87.3 | 99.2 | +11.9 | [+9.1, +14.6] | <0.05 |
| Two-Room | iCEM | Submission, run 1 | 93.3 | 99.8 | +6.5 | [+4.5, +8.5] | <0.05 |
| Two-Room | iCEM | Submission, run 2 | 92.5 | 99.7 | +7.2 | [+5.1, +9.3] | <0.05 |
| Two-Room | iCEM | New run | 93.2 | 99.5 | +6.3 | [+4.3, +8.4] | <0.05 |
| PointMaze | CEM | Submission, run 1 | 83.7 | 87.3 | +3.6 | [+0.7, +6.6] | <0.05 |
| PointMaze | CEM | Submission, run 2 | 88.3 | 87.5 | −0.8 | [−3.9, +2.2] | 0.668 |
| PointMaze | CEM | New run | 83.8 | 87.3 | +3.5 | [+0.5, +6.5] | <0.05 |
| PointMaze | iCEM | Submission, run 1 | 82.7 | 82.7 | +0.0 | [−3.0, +3.0] | 1.000 |
| PointMaze | iCEM | Submission, run 2 | 84.5 | 88.8 | +4.3 | [+1.1, +7.5] | <0.05 |
| PointMaze | iCEM | New run | 79.3 | 86.0 | +6.7 | [+3.6, +9.7] | <0.05 |
