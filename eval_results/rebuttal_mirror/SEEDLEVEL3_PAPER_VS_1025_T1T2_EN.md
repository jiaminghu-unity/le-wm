**Seed-level results, SCALE vs. LeWM: success rate (%) averaged over the two largest planning budgets (tiers T1 and T2), separately for each independently trained run.** Runs 1 and 2 are the two training runs of the original submission; the new run is one additional independently trained run with a new seed. Δ = SCALE − LeWM in percentage points. Within each run both methods share the training seed and are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | Training runs | LeWM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | Submission, run 1 | 91.4 | 94.6 | +3.2 | [+1.5, +4.8] | <0.05 |
| Push-T | CEM | Submission, run 2 | 90.2 | 92.3 | +2.1 | [+0.5, +3.7] | <0.05 |
| Push-T | CEM | New run | 90.4 | 93.8 | +3.4 | [+1.5, +5.3] | <0.05 |
| Push-T | iCEM | Submission, run 1 | 83.4 | 88.3 | +4.9 | [+2.8, +7.0] | <0.05 |
| Push-T | iCEM | Submission, run 2 | 82.4 | 87.1 | +4.7 | [+2.6, +6.8] | <0.05 |
| Push-T | iCEM | New run | 83.4 | 87.6 | +4.2 | [+2.0, +6.4] | <0.05 |
| Reacher | CEM | Submission, run 1 | 80.3 | 80.6 | +0.3 | [−2.6, +3.3] | 0.867 |
| Reacher | CEM | Submission, run 2 | 81.4 | 80.8 | −0.6 | [−3.3, +2.0] | 0.666 |
| Reacher | CEM | New run | 80.8 | 79.8 | −1.0 | [−3.7, +1.8] | 0.553 |
| Reacher | iCEM | Submission, run 1 | 83.7 | 86.4 | +2.7 | [+0.4, +5.1] | <0.05 |
| Reacher | iCEM | Submission, run 2 | 85.5 | 87.2 | +1.7 | [−0.3, +3.9] | 0.119 |
| Reacher | iCEM | New run | 86.2 | 83.9 | −2.3 | [−4.4, −0.0] | 0.052 |
| Cube | CEM | Submission, run 1 | 67.2 | 72.4 | +5.2 | [+2.6, +7.7] | <0.05 |
| Cube | CEM | Submission, run 2 | 66.1 | 72.6 | +6.5 | [+3.9, +9.1] | <0.05 |
| Cube | CEM | New run | 68.2 | 74.0 | +5.8 | [+3.5, +8.2] | <0.05 |
| Cube | iCEM | Submission, run 1 | 69.7 | 76.6 | +6.9 | [+4.9, +9.0] | <0.05 |
| Cube | iCEM | Submission, run 2 | 66.3 | 76.8 | +10.5 | [+8.0, +12.9] | <0.05 |
| Cube | iCEM | New run | 69.3 | 76.8 | +7.5 | [+5.4, +9.6] | <0.05 |
| Two-Room | CEM | Submission, run 1 | 87.8 | 99.6 | +11.8 | [+9.3, +14.2] | <0.05 |
| Two-Room | CEM | Submission, run 2 | 87.0 | 99.4 | +12.4 | [+10.0, +14.9] | <0.05 |
| Two-Room | CEM | New run | 87.5 | 99.2 | +11.7 | [+9.3, +14.2] | <0.05 |
| Two-Room | iCEM | Submission, run 1 | 92.3 | 99.9 | +7.6 | [+5.6, +9.6] | <0.05 |
| Two-Room | iCEM | Submission, run 2 | 91.7 | 99.8 | +8.1 | [+6.0, +10.2] | <0.05 |
| Two-Room | iCEM | New run | 91.8 | 99.7 | +7.9 | [+5.8, +9.9] | <0.05 |
| PointMaze | CEM | Submission, run 1 | 83.3 | 88.2 | +4.9 | [+2.5, +7.4] | <0.05 |
| PointMaze | CEM | Submission, run 2 | 88.3 | 86.8 | −1.5 | [−4.3, +1.1] | 0.273 |
| PointMaze | CEM | New run | 83.8 | 86.4 | +2.6 | [+0.1, +5.2] | <0.05 |
| PointMaze | iCEM | Submission, run 1 | 82.2 | 85.0 | +2.8 | [+0.2, +5.4] | <0.05 |
| PointMaze | iCEM | Submission, run 2 | 85.4 | 88.3 | +2.9 | [+0.2, +5.6] | <0.05 |
| PointMaze | iCEM | New run | 80.5 | 86.5 | +6.0 | [+3.5, +8.5] | <0.05 |
