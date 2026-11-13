**Seed-level results, SCALE vs. LeWM: success rate (%) averaged over the five budget tiers.** "Submission" pools the two training runs of the original submission, as in the paper; "New run" is one additional independently trained run with a new seed, reported on its own. Δ = SCALE − LeWM in percentage points. Within each training seed both methods are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | Training runs | LeWM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | Submission (2 runs) | 70.7 | 73.1 | +2.4 | [+1.2, +3.6] | <0.05 |
| Push-T | CEM | New run | 71.0 | 74.0 | +3.0 | [+1.4, +4.6] | <0.05 |
| Push-T | iCEM | Submission (2 runs) | 63.9 | 68.3 | +4.4 | [+3.2, +5.6] | <0.05 |
| Push-T | iCEM | New run | 64.6 | 67.9 | +3.3 | [+1.7, +4.9] | <0.05 |
| Reacher | CEM | Submission (2 runs) | 71.4 | 71.7 | +0.3 | [−1.0, +1.6] | 0.664 |
| Reacher | CEM | New run | 71.2 | 71.6 | +0.4 | [−1.4, +2.2] | 0.697 |
| Reacher | iCEM | Submission (2 runs) | 78.9 | 81.4 | +2.6 | [+1.5, +3.7] | <0.05 |
| Reacher | iCEM | New run | 80.0 | 78.3 | −1.7 | [−3.2, −0.1] | <0.05 |
| Cube | CEM | Submission (2 runs) | 60.9 | 64.1 | +3.1 | [+1.9, +4.3] | <0.05 |
| Cube | CEM | New run | 61.4 | 64.4 | +3.1 | [+1.6, +4.5] | <0.05 |
| Cube | iCEM | Submission (2 runs) | 64.3 | 70.8 | +6.5 | [+5.3, +7.8] | <0.05 |
| Cube | iCEM | New run | 65.7 | 71.0 | +5.3 | [+3.9, +6.8] | <0.05 |
| Two-Room | CEM | Submission (2 runs) | 77.5 | 89.2 | +11.8 | [+9.9, +13.7] | <0.05 |
| Two-Room | CEM | New run | 77.5 | 90.1 | +12.6 | [+10.5, +14.7] | <0.05 |
| Two-Room | iCEM | Submission (2 runs) | 82.5 | 92.7 | +10.3 | [+8.6, +12.0] | <0.05 |
| Two-Room | iCEM | New run | 82.6 | 92.9 | +10.3 | [+8.4, +12.2] | <0.05 |
| PointMaze | CEM | Submission (2 runs) | 83.5 | 84.9 | +1.5 | [+0.1, +2.8] | <0.05 |
| PointMaze | CEM | New run | 81.2 | 83.4 | +2.3 | [+0.4, +4.1] | <0.05 |
| PointMaze | iCEM | Submission (2 runs) | 80.9 | 83.4 | +2.5 | [+1.1, +3.9] | <0.05 |
| PointMaze | iCEM | New run | 76.9 | 81.6 | +4.7 | [+2.8, +6.7] | <0.05 |
