**SCALE vs. Aux: success rate (%) averaged over the five budget tiers, with paired significance tests (three runs per method).** Each method averages three independently trained runs: the two runs of the submission and one new run (seed 1025). Δ = SCALE − Aux in percentage points. Both methods are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | Aux | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 74.2 | 73.4 | −0.8 | [−1.8, +0.1] | 0.093 |
| Push-T | iCEM | 68.8 | 68.2 | −0.6 | [−1.5, +0.3] | 0.220 |
| Reacher | CEM | 72.0 | 71.6 | −0.4 | [−1.4, +0.7] | 0.513 |
| Reacher | iCEM | 80.0 | 80.4 | +0.4 | [−0.5, +1.2] | 0.408 |
| Cube | CEM | 63.2 | 64.2 | +1.0 | [+0.2, +1.8] | <0.05 |
| Cube | iCEM | 68.3 | 70.9 | +2.5 | [+1.8, +3.3] | <0.05 |
| Two-Room | CEM | 77.9 | 89.5 | +11.6 | [+9.8, +13.4] | <0.05 |
| Two-Room | iCEM | 83.2 | 92.8 | +9.6 | [+7.9, +11.2] | <0.05 |
| PointMaze | CEM | 84.1 | 84.4 | +0.3 | [−0.7, +1.3] | 0.588 |
| PointMaze | iCEM | 81.6 | 82.8 | +1.1 | [−0.0, +2.3] | 0.061 |
