**SCALE vs. Aux: success rate (%) averaged over the five budget tiers, with paired significance tests (three runs per method).** Each method averages three independently trained runs: the two runs of the submission and one new run (seed 1024). Δ = SCALE − Aux in percentage points. Both methods are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | Aux | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 74.2 | 73.5 | −0.7 | [−1.7, +0.2] | 0.136 |
| Push-T | iCEM | 69.2 | 68.2 | −1.0 | [−2.0, −0.0] | <0.05 |
| Reacher | CEM | 71.6 | 71.5 | −0.2 | [−1.2, +0.9] | 0.768 |
| Reacher | iCEM | 79.7 | 80.4 | +0.7 | [−0.2, +1.6] | 0.131 |
| Cube | CEM | 63.4 | 64.3 | +0.9 | [+0.1, +1.7] | <0.05 |
| Cube | iCEM | 68.6 | 70.6 | +2.0 | [+1.2, +2.8] | <0.05 |
| Two-Room | CEM | 77.6 | 89.3 | +11.7 | [+9.8, +13.6] | <0.05 |
| Two-Room | iCEM | 83.0 | 92.9 | +9.9 | [+8.2, +11.6] | <0.05 |
| PointMaze | CEM | 83.1 | 80.7 | −2.4 | [−3.7, −1.0] | <0.05 |
| PointMaze | iCEM | 80.1 | 79.0 | −1.0 | [−2.5, +0.4] | 0.162 |
