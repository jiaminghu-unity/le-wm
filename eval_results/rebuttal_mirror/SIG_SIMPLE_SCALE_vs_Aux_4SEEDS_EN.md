**SCALE vs. Aux: success rate (%) averaged over the five budget tiers, with paired significance tests (four runs per method).** Each method averages four independently trained runs: the two runs of the submission and two new runs (seeds 1024 and 1025). Δ = SCALE − Aux in percentage points. Both methods are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | Aux | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 74.2 | 73.6 | −0.6 | [−1.5, +0.3] | 0.199 |
| Push-T | iCEM | 68.9 | 68.1 | −0.8 | [−1.7, +0.1] | 0.080 |
| Reacher | CEM | 72.0 | 71.5 | −0.5 | [−1.5, +0.4] | 0.252 |
| Reacher | iCEM | 79.8 | 79.9 | +0.1 | [−0.7, +0.9] | 0.856 |
| Cube | CEM | 63.3 | 64.3 | +1.0 | [+0.2, +1.8] | <0.05 |
| Cube | iCEM | 68.5 | 70.7 | +2.2 | [+1.4, +2.9] | <0.05 |
| Two-Room | CEM | 77.8 | 89.5 | +11.7 | [+9.9, +13.5] | <0.05 |
| Two-Room | iCEM | 83.1 | 92.9 | +9.9 | [+8.2, +11.5] | <0.05 |
| PointMaze | CEM | 84.5 | 81.4 | −3.1 | [−4.1, −2.1] | <0.05 |
| PointMaze | iCEM | 82.0 | 79.7 | −2.3 | [−3.4, −1.2] | <0.05 |
