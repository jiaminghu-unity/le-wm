**SCALE vs. Aux: success rate (%) averaged over the five budget tiers, with paired significance tests (two runs per method).** Each method averages two independently trained runs. Δ = SCALE − Aux in percentage points. Both methods are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | Aux | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 74.3 | 73.1 | −1.2 | [−2.3, −0.1] | <0.05 |
| Push-T | iCEM | 69.1 | 68.3 | −0.8 | [−1.8, +0.3] | 0.169 |
| Reacher | CEM | 71.4 | 71.7 | +0.3 | [−0.9, +1.5] | 0.652 |
| Reacher | iCEM | 80.0 | 81.4 | +1.4 | [+0.4, +2.5] | <0.05 |
| Cube | CEM | 63.3 | 64.1 | +0.8 | [−0.2, +1.7] | 0.118 |
| Cube | iCEM | 68.4 | 70.8 | +2.4 | [+1.6, +3.3] | <0.05 |
| Two-Room | CEM | 77.7 | 89.2 | +11.5 | [+9.6, +13.5] | <0.05 |
| Two-Room | iCEM | 83.3 | 92.7 | +9.4 | [+7.8, +11.2] | <0.05 |
| PointMaze | CEM | 81.9 | 84.9 | +3.0 | [+1.5, +4.6] | <0.05 |
| PointMaze | iCEM | 78.7 | 83.4 | +4.7 | [+3.0, +6.4] | <0.05 |
