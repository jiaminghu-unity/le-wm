**Aux vs. LeWM: success rate (%) averaged over the five budget tiers, with paired significance tests (two runs per method).** Each method averages two independently trained runs. Δ = Aux − LeWM in percentage points. Both methods are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | LeWM | Aux | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 70.7 | 74.3 | +3.6 | [+2.4, +4.7] | <0.05 |
| Push-T | iCEM | 63.9 | 69.1 | +5.2 | [+3.9, +6.4] | <0.05 |
| Reacher | CEM | 71.4 | 71.4 | +0.0 | [−1.2, +1.2] | 1.000 |
| Reacher | iCEM | 78.8 | 80.0 | +1.2 | [+0.0, +2.2] | <0.05 |
| Cube | CEM | 60.9 | 63.3 | +2.4 | [+1.2, +3.5] | <0.05 |
| Cube | iCEM | 64.3 | 68.4 | +4.1 | [+3.0, +5.2] | <0.05 |
| Two-Room | CEM | 77.5 | 77.7 | +0.2 | [−0.7, +1.3] | 0.626 |
| Two-Room | iCEM | 82.5 | 83.3 | +0.8 | [−0.1, +1.8] | 0.089 |
| PointMaze | CEM | 83.5 | 81.9 | −1.6 | [−3.0, −0.2] | <0.05 |
| PointMaze | iCEM | 80.9 | 78.7 | −2.2 | [−3.8, −0.6] | <0.05 |
