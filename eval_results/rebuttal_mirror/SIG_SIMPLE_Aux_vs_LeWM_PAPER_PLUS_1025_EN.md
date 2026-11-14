**Aux vs. LeWM: success rate (%) averaged over the five budget tiers, with paired significance tests (three runs per method).** Each method averages three independently trained runs: the two runs of the submission and one new run (seed 1025). Δ = Aux − LeWM in percentage points. Both methods are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | LeWM | Aux | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 70.8 | 74.2 | +3.4 | [+2.4, +4.4] | <0.05 |
| Push-T | iCEM | 64.1 | 68.8 | +4.6 | [+3.5, +5.8] | <0.05 |
| Reacher | CEM | 71.3 | 72.0 | +0.7 | [−0.3, +1.7] | 0.194 |
| Reacher | iCEM | 79.2 | 80.0 | +0.8 | [−0.1, +1.6] | 0.085 |
| Cube | CEM | 61.1 | 63.2 | +2.1 | [+1.1, +3.1] | <0.05 |
| Cube | iCEM | 64.8 | 68.3 | +3.6 | [+2.5, +4.6] | <0.05 |
| Two-Room | CEM | 77.5 | 77.9 | +0.4 | [−0.3, +1.2] | 0.272 |
| Two-Room | iCEM | 82.5 | 83.2 | +0.7 | [+0.0, +1.5] | 0.0501 |
| PointMaze | CEM | 82.7 | 84.1 | +1.4 | [+0.4, +2.5] | <0.05 |
| PointMaze | iCEM | 79.6 | 81.6 | +2.1 | [+0.9, +3.3] | <0.05 |
