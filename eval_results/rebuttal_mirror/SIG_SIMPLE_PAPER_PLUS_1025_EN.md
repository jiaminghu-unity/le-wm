**SCALE vs. LeWM: success rate (%) averaged over the five budget tiers, with paired significance tests (three runs per method).** Each method averages three independently trained runs: the two runs of the submission and one new run (seed 1025). Δ = SCALE − LeWM in percentage points. Both methods are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | LeWM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 70.8 | 73.4 | +2.6 | [+1.6, +3.6] | <0.05 |
| Push-T | iCEM | 64.1 | 68.2 | +4.0 | [+3.0, +5.1] | <0.05 |
| Reacher | CEM | 71.3 | 71.6 | +0.3 | [−0.7, +1.4] | 0.541 |
| Reacher | iCEM | 79.2 | 80.4 | +1.1 | [+0.2, +2.1] | <0.05 |
| Cube | CEM | 61.1 | 64.2 | +3.1 | [+2.1, +4.2] | <0.05 |
| Cube | iCEM | 64.8 | 70.9 | +6.1 | [+4.9, +7.4] | <0.05 |
| Two-Room | CEM | 77.5 | 89.5 | +12.0 | [+10.2, +13.9] | <0.05 |
| Two-Room | iCEM | 82.5 | 92.8 | +10.3 | [+8.6, +12.0] | <0.05 |
| PointMaze | CEM | 82.7 | 84.4 | +1.7 | [+0.6, +2.8] | <0.05 |
| PointMaze | iCEM | 79.6 | 82.8 | +3.2 | [+2.0, +4.4] | <0.05 |
