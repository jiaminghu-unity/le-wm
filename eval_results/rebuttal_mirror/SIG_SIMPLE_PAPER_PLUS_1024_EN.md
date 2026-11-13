**SCALE vs. LeWM: success rate (%) averaged over the five budget tiers, with paired significance tests (three runs per method).** Each method averages three independently trained runs: the two runs of the submission and one new run (seed 1024). Δ = SCALE − LeWM in percentage points. Both methods are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | LeWM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 70.8 | 73.5 | +2.7 | [+1.6, +3.7] | <0.05 |
| Push-T | iCEM | 64.6 | 68.2 | +3.6 | [+2.6, +4.7] | <0.05 |
| Reacher | CEM | 71.3 | 71.5 | +0.2 | [−0.9, +1.3] | 0.714 |
| Reacher | iCEM | 78.8 | 80.4 | +1.6 | [+0.7, +2.5] | <0.05 |
| Cube | CEM | 61.0 | 64.3 | +3.3 | [+2.2, +4.4] | <0.05 |
| Cube | iCEM | 64.2 | 70.6 | +6.4 | [+5.1, +7.6] | <0.05 |
| Two-Room | CEM | 77.0 | 89.3 | +12.3 | [+10.4, +14.2] | <0.05 |
| Two-Room | iCEM | 82.1 | 92.9 | +10.8 | [+9.1, +12.6] | <0.05 |
| PointMaze | CEM | 82.1 | 80.7 | −1.4 | [−2.7, −0.2] | <0.05 |
| PointMaze | iCEM | 77.2 | 79.0 | +1.8 | [+0.5, +3.2] | <0.05 |
