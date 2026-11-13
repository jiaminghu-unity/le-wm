**SCALE vs. LeWM: success rate (%) averaged over the five budget tiers, with paired significance tests (four runs per method).** Each method averages four independently trained runs: the two runs of the submission and two new runs (seeds 1024 and 1025). Δ = SCALE − LeWM in percentage points. Both methods are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so outcomes are paired per episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | LeWM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 70.9 | 73.6 | +2.8 | [+1.8, +3.7] | <0.05 |
| Push-T | iCEM | 64.6 | 68.1 | +3.6 | [+2.5, +4.6] | <0.05 |
| Reacher | CEM | 71.2 | 71.5 | +0.3 | [−0.7, +1.2] | 0.598 |
| Reacher | iCEM | 79.1 | 79.9 | +0.8 | [−0.0, +1.6] | 0.054 |
| Cube | CEM | 61.1 | 64.3 | +3.2 | [+2.2, +4.3] | <0.05 |
| Cube | iCEM | 64.6 | 70.7 | +6.1 | [+4.9, +7.3] | <0.05 |
| Two-Room | CEM | 77.1 | 89.5 | +12.4 | [+10.5, +14.3] | <0.05 |
| Two-Room | iCEM | 82.2 | 92.9 | +10.7 | [+9.0, +12.4] | <0.05 |
| PointMaze | CEM | 81.9 | 81.4 | −0.5 | [−1.5, +0.5] | 0.340 |
| PointMaze | iCEM | 77.1 | 79.7 | +2.6 | [+1.4, +3.7] | <0.05 |
