**Table J. SCALE vs. LeWM across training seeds: success rate (%) averaged over the five budget tiers and 600 test episodes.** Five training seeds per method (four on Two-Room): the two runs pooled in the paper, one additional earlier run, and two new runs (seeds 1024 and 1025). Δ = SCALE − LeWM, computed within each training seed on the same test episodes with identical planner noise; the 95% CI is a t-interval over training seeds and p is from a two-sided paired t-test over training seeds.

| Task | Planner | LeWM | SCALE | Δ | 95% CI | p-value | Seeds with Δ > 0 |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | 70.8 | 73.4 | +2.7 | [+1.6, +3.7] | 0.0019 | 5/5 |
| Push-T | iCEM | 64.5 | 67.8 | +3.3 | [+2.0, +4.7] | 0.0021 | 5/5 |
| Reacher | CEM | 71.3 | 71.3 | +0.1 | [−0.7, +0.9] | 0.792 | 3/5 |
| Reacher | iCEM | 78.9 | 79.9 | +0.9 | [−1.4, +3.3] | 0.339 | 3/5 |
| Cube | CEM | 61.1 | 64.3 | +3.2 | [+2.9, +3.6] | <0.0001 | 5/5 |
| Cube | iCEM | 64.5 | 70.7 | +6.2 | [+4.8, +7.6] | 0.0002 | 5/5 |
| Two-Room | CEM | 77.1 | 89.5 | +12.4 | [+11.0, +13.8] | <0.0001 | 4/4 |
| Two-Room | iCEM | 82.2 | 92.9 | +10.7 | [+9.3, +12.1] | 0.0001 | 4/4 |
| PointMaze | CEM | 81.7 | 81.1 | −0.6 | [−5.9, +4.8] | 0.782 | 2/5 |
| PointMaze | iCEM | 76.2 | 79.0 | +2.7 | [+0.7, +4.7] | 0.019 | 5/5 |

**Table K. SCALE vs. LeWM across test episodes, all training seeds pooled.** Same runs as Table J. Each of the 600 test episodes is evaluated with both methods under identical planner noise; the per-episode difference is averaged over the training seeds and the five budget tiers. The 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test.

| Task | Planner | LeWM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 70.8 | 73.4 | +2.7 | [+1.8, +3.5] | <0.0001 |
| Push-T | iCEM | 64.5 | 67.8 | +3.3 | [+2.4, +4.3] | <0.0001 |
| Reacher | CEM | 71.3 | 71.3 | +0.1 | [−0.8, +0.9] | 0.842 |
| Reacher | iCEM | 78.9 | 79.9 | +0.9 | [+0.2, +1.7] | 0.018 |
| Cube | CEM | 61.1 | 64.3 | +3.2 | [+2.2, +4.3] | <0.0001 |
| Cube | iCEM | 64.5 | 70.7 | +6.2 | [+5.0, +7.4] | <0.0001 |
| Two-Room | CEM | 77.1 | 89.5 | +12.4 | [+10.5, +14.3] | <0.0001 |
| Two-Room | iCEM | 82.2 | 92.9 | +10.7 | [+9.0, +12.4] | <0.0001 |
| PointMaze | CEM | 81.7 | 81.1 | −0.6 | [−1.6, +0.4] | 0.271 |
| PointMaze | iCEM | 76.2 | 79.0 | +2.7 | [+1.6, +3.9] | <0.0001 |
