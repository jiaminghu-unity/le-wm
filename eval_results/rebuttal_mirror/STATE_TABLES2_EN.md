**Table H2. State-based world model vs. image-based LeWM, SCALE and DINO-WM: planning success rate (%) by planning-budget tier.** The state-based model is LeWM with the image encoder replaced by an MLP over the complete simulator state (Push-T 8-d, Reacher 8-d, Cube 22-d, Two-Room 2-d, PointMaze 4-d); the predictor, the regulariser and latent-space planning are unchanged. LeWM, SCALE and DINO-WM take images. State-based, LeWM and SCALE average two training seeds; DINO-WM averages the two runs of the submission. Each entry is the mean over 600 test episodes (6 sets × 100) per run; Avg. is the mean over the five tiers.

**CEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 91.3 | 90.7 | 80.9 | 58.7 | 33.3 | 71.0 |
| Push-T | SCALE | 95.3 | 92.3 | 84.5 | 63.8 | 34.6 | 74.1 |
| Push-T | DINO-WM | 74.8 | 72.1 | 65.3 | 46.1 | 24.3 | 56.5 |
| Push-T | State-based | 93.1 | 90.8 | 80.4 | 56.7 | 26.2 | 69.4 |
| Reacher | LeWM | 81.2 | 78.8 | 73.5 | 64.8 | 57.0 | 71.1 |
| Reacher | SCALE | 79.9 | 80.6 | 73.3 | 66.5 | 56.1 | 71.3 |
| Reacher | DINO-WM | 77.2 | 74.8 | 70.8 | 61.8 | 50.3 | 67.0 |
| Reacher | State-based | 99.5 | 98.6 | 91.0 | 76.7 | 51.9 | 83.5 |
| Cube | LeWM | 68.0 | 67.0 | 60.0 | 56.2 | 55.3 | 61.3 |
| Cube | SCALE | 75.5 | 72.2 | 64.8 | 54.9 | 55.7 | 64.6 |
| Cube | DINO-WM | 73.4 | 66.9 | 58.4 | 51.6 | 53.9 | 60.9 |
| Cube | State-based | 52.0 | 50.7 | 49.5 | 48.5 | 49.5 | 50.0 |
| Two-Room | LeWM | 86.8 | 85.4 | 81.7 | 71.8 | 58.2 | 76.8 |
| Two-Room | SCALE | 99.3 | 98.9 | 95.8 | 82.8 | 71.8 | 89.8 |
| Two-Room | DINO-WM | 100.0 | 99.8 | 98.3 | 94.1 | 85.8 | 95.6 |
| Two-Room | State-based | 91.4 | 91.2 | 87.8 | 77.4 | 65.0 | 82.6 |
| PointMaze | LeWM | 83.4 | 81.9 | 80.7 | 79.3 | 76.3 | 80.3 |
| PointMaze | SCALE | 79.7 | 80.2 | 79.2 | 77.0 | 73.3 | 77.9 |
| PointMaze | DINO-WM | 98.2 | 98.3 | 97.2 | 92.4 | 87.2 | 94.7 |
| PointMaze | State-based | 94.3 | 93.7 | 92.3 | 88.3 | 79.5 | 89.6 |

**iCEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 86.2 | 82.6 | 74.3 | 56.7 | 26.4 | 65.2 |
| Push-T | SCALE | 89.8 | 84.7 | 79.2 | 58.8 | 27.4 | 68.0 |
| Push-T | DINO-WM | 70.0 | 65.3 | 58.8 | 41.5 | 23.1 | 51.7 |
| Push-T | State-based | 85.1 | 82.7 | 74.3 | 53.0 | 20.6 | 63.1 |
| Reacher | LeWM | 85.4 | 84.3 | 86.3 | 79.8 | 61.2 | 79.4 |
| Reacher | SCALE | 84.8 | 82.6 | 84.2 | 78.8 | 61.7 | 78.4 |
| Reacher | DINO-WM | 78.7 | 78.2 | 80.8 | 76.0 | 53.0 | 73.3 |
| Reacher | State-based | 99.3 | 99.6 | 98.8 | 94.5 | 60.4 | 90.5 |
| Cube | LeWM | 68.9 | 67.8 | 69.3 | 61.8 | 56.8 | 64.9 |
| Cube | SCALE | 76.8 | 76.7 | 75.0 | 65.6 | 58.8 | 70.6 |
| Cube | DINO-WM | 69.9 | 67.4 | 66.0 | 57.7 | 51.4 | 62.5 |
| Cube | State-based | 51.8 | 50.8 | 52.1 | 53.2 | 50.0 | 51.6 |
| Two-Room | LeWM | 90.3 | 92.8 | 88.2 | 77.8 | 60.8 | 82.0 |
| Two-Room | SCALE | 99.8 | 99.7 | 98.2 | 90.5 | 77.3 | 93.1 |
| Two-Room | DINO-WM | 100.0 | 100.0 | 99.8 | 98.6 | 92.5 | 98.2 |
| Two-Room | State-based | 96.3 | 97.2 | 93.5 | 83.5 | 70.3 | 88.1 |
| PointMaze | LeWM | 74.3 | 72.8 | 73.2 | 73.4 | 73.1 | 73.4 |
| PointMaze | SCALE | 81.3 | 79.8 | 75.4 | 73.5 | 69.9 | 76.0 |
| PointMaze | DINO-WM | 93.4 | 92.2 | 90.8 | 90.3 | 85.3 | 90.4 |
| PointMaze | State-based | 94.0 | 94.4 | 93.8 | 89.8 | 80.2 | 90.4 |

**Table I2. Image-based world models against the state-based world model: success rate (%) averaged over the five budget tiers, with paired significance tests.** All models are evaluated on the same 600 test episodes with identical planner noise, so outcomes are paired per episode. Δ = method − state-based in percentage points (positive = the image-based method is better); the 95% CI is a paired t-interval over episodes; p is from an exact two-sided paired sign-flip permutation test. State-based, LeWM and SCALE average two training seeds; DINO-WM averages the two runs of the submission.

| Task | Planner | Method | State-based | Method success | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | LeWM | 69.4 | 71.0 | +1.6 | [−0.1, +3.2] | 0.061 |
| Push-T | CEM | SCALE | 69.4 | 74.1 | +4.7 | [+3.1, +6.2] | <0.05 |
| Push-T | CEM | DINO-WM | 69.4 | 56.5 | −12.9 | [−14.9, −10.9] | <0.05 |
| Push-T | iCEM | LeWM | 63.1 | 65.2 | +2.1 | [+0.2, +4.1] | <0.05 |
| Push-T | iCEM | SCALE | 63.1 | 68.0 | +4.8 | [+3.0, +6.7] | <0.05 |
| Push-T | iCEM | DINO-WM | 63.1 | 51.7 | −11.4 | [−13.6, −9.2] | <0.05 |
| Reacher | CEM | LeWM | 83.5 | 71.1 | −12.5 | [−14.2, −10.8] | <0.05 |
| Reacher | CEM | SCALE | 83.5 | 71.3 | −12.3 | [−14.0, −10.6] | <0.05 |
| Reacher | CEM | DINO-WM | 83.5 | 67.0 | −16.6 | [−18.4, −14.7] | <0.05 |
| Reacher | iCEM | LeWM | 90.5 | 79.4 | −11.2 | [−12.8, −9.5] | <0.05 |
| Reacher | iCEM | SCALE | 90.5 | 78.4 | −12.1 | [−13.8, −10.4] | <0.05 |
| Reacher | iCEM | DINO-WM | 90.5 | 73.3 | −17.2 | [−19.0, −15.4] | <0.05 |
| Cube | CEM | LeWM | 50.0 | 61.3 | +11.3 | [+9.6, +12.9] | <0.05 |
| Cube | CEM | SCALE | 50.0 | 64.6 | +14.6 | [+12.8, +16.4] | <0.05 |
| Cube | CEM | DINO-WM | 50.0 | 60.9 | +10.8 | [+9.2, +12.5] | <0.05 |
| Cube | iCEM | LeWM | 51.6 | 64.9 | +13.3 | [+11.2, +15.5] | <0.05 |
| Cube | iCEM | SCALE | 51.6 | 70.6 | +19.0 | [+16.7, +21.4] | <0.05 |
| Cube | iCEM | DINO-WM | 51.6 | 62.5 | +10.9 | [+8.1, +13.8] | <0.05 |
| Two-Room | CEM | LeWM | 82.6 | 76.8 | −5.8 | [−7.7, −4.0] | <0.05 |
| Two-Room | CEM | SCALE | 82.6 | 89.8 | +7.2 | [+5.4, +8.9] | <0.05 |
| Two-Room | CEM | DINO-WM | 82.6 | 95.6 | +13.0 | [+11.1, +15.0] | <0.05 |
| Two-Room | iCEM | LeWM | 88.1 | 82.0 | −6.2 | [−7.9, −4.5] | <0.05 |
| Two-Room | iCEM | SCALE | 88.1 | 93.1 | +5.0 | [+3.6, +6.3] | <0.05 |
| Two-Room | iCEM | DINO-WM | 88.1 | 98.2 | +10.0 | [+8.5, +11.6] | <0.05 |
| PointMaze | CEM | LeWM | 89.6 | 80.3 | −9.3 | [−11.1, −7.5] | <0.05 |
| PointMaze | CEM | SCALE | 89.6 | 77.9 | −11.8 | [−13.6, −9.9] | <0.05 |
| PointMaze | CEM | DINO-WM | 89.6 | 94.7 | +5.0 | [+3.7, +6.3] | <0.05 |
| PointMaze | iCEM | LeWM | 90.4 | 73.4 | −17.1 | [−19.2, −14.9] | <0.05 |
| PointMaze | iCEM | SCALE | 90.4 | 76.0 | −14.5 | [−16.5, −12.4] | <0.05 |
| PointMaze | iCEM | DINO-WM | 90.4 | 90.4 | −0.1 | [−1.7, +1.6] | 0.969 |
