**Table H. State-based world model vs. image-based LeWM and SCALE: planning success rate (%) by planning-budget tier.** The state-based model is LeWM with the image encoder replaced by an MLP over the complete simulator state (Push-T: agent and block position, block angle, agent velocity, 8-d; Reacher: joint angles, fingertip position, joint velocities, 8-d; Cube: end-effector pose, gripper, arm joints, block position and yaw, 22-d; Two-Room: agent position, 2-d; PointMaze: position and velocity, 4-d); the predictor, the regulariser and latent-space planning are unchanged. LeWM and SCALE take images. All numbers average two training seeds (1024, 1025). Each entry is the mean over 600 test episodes (6 sets × 100). Budgets (candidates × iterations): T1 300×30, T2 150×15, T3 50×10, T4 20×5, T5 10×3; Avg. is the mean over the five tiers.

**CEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 91.3 | 90.7 | 80.9 | 58.7 | 33.3 | 71.0 |
| Push-T | SCALE | 95.3 | 92.3 | 84.5 | 63.8 | 34.6 | 74.1 |
| Push-T | State-based | 93.1 | 90.8 | 80.4 | 56.7 | 26.2 | 69.4 |
| Reacher | LeWM | 81.2 | 78.8 | 73.5 | 64.8 | 57.0 | 71.1 |
| Reacher | SCALE | 79.9 | 80.6 | 73.3 | 66.5 | 56.1 | 71.3 |
| Reacher | State-based | 99.5 | 98.6 | 91.0 | 76.7 | 51.9 | 83.5 |
| Cube | LeWM | 68.0 | 67.0 | 60.0 | 56.2 | 55.3 | 61.3 |
| Cube | SCALE | 75.5 | 72.2 | 64.8 | 54.9 | 55.7 | 64.6 |
| Cube | State-based | 52.0 | 50.7 | 49.5 | 48.5 | 49.5 | 50.0 |
| Two-Room | LeWM | 86.8 | 85.4 | 81.7 | 71.8 | 58.2 | 76.8 |
| Two-Room | SCALE | 99.3 | 98.9 | 95.8 | 82.8 | 71.8 | 89.8 |
| Two-Room | State-based | 91.4 | 91.2 | 87.8 | 77.4 | 65.0 | 82.6 |
| PointMaze | LeWM | 83.4 | 81.9 | 80.7 | 79.3 | 76.3 | 80.3 |
| PointMaze | SCALE | 79.7 | 80.2 | 79.2 | 77.0 | 73.3 | 77.9 |
| PointMaze | State-based | 94.3 | 93.7 | 92.3 | 88.3 | 79.5 | 89.6 |

**iCEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 86.2 | 82.6 | 74.3 | 56.7 | 26.4 | 65.2 |
| Push-T | SCALE | 89.8 | 84.7 | 79.2 | 58.8 | 27.4 | 68.0 |
| Push-T | State-based | 85.1 | 82.7 | 74.3 | 53.0 | 20.6 | 63.1 |
| Reacher | LeWM | 85.4 | 84.3 | 86.3 | 79.8 | 61.2 | 79.4 |
| Reacher | SCALE | 84.8 | 82.6 | 84.2 | 78.8 | 61.7 | 78.4 |
| Reacher | State-based | 99.3 | 99.6 | 98.8 | 94.5 | 60.4 | 90.5 |
| Cube | LeWM | 68.9 | 67.8 | 69.3 | 61.8 | 56.8 | 64.9 |
| Cube | SCALE | 76.8 | 76.7 | 75.0 | 65.6 | 58.8 | 70.6 |
| Cube | State-based | 51.8 | 50.8 | 52.1 | 53.2 | 50.0 | 51.6 |
| Two-Room | LeWM | 90.3 | 92.8 | 88.2 | 77.8 | 60.8 | 82.0 |
| Two-Room | SCALE | 99.8 | 99.7 | 98.2 | 90.5 | 77.3 | 93.1 |
| Two-Room | State-based | 96.3 | 97.2 | 93.5 | 83.5 | 70.3 | 88.1 |
| PointMaze | LeWM | 74.3 | 72.8 | 73.2 | 73.4 | 73.1 | 73.4 |
| PointMaze | SCALE | 81.3 | 79.8 | 75.4 | 73.5 | 69.9 | 76.0 |
| PointMaze | State-based | 94.0 | 94.4 | 93.8 | 89.8 | 80.2 | 90.4 |

**Table I. SCALE against the state-based world model: success rate (%) averaged over the five budget tiers, with paired significance tests.** Both models are evaluated on the same 600 test episodes with identical planner noise, so outcomes are paired per episode. Δ = SCALE − state-based in percentage points (positive = SCALE better); the 95% CI is a paired t-interval over episodes; p is from an exact two-sided paired sign-flip permutation test. All numbers average two training seeds (1024, 1025).

| Task | Planner | State-based | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|
| Push-T | CEM | 69.4 | 74.1 | +4.7 | [+3.1, +6.2] | <0.0001 |
| Push-T | iCEM | 63.1 | 68.0 | +4.8 | [+3.0, +6.7] | <0.0001 |
| Reacher | CEM | 83.5 | 71.3 | −12.3 | [−14.0, −10.6] | <0.0001 |
| Reacher | iCEM | 90.5 | 78.4 | −12.1 | [−13.8, −10.4] | <0.0001 |
| Cube | CEM | 50.0 | 64.6 | +14.6 | [+12.8, +16.4] | <0.0001 |
| Cube | iCEM | 51.6 | 70.6 | +19.0 | [+16.7, +21.4] | <0.0001 |
| Two-Room | CEM | 82.6 | 89.8 | +7.2 | [+5.4, +8.9] | <0.0001 |
| Two-Room | iCEM | 88.1 | 93.1 | +5.0 | [+3.6, +6.3] | <0.0001 |
| PointMaze | CEM | 89.6 | 77.9 | −11.8 | [−13.6, −9.9] | <0.0001 |
| PointMaze | iCEM | 90.4 | 76.0 | −14.5 | [−16.5, −12.4] | <0.0001 |
