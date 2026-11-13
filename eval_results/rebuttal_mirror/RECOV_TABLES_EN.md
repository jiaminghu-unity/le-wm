**Table C. Component ablation: planning success rate (%) when the SCALE loss supervises all state components, only the high-recoverability components, or only the low-recoverability components.** The state components are split by how well the LeWM baseline already encodes them (held-out linear-probe R²). Push-T: high = block position (x, y); low = agent position (x, y) and block angle (cos θ, sin θ). Cube: high = end-effector position and block position (x, y, z each); low = end-effector yaw (cos 2ψ, sin 2ψ) and gripper opening. All numbers average two training seeds (1024, 1025). Each entry is the mean over 600 test episodes (6 sets × 100). Budgets (candidates × iterations): T1 300×30, T2 150×15, T3 50×10, T4 20×5, T5 10×3; Avg. is the mean over the five tiers.

**CEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 91.3 | 90.7 | 80.9 | 58.7 | 33.3 | 71.0 |
| Push-T | SCALE (all components) | 95.3 | 92.3 | 84.5 | 63.8 | 34.6 | 74.1 |
| Push-T | SCALE (high-recoverability only) | 93.2 | 91.2 | 81.5 | 61.6 | 30.8 | 71.6 |
| Push-T | SCALE (low-recoverability only) | 94.3 | 92.6 | 83.4 | 64.6 | 32.7 | 73.5 |
| Cube | LeWM | 68.0 | 67.0 | 60.0 | 56.2 | 55.3 | 61.3 |
| Cube | SCALE (all components) | 75.5 | 72.2 | 64.8 | 54.9 | 55.7 | 64.6 |
| Cube | SCALE (high-recoverability only) | 75.0 | 71.4 | 64.4 | 57.3 | 53.9 | 64.4 |
| Cube | SCALE (low-recoverability only) | 71.8 | 69.0 | 63.8 | 55.3 | 55.2 | 63.0 |

**iCEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Push-T | LeWM | 86.2 | 82.6 | 74.3 | 56.7 | 26.4 | 65.2 |
| Push-T | SCALE (all components) | 89.8 | 84.7 | 79.2 | 58.8 | 27.4 | 68.0 |
| Push-T | SCALE (high-recoverability only) | 89.1 | 83.9 | 76.8 | 58.6 | 25.4 | 66.8 |
| Push-T | SCALE (low-recoverability only) | 90.2 | 86.3 | 79.2 | 56.8 | 25.8 | 67.7 |
| Cube | LeWM | 68.9 | 67.8 | 69.3 | 61.8 | 56.8 | 64.9 |
| Cube | SCALE (all components) | 76.8 | 76.7 | 75.0 | 65.6 | 58.8 | 70.6 |
| Cube | SCALE (high-recoverability only) | 76.3 | 75.2 | 74.1 | 64.0 | 59.0 | 69.7 |
| Cube | SCALE (low-recoverability only) | 74.8 | 73.9 | 72.3 | 63.3 | 58.3 | 68.5 |

**Table D. Component ablation: success rate (%) averaged over the five budget tiers, each SCALE variant against the LeWM baseline, with paired significance tests.** All models are evaluated on the same 600 test episodes with identical planner noise, so outcomes are paired per episode. Δ = SCALE variant − LeWM in percentage points; the 95% CI is a paired t-interval over episodes; p is from an exact two-sided paired sign-flip permutation test. All numbers average two training seeds (1024, 1025).

| Task | Planner | Supervised components | LeWM | SCALE | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|
| Push-T | CEM | All | 71.0 | 74.1 | +3.1 | [+1.9, +4.3] | <0.0001 |
| Push-T | CEM | High-recoverability only | 71.0 | 71.6 | +0.7 | [−0.5, +1.8] | 0.275 |
| Push-T | CEM | Low-recoverability only | 71.0 | 73.5 | +2.6 | [+1.3, +3.8] | <0.0001 |
| Push-T | iCEM | All | 65.2 | 68.0 | +2.7 | [+1.4, +4.0] | <0.0001 |
| Push-T | iCEM | High-recoverability only | 65.2 | 66.8 | +1.5 | [+0.4, +2.7] | 0.010 |
| Push-T | iCEM | Low-recoverability only | 65.2 | 67.7 | +2.4 | [+1.1, +3.7] | 0.0003 |
| Cube | CEM | All | 61.3 | 64.6 | +3.3 | [+2.1, +4.6] | <0.0001 |
| Cube | CEM | High-recoverability only | 61.3 | 64.4 | +3.1 | [+2.0, +4.3] | <0.0001 |
| Cube | CEM | Low-recoverability only | 61.3 | 63.0 | +1.7 | [+0.5, +2.9] | 0.0058 |
| Cube | iCEM | All | 64.9 | 70.6 | +5.7 | [+4.4, +7.0] | <0.0001 |
| Cube | iCEM | High-recoverability only | 64.9 | 69.7 | +4.8 | [+3.7, +6.0] | <0.0001 |
| Cube | iCEM | Low-recoverability only | 64.9 | 68.5 | +3.6 | [+2.4, +4.9] | <0.0001 |
