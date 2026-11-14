**Table P. LeWM, Aux, Aux + L2, a state-based world model and SWIM on four multi-object manipulation tasks: planning success rate (%) at the five planning budgets (T1 largest … T5 smallest) and their average.** LeWM is the image-only baseline. Aux adds a head that regresses the full simulator state from the latent (weight 0.1); Aux + L2 adds a penalty β·‖W‖² on that head's weights inside the same term. State-based is LeWM with the image encoder replaced by an MLP over the full simulator state (27-d, 32-d, 37-d and 26-d for the four tasks); it sees no images. SWIM is SCALE with per-dimension state weights learned from an inverse (action-prediction) model. Every method was trained twice, with the same two training seeds (1024 and 1025). Each entry averages the two training runs over 600 test episodes (6 sets × 100); every model sees the same episodes and planner noise.

**CEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Cube-Double | LeWM | 16.7 | 17.2 | 13.8 | 12.7 | 11.8 | 14.4 |
| Cube-Double | Aux | 17.9 | 16.8 | 13.3 | 14.6 | 11.9 | 14.9 |
| Cube-Double | Aux + L2 (β = 0.01) | 18.9 | 15.9 | 15.0 | 14.6 | 11.8 | 15.2 |
| Cube-Double | Aux + L2 (β = 0.1) | 18.4 | 17.3 | 14.3 | 13.8 | 11.8 | 15.1 |
| Cube-Double | Aux + L2 (β = 1) | 15.8 | 14.8 | 14.4 | 14.5 | 12.1 | 14.3 |
| Cube-Double | State-based | 21.8 | 20.3 | 14.3 | 13.6 | 11.6 | 16.3 |
| Cube-Double | SWIM | 17.8 | 17.3 | 13.8 | 13.2 | 12.2 | 14.8 |
| Cube-Triple | LeWM | 33.0 | 30.8 | 30.4 | 29.8 | 29.1 | 30.6 |
| Cube-Triple | Aux | 33.4 | 31.7 | 29.9 | 29.5 | 30.1 | 30.9 |
| Cube-Triple | Aux + L2 (β = 0.01) | 32.6 | 31.6 | 31.3 | 29.7 | 29.1 | 30.8 |
| Cube-Triple | Aux + L2 (β = 0.1) | 31.8 | 31.2 | 29.7 | 29.2 | 30.9 | 30.6 |
| Cube-Triple | Aux + L2 (β = 1) | 31.6 | 29.8 | 30.1 | 28.8 | 30.3 | 30.1 |
| Cube-Triple | State-based | 37.2 | 33.6 | 31.3 | 29.9 | 29.0 | 32.2 |
| Cube-Triple | SWIM | 31.3 | 31.3 | 30.1 | 29.2 | 28.5 | 30.1 |
| Cube-Quadruple | LeWM | 30.0 | 28.6 | 27.8 | 28.8 | 27.7 | 28.6 |
| Cube-Quadruple | Aux | 31.6 | 29.9 | 28.6 | 28.3 | 27.5 | 29.2 |
| Cube-Quadruple | Aux + L2 (β = 0.01) | 30.1 | 29.3 | 28.7 | 29.6 | 26.3 | 28.8 |
| Cube-Quadruple | Aux + L2 (β = 0.1) | 28.9 | 28.3 | 27.4 | 27.0 | 27.6 | 27.9 |
| Cube-Quadruple | Aux + L2 (β = 1) | 28.6 | 29.1 | 27.4 | 28.1 | 27.2 | 28.1 |
| Cube-Quadruple | State-based | 36.3 | 32.8 | 30.1 | 28.2 | 27.8 | 31.0 |
| Cube-Quadruple | SWIM | 29.7 | 31.0 | 30.4 | 28.9 | 27.8 | 29.6 |
| Scene | LeWM | 31.7 | 29.3 | 27.3 | 25.2 | 18.5 | 26.4 |
| Scene | Aux | 42.8 | 38.4 | 34.8 | 28.6 | 20.8 | 33.1 |
| Scene | Aux + L2 (β = 0.01) | 41.8 | 39.2 | 33.3 | 27.4 | 21.3 | 32.6 |
| Scene | Aux + L2 (β = 0.1) | 36.8 | 34.6 | 29.4 | 25.8 | 19.3 | 29.2 |
| Scene | Aux + L2 (β = 1) | 30.3 | 27.7 | 25.8 | 23.3 | 19.5 | 25.3 |
| Scene | State-based | 37.3 | 29.3 | 22.5 | 19.8 | 15.2 | 24.8 |
| Scene | SWIM | 37.8 | 34.3 | 28.8 | 23.5 | 15.4 | 27.9 |
| Mean of four tasks | LeWM | 27.8 | 26.5 | 24.8 | 24.1 | 21.8 | 25.0 |
| Mean of four tasks | Aux | 31.4 | 29.2 | 26.6 | 25.2 | 22.6 | 27.0 |
| Mean of four tasks | Aux + L2 (β = 0.01) | 30.9 | 29.0 | 27.1 | 25.3 | 22.1 | 26.9 |
| Mean of four tasks | Aux + L2 (β = 0.1) | 29.0 | 27.8 | 25.2 | 23.9 | 22.4 | 25.7 |
| Mean of four tasks | Aux + L2 (β = 1) | 26.5 | 25.4 | 24.4 | 23.7 | 22.3 | 24.5 |
| Mean of four tasks | State-based | 33.1 | 29.0 | 24.5 | 22.9 | 20.9 | 26.1 |
| Mean of four tasks | SWIM | 29.1 | 28.4 | 25.8 | 23.7 | 21.0 | 25.6 |

**iCEM planner**

| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |
|---|---|---|---|---|---|---|---|
| Cube-Double | LeWM | 17.3 | 16.7 | 15.8 | 13.9 | 14.9 | 15.7 |
| Cube-Double | Aux | 20.8 | 18.0 | 15.3 | 15.5 | 15.2 | 17.0 |
| Cube-Double | Aux + L2 (β = 0.01) | 22.2 | 19.2 | 15.7 | 14.3 | 14.8 | 17.2 |
| Cube-Double | Aux + L2 (β = 0.1) | 19.3 | 18.0 | 15.7 | 14.2 | 14.2 | 16.3 |
| Cube-Double | Aux + L2 (β = 1) | 16.8 | 16.3 | 16.8 | 15.1 | 15.4 | 16.1 |
| Cube-Double | State-based | 30.5 | 22.8 | 17.3 | 15.3 | 14.3 | 20.0 |
| Cube-Double | SWIM | 21.9 | 19.2 | 16.6 | 13.3 | 13.5 | 16.9 |
| Cube-Triple | LeWM | 36.2 | 32.5 | 32.1 | 30.8 | 29.5 | 32.2 |
| Cube-Triple | Aux | 36.8 | 33.8 | 32.4 | 30.8 | 30.3 | 32.8 |
| Cube-Triple | Aux + L2 (β = 0.01) | 36.0 | 34.0 | 32.1 | 29.9 | 29.7 | 32.3 |
| Cube-Triple | Aux + L2 (β = 0.1) | 35.1 | 32.6 | 32.5 | 30.1 | 29.3 | 31.9 |
| Cube-Triple | Aux + L2 (β = 1) | 36.1 | 32.4 | 31.6 | 30.0 | 30.2 | 32.1 |
| Cube-Triple | State-based | 46.0 | 39.4 | 31.8 | 31.5 | 30.3 | 35.8 |
| Cube-Triple | SWIM | 34.7 | 31.7 | 30.9 | 30.6 | 29.5 | 31.5 |
| Cube-Quadruple | LeWM | 33.1 | 31.7 | 29.7 | 28.0 | 28.9 | 30.3 |
| Cube-Quadruple | Aux | 35.8 | 33.4 | 29.4 | 28.3 | 28.3 | 31.0 |
| Cube-Quadruple | Aux + L2 (β = 0.01) | 34.2 | 32.4 | 30.1 | 29.2 | 28.4 | 30.9 |
| Cube-Quadruple | Aux + L2 (β = 0.1) | 32.6 | 32.4 | 29.8 | 29.0 | 29.1 | 30.6 |
| Cube-Quadruple | Aux + L2 (β = 1) | 33.1 | 31.6 | 29.9 | 28.7 | 28.3 | 30.3 |
| Cube-Quadruple | State-based | 47.8 | 40.1 | 30.8 | 28.8 | 29.5 | 35.4 |
| Cube-Quadruple | SWIM | 34.6 | 32.4 | 30.5 | 28.6 | 29.3 | 31.1 |
| Scene | LeWM | 24.8 | 21.7 | 17.8 | 16.2 | 14.6 | 19.0 |
| Scene | Aux | 29.1 | 24.3 | 21.7 | 18.6 | 14.5 | 21.6 |
| Scene | Aux + L2 (β = 0.01) | 30.2 | 25.3 | 21.6 | 18.1 | 12.6 | 21.5 |
| Scene | Aux + L2 (β = 0.1) | 25.7 | 22.2 | 19.8 | 16.6 | 12.2 | 19.3 |
| Scene | Aux + L2 (β = 1) | 24.5 | 20.4 | 17.8 | 16.1 | 13.7 | 18.5 |
| Scene | State-based | 23.0 | 16.8 | 14.9 | 12.5 | 10.0 | 15.4 |
| Scene | SWIM | 27.2 | 22.7 | 18.0 | 15.6 | 11.0 | 18.9 |
| Mean of four tasks | LeWM | 27.8 | 25.6 | 23.9 | 22.2 | 22.0 | 24.3 |
| Mean of four tasks | Aux | 30.6 | 27.4 | 24.7 | 23.3 | 22.0 | 25.6 |
| Mean of four tasks | Aux + L2 (β = 0.01) | 30.6 | 27.7 | 24.9 | 22.9 | 21.4 | 25.5 |
| Mean of four tasks | Aux + L2 (β = 0.1) | 28.1 | 26.3 | 24.4 | 22.5 | 21.2 | 24.5 |
| Mean of four tasks | Aux + L2 (β = 1) | 27.6 | 25.2 | 24.0 | 22.5 | 21.9 | 24.2 |
| Mean of four tasks | State-based | 36.8 | 29.8 | 23.7 | 22.0 | 21.0 | 26.7 |
| Mean of four tasks | SWIM | 29.6 | 26.5 | 24.0 | 22.0 | 20.8 | 24.6 |

**Table Q. Each method against LeWM, with paired significance tests.** Δ = A − LeWM in percentage points (positive = A better than LeWM). Success is averaged over the five planning budgets and the two training runs (seeds 1024 and 1025 on both sides). Outcomes are paired per test episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test. "All four" pools the episodes of the four tasks.

| Task | Planner | A | B | A SR | B SR | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|---|
| Cube-Double | CEM | Aux | LeWM | 14.9 | 14.4 | +0.5 | [−0.4, +1.4] | 0.293 |
| Cube-Double | CEM | Aux + L2 (β = 0.01) | LeWM | 15.2 | 14.4 | +0.8 | [−0.1, +1.7] | 0.090 |
| Cube-Double | CEM | Aux + L2 (β = 0.1) | LeWM | 15.1 | 14.4 | +0.7 | [−0.2, +1.6] | 0.126 |
| Cube-Double | CEM | Aux + L2 (β = 1) | LeWM | 14.3 | 14.4 | −0.1 | [−1.0, +0.8] | 0.861 |
| Cube-Double | CEM | State-based | LeWM | 16.3 | 14.4 | +1.9 | [+0.8, +3.0] | <0.05 |
| Cube-Double | CEM | SWIM | LeWM | 14.8 | 14.4 | +0.4 | [−0.6, +1.4] | 0.428 |
| Cube-Double | iCEM | Aux | LeWM | 17.0 | 15.7 | +1.3 | [+0.4, +2.1] | <0.05 |
| Cube-Double | iCEM | Aux + L2 (β = 0.01) | LeWM | 17.2 | 15.7 | +1.5 | [+0.6, +2.3] | <0.05 |
| Cube-Double | iCEM | Aux + L2 (β = 0.1) | LeWM | 16.3 | 15.7 | +0.5 | [−0.3, +1.4] | 0.239 |
| Cube-Double | iCEM | Aux + L2 (β = 1) | LeWM | 16.1 | 15.7 | +0.4 | [−0.4, +1.2] | 0.390 |
| Cube-Double | iCEM | State-based | LeWM | 20.0 | 15.7 | +4.3 | [+3.1, +5.5] | <0.05 |
| Cube-Double | iCEM | SWIM | LeWM | 16.9 | 15.7 | +1.2 | [+0.2, +2.2] | <0.05 |
| Cube-Triple | CEM | Aux | LeWM | 30.9 | 30.6 | +0.3 | [−0.6, +1.1] | 0.538 |
| Cube-Triple | CEM | Aux + L2 (β = 0.01) | LeWM | 30.8 | 30.6 | +0.2 | [−0.6, +1.0] | 0.660 |
| Cube-Triple | CEM | Aux + L2 (β = 0.1) | LeWM | 30.6 | 30.6 | −0.1 | [−0.9, +0.7] | 0.875 |
| Cube-Triple | CEM | Aux + L2 (β = 1) | LeWM | 30.1 | 30.6 | −0.5 | [−1.3, +0.3] | 0.239 |
| Cube-Triple | CEM | State-based | LeWM | 32.2 | 30.6 | +1.6 | [+0.5, +2.6] | <0.05 |
| Cube-Triple | CEM | SWIM | LeWM | 30.1 | 30.6 | −0.6 | [−1.5, +0.3] | 0.200 |
| Cube-Triple | iCEM | Aux | LeWM | 32.8 | 32.2 | +0.6 | [−0.2, +1.5] | 0.142 |
| Cube-Triple | iCEM | Aux + L2 (β = 0.01) | LeWM | 32.3 | 32.2 | +0.1 | [−0.7, +1.0] | 0.788 |
| Cube-Triple | iCEM | Aux + L2 (β = 0.1) | LeWM | 31.9 | 32.2 | −0.3 | [−1.1, +0.5] | 0.502 |
| Cube-Triple | iCEM | Aux + L2 (β = 1) | LeWM | 32.1 | 32.2 | −0.2 | [−0.9, +0.6] | 0.702 |
| Cube-Triple | iCEM | State-based | LeWM | 35.8 | 32.2 | +3.6 | [+2.5, +4.8] | <0.05 |
| Cube-Triple | iCEM | SWIM | LeWM | 31.5 | 32.2 | −0.7 | [−1.5, +0.0] | 0.069 |
| Cube-Quadruple | CEM | Aux | LeWM | 29.2 | 28.6 | +0.6 | [−0.2, +1.4] | 0.160 |
| Cube-Quadruple | CEM | Aux + L2 (β = 0.01) | LeWM | 28.8 | 28.6 | +0.2 | [−0.6, +0.9] | 0.662 |
| Cube-Quadruple | CEM | Aux + L2 (β = 0.1) | LeWM | 27.9 | 28.6 | −0.7 | [−1.4, +0.0] | 0.070 |
| Cube-Quadruple | CEM | Aux + L2 (β = 1) | LeWM | 28.1 | 28.6 | −0.5 | [−1.2, +0.2] | 0.145 |
| Cube-Quadruple | CEM | State-based | LeWM | 31.0 | 28.6 | +2.5 | [+1.5, +3.3] | <0.05 |
| Cube-Quadruple | CEM | SWIM | LeWM | 29.6 | 28.6 | +1.0 | [+0.2, +1.8] | <0.05 |
| Cube-Quadruple | iCEM | Aux | LeWM | 31.0 | 30.3 | +0.8 | [−0.1, +1.6] | 0.076 |
| Cube-Quadruple | iCEM | Aux + L2 (β = 0.01) | LeWM | 30.9 | 30.3 | +0.6 | [−0.2, +1.4] | 0.155 |
| Cube-Quadruple | iCEM | Aux + L2 (β = 0.1) | LeWM | 30.6 | 30.3 | +0.3 | [−0.5, +1.1] | 0.479 |
| Cube-Quadruple | iCEM | Aux + L2 (β = 1) | LeWM | 30.3 | 30.3 | +0.1 | [−0.7, +0.8] | 0.931 |
| Cube-Quadruple | iCEM | State-based | LeWM | 35.4 | 30.3 | +5.1 | [+3.9, +6.2] | <0.05 |
| Cube-Quadruple | iCEM | SWIM | LeWM | 31.1 | 30.3 | +0.8 | [−0.0, +1.6] | 0.058 |
| Scene | CEM | Aux | LeWM | 33.1 | 26.4 | +6.7 | [+5.1, +8.2] | <0.05 |
| Scene | CEM | Aux + L2 (β = 0.01) | LeWM | 32.6 | 26.4 | +6.2 | [+4.8, +7.7] | <0.05 |
| Scene | CEM | Aux + L2 (β = 0.1) | LeWM | 29.2 | 26.4 | +2.8 | [+1.4, +4.1] | <0.05 |
| Scene | CEM | Aux + L2 (β = 1) | LeWM | 25.3 | 26.4 | −1.1 | [−2.2, +0.0] | 0.062 |
| Scene | CEM | State-based | LeWM | 24.8 | 26.4 | −1.6 | [−3.2, +0.0] | 0.055 |
| Scene | CEM | SWIM | LeWM | 27.9 | 26.4 | +1.6 | [+0.1, +3.0] | <0.05 |
| Scene | iCEM | Aux | LeWM | 21.6 | 19.0 | +2.6 | [+1.2, +4.0] | <0.05 |
| Scene | iCEM | Aux + L2 (β = 0.01) | LeWM | 21.5 | 19.0 | +2.5 | [+1.2, +3.9] | <0.05 |
| Scene | iCEM | Aux + L2 (β = 0.1) | LeWM | 19.3 | 19.0 | +0.3 | [−0.9, +1.5] | 0.683 |
| Scene | iCEM | Aux + L2 (β = 1) | LeWM | 18.5 | 19.0 | −0.5 | [−1.4, +0.4] | 0.288 |
| Scene | iCEM | State-based | LeWM | 15.4 | 19.0 | −3.6 | [−5.1, −2.0] | <0.05 |
| Scene | iCEM | SWIM | LeWM | 18.9 | 19.0 | −0.1 | [−1.4, +1.2] | 0.882 |
| All four | CEM | Aux | LeWM | 27.0 | 25.0 | +2.0 | [+1.5, +2.5] | <0.05 |
| All four | CEM | Aux + L2 (β = 0.01) | LeWM | 26.9 | 25.0 | +1.9 | [+1.3, +2.4] | <0.05 |
| All four | CEM | Aux + L2 (β = 0.1) | LeWM | 25.7 | 25.0 | +0.7 | [+0.2, +1.2] | <0.05 |
| All four | CEM | Aux + L2 (β = 1) | LeWM | 24.5 | 25.0 | −0.6 | [−1.0, −0.1] | <0.05 |
| All four | CEM | State-based | LeWM | 26.1 | 25.0 | +1.1 | [+0.5, +1.7] | <0.05 |
| All four | CEM | SWIM | LeWM | 25.6 | 25.0 | +0.6 | [+0.1, +1.1] | <0.05 |
| All four | iCEM | Aux | LeWM | 25.6 | 24.3 | +1.3 | [+0.8, +1.8] | <0.05 |
| All four | iCEM | Aux + L2 (β = 0.01) | LeWM | 25.5 | 24.3 | +1.2 | [+0.7, +1.7] | <0.05 |
| All four | iCEM | Aux + L2 (β = 0.1) | LeWM | 24.5 | 24.3 | +0.2 | [−0.3, +0.7] | 0.394 |
| All four | iCEM | Aux + L2 (β = 1) | LeWM | 24.2 | 24.3 | −0.1 | [−0.5, +0.3] | 0.768 |
| All four | iCEM | State-based | LeWM | 26.7 | 24.3 | +2.4 | [+1.7, +3.0] | <0.05 |
| All four | iCEM | SWIM | LeWM | 24.6 | 24.3 | +0.3 | [−0.2, +0.8] | 0.276 |

**Table R. SWIM against Aux, Aux + L2 and the state-based world model, with paired significance tests.** Δ = SWIM − B in percentage points (positive = SWIM better). Success is averaged over the five planning budgets and the two training runs (seeds 1024 and 1025 on both sides). Outcomes are paired per test episode; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test. "All four" pools the episodes of the four tasks.

| Task | Planner | A | B | A SR | B SR | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|---|
| Cube-Double | CEM | SWIM | Aux | 14.8 | 14.9 | −0.1 | [−1.0, +0.9] | 0.918 |
| Cube-Double | CEM | SWIM | Aux + L2 (β = 0.01) | 14.8 | 15.2 | −0.4 | [−1.3, +0.5] | 0.428 |
| Cube-Double | CEM | SWIM | Aux + L2 (β = 0.1) | 14.8 | 15.1 | −0.3 | [−1.3, +0.7] | 0.567 |
| Cube-Double | CEM | SWIM | Aux + L2 (β = 1) | 14.8 | 14.3 | +0.5 | [−0.5, +1.5] | 0.338 |
| Cube-Double | CEM | SWIM | State-based | 14.8 | 16.3 | −1.5 | [−2.5, −0.4] | <0.05 |
| Cube-Double | iCEM | SWIM | Aux | 16.9 | 17.0 | −0.1 | [−1.0, +0.9] | 0.918 |
| Cube-Double | iCEM | SWIM | Aux + L2 (β = 0.01) | 16.9 | 17.2 | −0.3 | [−1.2, +0.6] | 0.542 |
| Cube-Double | iCEM | SWIM | Aux + L2 (β = 0.1) | 16.9 | 16.3 | +0.7 | [−0.3, +1.6] | 0.172 |
| Cube-Double | iCEM | SWIM | Aux + L2 (β = 1) | 16.9 | 16.1 | +0.8 | [−0.2, +1.9] | 0.141 |
| Cube-Double | iCEM | SWIM | State-based | 16.9 | 20.0 | −3.1 | [−4.4, −1.8] | <0.05 |
| Cube-Triple | CEM | SWIM | Aux | 30.1 | 30.9 | −0.9 | [−1.7, −0.0] | <0.05 |
| Cube-Triple | CEM | SWIM | Aux + L2 (β = 0.01) | 30.1 | 30.8 | −0.8 | [−1.6, +0.0] | 0.070 |
| Cube-Triple | CEM | SWIM | Aux + L2 (β = 0.1) | 30.1 | 30.6 | −0.5 | [−1.4, +0.4] | 0.275 |
| Cube-Triple | CEM | SWIM | Aux + L2 (β = 1) | 30.1 | 30.1 | −0.1 | [−0.9, +0.8] | 0.878 |
| Cube-Triple | CEM | SWIM | State-based | 30.1 | 32.2 | −2.1 | [−3.1, −1.2] | <0.05 |
| Cube-Triple | iCEM | SWIM | Aux | 31.5 | 32.8 | −1.4 | [−2.2, −0.5] | <0.05 |
| Cube-Triple | iCEM | SWIM | Aux + L2 (β = 0.01) | 31.5 | 32.3 | −0.9 | [−1.7, −0.1] | <0.05 |
| Cube-Triple | iCEM | SWIM | Aux + L2 (β = 0.1) | 31.5 | 31.9 | −0.5 | [−1.3, +0.4] | 0.293 |
| Cube-Triple | iCEM | SWIM | Aux + L2 (β = 1) | 31.5 | 32.1 | −0.6 | [−1.3, +0.2] | 0.142 |
| Cube-Triple | iCEM | SWIM | State-based | 31.5 | 35.8 | −4.3 | [−5.5, −3.2] | <0.05 |
| Cube-Quadruple | CEM | SWIM | Aux | 29.6 | 29.2 | +0.4 | [−0.4, +1.2] | 0.314 |
| Cube-Quadruple | CEM | SWIM | Aux + L2 (β = 0.01) | 29.6 | 28.8 | +0.8 | [+0.0, +1.6] | 0.053 |
| Cube-Quadruple | CEM | SWIM | Aux + L2 (β = 0.1) | 29.6 | 27.9 | +1.7 | [+0.9, +2.5] | <0.05 |
| Cube-Quadruple | CEM | SWIM | Aux + L2 (β = 1) | 29.6 | 28.1 | +1.5 | [+0.7, +2.3] | <0.05 |
| Cube-Quadruple | CEM | SWIM | State-based | 29.6 | 31.0 | −1.5 | [−2.3, −0.5] | <0.05 |
| Cube-Quadruple | iCEM | SWIM | Aux | 31.1 | 31.0 | +0.0 | [−0.8, +0.8] | 0.968 |
| Cube-Quadruple | iCEM | SWIM | Aux + L2 (β = 0.01) | 31.1 | 30.9 | +0.2 | [−0.6, +1.0] | 0.626 |
| Cube-Quadruple | iCEM | SWIM | Aux + L2 (β = 0.1) | 31.1 | 30.6 | +0.5 | [−0.3, +1.3] | 0.251 |
| Cube-Quadruple | iCEM | SWIM | Aux + L2 (β = 1) | 31.1 | 30.3 | +0.8 | [−0.1, +1.6] | 0.086 |
| Cube-Quadruple | iCEM | SWIM | State-based | 31.1 | 35.4 | −4.3 | [−5.5, −3.1] | <0.05 |
| Scene | CEM | SWIM | Aux | 27.9 | 33.1 | −5.1 | [−6.4, −3.8] | <0.05 |
| Scene | CEM | SWIM | Aux + L2 (β = 0.01) | 27.9 | 32.6 | −4.7 | [−6.0, −3.4] | <0.05 |
| Scene | CEM | SWIM | Aux + L2 (β = 0.1) | 27.9 | 29.2 | −1.2 | [−2.5, −0.0] | 0.051 |
| Scene | CEM | SWIM | Aux + L2 (β = 1) | 27.9 | 25.3 | +2.7 | [+1.1, +4.2] | <0.05 |
| Scene | CEM | SWIM | State-based | 27.9 | 24.8 | +3.1 | [+1.7, +4.6] | <0.05 |
| Scene | iCEM | SWIM | Aux | 18.9 | 21.6 | −2.7 | [−3.7, −1.7] | <0.05 |
| Scene | iCEM | SWIM | Aux + L2 (β = 0.01) | 18.9 | 21.5 | −2.7 | [−3.6, −1.7] | <0.05 |
| Scene | iCEM | SWIM | Aux + L2 (β = 0.1) | 18.9 | 19.3 | −0.4 | [−1.4, +0.6] | 0.458 |
| Scene | iCEM | SWIM | Aux + L2 (β = 1) | 18.9 | 18.5 | +0.4 | [−1.0, +1.7] | 0.594 |
| Scene | iCEM | SWIM | State-based | 18.9 | 15.4 | +3.5 | [+2.1, +4.8] | <0.05 |
| All four | CEM | SWIM | Aux | 25.6 | 27.0 | −1.4 | [−1.9, −0.9] | <0.05 |
| All four | CEM | SWIM | Aux + L2 (β = 0.01) | 25.6 | 26.9 | −1.3 | [−1.8, −0.8] | <0.05 |
| All four | CEM | SWIM | Aux + L2 (β = 0.1) | 25.6 | 25.7 | −0.1 | [−0.6, +0.4] | 0.750 |
| All four | CEM | SWIM | Aux + L2 (β = 1) | 25.6 | 24.5 | +1.1 | [+0.6, +1.7] | <0.05 |
| All four | CEM | SWIM | State-based | 25.6 | 26.1 | −0.5 | [−1.0, +0.1] | 0.103 |
| All four | iCEM | SWIM | Aux | 24.6 | 25.6 | −1.0 | [−1.5, −0.6] | <0.05 |
| All four | iCEM | SWIM | Aux + L2 (β = 0.01) | 24.6 | 25.5 | −0.9 | [−1.3, −0.5] | <0.05 |
| All four | iCEM | SWIM | Aux + L2 (β = 0.1) | 24.6 | 24.5 | +0.1 | [−0.4, +0.5] | 0.738 |
| All four | iCEM | SWIM | Aux + L2 (β = 1) | 24.6 | 24.2 | +0.3 | [−0.2, +0.9] | 0.192 |
| All four | iCEM | SWIM | State-based | 24.6 | 26.7 | −2.1 | [−2.7, −1.4] | <0.05 |

**Table S. SWIM against LeWM over every training run that was evaluated at all five planning budgets.** Δ = SWIM − LeWM in percentage points. Same statistics as Table Q, with success averaged over these training seeds (the same seeds on both sides) — Cube-Double: 1024, 1025, 3072, 3073, 3074; Cube-Triple: 1024, 1025, 3072, 3073, 3074; Cube-Quadruple: 1024, 1025, 3072, 3073, 3074; Scene: 1024, 1025, 3072, 3074.

| Task | Planner | A | B | A SR | B SR | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|---|
| Cube-Double | CEM | SWIM | LeWM | 14.6 | 14.5 | +0.1 | [−0.5, +0.7] | 0.790 |
| Cube-Double | iCEM | SWIM | LeWM | 16.6 | 15.8 | +0.7 | [+0.0, +1.5] | <0.05 |
| Cube-Triple | CEM | SWIM | LeWM | 30.4 | 30.3 | +0.1 | [−0.5, +0.7] | 0.749 |
| Cube-Triple | iCEM | SWIM | LeWM | 32.1 | 32.0 | +0.1 | [−0.4, +0.7] | 0.667 |
| Cube-Quadruple | CEM | SWIM | LeWM | 29.2 | 28.5 | +0.6 | [+0.1, +1.1] | <0.05 |
| Cube-Quadruple | iCEM | SWIM | LeWM | 30.7 | 30.6 | +0.1 | [−0.5, +0.6] | 0.774 |
| Scene | CEM | SWIM | LeWM | 27.9 | 25.9 | +2.0 | [+0.8, +3.2] | <0.05 |
| Scene | iCEM | SWIM | LeWM | 18.7 | 18.8 | −0.2 | [−1.4, +1.0] | 0.759 |
| All four | CEM | SWIM | LeWM | 25.5 | 24.8 | +0.7 | [+0.3, +1.1] | <0.05 |
| All four | iCEM | SWIM | LeWM | 24.5 | 24.3 | +0.2 | [−0.2, +0.6] | 0.344 |
