**Table P. Aux, Aux + L2 and a state-based world model on four multi-object manipulation tasks: planning success rate (%) at the largest planning budget (T1: 300 candidates × 30 iterations).** LeWM is the image-only baseline. Aux adds a head that regresses the full simulator state from the latent (weight 0.1); Aux + L2 adds a penalty β·‖W‖² on that head's weights inside the same term. SCALE (full state) aligns latent distances with distances in the full simulator state. SWIM is SCALE with per-dimension state weights learned from an inverse (action-prediction) model. State-based is LeWM with the image encoder replaced by an MLP over the full simulator state (27-d, 32-d, 37-d and 26-d for the four tasks); it sees no images. LeWM, SCALE and SWIM: training seeds 3072, 3073, 3074, 3075, 3076; Aux, Aux + L2 and State-based: training seeds 1024 and 1025. Each entry averages the method's training runs over 600 test episodes (6 sets × 100); every model sees the same episodes and planner noise.

**CEM planner**

| Method | Runs | Cube-Double | Cube-Triple | Cube-Quadruple | Scene | Mean |
|---|---|---|---|---|---|---|
| LeWM | 5 | 16.4 | 30.6 | 28.3 | 30.5 | 26.4 |
| Aux | 1/2 | 18.0 | 33.4 | 30.0 | 42.8 | 31.0 |
| Aux + L2 (β = 0.01) | 1/2 | 18.9 | 32.6 | 30.1 | 40.2 | 30.4 |
| Aux + L2 (β = 0.1) | 1/2 | 18.4 | 31.3 | 28.9 | 36.8 | 28.9 |
| Aux + L2 (β = 1) | 1/2 | 15.8 | 31.6 | 28.7 | 30.3 | 26.6 |
| State-based | 2 | 21.8 | 37.2 | 36.3 | 37.3 | 33.1 |
| SCALE (full state) | 5 | 18.2 | 31.3 | 29.6 | 33.4 | 28.2 |
| SWIM | 5 | 18.2 | 32.5 | 30.7 | 36.5 | 29.4 |

**iCEM planner**

| Method | Runs | Cube-Double | Cube-Triple | Cube-Quadruple | Scene | Mean |
|---|---|---|---|---|---|---|
| LeWM | 5 | 18.0 | 35.3 | 33.9 | 24.6 | 27.9 |
| Aux | 1/2 | 20.0 | 36.8 | 35.3 | 29.1 | 30.3 |
| Aux + L2 (β = 0.01) | 1/2 | 22.2 | 36.0 | 34.2 | 29.2 | 30.4 |
| Aux + L2 (β = 0.1) | 1/2 | 19.3 | 34.3 | 32.6 | 25.7 | 28.0 |
| Aux + L2 (β = 1) | 1/2 | 16.8 | 36.1 | 34.3 | 24.5 | 27.9 |
| State-based | 2 | 30.5 | 46.0 | 47.8 | 23.0 | 36.8 |
| SCALE (full state) | 5 | 19.5 | 36.0 | 34.7 | 26.3 | 29.1 |
| SWIM | 5 | 21.1 | 36.6 | 35.2 | 26.7 | 29.9 |

**Table Q. SWIM against Aux, Aux + L2 and the state-based world model, with paired significance tests.** Δ = SWIM − baseline in percentage points (positive = SWIM better). Outcomes are paired per test episode; each method's per-episode outcome is first averaged over its training runs. The 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test. The training seeds of the two sides differ (see Table P), so the test covers test-episode variation only. "All four" pools the episodes of the four tasks.

| Task | Planner | Baseline | Baseline SR | SWIM SR | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|
| Cube-Double | CEM | Aux | 18.0 | 18.2 | +0.2 | [−2.2, +2.5] | 0.912 |
| Cube-Double | CEM | Aux + L2 (β = 0.01) | 18.9 | 18.2 | −0.8 | [−2.5, +1.0] | 0.410 |
| Cube-Double | CEM | Aux + L2 (β = 0.1) | 18.4 | 18.2 | −0.3 | [−2.0, +1.5] | 0.795 |
| Cube-Double | CEM | Aux + L2 (β = 1) | 15.8 | 18.2 | +2.4 | [+0.7, +4.2] | 0.0068 |
| Cube-Double | CEM | State-based | 21.8 | 18.2 | −3.6 | [−5.7, −1.5] | 0.0010 |
| Cube-Double | iCEM | Aux | 20.0 | 21.1 | +1.1 | [−1.2, +3.5] | 0.353 |
| Cube-Double | iCEM | Aux + L2 (β = 0.01) | 22.2 | 21.1 | −1.0 | [−2.9, +0.9] | 0.297 |
| Cube-Double | iCEM | Aux + L2 (β = 0.1) | 19.3 | 21.1 | +1.9 | [+0.2, +3.6] | 0.033 |
| Cube-Double | iCEM | Aux + L2 (β = 1) | 16.8 | 21.1 | +4.3 | [+2.5, +6.1] | <0.0001 |
| Cube-Double | iCEM | State-based | 30.5 | 21.1 | −9.4 | [−12.0, −6.7] | <0.0001 |
| Cube-Triple | CEM | Aux | 33.4 | 32.5 | −1.0 | [−2.7, +0.8] | 0.295 |
| Cube-Triple | CEM | Aux + L2 (β = 0.01) | 32.6 | 32.5 | −0.1 | [−1.8, +1.5] | 0.905 |
| Cube-Triple | CEM | Aux + L2 (β = 0.1) | 31.3 | 32.5 | +1.1 | [−0.9, +3.2] | 0.294 |
| Cube-Triple | CEM | Aux + L2 (β = 1) | 31.6 | 32.5 | +0.9 | [−0.8, +2.6] | 0.308 |
| Cube-Triple | CEM | State-based | 37.2 | 32.5 | −4.7 | [−6.9, −2.6] | <0.0001 |
| Cube-Triple | iCEM | Aux | 36.8 | 36.6 | −0.2 | [−2.0, +1.5] | 0.806 |
| Cube-Triple | iCEM | Aux + L2 (β = 0.01) | 36.0 | 36.6 | +0.6 | [−1.3, +2.5] | 0.570 |
| Cube-Triple | iCEM | Aux + L2 (β = 0.1) | 34.3 | 36.6 | +2.3 | [+0.1, +4.4] | 0.042 |
| Cube-Triple | iCEM | Aux + L2 (β = 1) | 36.1 | 36.6 | +0.5 | [−1.3, +2.4] | 0.586 |
| Cube-Triple | iCEM | State-based | 46.0 | 36.6 | −9.4 | [−12.3, −6.6] | <0.0001 |
| Cube-Quadruple | CEM | Aux | 30.0 | 30.7 | +0.7 | [−1.3, +2.6] | 0.548 |
| Cube-Quadruple | CEM | Aux + L2 (β = 0.01) | 30.1 | 30.7 | +0.6 | [−1.0, +2.1] | 0.460 |
| Cube-Quadruple | CEM | Aux + L2 (β = 0.1) | 28.9 | 30.7 | +1.8 | [+0.3, +3.2] | 0.022 |
| Cube-Quadruple | CEM | Aux + L2 (β = 1) | 28.7 | 30.7 | +2.0 | [+0.0, +4.1] | 0.048 |
| Cube-Quadruple | CEM | State-based | 36.3 | 30.7 | −5.6 | [−7.6, −3.4] | <0.0001 |
| Cube-Quadruple | iCEM | Aux | 35.3 | 35.2 | −0.2 | [−2.5, +2.1] | 0.887 |
| Cube-Quadruple | iCEM | Aux + L2 (β = 0.01) | 34.2 | 35.2 | +1.0 | [−0.6, +2.6] | 0.245 |
| Cube-Quadruple | iCEM | Aux + L2 (β = 0.1) | 32.6 | 35.2 | +2.6 | [+0.9, +4.2] | 0.0029 |
| Cube-Quadruple | iCEM | Aux + L2 (β = 1) | 34.3 | 35.2 | +0.8 | [−1.6, +3.2] | 0.529 |
| Cube-Quadruple | iCEM | State-based | 47.8 | 35.2 | −12.6 | [−15.4, −9.8] | <0.0001 |
| Scene | CEM | Aux | 42.8 | 36.5 | −6.3 | [−8.8, −3.8] | <0.0001 |
| Scene | CEM | Aux + L2 (β = 0.01) | 40.2 | 36.5 | −3.7 | [−6.8, −0.6] | 0.022 |
| Scene | CEM | Aux + L2 (β = 0.1) | 36.8 | 36.5 | −0.3 | [−2.6, +2.0] | 0.819 |
| Scene | CEM | Aux + L2 (β = 1) | 30.3 | 36.5 | +6.2 | [+3.7, +8.7] | <0.0001 |
| Scene | CEM | State-based | 37.3 | 36.5 | −0.9 | [−3.7, +2.0] | 0.560 |
| Scene | iCEM | Aux | 29.1 | 26.7 | −2.4 | [−4.0, −0.7] | 0.0043 |
| Scene | iCEM | Aux + L2 (β = 0.01) | 29.2 | 26.7 | −2.4 | [−4.6, −0.3] | 0.026 |
| Scene | iCEM | Aux + L2 (β = 0.1) | 25.7 | 26.7 | +1.1 | [−0.5, +2.7] | 0.195 |
| Scene | iCEM | Aux + L2 (β = 1) | 24.5 | 26.7 | +2.2 | [+0.1, +4.3] | 0.038 |
| Scene | iCEM | State-based | 23.0 | 26.7 | +3.7 | [+1.3, +6.2] | 0.0031 |
| All four | CEM | Aux | 31.0 | 29.4 | −1.6 | [−2.7, −0.5] | 0.0038 |
| All four | CEM | Aux + L2 (β = 0.01) | 30.4 | 29.4 | −1.0 | [−2.1, +0.1] | 0.066 |
| All four | CEM | Aux + L2 (β = 0.1) | 28.9 | 29.4 | +0.6 | [−0.4, +1.5] | 0.239 |
| All four | CEM | Aux + L2 (β = 1) | 26.6 | 29.4 | +2.9 | [+1.9, +3.9] | <0.0001 |
| All four | CEM | State-based | 33.1 | 29.4 | −3.7 | [−4.8, −2.5] | <0.0001 |
| All four | iCEM | Aux | 30.3 | 29.9 | −0.4 | [−1.4, +0.6] | 0.427 |
| All four | iCEM | Aux + L2 (β = 0.01) | 30.4 | 29.9 | −0.5 | [−1.4, +0.5] | 0.321 |
| All four | iCEM | Aux + L2 (β = 0.1) | 28.0 | 29.9 | +2.0 | [+1.0, +2.8] | <0.0001 |
| All four | iCEM | Aux + L2 (β = 1) | 27.9 | 29.9 | +2.0 | [+0.9, +3.0] | 0.0002 |
| All four | iCEM | State-based | 36.8 | 29.9 | −6.9 | [−8.3, −5.6] | <0.0001 |

**Table R. SCALE (full state) against Aux, Aux + L2 and the state-based world model, with paired significance tests.** Δ = SCALE (full state) − baseline in percentage points (positive = SCALE (full state) better). Outcomes are paired per test episode; each method's per-episode outcome is first averaged over its training runs. The 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired sign-flip permutation test. The training seeds of the two sides differ (see Table P), so the test covers test-episode variation only. "All four" pools the episodes of the four tasks.

| Task | Planner | Baseline | Baseline SR | SCALE (full state) SR | Δ | 95% CI | p-value |
|---|---|---|---|---|---|---|---|
| Cube-Double | CEM | Aux | 18.0 | 18.2 | +0.2 | [−2.2, +2.7] | 0.872 |
| Cube-Double | CEM | Aux + L2 (β = 0.01) | 18.9 | 18.2 | −0.7 | [−2.5, +1.2] | 0.482 |
| Cube-Double | CEM | Aux + L2 (β = 0.1) | 18.4 | 18.2 | −0.2 | [−2.0, +1.7] | 0.861 |
| Cube-Double | CEM | Aux + L2 (β = 1) | 15.8 | 18.2 | +2.5 | [+0.7, +4.3] | 0.0062 |
| Cube-Double | CEM | State-based | 21.8 | 18.2 | −3.5 | [−5.7, −1.3] | 0.0019 |
| Cube-Double | iCEM | Aux | 20.0 | 19.5 | −0.5 | [−2.6, +1.6] | 0.665 |
| Cube-Double | iCEM | Aux + L2 (β = 0.01) | 22.2 | 19.5 | −2.7 | [−4.6, −0.7] | 0.0068 |
| Cube-Double | iCEM | Aux + L2 (β = 0.1) | 19.3 | 19.5 | +0.3 | [−1.4, +1.9] | 0.783 |
| Cube-Double | iCEM | Aux + L2 (β = 1) | 16.8 | 19.5 | +2.7 | [+1.0, +4.4] | 0.0024 |
| Cube-Double | iCEM | State-based | 30.5 | 19.5 | −11.0 | [−13.6, −8.4] | <0.0001 |
| Cube-Triple | CEM | Aux | 33.4 | 31.3 | −2.1 | [−3.9, −0.3] | 0.022 |
| Cube-Triple | CEM | Aux + L2 (β = 0.01) | 32.6 | 31.3 | −1.3 | [−2.8, +0.3] | 0.129 |
| Cube-Triple | CEM | Aux + L2 (β = 0.1) | 31.3 | 31.3 | +0.0 | [−2.1, +2.1] | 1.000 |
| Cube-Triple | CEM | Aux + L2 (β = 1) | 31.6 | 31.3 | −0.3 | [−2.0, +1.5] | 0.792 |
| Cube-Triple | CEM | State-based | 37.2 | 31.3 | −5.8 | [−8.0, −3.7] | <0.0001 |
| Cube-Triple | iCEM | Aux | 36.8 | 36.0 | −0.9 | [−2.6, +0.9] | 0.332 |
| Cube-Triple | iCEM | Aux + L2 (β = 0.01) | 36.0 | 36.0 | −0.0 | [−1.8, +1.7] | 0.955 |
| Cube-Triple | iCEM | Aux + L2 (β = 0.1) | 34.3 | 36.0 | +1.6 | [−0.4, +3.7] | 0.133 |
| Cube-Triple | iCEM | Aux + L2 (β = 1) | 36.1 | 36.0 | −0.1 | [−1.9, +1.7] | 0.914 |
| Cube-Triple | iCEM | State-based | 46.0 | 36.0 | −10.0 | [−12.7, −7.5] | <0.0001 |
| Cube-Quadruple | CEM | Aux | 30.0 | 29.6 | −0.4 | [−2.4, +1.6] | 0.717 |
| Cube-Quadruple | CEM | Aux + L2 (β = 0.01) | 30.1 | 29.6 | −0.5 | [−2.0, +1.1] | 0.580 |
| Cube-Quadruple | CEM | Aux + L2 (β = 0.1) | 28.9 | 29.6 | +0.7 | [−0.8, +2.1] | 0.370 |
| Cube-Quadruple | CEM | Aux + L2 (β = 1) | 28.7 | 29.6 | +1.0 | [−1.0, +3.0] | 0.329 |
| Cube-Quadruple | CEM | State-based | 36.3 | 29.6 | −6.6 | [−8.6, −4.5] | <0.0001 |
| Cube-Quadruple | iCEM | Aux | 35.3 | 34.7 | −0.6 | [−3.0, +1.7] | 0.612 |
| Cube-Quadruple | iCEM | Aux + L2 (β = 0.01) | 34.2 | 34.7 | +0.6 | [−1.0, +2.0] | 0.502 |
| Cube-Quadruple | iCEM | Aux + L2 (β = 0.1) | 32.6 | 34.7 | +2.2 | [+0.4, +3.9] | 0.017 |
| Cube-Quadruple | iCEM | Aux + L2 (β = 1) | 34.3 | 34.7 | +0.4 | [−1.9, +2.6] | 0.771 |
| Cube-Quadruple | iCEM | State-based | 47.8 | 34.7 | −13.0 | [−15.8, −10.3] | <0.0001 |
| Scene | CEM | Aux | 42.8 | 33.4 | −9.4 | [−11.8, −6.9] | <0.0001 |
| Scene | CEM | Aux + L2 (β = 0.01) | 40.2 | 33.4 | −6.8 | [−9.9, −3.6] | <0.0001 |
| Scene | CEM | Aux + L2 (β = 0.1) | 36.8 | 33.4 | −3.4 | [−5.5, −1.2] | 0.0030 |
| Scene | CEM | Aux + L2 (β = 1) | 30.3 | 33.4 | +3.2 | [+0.7, +5.6] | 0.011 |
| Scene | CEM | State-based | 37.3 | 33.4 | −3.9 | [−6.7, −1.1] | 0.0059 |
| Scene | iCEM | Aux | 29.1 | 26.3 | −2.8 | [−4.4, −1.1] | 0.0015 |
| Scene | iCEM | Aux + L2 (β = 0.01) | 29.2 | 26.3 | −2.8 | [−4.9, −0.7] | 0.0083 |
| Scene | iCEM | Aux + L2 (β = 0.1) | 25.7 | 26.3 | +0.7 | [−0.9, +2.2] | 0.409 |
| Scene | iCEM | Aux + L2 (β = 1) | 24.5 | 26.3 | +1.8 | [−0.1, +3.8] | 0.066 |
| Scene | iCEM | State-based | 23.0 | 26.3 | +3.3 | [+0.9, +5.7] | 0.0063 |
| All four | CEM | Aux | 31.0 | 28.2 | −2.9 | [−4.0, −1.8] | <0.0001 |
| All four | CEM | Aux + L2 (β = 0.01) | 30.4 | 28.2 | −2.3 | [−3.4, −1.2] | <0.0001 |
| All four | CEM | Aux + L2 (β = 0.1) | 28.9 | 28.2 | −0.7 | [−1.7, +0.3] | 0.150 |
| All four | CEM | Aux + L2 (β = 1) | 26.6 | 28.2 | +1.6 | [+0.6, +2.6] | 0.0017 |
| All four | CEM | State-based | 33.1 | 28.2 | −5.0 | [−6.1, −3.8] | <0.0001 |
| All four | iCEM | Aux | 30.3 | 29.1 | −1.2 | [−2.2, −0.2] | 0.019 |
| All four | iCEM | Aux + L2 (β = 0.01) | 30.4 | 29.1 | −1.2 | [−2.2, −0.3] | 0.0070 |
| All four | iCEM | Aux + L2 (β = 0.1) | 28.0 | 29.1 | +1.2 | [+0.3, +2.0] | 0.0097 |
| All four | iCEM | Aux + L2 (β = 1) | 27.9 | 29.1 | +1.2 | [+0.2, +2.2] | 0.016 |
| All four | iCEM | State-based | 36.8 | 29.1 | −7.7 | [−9.0, −6.4] | <0.0001 |
