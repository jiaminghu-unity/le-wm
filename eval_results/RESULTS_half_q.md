# Reduced-q ablation

Both `L_obj` and the aux q-head train on privileged physical state q. This round
withholds roughly half of q and retrains each arm end to end, every other
hyperparameter unchanged, to ask how much of the gain needs the withheld half.

Both losses are invariant to q's dimension, so "same hyperparameters" is literally
true: the aux loss averages over dimensions (`train.py:83`) and q is z-scored per
component, and `L_obj = 1 - Pearson(||dz||^2, ||dq||^2)` is scale-free. The reduced
variants are strict coordinate subsets of the full ones, verified elementwise on the
training set, and their z-score statistics equal the full variant's restricted to the
kept indices (`scripts/prep_half_qstats.py`).

Six pre-registered episode seeds s101-s106, all reported. The baseline arm is absent
from the retraining because it never consumes q.

## Push-T  (q 6 -> 4: kept block xy + cos/sin theta; dropped pusher xy)

Success rate %, mean over 4 solvers x 5 tiers, +- SD across the six seeds.

| arm | SR | SD | cem | icem | mppi | gd | T1 | T2 | T3 | T4 | T5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | **65.15** | 2.77 | 70.5 | 63.9 | 55.2 | 71.0 | 84.6 | 80.2 | 70.8 | 54.7 | 35.4 |
| L_obj | **69.23** | 2.47 | 73.9 | 68.2 | 60.6 | 74.2 | 88.9 | 84.7 | 76.0 | 60.1 | 36.4 |
| aux q-head | **69.86** | 2.98 | 74.9 | 69.3 | 59.8 | 75.4 | 88.6 | 85.8 | 75.8 | 62.0 | 37.1 |
| L_obj half | **67.92** | 2.02 | 72.3 | 67.3 | 59.2 | 72.9 | 87.6 | 83.3 | 74.7 | 59.2 | 34.7 |
| aux half | **66.90** | 2.58 | 72.0 | 66.3 | 55.9 | 73.4 | 85.0 | 82.0 | 73.6 | 59.0 | 34.8 |

Per-seed paired differences (n=6, Wilcoxon signed-rank):

| contrast | delta pp | effect | p |
|---|---|---|---|
| L_obj - baseline | +4.08 | 8.1 sigma | 0.031* |
| L_obj half - baseline | +2.78 | 5.7 sigma | 0.031* |
| L_obj half - L_obj | -1.31 | 2.5 sigma | 0.062 |
| aux q-head - baseline | +4.71 | 9.3 sigma | 0.031* |
| aux half - baseline | +1.75 | 3.4 sigma | 0.031* |
| aux half - aux q-head | -2.96 | 5.0 sigma | 0.031* |

## Reacher  (q 4 -> 2: kept cos/sin of joint 0; dropped joint 1)

Success rate %, mean over 4 solvers x 5 tiers, +- SD across the six seeds.

| arm | SR | SD | cem | icem | mppi | gd | T1 | T2 | T3 | T4 | T5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | **64.27** | 0.96 | 70.6 | 78.0 | 47.1 | 61.4 | 67.3 | 69.8 | 67.2 | 62.6 | 54.5 |
| L_obj | **65.75** | 1.11 | 71.5 | 81.1 | 46.9 | 63.4 | 69.8 | 71.8 | 68.0 | 63.2 | 56.0 |
| aux q-head | **65.11** | 1.55 | 70.4 | 80.2 | 46.4 | 63.4 | 67.4 | 72.8 | 67.2 | 63.8 | 54.2 |
| L_obj half | **66.17** | 1.73 | 72.5 | 80.9 | 47.4 | 63.9 | 69.3 | 72.4 | 69.0 | 63.5 | 56.7 |
| aux half | **63.58** | 1.41 | 70.0 | 77.1 | 47.0 | 60.2 | 67.1 | 69.7 | 65.8 | 61.0 | 54.4 |

Per-seed paired differences (n=6, Wilcoxon signed-rank):

| contrast | delta pp | effect | p |
|---|---|---|---|
| L_obj - baseline | +1.48 | 5.0 sigma | 0.031* |
| L_obj half - baseline | +1.91 | 3.4 sigma | 0.062 |
| L_obj half - L_obj | +0.42 | 0.7 sigma | 0.562 |
| aux q-head - baseline | +0.84 | 2.1 sigma | 0.094 |
| aux half - baseline | -0.68 | 1.6 sigma | 0.156 |
| aux half - aux q-head | -1.53 | 5.4 sigma | 0.031* |

## OGBench Cube  (q 9 -> 5: kept effector xyz + cos/sin 2psi; dropped gripper and block xyz)

Success rate %, mean over 4 solvers x 5 tiers, +- SD across the six seeds.

| arm | SR | SD | cem | icem | mppi | gd | T1 | T2 | T3 | T4 | T5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | **59.64** | 4.09 | 61.4 | 65.2 | 49.5 | 62.5 | 64.0 | 62.2 | 60.1 | 57.8 | 54.1 |
| L_obj | **62.55** | 3.51 | 64.2 | 70.5 | 50.0 | 65.5 | 69.8 | 66.8 | 62.6 | 58.5 | 55.0 |
| aux q-head | **61.99** | 3.06 | 63.5 | 68.6 | 50.4 | 65.5 | 69.0 | 65.6 | 62.2 | 58.2 | 55.0 |
| L_obj half | **62.65** | 3.67 | 65.2 | 69.9 | 50.8 | 64.7 | 70.1 | 66.1 | 63.1 | 58.5 | 55.5 |
| aux half | **61.57** | 3.23 | 63.9 | 67.7 | 50.3 | 64.3 | 69.1 | 65.0 | 61.8 | 57.2 | 54.7 |

Per-seed paired differences (n=6, Wilcoxon signed-rank):

| contrast | delta pp | effect | p |
|---|---|---|---|
| L_obj - baseline | +2.91 | 5.0 sigma | 0.031* |
| L_obj half - baseline | +3.01 | 5.0 sigma | 0.031* |
| L_obj half - L_obj | +0.10 | 0.9 sigma | 0.438 |
| aux q-head - baseline | +2.35 | 3.4 sigma | 0.031* |
| aux half - baseline | +1.92 | 3.6 sigma | 0.031* |
| aux half - aux q-head | -0.42 | 2.5 sigma | 0.125 |

## Reading

- `L_obj` needs the withheld half only on Push-T (-1.31 pp, p=0.062). Reacher and
  Cube lose nothing (+0.42, p=0.562; +0.10, p=0.438). What was kept in those two is
  the AGENT's own state (joints, effector pose); Push-T is the one task where the
  agent -- the pusher -- was the half removed.
- The aux head shows the opposite pattern: it cares about how much q there is, not
  which coordinates. Push-T -2.96 pp and Reacher -1.53 pp, both significant; Cube
  unchanged. Reacher's q went from 4-d to 2-d and the arm fell below baseline.
- On Cube the 5-d q matches the 9-d one, and what was dropped is the cube whose
  position the success test measures. The retained coordinates are readable from the
  robot's own encoders; the dropped ones need object perception.

**Limit.** One training seed per arm; the six seeds are evaluation-episode seeds.
