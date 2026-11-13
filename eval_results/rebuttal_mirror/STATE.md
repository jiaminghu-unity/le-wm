# State-based world model vs pixel arms (training seeds 3072 / 1024 / 1025)

State model = LeWM with the image encoder replaced by an MLP over the complete simulator state;
predictor, regulariser and latent-space planning are unchanged. Success % = mean over T1-T5 and
six 100-episode sets. Each cell: CEM / iCEM. Differences are within the training seed; the last
column is their mean with a 95% t-interval over the three seeds. `...` = not finished yet.
State-model cells complete: 14 / 30.

## Push-T

State fed to the model: 8-d: agent and block position, block angle (cos/sin), agent velocity.

| | 3072 | 1024 | 1025 | mean |
|---|---|---|---|---|
| LeWM | 70.5 / 63.9 | 70.9 / 65.9 | 71.0 / 64.6 | 70.8 / 64.8 |
| SCALE | 73.9 / 68.2 | 74.2 / 68.0 | 74.0 / 67.9 | 74.0 / 68.0 |
| Aux | 74.9 / 69.3 | 74.0 / 69.4 | 74.1 / 68.2 | 74.3 / 69.0 |
| State | ... / ... | ... / ... | ... / ... | ... / ... |
| State − LeWM | ... / ... | ... / ... | ... / ... | ... / ... |
| State − SCALE | ... / ... | ... / ... | ... / ... | ... / ... |

## Reacher

State fed to the model: 8-d: joint angles (cos/sin), fingertip position, joint velocities.

| | 3072 | 1024 | 1025 | mean |
|---|---|---|---|---|
| LeWM | 70.6 / 78.0 | 70.9 / 78.7 | 71.2 / 80.0 | 70.9 / 78.9 |
| SCALE | 71.5 / 81.1 | 71.0 / 78.5 | 71.6 / 78.3 | 71.4 / 79.3 |
| Aux | 70.4 / 80.2 | 72.1 / 79.3 | 73.3 / 80.2 | 71.9 / 79.9 |
| State | ... / ... | ... / ... | ... / ... | ... / ... |
| State − LeWM | ... / ... | ... / ... | ... / ... | ... / ... |
| State − SCALE | ... / ... | ... / ... | ... / ... | ... / ... |

## Cube

State fed to the model: 22-d: effector pose, gripper, arm joints, block position and yaw.

| | 3072 | 1024 | 1025 | mean |
|---|---|---|---|---|
| LeWM | 61.3 / 65.3 | 61.2 / 64.1 | 61.4 / 65.7 | 61.3 / 65.0 |
| SCALE | 64.1 / 70.5 | 64.8 / 70.1 | 64.4 / 71.0 | 64.4 / 70.5 |
| Aux | 63.5 / 68.6 | 63.6 / 69.1 | 63.0 / 68.2 | 63.4 / 68.6 |
| State | 51.3 / 52.0 | ... / ... | ... / ... | ... / ... |
| State − LeWM | -10.0 / -13.3 | ... / ... | ... / ... | ... / ... |
| State − SCALE | -12.8 / -18.5 | ... / ... | ... / ... | ... / ... |

## Two-Room

State fed to the model: 2-d: agent position (the only moving quantity).

| | 3072 | 1024 | 1025 | mean |
|---|---|---|---|---|
| LeWM | 77.9 / 82.9 | 76.0 / 81.4 | 77.5 / 82.6 | 77.2 / 82.3 |
| SCALE | 89.2 / 92.9 | 89.4 / 93.3 | 90.1 / 92.9 | 89.6 / 93.0 |
| Aux | 78.1 / 83.1 | 77.4 / 82.6 | 78.3 / 83.1 | 77.9 / 82.9 |
| State | 82.6 / 88.3 | 82.2 / 88.0 | 83.0 / 88.2 | 82.6 / 88.2 |
| State − LeWM | +4.7 / +5.4 | +6.1 / +6.7 | +5.5 / +5.7 | +5.4 [+3.7, +7.2] / +5.9 [+4.3, +7.6] |
| State − SCALE | -6.6 / -4.6 | -7.3 / -5.3 | -7.1 / -4.6 | -7.0 [-7.9, -6.1] / -4.8 [-5.8, -3.9] |

## PointMaze

State fed to the model: 4-d: position and velocity.

| | 3072 | 1024 | 1025 | mean |
|---|---|---|---|---|
| LeWM | 82.1 / 80.2 | 79.5 / 69.8 | 81.2 / 76.9 | 80.9 / 75.6 |
| SCALE | 86.2 / 83.4 | 72.3 / 70.3 | 83.4 / 81.6 | 80.6 / 78.4 |
| Aux | 84.3 / 81.3 | 85.6 / 82.9 | 88.7 / 87.6 | 86.2 / 83.9 |
| State | 89.0 / 90.2 | 89.6 / 90.3 | 89.7 / 90.6 | 89.4 / 90.3 |
| State − LeWM | +6.8 / +10.0 | +10.1 / +20.5 | +8.5 / +13.7 | +8.5 [+4.4, +12.6] / +14.7 [+1.5, +27.9] |
| State − SCALE | +2.8 / +6.8 | +17.3 / +19.9 | +6.2 / +9.0 | +8.8 [-10.1, +27.6] / +11.9 [-5.6, +29.4] |
