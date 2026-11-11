# Rebuttal round: training seeds 1024 / 1025

Overall success (%), mean over T1-T5 and 6 episode seeds x 100 episodes. Each cell: seed 1024 / seed 1025.
Delta = arm minus its baseline (LeWM for SCALE and Aux, LpWM for LpWM+SCALE), same seed and solver.
`...` = not finished yet.
Cells complete: 80 / 104.

## Push-T

| Arm | CEM | iCEM | Delta CEM | Delta iCEM |
|---|---|---|---|---|
| LeWM | ... / 71.0 | ... / 64.6 |  |  |
| SCALE | ... / 74.0 | ... / 67.9 | ... / +3.0 | ... / +3.3 |
| Aux | 74.0 / 74.1 | 69.4 / 68.2 | ... / +3.1 | ... / +3.6 |
| LpWM | ... / 51.3 | ... / 48.7 |  |  |
| LpWM+SCALE | 61.8 / ... | 54.3 / ... | ... / ... | ... / ... |

## Reacher

| Arm | CEM | iCEM | Delta CEM | Delta iCEM |
|---|---|---|---|---|
| LeWM | 70.9 / 71.2 | 78.7 / 80.0 |  |  |
| SCALE (0.15) | 71.0 / ... | 78.5 / ... | +0.1 / ... | -0.3 / ... |
| SCALE (0.1) | ... / ... | ... / ... | ... / ... | ... / ... |
| Aux | 72.1 / 73.3 | 79.3 / 80.2 | +1.1 / +2.1 | +0.5 / +0.1 |
| LpWM | ... / 68.4 | ... / 75.4 |  |  |
| LpWM+SCALE | ... / 69.4 | ... / 75.7 | ... / +1.0 | ... / +0.4 |

## Cube

| Arm | CEM | iCEM | Delta CEM | Delta iCEM |
|---|---|---|---|---|
| LeWM | 61.2 / 61.4 | 64.1 / 65.7 |  |  |
| SCALE | 64.8 / ... | 70.1 / ... | +3.6 / ... | +6.0 / ... |
| Aux | 63.6 / 63.0 | 69.1 / 68.2 | +2.4 / +1.6 | +5.0 / +2.6 |
| LpWM | 61.4 / ... | 66.8 / ... |  |  |
| LpWM+SCALE | 62.6 / ... | 70.2 / ... | +1.3 / ... | +3.4 / ... |

## Two-Room

| Arm | CEM | iCEM | Delta CEM | Delta iCEM |
|---|---|---|---|---|
| LeWM | 76.0 / 77.5 | 81.4 / 82.6 |  |  |
| SCALE | 89.4 / 90.1 | 93.3 / 92.9 | +13.4 / +12.6 | +11.9 / +10.3 |
| Aux | 77.4 / 78.3 | 82.6 / 83.1 | +1.4 / +0.8 | +1.2 / +0.5 |
| LpWM | 85.5 / 85.2 | 88.6 / 88.3 |  |  |
| LpWM+SCALE | 92.8 / 91.8 | 94.8 / 94.2 | +7.2 / +6.7 | +6.2 / +5.9 |

## PointMaze

| Arm | CEM | iCEM | Delta CEM | Delta iCEM |
|---|---|---|---|---|
| LeWM | 79.5 / 81.2 | 69.8 / 76.9 |  |  |
| SCALE | 72.3 / 83.4 | 70.3 / 81.6 | -7.2 / +2.3 | +0.5 / +4.7 |
| Aux | 85.6 / 88.7 | 82.9 / 87.6 | +6.1 / +7.5 | +13.1 / +10.7 |
| LpWM | 67.2 / 74.2 | 60.1 / 66.4 |  |  |
| LpWM+SCALE | 80.6 / 78.8 | 75.0 / 72.3 | +13.4 / +4.6 | +14.9 / +5.9 |

