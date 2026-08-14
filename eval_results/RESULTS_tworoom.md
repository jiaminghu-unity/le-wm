# two-room

The fourth LeWM environment, and the one this study had not covered. It differs from
the other three in kind: no physics engine -- the env renders 224x224 frames from
torch directly -- the action is a 2-d velocity, and success is
`||agent - target|| < 16 px` evaluated at any step within the budget.

**q = agent xy (2-d)**, following the same rule as the other three: q is what moves.
`pos_target` and the door centres are per-episode configuration, and `pos_target` IS
the goal -- putting it in q would hand the loss the success criterion. This is the
cleanest q in the study: no periodic coordinate, and none of Cube's problem where a
mostly-static object lets z-scoring amplify the rare frames in which it moves.

**Hyperparameters were not tuned here.** obj 0.1 is the value Push-T and Cube used;
aux 0.1 is Cube's and the config default (Push-T used 0.3, Reacher 0.4, so there is no
consensus value). Both were fixed before any two-room result existed. If the aux arm
underperforms, an untuned weight is a live explanation.

**Scene reconstruction check:** frame MAE 0.000 against the dataset (reference scale:
reacher 0.0001, cube 0.175, pusht 0.474), agent position restored to 0.000000 px. Only
`agent.position` and `target.position` are randomised per episode
(`DEFAULT_VARIATIONS`), so the wall and doors are fixed and `_set_state` is sufficient.
670,809 valid starting points = 920,809 frames - 10,000 episodes x 25, confirming the
goal offset never runs past an episode boundary.

## Status

Sweep incomplete: 0 of 72 (arm, solver, seed) cells present, and
no solver yet has all six seeds for all three arms, so no table is produced.

Present so far: none
