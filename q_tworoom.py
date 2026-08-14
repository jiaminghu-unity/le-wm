"""The q variant for two-room, the fourth LeWM environment.

A separate module so utils.py, train.py and every existing q_stats artifact stay
byte-identical; train_tworoom.py merges the dict below into utils.Q_VARIANTS at import
time, which adds a key and overwrites nothing.

WHAT q IS, and why. Same rule the other three tasks follow: q is what MOVES.

    pusht    agent xy + block xy + cos/sin block angle   (6)
    reacher  cos/sin of both joints                      (4)
    cube     effector xyz + yaw + gripper + block xyz    (9)
    tworoom  agent xy                                    (2)

In two-room only the agent moves. `pos_target` and the door centres are per-episode
configuration, fixed for the whole trajectory -- and `pos_target` IS the goal, so
putting it in q would hand the loss the success criterion directly, which is exactly
what every other variant here avoids.

This makes two-room the cleanest q in the study: 2-d, no periodic coordinate (no
cos/sin folding to get right), and none of cube's problem where a mostly-static object
lets z-scoring blow up the rare frames in which it moves. If L_obj's mechanism really is
"calibrate latent distance against physical distance", this is where it should work best;
if it does nothing here, the mechanism story is in trouble.

The dataset ships pos_agent as its own column (confirmed by scripts/ray_prep_tworoom.sh's
schema probe, not assumed), so no slicing out of a packed state vector is needed --
unlike pusht, where q comes out of a 7-vector.
"""

import torch


def build_q_tworoom_agent(pos_agent):
    """two-room: (..., 2) agent position -> (..., 2), unchanged.

    Position is not periodic, so no cos/sin is involved, and there is nothing to drop:
    ||dq|| is plain Euclidean displacement in the env's pixel coordinates (the column
    spans roughly [14, 209] on both axes, i.e. the playable area inside the border).

    The [..., :2] slice is defensive rather than cosmetic: if the column ever carries
    more than two components the shape assertion in ray_train_tworoom.sh fires instead
    of a wrong q training silently.
    """
    return pos_agent[..., :2]


# variant -> (builder fn, source columns, angle unit check: col, idx, lo, hi)
#
# The unit check exists to catch a degrees-vs-radians mix-up on angular columns. q here
# has no angle at all, so the check is pointed at pos_agent's first component with the
# bounds the schema probe reported for the playable area. That still catches the failure
# it is there to catch -- a column silently swapped for one on a different scale.
Q_VARIANTS_TWOROOM = {
    "tworoom_agent": (build_q_tworoom_agent, ["pos_agent"], ("pos_agent", 0, 0.0, 224.0)),
}

Q_DIM = {"tworoom_agent": 2}
