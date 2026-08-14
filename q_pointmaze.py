"""The q variant for DINO-WM's PointMaze (UMaze).

A separate module so utils.py, train.py and every existing q_stats artifact stay
byte-identical; train_pointmaze.py merges the dict below into utils.Q_VARIANTS at import
time, which adds a key and overwrites nothing.

WHAT q IS. Position only: q = (x, y). The same rule the other tasks follow -- q is what
moves -- plus two task-specific reasons to drop the velocity the state also carries:

  * The success test is position-only. DINO-WM's wrapper decides success as
    ||goal_state[:2] - cur_state[:2]|| < 0.5, ignoring velocity outright. Putting velocity
    in q would ask L_obj to calibrate against a quantity the task does not score.
  * Velocity would dominate. Measured on all 200,000 frames: position std is 1.0055 and
    0.9630, velocity std is 1.6803 and 1.9319 -- 1.7-2.0x larger. Since q is z-scored per
    component, a 4-d q would put roughly half of ||dq||^2 into velocity. This is the same
    failure mode as cube's IK-pinned joint, in a different guise: there a frozen dimension
    would have dominated, here an irrelevant one would.

Every other task drops velocity too (Push-T keeps 5 of 7 raw state components, Reacher uses
joint angles without joint velocities, cube's q contains no qvel), so this is consistent
rather than special-cased.

The result is a 2-d planar-position q, the same shape as two-room's. That is the point: on
two-room this q gave the largest L_obj effect in the study (+7.50 pp, obj - aux +7.36 pp,
the only task where obj and aux separate), and that rested on a single training seed in a
single environment. PointMaze is an independent environment with the same kind of q.
"""

import torch


def build_q_pointmaze_pos(pos):
    """PointMaze: (..., 2) position -> (..., 2), unchanged.

    Position is not periodic, so no cos/sin is involved. ||dq|| is plain Euclidean
    displacement in the maze's own units (x spans [0.342, 3.250], y [0.344, 3.257]).

    The [..., :2] slice is defensive: the dataset writes a dedicated 2-d `pos` column, and
    if that ever changed shape the dimension assertion in ray_train_pointmaze.sh fires
    instead of a wrong q training silently.
    """
    return pos[..., :2]


# variant -> (builder fn, source columns, angle unit check: col, idx, lo, hi)
#
# q has no angular component, so the unit check is pointed at the x coordinate with the
# maze's own bounds. It still catches what such a check is for: a column silently swapped
# for one on a different scale (velocity spans +-6.1, four times wider).
Q_VARIANTS_POINTMAZE = {
    "pointmaze_pos": (build_q_pointmaze_pos, ["pos"], ("pos", 0, -1.0, 5.0)),
}

Q_DIM = {"pointmaze_pos": 2}
