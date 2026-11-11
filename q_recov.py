"""Recoverability-split q variants (rebuttal, reviewer 9CWa Q2).

The reviewer asks whether SCALE's gain comes from state components LeWM already recovers
well or from components it recovers poorly. Each variant below supervises L_corr on a strict
coordinate subset of the paper's q, split by the baseline's held-out linear-probe R^2
(paper Tables 6 and 8):

    pusht_recov_high   block x, y                     (R^2 .97, .96)        indices 2, 3
    pusht_recov_low    agent x, y, cos/sin theta      (R^2 .92, .92, .90, .87)  indices 0, 1, 4, 5
    cube_recov_high    effector xyz, block xyz        (R^2 .97-.99)         indices 0, 1, 2, 6, 7, 8
    cube_recov_low     cos/sin 2psi, gripper opening  (R^2 .76, -.26, .91)  indices 3, 4, 5

Reacher, Two-Room and PointMaze are not split: every component of their q is recovered at
R^2 >= .99 by the baseline, so there is no "low" group.

A subset is taken by indexing the full variant's builder output, so the surviving components
are built -- and, because q statistics are per component, standardized -- exactly as in the
full-q runs. The only difference from the paper's SCALE arm is which coordinates the loss sees.
Same pattern as q_half.py: new keys merged into utils.Q_VARIANTS in-process by train_recov.py.
"""

from functools import partial

import utils


def _subset(full_builder, idx, *raws):
    """Module-level so that partial(_subset, ...) pickles for DataLoader workers."""
    return full_builder(*raws)[..., list(idx)]


# variant -> (full variant, kept indices of the full q)
RECOV_OF = {
    "pusht_recov_high": ("pusht_state", (2, 3)),
    "pusht_recov_low": ("pusht_state", (0, 1, 4, 5)),
    "cube_recov_high": ("cube_effector", (0, 1, 2, 6, 7, 8)),
    "cube_recov_low": ("cube_effector", (3, 4, 5)),
}

Q_VARIANTS_RECOV = {}
for _name, (_full, _idx) in RECOV_OF.items():
    _builder, _sources, _check = utils.Q_VARIANTS[_full]
    Q_VARIANTS_RECOV[_name] = (partial(_subset, _builder, _idx), _sources, _check)
