"""Reduced-q variants: the same three tasks with roughly half of q withheld.

A separate module so that utils.py, train.py and every existing q_stats artifact
stay byte-identical. train_half.py merges the dict below into utils.Q_VARIANTS at
import time; that is an in-process addition of new keys, so no existing variant,
checkpoint or stats file can be affected.

Why. In the main experiments L_obj and the aux head are handed the full physical
pose q, which is more state than any deployed system has. If their advantage
survives on a q that is missing the task-relevant object entirely, the mechanism
cannot be "the loss smuggled in the success criterion" -- it has to be something
about geometry/decodability of the part of the state that IS supplied. If the
advantage dies, the advantage was riding on the withheld coordinates.

What is withheld, per task. Each variant drops the coordinates that name the thing
the task's success test measures, keeping the agent's own configuration:

    pusht_block_only     6 -> 4   drops pusher xy; keeps block xy + cos/sin theta
    reacher_joint0_only  4 -> 2   keeps cos/sin of joint 0 only
    cube_effector_only   9 -> 5   drops block xyz and gripper opening;
                                  keeps effector xyz + cos/sin 2psi

Push-T is the exception on purpose: it keeps the block and drops the pusher, because
Push-T's pusher is directly actuated (its position IS the action integrated) and is
the trivially predictable half -- dropping the block instead would leave a q that a
predictor can nail without learning anything about the scene.

Both losses are invariant to q's dimension, so "same hyperparameters" is literally
true rather than approximately:
  - aux_loss = (q_hat - q).pow(2).mean() averages over dimensions, and q is z-scored
    per component, so each dimension contributes ~1 in expectation. A sum reduction
    would have silently scaled the weight with dim; it is a mean (train.py:83).
  - L_obj = 1 - Pearson(||dz||^2, ||dq||^2) is scale-free and stays in [0, 2].

The angle-unit check tuple is unchanged from each task's full variant: it validates a
RAW source column, not q, so it is unaffected by which components survive into q.
"""

import torch

from utils import _CUBE_SOURCES  # noqa: F401  (kept for reference/symmetry)


def build_q_pusht_block_only(state):
    """Push-T, block only: (..., 7) raw state -> (..., 4)
    [block x, block y, cos angle, sin angle].

    Drops agent_x, agent_y (state[..., :2]) relative to build_q_raw. Velocities were
    never in q. The angle still enters only as (cos, sin) -- raw theta wraps at 2pi.
    """
    return torch.cat(
        [state[..., 2:4], torch.cos(state[..., 4:5]), torch.sin(state[..., 4:5])],
        dim=-1,
    )


def build_q_reacher_joint0_only(qpos):
    """Reacher, first joint only: (..., 2) qpos -> (..., 2) [cos q0, sin q0].

    q0 is the shoulder, which is unbounded and accumulates past +-pi, so (cos, sin)
    is mandatory here and not a stylistic choice.
    """
    return torch.cat([torch.cos(qpos[..., :1]), torch.sin(qpos[..., :1])], dim=-1)


def build_q_cube_effector_only(effector_pos, effector_yaw):
    """Cube, effector only: -> (..., 5) [eff x, y, z, cos 2psi, sin 2psi].

    Drops gripper opening and block xyz relative to build_q_cube_effector, i.e. q no
    longer contains the cube whose position the success test checks
    (||block - target|| <= 0.04 m).

    Yaw keeps the DOUBLE angle: a parallel-jaw gripper rotated by pi is the same
    physical configuration (the fingers swap), so psi lives on a pi-periodic circle
    and plain cos/sin would call the most similar pair the most distant.
    """
    two_psi = 2.0 * effector_yaw[..., :1]
    return torch.cat(
        [effector_pos[..., :3], torch.cos(two_psi), torch.sin(two_psi)], dim=-1
    )


# variant -> (builder fn, source columns, angle unit check: col, idx, lo, hi)
# Merged into utils.Q_VARIANTS by train_half.py. Keys are new; nothing is overwritten.
Q_VARIANTS_HALF = {
    "pusht_block_only": (
        build_q_pusht_block_only, ["state"], ("state", 4, -3.15, 6.30),
    ),
    "reacher_joint0_only": (
        # check rides on qpos[1] (the bounded elbow) exactly as reacher_joints_only
        # does -- qpos[0] is unbounded and would fail a radian range check
        build_q_reacher_joint0_only, ["qpos"], ("qpos", 1, -3.15, 3.15),
    ),
    "cube_effector_only": (
        build_q_cube_effector_only,
        ["proprio_effector_pos", "proprio_effector_yaw"],
        ("proprio_effector_yaw", 0, -3.15, 3.15),
    ),
}

# half variant -> (full variant, dim_full, dim_half, indices of the full q that survive)
#
# Every reduced variant is a strict coordinate subset of its full counterpart, which
# makes two things checkable rather than assumed (scripts/prep_half_qstats.py does both):
#   1. builder(x)[half] == builder_full(x)[kept], elementwise, on real data;
#   2. the new q_stats mean/std equal the full variant's restricted to `kept` -- so the
#      surviving components are z-scored by exactly the numbers the original runs used,
#      and the only difference between the two experiments is which coordinates exist.
HALF_OF = {
    "pusht_block_only": ("pusht_state", 6, 4, (2, 3, 4, 5)),
    "reacher_joint0_only": ("reacher_joints_only", 4, 2, (0, 1)),
    "cube_effector_only": ("cube_effector", 9, 5, (0, 1, 2, 3, 4)),
}
