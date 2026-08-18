"""Full-configuration q for Cube: EVERY non-velocity, non-leaking state quantity the
dataset carries, encoded under the repo's angle conventions. 22 dims.

The canonical 9-d cube q is the "success-criterion-aligned" subset. The schema audit
(2026-08-18) showed the dataset also carries block orientation, the 6 arm joints and
a binary gripper contact flag. This variant hands the aux head ALL of it, to test
whether aux's null planning effect is an information-quantity problem (prediction
from five probe families: it is not -- the information lands in low-variance tail
directions regardless of how much is provided).

Layout (22):
    [0:3]   effector xyz                       raw positions
    [3:5]   cos 2psi, sin 2psi                 gripper yaw, DOUBLE angle (parallel jaw
                                               is pi-symmetric)
    [5]     gripper opening                    raw in [0, 1]
    [6]     gripper contact                    binary flag, raw
    [7:17]  cos/sin of arm joints 0,1,2,3,5    joint 4 is frozen (range ~[-1.58,-1.55])
                                               and excluded; joint 5 sweeps +-2pi so
                                               cos/sin is mandatory (utils.CUBE_ARM_JOINTS)
    [17:20] block xyz                          raw positions
    [20:22] cos 4theta, sin 4theta             block yaw at QUADRUPLE angle: the block
                                               is a cube, pi/2-symmetric, so yaw is only
                                               defined modulo pi/2

Excluded on principle: every velocity column (qvel, joint_vel, gripper_vel, prev_*),
and privileged_target_* (the target block pose IS the goal -- putting it in q leaks
the answer, same rule as two-room's target position).
"""

import torch

from utils import CUBE_ARM_JOINTS


def build_q_cube_full(effector_pos, effector_yaw, gripper_opening, gripper_contact,
                      joint_pos, block_pos, block_yaw):
    parts = [
        effector_pos[..., :3],
        torch.cos(2.0 * effector_yaw[..., :1]),
        torch.sin(2.0 * effector_yaw[..., :1]),
        gripper_opening[..., :1],
        gripper_contact[..., :1],
    ]
    for i in CUBE_ARM_JOINTS:
        parts.append(torch.cos(joint_pos[..., i : i + 1]))
        parts.append(torch.sin(joint_pos[..., i : i + 1]))
    parts += [
        block_pos[..., :3],
        torch.cos(4.0 * block_yaw[..., :1]),
        torch.sin(4.0 * block_yaw[..., :1]),
    ]
    q = torch.cat(parts, dim=-1)
    assert q.shape[-1] == 22, q.shape
    return q


# variant -> (builder, source columns, angle unit check: col, idx, lo, hi)
# The unit check rides on block yaw, which the dataset stores wrapped to [-pi, pi].
Q_VARIANTS_CUBE_FULL = {
    "cube_full_config": (
        build_q_cube_full,
        ["proprio_effector_pos", "proprio_effector_yaw", "proprio_gripper_opening",
         "proprio_gripper_contact", "proprio_joint_pos", "privileged_block_0_pos",
         "privileged_block_0_yaw"],
        ("privileged_block_0_yaw", 0, -3.15, 3.15),
    ),
}
