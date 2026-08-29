"""Native-FULL q variant for Reacher SCALE: joints (cos/sin, shoulder unbounded) +
fingertip position + joint velocities. 8 dims. The existing reacher variants stop
at joints(+finger); the qgate probes confirmed the h5 carries qvel, and a q-INPUT/
full-metric arm should see the complete Markovian state.

    reacher_native_full: qpos (...,2) + finger (...,2) + qvel (...,2) -> (..., 8)
        [cos q0, sin q0, cos q1, sin q1, finger_x, finger_y, qvel_0, qvel_1]

Registered at runtime by train_qnative.py; utils.Q_VARIANTS untouched.
"""

import torch

from utils import build_q_reacher_joints


def build_q_reacher_native_full(qpos, finger, qvel):
    return torch.cat([build_q_reacher_joints(qpos), finger[..., :2], qvel[..., :2]], dim=-1)


Q_VARIANTS_REACHER_FULL = {
    "reacher_native_full": (
        build_q_reacher_native_full, ["qpos", "finger_pos", "qvel"],
        ("qpos", 1, -3.15, 3.15),
    ),
}
