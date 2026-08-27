"""Native-full q variant for the q-only-INPUT model family: Push-T's complete raw
state, velocities included.

The canonical pusht_state variant (6-d) drops the velocities; a PIXEL model can't
see them anyway, so for L_obj/aux they were rightly excluded. A q-INPUT model is a
different animal: its encoder input should be the full MARKOVIAN state, or the
predictor is asked to model dynamics with hidden variables. (The cube q-only
teacher's planning collapse at 9-d effector q -- no joint angles -- is the
motivating example; this file is the pusht side of the same fix.)

    pusht_state_native: (..., 7) state -> (..., 8)
        [agent_x, agent_y, block_x, block_y, cos(theta), sin(theta), vx, vy]

Registered at runtime by train_qnative.py; utils.Q_VARIANTS itself is untouched.
"""

import torch


def build_q_pusht_native(state):
    pos = state[..., :4]
    theta = state[..., 4:5]
    vel = state[..., 5:7]
    return torch.cat([pos, torch.cos(theta), torch.sin(theta), vel], dim=-1)


def build_q_pointmaze_native(state):
    """PointMaze native-full: (..., 4) state = [x, y, vx, vy], used as-is (the ball
    has momentum; position-only q is non-Markovian for the predictor)."""
    return state[..., :4]


Q_VARIANTS_NATIVE = {
    "pusht_state_native": (build_q_pusht_native, ["state"], ("state", 4, -3.15, 6.30)),
    "pointmaze_state_native": (build_q_pointmaze_native, ["state"], ("state", 0, -1.0, 5.0)),
}
