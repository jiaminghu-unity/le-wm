"""Full-config q variants for the self-collected OGBench multi-object datasets
(slash-namespaced lance columns, names verified by the collection smoke runs).

Arm part (17-d), same construction as q_cube_full's cube_single 22-d:
  effector_pos 3 + cos2psi/sin2psi 2 + gripper_opening 1 + gripper_contact 1 +
  5 free joints as cos/sin 10
Per block (5-d): pos 3 + yaw as cos4th/sin4th 2 (pi/2 symmetry fold).
Scene extras: drawer_pos, window_pos (slide positions), button states x2.

  cube_double_full : 17 + 5*2 = 27
  scene_full       : 17 + 5 + 1 + 1 + 2 = 26

Registered at runtime by train_qnative.py; utils.Q_VARIANTS untouched.
"""

import torch


def _arm17(eff_pos, eff_yaw, grip_open, grip_contact, joint_pos):
    psi2 = 2.0 * eff_yaw.reshape(*eff_yaw.shape[:-1], -1)[..., :1]
    joints = joint_pos.reshape(*joint_pos.shape[:-1], -1)[..., :5]
    return torch.cat([
        eff_pos[..., :3],
        torch.cos(psi2), torch.sin(psi2),
        grip_open.reshape(*grip_open.shape[:-1], -1)[..., :1],
        grip_contact.reshape(*grip_contact.shape[:-1], -1)[..., :1],
        torch.cos(joints), torch.sin(joints),
    ], dim=-1)


def _block5(pos, yaw):
    th4 = 4.0 * yaw.reshape(*yaw.shape[:-1], -1)[..., :1]
    return torch.cat([pos[..., :3], torch.cos(th4), torch.sin(th4)], dim=-1)


def build_q_cube_double_full(eff_pos, eff_yaw, grip_open, grip_contact, joint_pos,
                             b0_pos, b0_yaw, b1_pos, b1_yaw):
    return torch.cat([
        _arm17(eff_pos, eff_yaw, grip_open, grip_contact, joint_pos),
        _block5(b0_pos, b0_yaw), _block5(b1_pos, b1_yaw),
    ], dim=-1)


def build_q_scene_full(eff_pos, eff_yaw, grip_open, grip_contact, joint_pos,
                       b0_pos, b0_yaw, drawer_pos, window_pos, btn0, btn1):
    flat = lambda x: x.reshape(*x.shape[:-1], -1)[..., :1]
    return torch.cat([
        _arm17(eff_pos, eff_yaw, grip_open, grip_contact, joint_pos),
        _block5(b0_pos, b0_yaw),
        flat(drawer_pos), flat(window_pos), flat(btn0), flat(btn1),
    ], dim=-1)


_ARM_SRC = ["proprio/effector_pos", "proprio/effector_yaw", "proprio/gripper_opening",
            "proprio/gripper_contact", "proprio/joint_pos"]

Q_VARIANTS_OGB_MULTI = {
    "cube_double_full": (
        build_q_cube_double_full,
        _ARM_SRC + ["privileged/block_0_pos", "privileged/block_0_yaw",
                    "privileged/block_1_pos", "privileged/block_1_yaw"],
        ("proprio/effector_yaw", 0, -3.15, 3.15),
    ),
    "scene_full": (
        build_q_scene_full,
        _ARM_SRC + ["privileged/block_0_pos", "privileged/block_0_yaw",
                    "privileged/drawer_pos", "privileged/window_pos",
                    "privileged/button_0_state", "privileged/button_1_state"],
        ("proprio/effector_yaw", 0, -3.15, 3.15),
    ),
}
