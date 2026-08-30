"""budget_sweep ENV_PRESETS for the self-collected OGBench multi-object tasks
(cube_double / scene), NEW FILE -- budget_sweep.py itself is untouched.

Three pieces:
  1. presets: env + lance dataset + reset callables. The goal is imposed from the
     dataset frame at start+25 exactly like the cube preset: set_state(qpos,qvel)
     restores the start, then the target setters plant the goal, and the env's own
     success flag (mode='task': ALL elements at target) is the SR criterion.
  2. SceneEvalEnv: SceneEnv subclass whose scalar target setters coerce the
     dataset's shape-(1,) float columns to python scalars (button states to int),
     so _compute_successes never compares against arrays. Registered as
     swm/OGBSceneEval-v0 on import.
  3. install_lance_dispatch(): budget_sweep instantiates swm.data.HDF5Dataset;
     these datasets exist only as lance. The dispatch routes *.lance names to
     LanceDataset (same Dataset abstraction: get_col_data / keys_to_load) and
     leaves every other name on the original class.
"""

from pathlib import Path

import numpy as np


_ARM = ["proprio/effector_pos", "proprio/effector_yaw", "proprio/gripper_opening",
        "proprio/gripper_contact", "proprio/joint_pos"]
_COMMON_KEYS = ["pixels", "action", "qpos", "qvel"] + _ARM

PRESETS = {
    "cube_double": {
        "env_name": "swm/OGBCube-v0",
        "env_kwargs": {
            "env_type": "double", "ob_type": "states", "multiview": False,
            "visualize_info": False, "terminate_at_goal": True,
        },
        "dataset": "ogbench/cube_double_play.lance",
        "process_cols": ["action"],
        "keys_to_load": _COMMON_KEYS + [
            "privileged/block_0_pos", "privileged/block_0_yaw", "privileged/block_0_quat",
            "privileged/block_1_pos", "privileged/block_1_yaw", "privileged/block_1_quat",
        ],
        "callables": [
            {"method": "set_state",
             "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
            {"method": "set_target_pos",
             "args": {"cube_id": {"value": 0, "in_dataset": False},
                      "target_pos": {"value": "goal_privileged/block_0_pos"},
                      "target_quat": {"value": "goal_privileged/block_0_quat"}}},
            {"method": "set_target_pos",
             "args": {"cube_id": {"value": 1, "in_dataset": False},
                      "target_pos": {"value": "goal_privileged/block_1_pos"},
                      "target_quat": {"value": "goal_privileged/block_1_quat"}}},
        ],
    },
    "cube_triple": {
        "env_name": "swm/OGBCube-v0",
        "env_kwargs": {
            "env_type": "triple", "ob_type": "states", "multiview": False,
            "visualize_info": False, "terminate_at_goal": True,
        },
        "dataset": "ogbench/cube_triple_play.lance",
        "process_cols": ["action"],
        "keys_to_load": _COMMON_KEYS + [
            "privileged/block_0_pos", "privileged/block_0_yaw", "privileged/block_0_quat",
            "privileged/block_1_pos", "privileged/block_1_yaw", "privileged/block_1_quat",
            "privileged/block_2_pos", "privileged/block_2_yaw", "privileged/block_2_quat",
        ],
        "callables": [
            {"method": "set_state",
             "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
            {"method": "set_target_pos",
             "args": {"cube_id": {"value": 0, "in_dataset": False},
                      "target_pos": {"value": "goal_privileged/block_0_pos"},
                      "target_quat": {"value": "goal_privileged/block_0_quat"}}},
            {"method": "set_target_pos",
             "args": {"cube_id": {"value": 1, "in_dataset": False},
                      "target_pos": {"value": "goal_privileged/block_1_pos"},
                      "target_quat": {"value": "goal_privileged/block_1_quat"}}},
            {"method": "set_target_pos",
             "args": {"cube_id": {"value": 2, "in_dataset": False},
                      "target_pos": {"value": "goal_privileged/block_2_pos"},
                      "target_quat": {"value": "goal_privileged/block_2_quat"}}},
        ],
    },
    "cube_quadruple": {
        "env_name": "swm/OGBCube-v0",
        "env_kwargs": {
            "env_type": "quadruple", "ob_type": "states", "multiview": False,
            "visualize_info": False, "terminate_at_goal": True,
        },
        "dataset": "ogbench/cube_quadruple_play.lance",
        "process_cols": ["action"],
        "keys_to_load": _COMMON_KEYS + [
            "privileged/block_0_pos", "privileged/block_0_yaw", "privileged/block_0_quat",
            "privileged/block_1_pos", "privileged/block_1_yaw", "privileged/block_1_quat",
            "privileged/block_2_pos", "privileged/block_2_yaw", "privileged/block_2_quat",
            "privileged/block_3_pos", "privileged/block_3_yaw", "privileged/block_3_quat",
        ],
        "callables": [
            {"method": "set_state",
             "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
            {"method": "set_target_pos",
             "args": {"cube_id": {"value": 0, "in_dataset": False},
                      "target_pos": {"value": "goal_privileged/block_0_pos"},
                      "target_quat": {"value": "goal_privileged/block_0_quat"}}},
            {"method": "set_target_pos",
             "args": {"cube_id": {"value": 1, "in_dataset": False},
                      "target_pos": {"value": "goal_privileged/block_1_pos"},
                      "target_quat": {"value": "goal_privileged/block_1_quat"}}},
            {"method": "set_target_pos",
             "args": {"cube_id": {"value": 2, "in_dataset": False},
                      "target_pos": {"value": "goal_privileged/block_2_pos"},
                      "target_quat": {"value": "goal_privileged/block_2_quat"}}},
            {"method": "set_target_pos",
             "args": {"cube_id": {"value": 3, "in_dataset": False},
                      "target_pos": {"value": "goal_privileged/block_3_pos"},
                      "target_quat": {"value": "goal_privileged/block_3_quat"}}},
        ],
    },
    "scene": {
        "env_name": "swm/OGBSceneEval-v0",
        "env_kwargs": {
            "ob_type": "states", "multiview": False,
            "visualize_info": False, "terminate_at_goal": True,
        },
        "dataset": "ogbench/scene_play.lance",
        "process_cols": ["action"],
        "keys_to_load": _COMMON_KEYS + [
            "privileged/block_0_pos", "privileged/block_0_yaw", "privileged/block_0_quat",
            "privileged/drawer_pos", "privileged/window_pos",
            "privileged/button_0_state", "privileged/button_1_state",
        ],
        "callables": [
            {"method": "set_state",
             "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"},
                      "button_state_0": {"value": "privileged/button_0_state"},
                      "button_state_1": {"value": "privileged/button_1_state"}}},
            {"method": "set_cube_target_pos",
             "args": {"cube_id": {"value": 0, "in_dataset": False},
                      "target_pos": {"value": "goal_privileged/block_0_pos"},
                      "target_quat": {"value": "goal_privileged/block_0_quat"}}},
            {"method": "set_target_drawer_pos",
             "args": {"target_pos": {"value": "goal_privileged/drawer_pos"}}},
            {"method": "set_target_window_pos",
             "args": {"target_pos": {"value": "goal_privileged/window_pos"}}},
            {"method": "set_target_button_state",
             "args": {"button_id": {"value": 0, "in_dataset": False},
                      "target_state": {"value": "goal_privileged/button_0_state"}}},
            {"method": "set_target_button_state",
             "args": {"button_id": {"value": 1, "in_dataset": False},
                      "target_state": {"value": "goal_privileged/button_1_state"}}},
        ],
    },
}


def _scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def _register_scene_eval_env():
    import gymnasium as gym
    from stable_worldmodel.envs.ogbench.scene_env import SceneEnv

    class SceneEvalEnv(SceneEnv):
        def set_state(self, qpos, qvel, **kw):
            # dataset 的按钮列是 shape-(1,) float;set_state 期望 int 状态
            kw = {k: (int(round(_scalar(v))) if k.startswith("button_state_") else v)
                  for k, v in kw.items()}
            super().set_state(qpos, qvel, **kw)

        def set_target_drawer_pos(self, target_pos):
            super().set_target_drawer_pos(_scalar(target_pos))

        def set_target_window_pos(self, target_pos):
            super().set_target_window_pos(_scalar(target_pos))

        def set_target_button_state(self, button_id, target_state):
            super().set_target_button_state(int(button_id), int(round(_scalar(target_state))))

    if "swm/OGBSceneEval-v0" not in gym.registry:
        gym.register(id="swm/OGBSceneEval-v0", entry_point=SceneEvalEnv)


def install_lance_dispatch(budget_sweep):
    import stable_worldmodel as swm
    from stable_worldmodel.data.formats.lance import LanceDataset

    orig = swm.data.HDF5Dataset

    def dispatch(name, *a, keys_to_cache=None, cache_dir=None, keys_to_load=None, **kw):
        if str(name).endswith(".lance"):
            # HDF5Dataset 自己在 cache 根下拼 datasets/ 子目录,这里同样拼上
            root = Path(swm.data.utils.get_cache_dir(cache_dir, sub_folder="datasets"))
            return LanceDataset(path=str(root / name),
                                keys_to_load=keys_to_load, keys_to_cache=keys_to_cache, **kw)
        return orig(name, *a, keys_to_cache=keys_to_cache, cache_dir=cache_dir,
                    keys_to_load=keys_to_load, **kw)

    swm.data.HDF5Dataset = dispatch
    print("[ogbmulti] lance dispatch installed over swm.data.HDF5Dataset", flush=True)


def register(presets_dict):
    _register_scene_eval_env()
    presets_dict.update(PRESETS)
    print(f"[ogbmulti] presets registered: {sorted(PRESETS)}", flush=True)
