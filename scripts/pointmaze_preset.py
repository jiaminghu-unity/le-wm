"""The budget_sweep ENV_PRESETS entry for PointMaze, kept out of budget_sweep.py.

budget_sweep.py, its presets and every existing result stay untouched; the wrapper merges
the entry below into the in-process dict, adding a key only -- the same arrangement
two-room used (scripts/tworoom_preset.py).

The callables mirror the other tasks' contract:
    _set_state      <- dataset column `state`  (4-d: x, y, vx, vy)
    _set_goal_state <- goal_state              (the same column at start_idx + 25)
Success is decided inside the adapter with DINO-WM's own criterion
(||pos - goal_xy|| < 0.5, velocities ignored), evaluated at any step within the
budget -- the same "reach where the trajectory was 25 steps later" contract as
everywhere else.

keys_to_load is pinned because _extract_init_goal walks every loaded column; `pos`
rides along so the q used in training is visible in eval logs, and nothing else is
loaded.
"""

from scripts.pointmaze_env import register as _register_env

POINTMAZE_PRESET = {
    "env_name": "swm/DWMPointMaze-v0",
    "env_kwargs": {},
    "dataset": "pointmaze",
    "process_cols": ["action", "state", "pos"],
    "keys_to_load": ["pixels", "action", "state", "pos"],
    "callables": [
        {"method": "_set_state", "args": {"state": {"value": "state"}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
    ],
}


def register(presets):
    """Register the gymnasium env and add the preset, refusing to shadow anything."""
    _register_env()
    if "pointmaze" in presets:
        raise RuntimeError("budget_sweep already defines a 'pointmaze' preset; refusing "
                           "to shadow it")
    presets["pointmaze"] = POINTMAZE_PRESET
    print("[preset] registered pointmaze (swm/DWMPointMaze-v0)", flush=True)
    return presets
