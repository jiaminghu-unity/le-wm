"""The budget_sweep ENV_PRESETS entry for two-room, kept out of budget_sweep.py.

budget_sweep.py, its presets and every existing result are untouched; the wrappers in
this directory merge the entry below into the in-process dict, adding a key only.

Everything here is transcribed from config/eval/tworoom.yaml, not invented:

    world.env_name              swm/TwoRoom-v1
    callables                   _set_state <- proprio, _set_goal_state <- goal_proprio

and both methods were confirmed to exist on TwoRoomEnv (env.py:702, 705). They take the
agent's 2-d position, which is exactly what the proprio column holds -- the schema probe
found proprio and pos_agent carry identical ranges, min [14.03 14.02] max [208.98 208.97].

_set_goal_state does more than store a number: it also re-renders the target image, so the
goal frame the planner is scored against is the env's own rendering of the goal position,
consistent with how the dataset frames were produced.

Success in this env is ||agent - target|| < 16 px (env.py:274), evaluated at ANY step
within the budget, and the goal is the expert's position 25 steps later -- the same
"reach where the expert was" contract the other three tasks use.

keys_to_load is pinned for the same reason cube pins it: _extract_init_goal walks every
loaded column, and this dataset carries `id` (int64 near 9.2e18) plus render_time and
reward that the protocol never touches.
"""

TWOROOM_PRESET = {
    "env_name": "swm/TwoRoom-v1",
    "env_kwargs": {},
    "dataset": "tworoom",
    "process_cols": ["action", "proprio", "pos_agent"],
    "keys_to_load": ["pixels", "action", "proprio", "pos_agent"],
    "callables": [
        {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}},
    ],
}


def register(presets):
    """Add the two-room entry to a presets dict, refusing to shadow an existing one."""
    if "tworoom" in presets:
        raise RuntimeError("budget_sweep already defines a 'tworoom' preset; refusing "
                           "to shadow it")
    presets["tworoom"] = TWOROOM_PRESET
    print("[preset] registered tworoom (swm/TwoRoom-v1)", flush=True)
    return presets
