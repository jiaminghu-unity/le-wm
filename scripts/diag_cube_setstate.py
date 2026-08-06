"""Is set_state actually taking effect on the cube env?

The montage says no: across eight episodes the dataset frames show the arm in wildly
different poses while the rendered frames all show roughly the same vertical pose, even
though the block clearly does move. Pixel MAE cannot tell which body is wrong, so
compare the state numerically instead — dataset qpos vs what the env reports after the
callables are applied, joint by joint.

    usage: diag_cube_setstate.py [n]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_render_fidelity import TASKS  # noqa: E402
from scripts.check_render_fidelity import _apply_callables, _extract_init_goal, swm  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
spec = TASKS["cube"]

ds = swm.data.HDF5Dataset(spec["dataset"], keys_to_cache=["action"],
                          cache_dir=Path(swm.data.utils.get_cache_dir()),
                          keys_to_load=spec["keys_to_load"])
world = swm.World(env_name=spec["env"], num_envs=1, image_shape=(224, 224),
                  max_episode_steps=100, **spec["env_kwargs"])
env = world.envs.envs[0].unwrapped
print("env has:", {c["method"]: hasattr(env, c["method"]) for c in spec["callables"]})
print("dataset columns loaded:", spec["keys_to_load"])

home = np.asarray(getattr(env, "_home_qpos", np.full(21, np.nan)), dtype=np.float64).ravel()
print(f"env._home_qpos[:7] = {np.round(home[:7], 4)}\n")

eps = json.loads(Path(spec["episodes"]).read_text())["episodes"][:N]
for i, e in enumerate(eps):
    init, goal, _ = _extract_init_goal(ds, [e["traj_id"]], [e["start_idx"]], 25)
    ds_qpos = np.asarray(init["qpos"][0], dtype=np.float64).ravel()
    ds_blk = np.asarray(init["privileged_block_0_pos"][0], dtype=np.float64).ravel()[:3]

    world.reset(seed=[e["env_seed"]])
    after_reset = np.asarray(world.infos["qpos"], dtype=np.float64).ravel().copy()
    merged = {**init, **goal}
    _apply_callables(env, spec["callables"],
                     {k: v[0] for k, v in merged.items() if hasattr(v, "__len__")})
    after_cb = np.asarray(world.infos["qpos"], dtype=np.float64).ravel()
    env_blk = np.asarray(world.infos["privileged/block_0_pos"], dtype=np.float64).ravel()[:3]

    n = min(len(ds_qpos), len(after_cb))
    print(f"--- ep{i}  (traj {e['traj_id']}, start {e['start_idx']}) ---")
    print(f"  dataset qpos      len={len(ds_qpos):3d}  [:7] {np.round(ds_qpos[:7], 4)}")
    print(f"  env after reset   len={len(after_reset):3d}  [:7] {np.round(after_reset[:7], 4)}")
    print(f"  env after callabl len={len(after_cb):3d}  [:7] {np.round(after_cb[:7], 4)}")
    print(f"  |env_after − dataset| over first {n}: max {np.abs(after_cb[:n] - ds_qpos[:n]).max():.5f}"
          f"   mean {np.abs(after_cb[:n] - ds_qpos[:n]).mean():.5f}")
    print(f"  |env_after − env_reset|      : max {np.abs(after_cb[:len(after_reset)] - after_reset).max():.5f}"
          f"   -> {'callables CHANGED the state' if np.abs(after_cb[:len(after_reset)] - after_reset).max() > 1e-6 else 'callables did NOTHING'}")
    print(f"  block: dataset {np.round(ds_blk, 4)}  env {np.round(env_blk, 4)}  "
          f"|diff| {np.abs(ds_blk - env_blk).max():.5f}")
world.close()
