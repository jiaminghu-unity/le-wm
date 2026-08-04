"""Gate every evaluation on render fidelity.

The model is trained on the pixels stored in the dataset but evaluated on pixels
the environment renders live. If those two disagree, the encoder is fed
out-of-distribution images and the success rate drops for reasons that have
nothing to do with the model. This check sets the env to a dataset frame's exact
simulator state, renders, and compares against the stored frame.

It is what would have caught the Reacher discrepancy immediately: with EGL silently
falling back to software rendering the MAE was 4.83; with the NVIDIA GL driver
installed it is 2.34.

    usage: check_render_fidelity.py <task> [n_frames] [--max-mae X]
    exit 0 if mean absolute pixel error <= --max-mae, else 1.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TASKS = {
    "reacher": dict(
        env="swm/ReacherDMControl-v0", env_kwargs=dict(task="qpos_match"),
        dataset="reacher", episodes="scripts/episodes_reacher_250.json",
        callables=[{"method": "set_state",
                    "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
                   {"method": "set_target_qpos",
                    "args": {"target_qpos": {"value": "goal_qpos"}}}]),
    "pusht": dict(
        env="swm/PushT-v1", env_kwargs={},
        dataset="pusht_expert_train", episodes="scripts/episodes_pusht_200.json",
        callables=[{"method": "_set_state", "args": {"state": {"value": "state"}}},
                   {"method": "_set_goal_state",
                    "args": {"goal_state": {"value": "goal_state"}}}]),
    "cube": dict(
        env="swm/OGBCube-v0",
        env_kwargs=dict(env_type="single", ob_type="states", multiview=False,
                        visualize_info=False, terminate_at_goal=True),
        dataset="ogbench/cube_single_expert",
        episodes="scripts/episodes_cube_s101_100.json",
        keys_to_load=["pixels", "action", "qpos", "qvel",
                      "privileged_block_0_pos", "privileged_block_0_quat"],
        callables=[{"method": "set_state",
                    "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
                   {"method": "set_target_pos",
                    "args": {"cube_id": {"value": 0, "in_dataset": False},
                             "target_pos": {"value": "goal_privileged_block_0_pos"},
                             "target_quat": {"value": "goal_privileged_block_0_quat"}}}]),
}

ap = argparse.ArgumentParser()
ap.add_argument("task", choices=list(TASKS))
ap.add_argument("n", nargs="?", type=int, default=8)
ap.add_argument("--max-mae", type=float, default=3.0)
args = ap.parse_args()

os.environ.setdefault("MUJOCO_GL", "egl")
import hdf5plugin  # noqa: F401,E402
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal  # noqa: E402

spec = TASKS[args.task]
kw = {}
if spec.get("keys_to_load"):
    kw["keys_to_load"] = spec["keys_to_load"]
ds = swm.data.HDF5Dataset(spec["dataset"], keys_to_cache=["action"],
                          cache_dir=Path(swm.data.utils.get_cache_dir()), **kw)
world = swm.World(env_name=spec["env"], num_envs=1, image_shape=(224, 224),
                  max_episode_steps=100, **spec["env_kwargs"])
env = world.envs.envs[0].unwrapped
for cb in spec["callables"]:
    print(f"  env.{cb['method']}: {hasattr(env, cb['method'])}"
          f"{'   <-- MISSING, callable silently skipped' if not hasattr(env, cb['method']) else ''}")

eps = json.loads(Path(spec["episodes"]).read_text())["episodes"][: args.n]
maes = []
for e in eps:
    init, goal, _ = _extract_init_goal(ds, [e["traj_id"]], [e["start_idx"]], 25)
    world.reset(seed=[e["env_seed"]])
    merged = {**init, **goal}
    _apply_callables(env, spec["callables"],
                     {k: v[0] for k, v in merged.items() if hasattr(v, "__len__")})
    r = np.asarray(world.infos["pixels"][0, 0]).astype(np.int32)
    s = np.asarray(init["pixels"][0]).astype(np.int32)
    if r.shape != s.shape:
        r = r.reshape(s.shape)
    maes.append(float(np.abs(r - s).mean()))

mae = float(np.mean(maes))
frac = None
print(f"RENDER_FIDELITY task={args.task} backend={os.environ['MUJOCO_GL']} "
      f"n={len(maes)} MAE={mae:.4f} per-frame={[round(m, 2) for m in maes]}")
if mae > args.max_mae:
    print(f"FAIL: MAE {mae:.3f} > {args.max_mae} — env renders disagree with the dataset "
          f"the model was trained on; absolute SR will be biased low.", file=sys.stderr)
    sys.exit(1)
print(f"OK: MAE {mae:.3f} <= {args.max_mae}")
