"""Does stepping refresh world.infos["pixels"], and does that fix the MAE?

The live-state probe showed the sim state is set exactly (|live-ds| = 0) while the
rendered frame is bit-identical to the post-reset one (px changed = 0). So the gate's
9.04 measures a stale pixel buffer, not a bad renderer. Two things follow, and both need
confirming rather than assuming:

  * P4/P5 read pixels only AFTER stepping 25 actions, so their z_true should be current.
  * the gate must force a refresh before comparing.

Measure MAE right after the callables, then after one zero-action step, then after
re-rendering explicitly.

    usage: diag_cube_pixrefresh.py [n]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_render_fidelity import TASKS
from scripts.check_render_fidelity import _apply_callables, _extract_init_goal, swm

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
spec = TASKS["cube"]
ds = swm.data.HDF5Dataset(spec["dataset"], keys_to_cache=["action"],
                          cache_dir=Path(swm.data.utils.get_cache_dir()),
                          keys_to_load=spec["keys_to_load"])
world = swm.World(env_name=spec["env"], num_envs=1, image_shape=(224, 224),
                  max_episode_steps=100, **spec["env_kwargs"])
env = world.envs.envs[0].unwrapped
# infos is only populated once the env has been reset, so take the action dim from the
# action space instead of from infos["action"]
adim = int(np.prod(env.action_space.shape)) if getattr(env, "action_space", None) is not None \
    else int(np.prod(world.envs.single_action_space.shape))
print(f"action dim = {adim}")

eps = json.loads(Path(spec["episodes"]).read_text())["episodes"][:N]
print(f"\n{'ep':4s}{'MAE 无步进':>12s}{'MAE 步进1次':>13s}{'MAE env.render':>16s}")
rows = []
for i, e in enumerate(eps):
    init, goal, _ = _extract_init_goal(ds, [e["traj_id"]], [e["start_idx"]], 25)
    stored = np.asarray(init["pixels"][0]).astype(np.int32)
    world.reset(seed=[e["env_seed"]])
    _apply_callables(env, spec["callables"],
                     {k: v[0] for k, v in {**init, **goal}.items() if hasattr(v, "__len__")})
    m0 = float(np.abs(np.asarray(world.infos["pixels"][0, 0]).astype(np.int32) - stored).mean())
    # explicit re-render through the env, no dynamics advanced
    m2 = float("nan")
    try:
        r = np.asarray(env.render())
        if r.shape != stored.shape:
            r = r.reshape(stored.shape)
        m2 = float(np.abs(r.astype(np.int32) - stored).mean())
    except Exception as exc:
        print(f"  env.render() raised {type(exc).__name__}: {exc}")
    # one zero-action step: refreshes infos but advances physics slightly
    world.envs.step(np.zeros((1, adim), dtype=np.float32))
    m1 = float(np.abs(np.asarray(world.infos["pixels"][0, 0]).astype(np.int32) - stored).mean())
    print(f"{i:<4d}{m0:12.2f}{m1:13.2f}{m2:16.2f}")
    rows.append((m0, m1, m2))
a = np.array(rows)
print(f"\n均值   无步进 {a[:,0].mean():.2f}   步进1次 {a[:,1].mean():.2f}   env.render {a[:,2].mean():.2f}")
print("\n若 env.render 或步进后的 MAE 落到 ~2-3，则渲染本身没问题，"
      "门只需在比较前强制刷新一次。")
world.close()
