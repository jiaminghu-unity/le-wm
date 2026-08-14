"""Does the eval path's reconstructed PointMaze match the dataset frame it restores?

The earlier gate (ray_gate_pointmaze.sh) proved the raw env reproduces dataset pixels at
MAE 0.039. This one checks the same thing THROUGH the path evaluation actually takes --
swm.World + the preset's callables + the adapter -- so a wiring mistake in any of those
layers is caught before GPU-hours are spent, not after.

    usage: PMENV_DIR=... check_render_pointmaze.py [n] --episodes FILE [--max-mae 3.0]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hdf5plugin  # noqa: F401,E402
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal  # noqa: E402

from scripts import budget_sweep  # noqa: E402
from scripts.pointmaze_preset import POINTMAZE_PRESET, register  # noqa: E402

GOAL_OFFSET = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=12)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--max-mae", type=float, default=3.0)
    args = ap.parse_args()

    register(budget_sweep.ENV_PRESETS)
    P = POINTMAZE_PRESET
    eps = json.loads(Path(args.episodes).read_text())["episodes"][: args.n]
    ds = swm.data.HDF5Dataset(P["dataset"], keys_to_cache=P["process_cols"],
                              cache_dir=Path(swm.data.utils.get_cache_dir()),
                              keys_to_load=P["keys_to_load"])
    world = swm.World(env_name=P["env_name"], num_envs=1, image_shape=(224, 224),
                      max_episode_steps=10_000, **P["env_kwargs"])
    env = world.envs.envs[0].unwrapped

    maes, serr = [], []
    for ep in eps:
        init_state, goal_state, _ = _extract_init_goal(
            ds, [ep["traj_id"]], [ep["start_idx"]], GOAL_OFFSET)
        ref = np.asarray(init_state["pixels"][0]).astype(np.int32)
        world.reset(seed=[ep["env_seed"]])
        _apply_callables(env, P["callables"],
                         {k: v[0] for k, v in {**init_state, **goal_state}.items()
                          if hasattr(v, "__len__")})
        got = np.asarray(env.render()).astype(np.int32)
        if got.shape != ref.shape:
            got = got.reshape(ref.shape)
        maes.append(float(np.abs(got - ref).mean()))
        want = np.asarray(init_state["state"][0], dtype=np.float64).reshape(-1)[:2]
        have = np.asarray(env._state, dtype=np.float64).reshape(-1)[:2]
        serr.append(float(np.abs(have - want).max()))
    world.close()

    print(f"[pointmaze] state restore max|diff| = {max(serr):.6f}")
    print(f"[pointmaze] frame MAE vs dataset: mean {np.mean(maes):.3f} max {max(maes):.3f} "
          f"(n={len(maes)}; raw-env gate was 0.039; threshold {args.max_mae})")
    if max(serr) > 1e-3 or np.mean(maes) > args.max_mae:
        print("GATE FAIL", file=sys.stderr)
        return 1
    print("POINTMAZE EVAL-PATH GATE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
