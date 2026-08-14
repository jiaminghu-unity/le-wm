"""Does the reconstructed two-room scene match the dataset frame it claims to restore?

This is not a copy of check_render_fidelity.py's question. There the risk was a rendering
BACKEND mismatch (EGL vs software). Two-room renders deterministically from torch with no
GL involved, so that failure mode does not exist. The risk here is different and specific:

    _set_state sets ONLY agent_position (env.py:702).
    _set_goal_state sets the target and re-renders the target image (env.py:705).

But two-room's SCENE ITSELF varies per episode -- wall thickness, wall axis, door count and
door positions all live in variation_space and are drawn by reset(seed=...). The other three
tasks have fixed geometry, so restoring the physical state restores the picture. Here it may
not: the env resets with env_seed = 40000 + ... from the episode file, which has no relation
to whatever seed produced the dataset trajectory, so the walls and doors can simply differ.

If they do, the protocol is broken in a way that biases every two-room number: budget_sweep
overwrites the t=0 observation with the dataset frame, so the planner would be shown a scene
with one set of doors and then made to act in a scene with another.

The measurement is the same MAE-per-pixel comparison the other gate uses, so the numbers are
read on the same scale: the published gate threshold is 3.0, and reacher/cube/pusht come in
at 0.0001 / 0.175 / 0.474 when the restore is correct.

    usage: check_render_tworoom.py <n_episodes> [--episodes FILE] [--max-mae 3.0]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

import hdf5plugin  # noqa: F401,E402
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal  # noqa: E402

from scripts.tworoom_preset import TWOROOM_PRESET  # noqa: E402

GOAL_OFFSET = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=8)
    ap.add_argument("--episodes", default="scripts/episodes_tworoom_s101_100.json")
    ap.add_argument("--max-mae", type=float, default=3.0)
    args = ap.parse_args()

    eps = json.loads(Path(args.episodes).read_text())["episodes"][: args.n]
    ds = swm.data.HDF5Dataset(TWOROOM_PRESET["dataset"],
                              keys_to_cache=TWOROOM_PRESET["process_cols"],
                              cache_dir=Path(swm.data.utils.get_cache_dir()),
                              keys_to_load=TWOROOM_PRESET["keys_to_load"])
    world = swm.World(env_name=TWOROOM_PRESET["env_name"], num_envs=1,
                     image_shape=(224, 224), max_episode_steps=10_000,
                     **TWOROOM_PRESET["env_kwargs"])
    env = world.envs.envs[0].unwrapped

    maes, agent_err = [], []
    for ep in eps:
        init_state, goal_state, _ = _extract_init_goal(
            ds, [ep["traj_id"]], [ep["start_idx"]], GOAL_OFFSET)
        ref = np.asarray(init_state["pixels"][0]).astype(np.int32)
        world.reset(seed=[ep["env_seed"]])
        _apply_callables(env, TWOROOM_PRESET["callables"],
                         {k: v[0] for k, v in {**init_state, **goal_state}.items()
                          if hasattr(v, "__len__")})
        # render explicitly: infos["pixels"] is a buffer filled at step/reset time and can
        # be stale after callables mutate the state, which is what made the earlier gate
        # report a healthy renderer as broken
        got = np.asarray(env.render()).astype(np.int32)
        if got.shape != ref.shape:
            got = got.reshape(ref.shape)
        maes.append(float(np.abs(got - ref).mean()))
        want = np.asarray(init_state["proprio"][0], dtype=np.float64).reshape(-1)[:2]
        have = np.asarray(env.agent_position, dtype=np.float64).reshape(-1)[:2]
        agent_err.append(float(np.abs(have - want).max()))
    world.close()

    mae = float(np.mean(maes))
    print(f"[tworoom] agent position restored to max|diff| = {max(agent_err):.6f} px")
    print(f"[tworoom] frame MAE vs dataset: mean {mae:.3f}  max {max(maes):.3f}  "
          f"(n={len(maes)}; reference scale: reacher 0.0001, cube 0.175, pusht 0.474)")
    if max(agent_err) > 1e-3:
        print("FATAL: _set_state did not restore the agent position", file=sys.stderr)
        return 1
    if mae > args.max_mae:
        print(f"FATAL: frame MAE {mae:.3f} > {args.max_mae}. The agent is in the right "
              f"place, so what differs is the SCENE -- wall thickness/axis, door count or "
              f"door positions are drawn by reset(seed) and the episode file's env_seed "
              f"has no relation to the seed that produced the trajectory. Two-room results "
              f"would be biased: budget_sweep shows the planner the dataset frame at t=0 "
              f"and then steps a differently-shaped room.", file=sys.stderr)
        return 1
    print("TWOROOM RENDER OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
