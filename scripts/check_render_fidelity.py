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

# Imports and env setup stay at module level; everything that RUNS lives in main().
# Previously the argparse and the whole check sat at module level, so `from
# check_render_fidelity import TASKS` executed the check and parsed the *importer's*
# argv — which is how scripts/diag_render_cube.py died on "invalid choice: '8'".
os.environ.setdefault("MUJOCO_GL", "egl")
import hdf5plugin  # noqa: F401,E402
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal  # noqa: E402


def measure(task, n, callables=None, env_kwargs_override=None, want_pairs=False):
    """Per-frame |rendered - stored| for the first n episodes of `task`.

    callables / env_kwargs_override let a caller probe variants (e.g. skipping
    set_target_pos) without duplicating the task table.
    """
    spec = TASKS[task]
    kw = dict(spec["env_kwargs"])
    kw.update(env_kwargs_override or {})
    cbs = spec["callables"] if callables is None else callables
    ds_kw = {"keys_to_load": spec["keys_to_load"]} if spec.get("keys_to_load") else {}
    ds = swm.data.HDF5Dataset(spec["dataset"], keys_to_cache=["action"],
                              cache_dir=Path(swm.data.utils.get_cache_dir()), **ds_kw)
    world = swm.World(env_name=spec["env"], num_envs=1, image_shape=(224, 224),
                      max_episode_steps=100, **kw)
    env = world.envs.envs[0].unwrapped
    missing = [c["method"] for c in cbs if not hasattr(env, c["method"])]
    eps = json.loads(Path(spec["episodes"]).read_text())["episodes"][:n]
    maes, extra, pairs, stale_reads = [], [], [], []
    for e in eps:
        init, goal, _ = _extract_init_goal(ds, [e["traj_id"]], [e["start_idx"]], 25)
        world.reset(seed=[e["env_seed"]])
        merged = {**init, **goal}
        _apply_callables(env, cbs,
                         {k: v[0] for k, v in merged.items() if hasattr(v, "__len__")})
        st = np.asarray(init["pixels"][0]).astype(np.int32)
        # Render explicitly. world.infos["pixels"] is a snapshot refreshed on
        # reset/step, so reading it straight after set_state returns the POST-RESET
        # frame — which made this gate report cube at MAE 9.04 (a reset-pose arm
        # compared against a dataset-pose arm) when the renderer was in fact fine:
        # env.render() on the same state gives 0.19, and one env step gives 0.67.
        # Stepping is not used here because it advances physics and so measures drift
        # on top of render fidelity.
        r = None
        if hasattr(env, "render"):
            try:
                cand = np.asarray(env.render())
                if cand.size == st.size:
                    r = cand.reshape(st.shape).astype(np.int32)
            except Exception:
                r = None
        if r is None:
            r = np.asarray(world.infos["pixels"][0, 0]).astype(np.int32)
            if r.shape != st.shape:
                r = r.reshape(st.shape)
            stale_reads.append(True)
        maes.append(float(np.abs(r - st).mean()))
        extra.append((init, goal))
        if want_pairs:
            pairs.append((st.astype(np.uint8), r.astype(np.uint8)))
    world.close()
    if stale_reads:
        print(f"  NOTE: env.render() unavailable for {task}; fell back to "
              f"world.infos['pixels'] on {len(stale_reads)} frame(s), which may be stale")
    return maes, missing, extra, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("n", nargs="?", type=int, default=8)
    ap.add_argument("--max-mae", type=float, default=3.0)
    args = ap.parse_args()

    maes, missing, _, _ = measure(args.task, args.n)
    for cb in TASKS[args.task]["callables"]:
        flag = "   <-- MISSING, callable silently skipped" if cb["method"] in missing else ""
        print(f"  env.{cb['method']}: {cb['method'] not in missing}{flag}")
    mae = float(np.mean(maes))
    print(f"RENDER_FIDELITY task={args.task} backend={os.environ['MUJOCO_GL']} "
          f"n={len(maes)} MAE={mae:.4f} per-frame={[round(m, 2) for m in maes]}")
    if mae > args.max_mae:
        print(f"FAIL: MAE {mae:.3f} > {args.max_mae} — env renders disagree with the "
              f"dataset the model was trained on; absolute SR will be biased low.",
              file=sys.stderr)
        sys.exit(1)
    print(f"OK: MAE {mae:.3f} <= {args.max_mae}")


if __name__ == "__main__":
    main()
