"""Self-collect OGBench multi-object manipulation datasets at 224x224 through
stable_worldmodel's own pipeline: World + ExpertPolicy (OGBench's official
markov oracle) + World.collect -> lance.

Why self-collection instead of converting the released npz: the Berkeley host
(rail.eecs.berkeley.edu) is redirecting to an infrastructure-incident page
(CIFS shares down), and — more decisively — swm ships the exact collection stack
(envs/ogbench/expert_policy.py wraps OGBench's own oracles, with env_type branches
up to octuple), which is how cube_single_expert was evidently produced in the
first place. Same route = same data-generating process as the task we already run.

Episode length and count default to OGBench's official play-collection shape
(1001 steps/episode) scaled down to a --max-frames budget; if the reference
cube_single_expert lance is staged locally, its episode length is printed for
comparison but not enforced.

    usage: ogb_collect_multiobj.py cube_double --out <dir.lance> [--smoke]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.envs.ogbench.expert_policy import ExpertPolicy  # noqa: E402

import swm_ext.register  # noqa: E402,F401  (adds swm/OGBPuzzle-v0)
from swm_ext.expert_policy import PuzzleExpertPolicy  # noqa: E402

TASKS = {
    "cube_double": dict(env="swm/OGBCube-v0",
                        env_kwargs=dict(env_type="double", ob_type="states", multiview=False, mode="data_collection",
                                        visualize_info=False, terminate_at_goal=False)),
    "cube_triple": dict(env="swm/OGBCube-v0",
                        env_kwargs=dict(env_type="triple", ob_type="states", multiview=False, mode="data_collection",
                                        visualize_info=False, terminate_at_goal=False)),
    "cube_quadruple": dict(env="swm/OGBCube-v0",
                           env_kwargs=dict(env_type="quadruple", ob_type="states", multiview=False, mode="data_collection",
                                           visualize_info=False, terminate_at_goal=False)),
    "scene": dict(env="swm/OGBScene-v0",
                  env_kwargs=dict(ob_type="states", multiview=False, mode="data_collection",
                                  visualize_info=False, terminate_at_goal=False)),
    "puzzle_3x3": dict(env="swm/OGBPuzzle-v0",
                       env_kwargs=dict(env_type="3x3", ob_type="states", multiview=False, mode="data_collection",
                                       visualize_info=False, terminate_at_goal=False)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("--out", required=True, help="output .lance directory")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=200,
                    help="max env steps per episode (OGBench official play uses 1001; "
                         "200 x 2000 episodes matches our 400k-frame task scale)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    spec = TASKS[args.task]

    # reference: report cube_single_expert episode shape when available (informational)
    try:
        ref = swm.data.load_dataset("ogbench/cube_single_expert.lance", keys_to_load=["action"])
        ls = np.asarray(ref.lengths)
        print(f"[ref] cube_single_expert: {len(ls)} episodes, len {ls.min()}..{ls.max()} "
              f"(median {int(np.median(ls))})", flush=True)
    except Exception as e:
        print(f"[ref] cube_single_expert not staged locally ({type(e).__name__}); skipping", flush=True)

    n_ep = 2 if args.smoke else args.episodes
    world = swm.World(env_name=spec["env"], num_envs=1, image_shape=(224, 224),
                      max_episode_steps=args.steps, **spec["env_kwargs"])
    pol_cls = PuzzleExpertPolicy if args.task.startswith("puzzle") else ExpertPolicy
    policy = pol_cls(policy_type="markov_oracle", seed=args.seed)
    world.set_policy(policy)

    out = Path(args.out)
    print(f"[collect] {args.task}: {n_ep} episodes x <= {args.steps} steps -> {out}", flush=True)
    world.collect(path=str(out), episodes=n_ep, seed=args.seed, format="lance")
    world.close()

    ds = swm.data.load_dataset(str(out))
    ls = np.asarray(ds.lengths)
    print(f"[verify] episodes {len(ls)}, frames {int(ls.sum())}, len {ls.min()}..{ls.max()}", flush=True)
    row = ds.get_row_data([0, 1])
    keys = sorted(row.keys())
    print(f"[verify] columns: {keys}", flush=True)
    if "pixels" in row:
        px = ds._decode_images(row["pixels"].tolist()) if hasattr(ds, "_decode_images") else None
        if px is not None:
            px = np.asarray(px)
            print(f"[verify] pixels decoded {px.shape}, std {float(px.astype(np.float32).std()):.2f}", flush=True)
            assert px.std() > 5, "pixels look constant"
    needed = [k for k in keys if "effector" in k or "block" in k or k in ("qpos", "qvel", "action")]
    print(f"[verify] state-ish columns present: {needed}", flush=True)
    marker = "button" if args.task.startswith("puzzle") else "block"
    assert any(marker in k for k in keys) and "action" in keys, "missing q/action columns"
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
