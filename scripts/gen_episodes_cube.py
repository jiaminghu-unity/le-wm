"""Cube episode sampler with a non-trivial-goal filter.

Same protocol as scripts/gen_episodes.py (GOAL_OFFSET=25, uniform over valid starts,
fixed seed) plus one cube-specific constraint:

    ||block_pos(start + GOAL_OFFSET) - block_pos(start)|| > SUCCESS_RADIUS

Without it ~32% of uniformly sampled cube episodes are already solved at t=0: the
goal is the cube's own position 25 steps later, and the expert spends the early part
of every episode *reaching* before it ever touches the cube, so the target mocap gets
planted exactly where the cube already sits. Those episodes succeed for every policy,
inflating absolute SR by ~30pp and wasting a third of the sample. (Push-T never hits
this — its expert is pushing the whole time.)

Paired McNemar is unaffected by them either way; this filter is about absolute SR
being meaningful and about not throwing away statistical power.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

GOAL_OFFSET = 25
SUCCESS_RADIUS = 0.04  # CubeEnv._compute_successes threshold, metres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--env-seed-base", type=int, default=30000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    f = h5py.File(args.h5, "r")
    ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
    block = np.asarray(f["privileged_block_0_pos"][:], dtype=np.float64)

    # valid starts: the whole [start, start+GOAL_OFFSET] window inside the episode
    rows, traj, step = [], [], []
    for ep, (L, O) in enumerate(zip(ep_len, ep_off)):
        n = int(L) - GOAL_OFFSET - 1
        if n <= 0:
            continue
        idx = O + np.arange(n)
        rows.append(idx); traj.append(np.full(n, ep)); step.append(np.arange(n))
    rows = np.concatenate(rows); traj = np.concatenate(traj); step = np.concatenate(step)

    disp = np.linalg.norm(block[rows + GOAL_OFFSET] - block[rows], axis=1)
    keep = disp > SUCCESS_RADIUS
    print(f"valid starts: {len(rows)}")
    print(f"  goal already met at t=0 (<= {SUCCESS_RADIUS} m): "
          f"{int((~keep).sum())} ({100 * (~keep).mean():.1f}%) -- excluded")
    print(f"  eligible: {int(keep.sum())}")
    print("  displacement quantiles over eligible:",
          np.round(np.quantile(disp[keep], [0, .25, .5, .75, 1]), 4).tolist())

    elig = np.nonzero(keep)[0]
    g = np.random.default_rng(args.seed)
    pick = np.sort(g.choice(elig, size=args.num, replace=False))

    episodes = [
        {
            "episode_id": i,
            "traj_id": int(traj[r]),
            "start_idx": int(step[r]),
            "goal_idx": int(step[r]) + GOAL_OFFSET,
            "env_seed": args.env_seed_base + i,
            "block_displacement_m": round(float(disp[r]), 5),
        }
        for i, r in enumerate(pick)
    ]
    payload = json.dumps(
        {
            "seed": args.seed,
            "goal_offset": GOAL_OFFSET,
            "filter": {"min_block_displacement_m": SUCCESS_RADIUS},
            "episodes": episodes,
        },
        indent=2, sort_keys=True,
    )
    out = Path(args.out); out.write_text(payload)
    print(f"wrote {out} ({args.num} episodes) "
          f"sha256[:12]={hashlib.sha256(payload.encode()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
