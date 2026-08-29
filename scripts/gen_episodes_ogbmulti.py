"""Episode sampler for the self-collected OGBench multi-object lance datasets
(cube_double / scene), mirroring gen_episodes_cube.py's protocol: GOAL_OFFSET=25,
uniform over valid starts, fixed seed, plus a non-trivial-goal filter.

Non-trivial means the goal frame differs from the start frame by more than the
env's own success tolerance in at least one goal-relevant coordinate:
    cube_double : any block moved > 0.04 m
    scene       : block moved > 0.04 m, or |drawer|/|window| moved > 0.04,
                  or any button toggled
Without it a large share of episodes is already solved at t=0 (the expert spends
the first steps reaching), which inflates absolute SR and wastes sample.

    usage: gen_episodes_ogbmulti.py --task cube_double --lance <dir> --num 100
           --seed 101 --env-seed-base 30000 --out episodes_cube_double_s101_100.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

GOAL_OFFSET = 25
TOL = 0.04  # env success tolerance (metres / slide units)

COLS = {
    "cube_double": ["privileged/block_0_pos", "privileged/block_1_pos"],
    "scene": ["privileged/block_0_pos", "privileged/drawer_pos",
              "privileged/window_pos", "privileged/button_0_state",
              "privileged/button_1_state"],
}


def nontrivial(task, cols, rows):
    a, b = rows, rows + GOAL_OFFSET
    if task == "cube_double":
        d0 = np.linalg.norm(cols["privileged/block_0_pos"][b] - cols["privileged/block_0_pos"][a], axis=-1)
        d1 = np.linalg.norm(cols["privileged/block_1_pos"][b] - cols["privileged/block_1_pos"][a], axis=-1)
        return np.maximum(d0, d1) > TOL
    blk = np.linalg.norm(cols["privileged/block_0_pos"][b] - cols["privileged/block_0_pos"][a], axis=-1) > TOL
    drw = np.abs(cols["privileged/drawer_pos"][b] - cols["privileged/drawer_pos"][a]).reshape(len(rows), -1)[:, 0] > TOL
    win = np.abs(cols["privileged/window_pos"][b] - cols["privileged/window_pos"][a]).reshape(len(rows), -1)[:, 0] > TOL
    bt0 = (cols["privileged/button_0_state"][b] != cols["privileged/button_0_state"][a]).reshape(len(rows), -1)[:, 0]
    bt1 = (cols["privileged/button_1_state"][b] != cols["privileged/button_1_state"][a]).reshape(len(rows), -1)[:, 0]
    return blk | drw | win | bt0 | bt1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(COLS))
    ap.add_argument("--lance", required=True)
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--env-seed-base", type=int, default=30000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from stable_worldmodel.data.formats.lance import LanceDataset
    need = COLS[args.task]
    ds = LanceDataset(path=args.lance, keys_to_load=need)
    ep = np.asarray(ds.get_col_data("episode_idx")).reshape(-1)
    st = np.asarray(ds.get_col_data("step_idx")).reshape(-1)
    cols = {c: np.asarray(ds.get_col_data(c), dtype=np.float64) for c in need}

    # valid starts: whole [start, start+GOAL_OFFSET] window inside one episode
    order = np.lexsort((st, ep))
    assert (order == np.arange(len(ep))).all(), "lance rows not episode-major sorted"
    ok = np.zeros(len(ep), bool)
    n = len(ep) - GOAL_OFFSET
    ok[:n] = ep[:n] == ep[GOAL_OFFSET:]
    rows = np.nonzero(ok)[0]

    keep = nontrivial(args.task, cols, rows)
    print(f"valid starts: {len(rows)}; trivial (goal met at t=0): "
          f"{int((~keep).sum())} ({100 * (~keep).mean():.1f}%) -- excluded; eligible {int(keep.sum())}")

    g = np.random.default_rng(args.seed)
    pick = np.sort(g.choice(rows[keep], size=args.num, replace=False))
    episodes = [
        {"episode_id": i, "traj_id": int(ep[r]), "start_idx": int(st[r]),
         "goal_idx": int(st[r]) + GOAL_OFFSET, "env_seed": args.env_seed_base + i}
        for i, r in enumerate(pick)
    ]
    payload = json.dumps(
        {"seed": args.seed, "task": args.task, "goal_offset": GOAL_OFFSET,
         "filter": {"tolerance": TOL}, "episodes": episodes},
        indent=2, sort_keys=True)
    Path(args.out).write_text(payload)
    print(f"wrote {args.out} ({args.num} eps) sha256[:12]="
          f"{hashlib.sha256(payload.encode()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
