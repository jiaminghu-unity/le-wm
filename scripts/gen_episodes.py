"""Generate a fixed paired-evaluation episode set for the budget sweep.

Follows the repo's Push-T eval protocol exactly (eval.py): valid start frames
are those with >= goal_offset+1 steps left in their trajectory; ``num`` rows
are drawn without replacement and sorted. Goal = state 25 env steps later in
the SAME trajectory. Each set is generated ONCE and shared by every
(config x tier) cell — never resample.

Default (no args) reproduces the original 50-episode set (seed 42). The
200-episode replication set uses an independent seed: --num 200 --seed 43.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import stable_worldmodel as swm

GOAL_OFFSET = 25
DEFAULT_OUT = Path(__file__).resolve().parent / "episodes_pusht_50.json"


def generate(num, seed, out):
    dataset = swm.data.HDF5Dataset(
        "pusht_expert_train",
        keys_to_cache=["action", "proprio", "state"],
        cache_dir=Path(swm.data.utils.get_cache_dir()),
    )
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_col = dataset.get_col_data(col_name)
    step_col = dataset.get_col_data("step_idx")

    ep_indices = np.unique(ep_col)
    lengths = {ep: np.max(step_col[ep_col == ep]) + 1 for ep in ep_indices}
    max_start_per_row = np.array([lengths[ep] - GOAL_OFFSET - 1 for ep in ep_col])
    valid_indices = np.nonzero(step_col <= max_start_per_row)[0]
    print(f"{len(valid_indices)} valid starting points")

    g = np.random.default_rng(seed)
    rows = g.choice(len(valid_indices) - 1, size=num, replace=False)
    rows = np.sort(valid_indices[rows])

    episodes = []
    for i, r in enumerate(rows):
        episodes.append(
            {
                "episode_id": i,
                "traj_id": int(ep_col[r]),
                "start_idx": int(step_col[r]),
                "goal_idx": int(step_col[r]) + GOAL_OFFSET,
                "env_seed": 20000 + i,
            }
        )

    payload = json.dumps(
        {"seed": seed, "goal_offset": GOAL_OFFSET, "episodes": episodes},
        indent=2,
        sort_keys=True,
    )
    out = Path(out)
    out.write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    print(f"wrote {out} ({num} episodes), sha256[:12]={digest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    generate(args.num, args.seed, args.out)


if __name__ == "__main__":
    main()
