"""Cost-quality diagnostic (instructions §7.3).

On ~500 held-out frame pairs (t, goal) sampled like the eval protocol
(goal = 25 env steps ahead in the SAME episode), correlate the planning cost
||z_t - z_g||^2 with the physical distance ||q_t - q_g||^2 (standardized q,
training-time stats). Reports Pearson and Spearman — the direct evidence for
whether latent distance behaves like physical distance.

Usage: python scripts/cost_quality.py --config c1 lewm_c1_s3072/weights_epoch_10.pt
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import build_q_raw  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from scripts.probe import ENCODE_BS, SPLIT_SEED, TEST_EPISODE_FRAC, encode, load_frames  # noqa: E402

N_PAIRS = 500
GOAL_OFFSET = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs=2, required=True, metavar=("NAME", "CKPT"))
    ap.add_argument("--out", default="eval_results/cost_quality.csv")
    args = ap.parse_args()
    name, ckpt = args.config
    device = "cuda"

    dataset = swm.data.load_dataset(
        "pusht_expert_train.lance", keys_to_load=["pixels", "state"]
    )
    stats = json.loads(
        Path(
            swm.data.utils.get_cache_dir(sub_folder="datasets"),
            "pusht_expert_train.lance.q_stats.json",
        ).read_text()
    )
    q_mean = torch.tensor(stats["mean"])
    q_std = torch.tensor(stats["std"])

    # same held-out episodes as the probing split
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    test_eps = perm[: int(n_ep * TEST_EPISODE_FRAC)]

    # sample (episode, t) with t + GOAL_OFFSET inside the episode
    lengths = np.asarray(dataset.lengths)
    offsets = np.asarray(dataset.offsets)
    valid_eps = test_eps[lengths[test_eps] > GOAL_OFFSET + 1]
    eps = g.choice(valid_eps, N_PAIRS, replace=True)
    ts = g.integers(0, lengths[eps] - GOAL_OFFSET - 1)
    rows_t = offsets[eps] + ts
    rows_g = rows_t + GOAL_OFFSET

    model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
    model.requires_grad_(False)

    # duplicate rows are possible (pairs share frames); fetch unique rows once
    all_rows = np.concatenate([rows_t, rows_g])
    uniq_rows, inverse = np.unique(all_rows, return_inverse=True)
    pix, state = load_frames(dataset, uniq_rows, device)
    z = encode(model, pix, device)[torch.from_numpy(inverse)]
    state = state[inverse]

    q = (build_q_raw(torch.from_numpy(state)) - q_mean) / q_std
    z_t, z_g = z[:N_PAIRS], z[N_PAIRS:]
    q_t, q_g = q[:N_PAIRS], q[N_PAIRS:]

    dz = (z_t - z_g).pow(2).sum(-1).numpy()
    dq = (q_t - q_g).pow(2).sum(-1).numpy()

    pear = pearsonr(dz, dq)
    spear = spearmanr(dz, dq)
    print(f"[{name}] cost-quality over {N_PAIRS} pairs: "
          f"pearson={pear.statistic:.4f} (p={pear.pvalue:.2e})  "
          f"spearman={spear.statistic:.4f} (p={spear.pvalue:.2e})", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "n_pairs", "goal_offset", "pearson", "spearman"])
        if write_header:
            w.writeheader()
        w.writerow({
            "config": name, "n_pairs": N_PAIRS, "goal_offset": GOAL_OFFSET,
            "pearson": round(float(pear.statistic), 4),
            "spearman": round(float(spear.statistic), 4),
        })


if __name__ == "__main__":
    main()
