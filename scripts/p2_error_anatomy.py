"""P2: error-direction anatomy + per-start paired test for P1.

Consumes p1_cache_<model>.npz (imagined vs true terminal embeddings for the
same random candidates). For each model:
  - fit a linear pose probe q ~ W^T z on held-out frames (ridge),
  - project the imagination error e = z_hat - z_true onto the pose subspace
    col(W) vs its orthogonal complement,
  - report pose-fraction ||P e||^2 / ||e||^2 (random-direction baseline: 6/192).
Pre-registration: C5's pose-fraction < C1's (aux welds pose; errors pushed into
cost-harmless directions). Also runs the per-start Wilcoxon on P1's comparison
noise (C5 vs C1, C3 vs C1).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402
from utils import build_q_raw  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, encode, load_frames  # noqa: E402

MODELS = {
    "c1": "lewm_c1_s3072/weights_epoch_10.pt",
    "c3": "lewm_c3_sig_obj0.1_s3072/weights_epoch_10.pt",
    "c5_w03": "lewm_c5_qhead0.3_s3072/weights_epoch_10.pt",
}
N_FIT = 15000


def main():
    device = "cuda"
    dataset = swm.data.load_dataset("pusht_expert_train.lance", keys_to_load=["pixels", "state"])
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    test_eps = perm[: int(n_ep * TEST_EPISODE_FRAC)]
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)
    pool = np.concatenate([offsets[e] + np.arange(lengths[e]) for e in test_eps])
    rows = np.sort(g.choice(pool, N_FIT, replace=False))
    pix, cols = load_frames(dataset, rows, device)
    q = build_q_raw(torch.from_numpy(cols["state"]))
    q = ((q - q.mean(0)) / q.std(0)).numpy()

    # per-start comparison noise paired test
    caches = {n: np.load(f"eval_results/p1_cache_{n}.npz") for n in MODELS}
    per_start = {}
    for n, c in caches.items():
        eps = c["eps"]  # (S, N)
        per_start[n] = np.sqrt(2) * eps.std(axis=1)  # within-start (common-mode removed)
    print("=== P1 补充: 逐起点比较噪声 (Wilcoxon 配对, 20 起点) ===")
    for a in ["c3", "c5_w03"]:
        w = wilcoxon(per_start[a], per_start["c1"])
        print(f"  {a:7s} median={np.median(per_start[a]):.4f} vs c1 {np.median(per_start['c1']):.4f}"
              f"  p={w.pvalue:.4f}  (低于c1的起点数: {(per_start[a] < per_start['c1']).sum()}/20)")

    print("\n=== P2: 想象误差的位姿子空间占比 (随机方向基线 = 6/192 = 3.1%) ===")
    for n, ckpt in MODELS.items():
        model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        model.requires_grad_(False)
        z = encode(model, pix, device).numpy().astype(np.float64)
        del model
        torch.cuda.empty_cache()
        zc = z - z.mean(0)
        W = np.linalg.solve(zc.T @ zc + 1e-4 * np.eye(zc.shape[1]), zc.T @ (q - q.mean(0)))  # (192,6)
        Uq, _ = np.linalg.qr(W)  # orthonormal basis of pose subspace
        c = caches[n]
        e = (c["z_hat"] - c["z_final"]).reshape(-1, z.shape[1])  # imagination errors
        e_pose = e @ Uq  # coords in pose subspace
        frac = (e_pose ** 2).sum() / (e ** 2).sum()
        # also: error energy along the goal-difference direction (ranking-relevant axis)
        zg = np.repeat(c["z_goal"][:, None, :], c["z_hat"].shape[1], axis=1).reshape(-1, z.shape[1])
        gdir = zg - c["z_final"].reshape(-1, z.shape[1])
        gdir /= np.linalg.norm(gdir, axis=1, keepdims=True) + 1e-9
        frac_goal = ((e * gdir).sum(1) ** 2).sum() / (e ** 2).sum()
        print(f"  {n:7s} pose-fraction={100*frac:.1f}%   goal-axis fraction={100*frac_goal:.1f}%"
              f"  (随机基线 {100*6/192:.1f}% / {100/192:.1f}%)")


if __name__ == "__main__":
    main()
