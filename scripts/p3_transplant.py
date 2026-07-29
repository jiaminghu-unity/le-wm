"""P3: aligned predictor cross-transplant (space vs predictor attribution).

1. Fit linear maps between model embedding spaces on held-out frames
   (ridge + orthogonal Procrustes). High ridge R^2 with low Procrustes R^2
   would mean: same information, different metric geometry.
2. Transplant: feed model A's (mapped) start embeddings to model B's predictor
   and measure imagination error against A's (mapped) true terminal embeddings,
   using the SAME random candidates as P1 (generator seed 7). If C5's predictor
   keeps its advantage on C1-mapped inputs, the advantage lives in the predictor
   weights; if it collapses to C1's level, it lives in the space.
Alignment floor per (A->B): ||R z^A_final - z^B_final||^2 / scale_B.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, encode, load_frames  # noqa: E402
from scripts.p1_comparison_noise import MODELS, N_STARTS, N_CAND, HORIZON, ACTION_BLOCK, imagined_terminal  # noqa: E402

N_FIT = 15000
D = 192


def fit_maps(Za, Zb):
    """Return (ridge map incl. means, ridge R2, procrustes R2) for Za -> Zb, on a held-out half."""
    n = len(Za)
    tr, va = slice(0, n // 2), slice(n // 2, n)
    ma, mb = Za[tr].mean(0), Zb[tr].mean(0)
    A, B = Za[tr] - ma, Zb[tr] - mb
    R = np.linalg.solve(A.T @ A + 1e-3 * np.eye(D), A.T @ B)
    res = Zb[va] - ((Za[va] - ma) @ R + mb)
    r2 = 1 - (res ** 2).sum() / ((Zb[va] - Zb[va].mean(0)) ** 2).sum()
    Ro, sc = orthogonal_procrustes(A, B)
    res_o = (Zb[va] - mb) - (Za[va] - ma) @ Ro * (sc / (A ** 2).sum())
    r2_o = 1 - (res_o ** 2).sum() / ((Zb[va] - Zb[va].mean(0)) ** 2).sum()
    return (R, ma, mb), r2, r2_o


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
    pix, _ = load_frames(dataset, rows, device)

    gen = torch.Generator().manual_seed(7)
    cands = torch.randn(N_STARTS, N_CAND, HORIZON, ACTION_BLOCK * 2, generator=gen)

    caches = {n: np.load(f"eval_results/p1_cache_{n}.npz") for n in MODELS}
    Z, preds = {}, {}
    for n, ckpt in MODELS.items():
        m = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        m.requires_grad_(False)
        Z[n] = encode(m, pix, device).numpy().astype(np.float64)
        preds[n] = m

    names = list(MODELS)
    maps, floors = {}, {}
    print("=== 对齐质量 (A->B, held-out R^2) ===")
    for a in names:
        for b in names:
            if a == b:
                continue
            maps[(a, b)], r2, r2o = fit_maps(Z[a], Z[b])
            R, ma, mb = maps[(a, b)]
            zfa = caches[a]["z_final"].reshape(-1, D)
            mapped = (zfa - ma) @ R + mb
            fl = ((mapped - caches[b]["z_final"].reshape(-1, D)) ** 2).sum(1).mean() / caches[b]["scale"]
            floors[(a, b)] = fl
            print(f"  {a:7s}->{b:7s} ridge R2={r2:.3f}  procrustes R2={r2o:.3f}  floor={fl:.4f}")

    print("\n=== 移植矩阵: mean ||pred_B(inputs from A) - truth from A||^2 / scale_B ===")
    hdr = "enc/pred"
    print(f"{hdr:12s}" + "".join(f"{b:>10s}" for b in names))
    for a in names:
        row = []
        for b in names:
            c_a, c_b = caches[a], caches[b]
            if a == b:
                z0 = torch.from_numpy(c_a["z_start"]).float()
                zt = torch.from_numpy(c_a["z_final"]).float()
            else:
                R, ma, mb = maps[(a, b)]
                z0 = torch.from_numpy((c_a["z_start"] - ma) @ R + mb).float()
                zt = torch.from_numpy((c_a["z_final"].reshape(-1, D) - ma) @ R + mb).float()
                zt = zt.reshape(N_STARTS, N_CAND, D)
            errs = []
            for si in range(N_STARTS):
                zh = imagined_terminal(preds[b], z0[si:si + 1].to(device),
                                       cands[si].to(device), device).cpu()
                errs.append((zh - zt[si]).pow(2).sum(-1).mean().item())
            row.append(np.mean(errs) / c_b["scale"])
        print(f"{a:12s}" + "".join(f"{v:10.4f}" for v in row))


if __name__ == "__main__":
    main()
