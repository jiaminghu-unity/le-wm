"""General representation diagnostics for the Reacher models
(R1 baseline / R2 PAEP l=0.15 / R5 aux w=0.2), plus physics-referenced RSA.

Same battery as visualize_general.py: spectrum, distance & cosine histograms,
layer-wise CKA, cluster-ordered RDMs; RSA uses the joints (cos/sin) pose.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402
from utils import build_q_reacher_joints  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, load_frames  # noqa: E402
from scripts.visualize_general import cka, encode_layers  # noqa: E402

MODELS = {
    "r1": ("R1 baseline", "lewm_r1_reacher_s3072/weights_epoch_10.pt", "#2a78d6"),
    "r2_l015": ("R2 PAEP l=.15", "lewm_r2_reacher_paep_l015_s3072/weights_epoch_10.pt", "#eb6834"),
    "r5_l02": ("R5 aux w=.2", "lewm_r5_qhead0.2_s3072/weights_epoch_10.pt", "#e87ba4"),
}
N = 1500
N_RDM = 350
TEXT, MUTED = "#3d3d3c", "#6f6e66"


def main():
    device = "cuda"
    dataset = swm.data.load_dataset("reacher.lance", keys_to_load=["pixels", "qpos"])
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    test_eps = perm[: int(n_ep * TEST_EPISODE_FRAC)]
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)
    pool = np.concatenate([offsets[e] + np.arange(lengths[e]) for e in test_eps])
    rows = np.sort(g.choice(pool, N, replace=False))
    pix, cols = load_frames(dataset, rows, device, cols=("qpos",))
    q = build_q_reacher_joints(torch.from_numpy(cols["qpos"])).numpy()
    qs = (q - q.mean(0)) / q.std(0)

    layers, finals = {}, {}
    for key, (label, ckpt, color) in MODELS.items():
        m = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        m.requires_grad_(False)
        layers[key], finals[key] = encode_layers(m, pix, device)
        del m; torch.cuda.empty_cache()

    # RSA vs joints pose
    sub = g.choice(N, N_RDM, replace=False)
    ideal = ((qs[sub][:, None, :] - qs[sub][None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(N_RDM, 1)
    print("RSA vs joints-pose ideal:")
    for key in MODELS:
        z = finals[key][sub]
        rdm = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        print(f"  {key:8s} RSA={spearmanr(rdm[iu], ideal[iu]).statistic:.3f}")

    fig = plt.figure(figsize=(13, 9.5), dpi=150)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15], hspace=0.42, wspace=0.3)

    axA = fig.add_subplot(gs[0, 0])
    for key, (label, _, color) in MODELS.items():
        z = finals[key] - finals[key].mean(0)
        ev = np.linalg.eigvalsh(np.cov(z.T))[::-1]
        axA.plot(np.arange(1, len(ev) + 1), ev / ev.sum(), color=color, lw=1.8, label=label)
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_title("A. covariance spectrum", fontsize=9, color=TEXT)
    axA.legend(frameon=False, fontsize=7, labelcolor=TEXT)

    axB1 = fig.add_subplot(gs[0, 1]); axB2 = fig.add_subplot(gs[0, 2])
    idx = g.integers(0, N, size=(20000, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    for key, (label, _, color) in MODELS.items():
        z = finals[key]
        d = np.linalg.norm(z[idx[:, 0]] - z[idx[:, 1]], axis=1)
        axB1.hist(d / d.mean(), bins=60, density=True, histtype="step", color=color, lw=1.6, label=label)
        zn = z / np.linalg.norm(z, axis=1, keepdims=True)
        cos = (zn[idx[:, 0]] * zn[idx[:, 1]]).sum(1)
        axB2.hist(cos, bins=60, density=True, histtype="step", color=color, lw=1.6)
    axB1.set_title("B1. pairwise distance (/mean)", fontsize=9, color=TEXT)
    axB2.set_title("B2. pairwise cosine similarity", fontsize=9, color=TEXT)
    axB1.legend(frameon=False, fontsize=7, labelcolor=TEXT)

    pairs = [("r1", "r2_l015"), ("r1", "r5_l02"), ("r2_l015", "r5_l02")]
    L = layers["r1"].shape[1]
    for j, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, j])
        M = np.zeros((L, L))
        for i in range(L):
            for k in range(L):
                M[i, k] = cka(layers[a][:, i], layers[b][:, k])
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set_title(f"C. layer CKA: {a} vs {b}", fontsize=8, color=TEXT)
        if j == 2:
            fig.colorbar(im, ax=ax, fraction=0.046)

    from scipy.cluster.hierarchy import linkage, leaves_list
    ref = finals["r1"][sub]
    order = leaves_list(linkage(ref, method="ward"))
    for j, (key, (label, _, _)) in enumerate(MODELS.items()):
        ax = fig.add_subplot(gs[2, j])
        z = finals[key][sub][order]
        rdm = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        rdm = rdm / np.median(rdm)
        ax.imshow(np.clip(rdm, 0, 3), cmap="magma")
        ax.set_title(f"D. RDM ({label})", fontsize=9, color=TEXT)
        ax.set_xticks([]); ax.set_yticks([])

    fig.savefig("eval_results/viz_general_reacher.png", facecolor="white", bbox_inches="tight")
    print("wrote eval_results/viz_general_reacher.png")


if __name__ == "__main__":
    main()
