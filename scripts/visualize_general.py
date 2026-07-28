"""General-purpose representation diagnostics panel (no ground-truth labels used).

  A. eigenvalue decay curves (covariance spectrum, log-log)
  B. pairwise distance & cosine-similarity distributions
  C. layer-wise CKA heatmaps between model pairs (where do the nets diverge)
  D. RDMs ordered by hierarchical clustering of a reference space (label-free)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, load_frames  # noqa: E402

MODELS = {
    "c1": ("C1 baseline", "lewm_c1_s3072/weights_epoch_10.pt", "#2a78d6"),
    "c3": ("C3 L_obj", "lewm_c3_sig_obj0.1_s3072/weights_epoch_10.pt", "#eda100"),
    "c5": ("C5 aux-MSE", "lewm_c5_qhead0.2_s3072/weights_epoch_10.pt", "#e87ba4"),
}
N = 1500
N_RDM = 350
TEXT, MUTED = "#3d3d3c", "#6f6e66"


@torch.no_grad()
def encode_layers(model, pix_list, device):
    """CLS token of every encoder layer + final projected embedding."""
    per_layer, final = [], []
    for pix in pix_list:
        out = model.encoder(pix.to(device), interpolate_pos_encoding=True,
                            output_hidden_states=True)
        cls = torch.stack([h[:, 0] for h in out.hidden_states], dim=1)  # (B, L+1, D)
        per_layer.append(cls.float().cpu())
        final.append(model.projector(out.last_hidden_state[:, 0]).float().cpu())
    return torch.cat(per_layer).numpy(), torch.cat(final).numpy()


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return np.linalg.norm(X.T @ Y, "fro") ** 2 / (
        np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro"))


def main():
    device = "cuda"
    dataset = swm.data.load_dataset("pusht_expert_train.lance", keys_to_load=["pixels", "state"])
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    test_eps = perm[: int(n_ep * TEST_EPISODE_FRAC)]
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)
    pool = np.concatenate([offsets[e] + np.arange(lengths[e]) for e in test_eps])
    rows = np.sort(g.choice(pool, N, replace=False))
    pix, _ = load_frames(dataset, rows, device)

    layers, finals = {}, {}
    for key, (label, ckpt, color) in MODELS.items():
        m = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        m.requires_grad_(False)
        layers[key], finals[key] = encode_layers(m, pix, device)
        del m; torch.cuda.empty_cache()

    fig = plt.figure(figsize=(13, 9.5), dpi=150)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15], hspace=0.42, wspace=0.3)

    # --- A: eigenvalue decay ---
    axA = fig.add_subplot(gs[0, 0])
    for key, (label, _, color) in MODELS.items():
        z = finals[key] - finals[key].mean(0)
        ev = np.linalg.eigvalsh(np.cov(z.T))[::-1]
        axA.plot(np.arange(1, len(ev) + 1), ev / ev.sum(), color=color, lw=1.8, label=label)
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel("eigenvalue rank", fontsize=8, color=MUTED)
    axA.set_ylabel("normalized variance", fontsize=8, color=MUTED)
    axA.set_title("A. covariance spectrum", fontsize=9, color=TEXT)
    axA.legend(frameon=False, fontsize=7, labelcolor=TEXT)

    # --- B: distance & cosine distributions ---
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
    for ax in (axB1, axB2):
        ax.tick_params(labelsize=7, colors=MUTED)
    axB1.legend(frameon=False, fontsize=7, labelcolor=TEXT)

    # --- C: layer-wise CKA heatmaps ---
    pairs = [("c1", "c3"), ("c1", "c5"), ("c3", "c5")]
    L = layers["c1"].shape[1]
    for j, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, j])
        M = np.zeros((L, L))
        for i in range(L):
            for k in range(L):
                M[i, k] = cka(layers[a][:, i], layers[b][:, k])
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set_title(f"C. layer CKA: {a} vs {b}", fontsize=9, color=TEXT)
        ax.set_xlabel(b, fontsize=7, color=MUTED); ax.set_ylabel(a, fontsize=7, color=MUTED)
        ax.tick_params(labelsize=6, colors=MUTED)
        if j == 2:
            fig.colorbar(im, ax=ax, fraction=0.046)

    # --- D: cluster-ordered RDMs (order from C1's own clustering, label-free) ---
    from scipy.cluster.hierarchy import linkage, leaves_list
    sub = g.choice(N, N_RDM, replace=False)
    ref = finals["c1"][sub]
    order = leaves_list(linkage(ref, method="ward"))
    for j, (key, (label, _, _)) in enumerate(MODELS.items()):
        ax = fig.add_subplot(gs[2, j])
        z = finals[key][sub][order]
        rdm = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        rdm = rdm / np.median(rdm)
        ax.imshow(np.clip(rdm, 0, 3), cmap="magma")
        ax.set_title(f"D. RDM ({label}), C1-cluster order", fontsize=9, color=TEXT)
        ax.set_xticks([]); ax.set_yticks([])

    fig.savefig("eval_results/viz_general_panel.png", facecolor="white", bbox_inches="tight")
    print("wrote eval_results/viz_general_panel.png")


if __name__ == "__main__":
    main()
