"""t-SNE panels of the latent manifold per model (Push-T, held-out frames).

Qualitative companion to viz_pca_angle.png: t-SNE preserves local neighborhoods
but distorts global distances — use for 'is the manifold locally organized by
physical state', never for metric claims (that's the distance-scatter figure).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402
from utils import build_q_raw  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, encode, load_frames  # noqa: E402

MODELS = {
    "c1": ("C1 baseline", "lewm_c1_s3072/weights_epoch_10.pt"),
    "c3": ("C3 L_obj", "lewm_c3_sig_obj0.1_s3072/weights_epoch_10.pt"),
    "c5_l02": ("C5 aux-MSE", "lewm_c5_qhead0.2_s3072/weights_epoch_10.pt"),
    "c6_combo": ("C6 combo", "lewm_c6_combo_s3072/weights_epoch_10.pt"),
}
N_MAP = 3000
TEXT = "#3d3d3c"


def main():
    device = "cuda"
    dataset = swm.data.load_dataset("pusht_expert_train.lance", keys_to_load=["pixels", "state"])
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    test_eps = perm[: int(n_ep * TEST_EPISODE_FRAC)]
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)
    pool = np.concatenate([offsets[e] + np.arange(lengths[e]) for e in test_eps])
    rows = np.sort(g.choice(pool, N_MAP, replace=False))

    pix, cols = load_frames(dataset, rows, device)
    q = build_q_raw(torch.from_numpy(cols["state"])).numpy()
    colorings = [
        ("block angle", np.arctan2(q[:, 5], q[:, 4]), "twilight"),
        ("block x", q[:, 2], "viridis"),
        ("block y", q[:, 3], "viridis"),
    ]

    fig, axs = plt.subplots(len(colorings), len(MODELS),
                            figsize=(3.0 * len(MODELS), 2.9 * len(colorings)), dpi=150)

    for k, (mkey, (label, ckpt)) in enumerate(MODELS.items()):
        model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        model.requires_grad_(False)
        z = encode(model, pix, device).numpy()
        del model
        torch.cuda.empty_cache()

        z50 = PCA(n_components=50, random_state=0).fit_transform(z)
        xy = TSNE(n_components=2, perplexity=30, init="pca",
                  random_state=0).fit_transform(z50)

        for r, (cname, cval, cmap) in enumerate(colorings):
            ax = axs[r, k]
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=cval, cmap=cmap, s=4, alpha=0.7, edgecolors="none")
            if r == 0:
                ax.set_title(label, fontsize=10, color=TEXT)
            if k == 0:
                ax.set_ylabel(cname, fontsize=9, color=TEXT)
            ax.set_xticks([]); ax.set_yticks([])
            if k == len(MODELS) - 1:
                fig.colorbar(sc, ax=ax, fraction=0.05)

    fig.suptitle("t-SNE of held-out latents (local structure only — distances not metric)",
                 fontsize=10, color=TEXT)
    fig.tight_layout()
    fig.savefig("eval_results/viz_tsne.png", facecolor="white", bbox_inches="tight")
    print("wrote eval_results/viz_tsne.png")


if __name__ == "__main__":
    main()
