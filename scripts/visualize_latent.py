"""Visualize latent-space geometry differences across models (Push-T).

Three eval-only figures over held-out episodes (same split as probing):
  A. distance scatter: ||dq||^2 vs ||dz||^2 for 500 pairs (the cost-quality
     number rendered as its raw cloud), one panel per model
  B. cost-to-goal profile: latent distance to the goal frame as a function of
     steps-to-goal along real trajectories (the terrain CEM descends)
  C. PCA map of the latent manifold colored by a physical coordinate
     (PCA, not t-SNE: nonlinear embeddings distort the very metric we study)
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
from utils import build_q_raw  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, encode, load_frames  # noqa: E402

MODELS = {
    "c1": ("C1 baseline", "lewm_c1_s3072/weights_epoch_10.pt", "#2a78d6"),
    "c3": ("C3 L_obj", "lewm_c3_sig_obj0.1_s3072/weights_epoch_10.pt", "#eda100"),
    "c5_l02": ("C5 aux-MSE", "lewm_c5_qhead0.2_s3072/weights_epoch_10.pt", "#e87ba4"),
    "c6_combo": ("C6 combo", "lewm_c6_combo_s3072/weights_epoch_10.pt", "#1baf7a"),
}
N_PAIRS = 500
GOAL_OFFSET = 25
N_TRACE_EP = 40
TRACE_LEN = 50
N_MAP = 3000
TEXT, MUTED = "#3d3d3c", "#6f6e66"


def main():
    device = "cuda"
    dataset = swm.data.load_dataset("pusht_expert_train.lance", keys_to_load=["pixels", "state"])
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    test_eps = perm[: int(n_ep * TEST_EPISODE_FRAC)]
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)

    # ---- row sets (identical across models) ----
    valid = test_eps[lengths[test_eps] > GOAL_OFFSET + 1]
    eps_pairs = g.choice(valid, N_PAIRS, replace=True)
    ts = g.integers(0, lengths[eps_pairs] - GOAL_OFFSET - 1)
    rows_pair = np.concatenate([offsets[eps_pairs] + ts, offsets[eps_pairs] + ts + GOAL_OFFSET])

    trace_eps = g.choice(test_eps[lengths[test_eps] > TRACE_LEN + 1], N_TRACE_EP, replace=False)
    trace_starts = g.integers(0, lengths[trace_eps] - TRACE_LEN - 1)
    rows_trace = np.concatenate(
        [offsets[e] + s + np.arange(TRACE_LEN + 1) for e, s in zip(trace_eps, trace_starts)]
    )

    map_pool = np.concatenate([offsets[e] + np.arange(lengths[e]) for e in test_eps])
    rows_map = np.sort(g.choice(map_pool, N_MAP, replace=False))

    all_rows = np.concatenate([rows_pair, rows_trace, rows_map])
    uniq, inverse = np.unique(all_rows, return_inverse=True)
    pix, cols = load_frames(dataset, uniq, device)
    state = cols["state"]
    q = build_q_raw(torch.from_numpy(state)).numpy()
    q_std = (q - q.mean(0)) / q.std(0)

    i_pair = inverse[: 2 * N_PAIRS]
    i_trace = inverse[2 * N_PAIRS : 2 * N_PAIRS + len(rows_trace)]
    i_map = inverse[2 * N_PAIRS + len(rows_trace):]

    dq = ((q_std[i_pair[:N_PAIRS]] - q_std[i_pair[N_PAIRS:]]) ** 2).sum(-1)

    from scipy.stats import pearsonr, spearmanr

    figA, axsA = plt.subplots(1, len(MODELS), figsize=(3.1 * len(MODELS), 3.2), dpi=150)
    figB, axB = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    figC, axsC = plt.subplots(1, len(MODELS), figsize=(3.1 * len(MODELS), 3.4), dpi=150)

    for k, (mkey, (label, ckpt, color)) in enumerate(MODELS.items()):
        model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        model.requires_grad_(False)
        z = encode(model, pix, device).numpy()
        del model
        torch.cuda.empty_cache()

        # --- A: distance scatter ---
        dz = ((z[i_pair[:N_PAIRS]] - z[i_pair[N_PAIRS:]]) ** 2).sum(-1)
        ax = axsA[k]
        ax.scatter(dq, dz / dz.mean(), s=6, alpha=0.35, color=color, edgecolors="none")
        ax.set_title(f"{label}\nr={pearsonr(dz, dq).statistic:.2f}  "
                     f"rho={spearmanr(dz, dq).statistic:.2f}", fontsize=9, color=TEXT)
        ax.set_xlabel("||dq||^2 (physical)", fontsize=8, color=MUTED)
        if k == 0:
            ax.set_ylabel("||dz||^2 (latent, /mean)", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)

        # --- B: cost-to-goal profile ---
        zt = z[i_trace].reshape(N_TRACE_EP, TRACE_LEN + 1, -1)
        d2goal = ((zt - zt[:, -1:, :]) ** 2).sum(-1)
        d2goal = d2goal[:, :-1] / d2goal[:, :1].clip(min=1e-9)
        steps_to_goal = np.arange(TRACE_LEN, 0, -1)
        mean = d2goal.mean(0)
        se = d2goal.std(0) / np.sqrt(N_TRACE_EP)
        axB.plot(steps_to_goal, mean, color=color, lw=2, label=label)
        axB.fill_between(steps_to_goal, mean - se, mean + se, color=color, alpha=0.15, lw=0)

        # --- C: PCA map colored by block angle ---
        zm = z[i_map] - z[i_map].mean(0)
        _, _, vt = np.linalg.svd(zm, full_matrices=False)
        xy = zm @ vt[:2].T
        ang = np.arctan2(q[i_map][:, 5], q[i_map][:, 4])
        ax = axsC[k]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=ang, cmap="twilight", s=5, alpha=0.6, edgecolors="none")
        ax.set_title(label, fontsize=9, color=TEXT)
        ax.set_xticks([]); ax.set_yticks([])
        if k == len(MODELS) - 1:
            figC.colorbar(sc, ax=ax, fraction=0.05, label="block angle")

    axB.invert_xaxis()
    axB.set_xlabel("steps to goal", fontsize=9, color=TEXT)
    axB.set_ylabel("latent cost to goal (normalized)", fontsize=9, color=TEXT)
    axB.set_title("Cost-to-goal terrain along held-out trajectories", fontsize=10, color=TEXT)
    axB.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    axB.grid(True, axis="y", color="#e8e7e0", lw=0.8)
    for s in ["top", "right"]:
        axB.spines[s].set_visible(False)

    out = Path("eval_results")
    figA.tight_layout(); figA.savefig(out / "viz_distance_scatter.png", facecolor="white", bbox_inches="tight")
    figB.tight_layout(); figB.savefig(out / "viz_cost_to_goal.png", facecolor="white", bbox_inches="tight")
    figC.tight_layout(); figC.savefig(out / "viz_pca_angle.png", facecolor="white", bbox_inches="tight")
    print("wrote eval_results/viz_{distance_scatter,cost_to_goal,pca_angle}.png")


if __name__ == "__main__":
    main()
