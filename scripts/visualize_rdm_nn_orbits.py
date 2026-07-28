"""Three cheap latent-space visualizations sharing one encoding pass (Push-T):

  1. RDM heatmaps: frames sorted by block angle -> frame x frame latent distance
     matrix. Metric-aligned spaces show banded structure along the diagonal;
     an 'ideal' RDM computed from q distances is included as reference.
  2. Nearest-neighbor retrieval strips: query frame + top-5 latent neighbors
     rendered as raw images per model.
  3. Trajectory orbits: episode latent sequences in each model's own PCA plane.
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
    "c1": ("C1 baseline", "lewm_c1_s3072/weights_epoch_10.pt"),
    "c3": ("C3 L_obj", "lewm_c3_sig_obj0.1_s3072/weights_epoch_10.pt"),
    "c5_l02": ("C5 aux-MSE", "lewm_c5_qhead0.2_s3072/weights_epoch_10.pt"),
    "c6_combo": ("C6 combo", "lewm_c6_combo_s3072/weights_epoch_10.pt"),
}
N_RDM = 400
N_QUERY = 4
N_NN = 5
N_ORBIT_EP = 6
ORBIT_LEN = 80
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
    rows_rdm = np.sort(g.choice(pool, N_RDM, replace=False))
    rows_nnpool = np.sort(g.choice(pool, 2000, replace=False))
    orbit_eps = g.choice(test_eps[lengths[test_eps] > ORBIT_LEN], N_ORBIT_EP, replace=False)
    rows_orbit = np.concatenate([offsets[e] + np.arange(ORBIT_LEN) for e in orbit_eps])

    all_rows = np.concatenate([rows_rdm, rows_nnpool, rows_orbit])
    uniq, inverse = np.unique(all_rows, return_inverse=True)
    pix, cols = load_frames(dataset, uniq, device)
    q = build_q_raw(torch.from_numpy(cols["state"])).numpy()
    q_std = (q - q.mean(0)) / q.std(0)

    i_rdm = inverse[:N_RDM]
    i_nn = inverse[N_RDM : N_RDM + 2000]
    i_orb = inverse[N_RDM + 2000:]

    # sort RDM frames by block angle
    ang = np.arctan2(q[i_rdm][:, 5], q[i_rdm][:, 4])
    order = np.argsort(ang)
    ideal = ((q_std[i_rdm][order][:, None, :] - q_std[i_rdm][order][None, :, :]) ** 2).sum(-1)

    # raw pixel access for NN strips (denormalize from the ImageNet-normalized cache)
    import stable_pretraining as spt
    imagenet = spt.data.dataset_stats.ImageNet
    mean = torch.tensor(imagenet["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(imagenet["std"]).view(1, 3, 1, 1)
    pix_all = torch.cat(pix)  # normalized (N,3,224,224) on cpu
    def img(idx):
        x = pix_all[idx].unsqueeze(0) * std + mean
        return x.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()

    queries = g.choice(2000, N_QUERY, replace=False)

    figR, axsR = plt.subplots(1, len(MODELS) + 1, figsize=(3.0 * (len(MODELS) + 1), 3.2), dpi=150)
    figN, axsN = plt.subplots(N_QUERY * len(MODELS), N_NN + 1,
                              figsize=(1.15 * (N_NN + 1), 1.15 * N_QUERY * len(MODELS)), dpi=130)
    figO, axsO = plt.subplots(1, len(MODELS), figsize=(3.0 * len(MODELS), 3.2), dpi=150)

    axsR[0].imshow(ideal, cmap="magma")
    axsR[0].set_title("ideal (q-space)", fontsize=9, color=TEXT)
    axsR[0].set_xticks([]); axsR[0].set_yticks([])

    for k, (mkey, (label, ckpt)) in enumerate(MODELS.items()):
        model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        model.requires_grad_(False)
        z = encode(model, pix, device).numpy()
        del model
        torch.cuda.empty_cache()

        # --- RDM ---
        zr = z[i_rdm][order]
        rdm = ((zr[:, None, :] - zr[None, :, :]) ** 2).sum(-1)
        ax = axsR[k + 1]
        ax.imshow(rdm, cmap="magma")
        ax.set_title(label, fontsize=9, color=TEXT)
        ax.set_xticks([]); ax.set_yticks([])

        # --- NN strips ---
        zn = z[i_nn]
        for qi, qq in enumerate(queries):
            d = ((zn - zn[qq]) ** 2).sum(-1)
            d[qq] = np.inf
            nn = np.argsort(d)[:N_NN]
            r = qi * len(MODELS) + k
            axsN[r, 0].imshow(img(i_nn[qq]))
            axsN[r, 0].set_ylabel(label.split()[0], fontsize=6, color=TEXT)
            for j, nb in enumerate(nn):
                axsN[r, j + 1].imshow(img(i_nn[nb]))
            for a in axsN[r]:
                a.set_xticks([]); a.set_yticks([])
            if r == 0:
                axsN[0, 0].set_title("query", fontsize=7, color=TEXT)
                for j in range(N_NN):
                    axsN[0, j + 1].set_title(f"nn{j+1}", fontsize=7, color=TEXT)

        # --- orbits ---
        zo = z[i_orb].reshape(N_ORBIT_EP, ORBIT_LEN, -1)
        flat = zo.reshape(-1, zo.shape[-1])
        flat = flat - flat.mean(0)
        _, _, vt = np.linalg.svd(flat, full_matrices=False)
        xy = (flat @ vt[:2].T).reshape(N_ORBIT_EP, ORBIT_LEN, 2)
        ax = axsO[k]
        for e in range(N_ORBIT_EP):
            ax.plot(xy[e, :, 0], xy[e, :, 1], lw=1.2, alpha=0.8)
            ax.scatter(xy[e, 0, 0], xy[e, 0, 1], s=14, marker="o")
        ax.set_title(label, fontsize=9, color=TEXT)
        ax.set_xticks([]); ax.set_yticks([])

    figR.suptitle("RDM: frames sorted by block angle — latent pairwise distance", fontsize=10, color=TEXT)
    figR.tight_layout()
    figR.savefig("eval_results/viz_rdm.png", facecolor="white", bbox_inches="tight")
    figN.tight_layout()
    figN.savefig("eval_results/viz_nn_strips.png", facecolor="white", bbox_inches="tight")
    figO.suptitle("Episode orbits in each model's PCA plane", fontsize=10, color=TEXT)
    figO.tight_layout()
    figO.savefig("eval_results/viz_orbits.png", facecolor="white", bbox_inches="tight")
    print("wrote eval_results/viz_{rdm,nn_strips,orbits}.png")


if __name__ == "__main__":
    main()
