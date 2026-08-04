"""General representation diagnostics for the OGBench-Cube models, mirroring
visualize_general_reacher.py: covariance spectrum, pairwise distance & cosine
histograms, layer-wise CKA, cluster-ordered RDMs, plus physics-referenced RSA.

Four models, chosen so the panel answers the round's central question directly:
    k1  baseline                    (SIGReg only)
    k2  L_obj, 9-dim effector q, l=0.1      -- geometry arm
    k4  aux q-head, same q,        w=0.1    -- information-injection control
    k7  L_obj, same q,             l=0.2    -- dose point for the geometry arm
The k2-vs-k4 CKA/RDM pair is the one that matters: on Push-T those two arms were
statistically indistinguishable in planning, so any representational difference
here is the first sign the two routes are not the same thing.

RSA reference is the 9-dim cube q (effector xyz, cos2psi/sin2psi, gripper opening,
block xyz) -- the same vector L_obj aligns to and the aux head regresses.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402
from utils import build_q_cube_effector  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, load_frames  # noqa: E402
from scripts.visualize_general import cka, encode_layers  # noqa: E402

MODELS = {
    "k1": ("K1 baseline", "lewm_k1_cube_s3072/weights_epoch_10.pt", "#2a78d6"),
    "k2": ("K2 L_obj l=.1", "lewm_k2_cube_obj_eff0.1_s3072/weights_epoch_10.pt", "#eb6834"),
    "k4": ("K4 aux w=.1", "lewm_k4_cube_qhead_eff0.1_s3072/weights_epoch_10.pt", "#e87ba4"),
    "k7": ("K7 L_obj l=.2", "lewm_k7_cube_obj_eff0.2_s3072/weights_epoch_10.pt", "#3f9e6a"),
}
Q_COLS = ("proprio_effector_pos", "proprio_effector_yaw",
          "proprio_gripper_opening", "privileged_block_0_pos")
N = 1500
N_RDM = 350
TEXT, MUTED = "#3d3d3c", "#6f6e66"
CKA_PAIRS = [("k1", "k2"), ("k1", "k4"), ("k2", "k4"), ("k2", "k7")]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    dataset = swm.data.load_dataset(
        "ogbench/cube_single_expert.lance", keys_to_load=["pixels", *Q_COLS]
    )
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    test_eps = perm[: int(n_ep * TEST_EPISODE_FRAC)]
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)
    pool = np.concatenate([offsets[e] + np.arange(lengths[e]) for e in test_eps])
    rows = np.sort(g.choice(pool, N, replace=False))
    print(f"held-out episodes={len(test_eps)} frames sampled={N}", flush=True)

    pix, cols = load_frames(dataset, rows, device, cols=Q_COLS)
    q = build_q_cube_effector(*[torch.from_numpy(cols[c]) for c in Q_COLS]).numpy()
    qs = (q - q.mean(0)) / q.std(0)
    print(f"q dim={q.shape[1]}", flush=True)

    layers, finals = {}, {}
    for key, (label, ckpt, _) in MODELS.items():
        m = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        m.requires_grad_(False)
        layers[key], finals[key] = encode_layers(m, pix, device)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"encoded {key}", flush=True)

    # ---- RSA vs the cube pose ----
    sub = g.choice(N, N_RDM, replace=False)
    ideal = ((qs[sub][:, None, :] - qs[sub][None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(N_RDM, 1)
    rsa = {}
    print("RSA vs cube-pose ideal (Spearman):")
    for key in MODELS:
        z = finals[key][sub]
        rdm = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        rsa[key] = spearmanr(rdm[iu], ideal[iu]).statistic
        print(f"  {key:4s} RSA={rsa[key]:.3f}", flush=True)

    fig = plt.figure(figsize=(17, 9.5), dpi=150)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.15], hspace=0.42, wspace=0.3)

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

    axE = fig.add_subplot(gs[0, 3])
    keys = list(MODELS)
    axE.bar(range(len(keys)), [rsa[k] for k in keys],
            color=[MODELS[k][2] for k in keys], width=0.6)
    axE.set_xticks(range(len(keys))); axE.set_xticklabels(keys, fontsize=8, color=TEXT)
    axE.set_ylim(0, 1)
    axE.set_title("E. RSA vs cube pose (Spearman)", fontsize=9, color=TEXT)
    for i, k in enumerate(keys):
        axE.text(i, rsa[k] + 0.02, f"{rsa[k]:.2f}", ha="center", fontsize=7, color=TEXT)

    L = layers["k1"].shape[1]
    for j, (a, b) in enumerate(CKA_PAIRS):
        ax = fig.add_subplot(gs[1, j])
        M = np.zeros((L, L))
        for i in range(L):
            for k in range(L):
                M[i, k] = cka(layers[a][:, i], layers[b][:, k])
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set_title(f"C. layer CKA: {a} vs {b}", fontsize=8, color=TEXT)
        if j == len(CKA_PAIRS) - 1:
            fig.colorbar(im, ax=ax, fraction=0.046)

    ref = finals["k1"][sub]
    order = leaves_list(linkage(ref, method="ward"))
    for j, (key, (label, _, _)) in enumerate(MODELS.items()):
        ax = fig.add_subplot(gs[2, j])
        z = finals[key][sub][order]
        rdm = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        rdm = rdm / np.median(rdm)
        ax.imshow(np.clip(rdm, 0, 3), cmap="magma")
        ax.set_title(f"D. RDM ({label})", fontsize=9, color=TEXT)
        ax.set_xticks([]); ax.set_yticks([])

    out = Path("eval_results/viz_general_cube.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
