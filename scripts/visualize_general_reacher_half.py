"""Representation diagnostics for the Reacher reduced-q question: what did L_obj
trained on HALF the q (shoulder cos/sin only) do to the geometry, next to baseline,
full-q L_obj, and the aux arm?

Same battery as visualize_general_reacher.py (untouched): covariance spectrum,
pairwise distance & cosine histograms, layer CKA, base-cluster-ordered RDMs. The
E panel is where the half-q question lives, so it is split three ways instead of
one RSA bar per model:

    full q     [cos q0, sin q0, cos q1, sin q1]   what full-q L_obj aligns to
    kept half  [cos q0, sin q0]  (shoulder)       what half-q L_obj aligns to
    dropped    [cos q1, sin q1]  (elbow)          never in half-q training

If the half-q model's geometry carries the elbow anyway, the alignment generalised
beyond its training signal; if its "dropped" bar sits at baseline level, the loss
sculpted only what it was shown. That is the figure's single question.

Arms (canonical grid checkpoints; hq trained by the half-q round):
    base    lewm_r1_reacher_s3072
    obj     lewm_r2_reacher_paep_l015_s3072      L_obj l=0.15, full 4-d q
    obj_h   lewm_hq_obj_reacher_s3072            L_obj l=0.15, shoulder-only q
    aux     lewm_r5_qhead0.4_s3072               aux w=0.4, full q
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
from utils import build_q_reacher_joints  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, load_frames  # noqa: E402
from scripts.visualize_general import cka, encode_layers  # noqa: E402

MODELS = {
    "base": ("baseline", "lewm_r1_reacher_s3072/weights_epoch_10.pt", "#6b7280"),
    "obj": ("L_obj l=.15 (full q)", "lewm_r2_reacher_paep_l015_s3072/weights_epoch_10.pt", "#4f46e5"),
    "obj_h": ("L_obj l=.15 (half q)", "lewm_hq_obj_reacher_s3072/weights_epoch_10.pt", "#8a87ec"),
    "aux": ("aux w=.4", "lewm_r5_qhead0.4_s3072/weights_epoch_10.pt", "#d97706"),
}
# q slices in build_q_reacher_joints order [cos q0, sin q0, cos q1, sin q1]
REFS = {"full q (4d)": slice(0, 4), "kept: shoulder": slice(0, 2), "dropped: elbow": slice(2, 4)}
N = 1500
N_RDM = 350
TEXT, MUTED = "#3d3d3c", "#6f6e66"
CKA_PAIRS = [("base", "obj"), ("base", "obj_h"), ("obj", "obj_h"), ("obj_h", "aux")]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
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
    for key, (label, ckpt, _) in MODELS.items():
        m = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        m.requires_grad_(False)
        layers[key], finals[key] = encode_layers(m, pix, device)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"encoded {key}", flush=True)

    # ---- RSA against the three references ----
    sub = g.choice(N, N_RDM, replace=False)
    iu = np.triu_indices(N_RDM, 1)
    rsa = {r: {} for r in REFS}
    for rname, sl in REFS.items():
        ref = qs[sub][:, sl]
        ideal = ((ref[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        for key in MODELS:
            z = finals[key][sub]
            rdm = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
            rsa[rname][key] = spearmanr(rdm[iu], ideal[iu]).statistic
        print(f"RSA vs {rname}: " +
              "  ".join(f"{k}={rsa[rname][k]:.3f}" for k in MODELS), flush=True)

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
        axB1.hist(d / d.mean(), bins=60, density=True, histtype="step", color=color,
                  lw=1.6, label=label)
        zn = z / np.linalg.norm(z, axis=1, keepdims=True)
        cos = (zn[idx[:, 0]] * zn[idx[:, 1]]).sum(1)
        axB2.hist(cos, bins=60, density=True, histtype="step", color=color, lw=1.6)
    axB1.set_title("B1. pairwise distance (/mean)", fontsize=9, color=TEXT)
    axB2.set_title("B2. pairwise cosine similarity", fontsize=9, color=TEXT)
    axB1.legend(frameon=False, fontsize=7, labelcolor=TEXT)

    axE = fig.add_subplot(gs[0, 3])
    keys = list(MODELS)
    width = 0.26
    for j, (rname, _) in enumerate(REFS.items()):
        xs = np.arange(len(keys)) + (j - 1) * width
        vals = [rsa[rname][k] for k in keys]
        axE.bar(xs, vals, width=width * 0.92,
                color=[MODELS[k][2] for k in keys],
                alpha=(1.0, 0.62, 0.3)[j], label=rname)
        for x, v in zip(xs, vals):
            axE.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=5.6, color=TEXT)
    axE.set_xticks(range(len(keys)))
    axE.set_xticklabels(["base", "obj", "obj half", "aux"], fontsize=7.5, color=TEXT)
    axE.set_ylim(0, 1)
    axE.legend(frameon=False, fontsize=6.4, labelcolor=TEXT, loc="upper left")
    axE.set_title("E. RSA vs joints pose (Spearman)", fontsize=9, color=TEXT)

    L = layers["base"].shape[1]
    for j, (a, b) in enumerate(CKA_PAIRS):
        ax = fig.add_subplot(gs[1, j])
        M = np.zeros((L, L))
        for i in range(L):
            for k in range(L):
                M[i, k] = cka(layers[a][:, i], layers[b][:, k])
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set_title(f"C. layer CKA: {a} vs {b}", fontsize=8, color=TEXT)
        ax.set_xlabel(b, fontsize=7, color=MUTED); ax.set_ylabel(a, fontsize=7, color=MUTED)
        if j == len(CKA_PAIRS) - 1:
            fig.colorbar(im, ax=ax, fraction=0.046)

    order = leaves_list(linkage(finals["base"][sub], method="ward"))
    for j, (key, (label, _, _)) in enumerate(MODELS.items()):
        ax = fig.add_subplot(gs[2, j])
        z = finals[key][sub][order]
        rdm = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        rdm = rdm / np.median(rdm)
        ax.imshow(np.clip(rdm, 0, 3), cmap="magma")
        ax.set_title(f"D. RDM ({label}), base-cluster order", fontsize=9, color=TEXT)
        ax.set_xticks([]); ax.set_yticks([])

    out = Path("eval_results/viz_general_reacher_half.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
