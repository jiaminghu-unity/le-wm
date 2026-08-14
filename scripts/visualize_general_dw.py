"""General representation-diagnostics panel for the navigation tasks (two-room /
PointMaze), four models: baseline, L_obj, aux q-head, and the DINO-WM baseline.
Mirrors visualize_general_cube.py: covariance spectrum, pairwise distance & cosine
histograms, RSA against the physical q, layer-wise CKA, cluster-ordered RDMs.

    usage: python scripts/visualize_general_dw.py {tworoom|pointmaze}

DINO-WM is not a drop-in fourth checkpoint; two representation choices are made
explicitly rather than silently:

  * Its "final" embedding is the flattened 256x384 patch grid -- the exact vector
    its planner computes costs on. Mean-pooling the patches would be prettier but
    would diagnose a representation nothing ever plans in. The 98k dimensionality
    makes naive pairwise math explode, so every distance/cosine/spectrum panel is
    computed from the NxN Gram matrix instead (d2_ij = G_ii + G_jj - 2 G_ij;
    spectrum = eigvalsh of the double-centered Gram, whose nonzero eigenvalues
    equal the covariance spectrum's).
  * Its per-layer representation for CKA is the CLS token of each DINOv2 layer,
    the same summary token the LeWM ViT-Tiny panels use, so the CKA rows compare
    like with like. CKA itself is dimension-agnostic (192-d vs 384-d is fine).

The colors are the artifact pages' validated arm palette (base=gray as the neutral
reference arm, obj=indigo, aux=amber; teal, the spare validated hue, = DINO-WM).
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
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, load_frames  # noqa: E402
from scripts.visualize_general import cka, encode_layers  # noqa: E402

TASKS = {
    "tworoom": {
        "lance": "tworoom.lance",
        "qcol": "pos_agent",
        "qname": "agent position",
        "models": {
            "base": ("baseline", "lewm_t1_tworoom_s3072/weights_epoch_10.pt", "#6b7280"),
            "obj": ("L_obj l=.1", "lewm_t2_tworoom_obj0.1_s3072/weights_epoch_10.pt", "#4f46e5"),
            "aux": ("aux w=.1", "lewm_t5_tworoom_qhead0.1_s3072/weights_epoch_10.pt", "#d97706"),
            "dw": ("DINO-WM", "dinowm_tworoom_s3072/weights_epoch_10.pt", "#0d9488"),
        },
    },
    "pointmaze": {
        "lance": "pointmaze.lance",
        "qcol": "pos",
        "qname": "agent position",
        "models": {
            "base": ("baseline", "lewm_p1_pointmaze_s3072/weights_epoch_10.pt", "#6b7280"),
            "obj": ("L_obj l=.1", "lewm_p2_pointmaze_s3072/weights_epoch_10.pt", "#4f46e5"),
            "aux": ("aux w=.1", "lewm_p5_pointmaze_s3072/weights_epoch_10.pt", "#d97706"),
            "dw": ("DINO-WM", "dinowm_pointmaze_s3072/weights_epoch_10.pt", "#0d9488"),
        },
    },
}
N = 1500
N_RDM = 350
TEXT, MUTED = "#3d3d3c", "#6f6e66"
CKA_PAIRS = [("base", "obj"), ("base", "aux"), ("obj", "aux"), ("base", "dw")]


@torch.no_grad()
def encode_layers_dinowm(model, pix_list, device):
    """Per-layer CLS of the frozen DINOv2 backbone + the flattened patch grid the
    planner costs on. `interpolate_pos_encoding` is not passed: HF's Dinov2Model
    interpolates internally and its forward has no such kwarg."""
    per_layer, final = [], []
    for pix in pix_list:
        out = model.backbone(pix.to(device), output_hidden_states=True)
        cls = torch.stack([h[:, 0] for h in out.hidden_states], dim=1)  # (B, L+1, 384)
        per_layer.append(cls.float().cpu())
        final.append(out.last_hidden_state[:, 1:, :].flatten(1).float().cpu())
    return torch.cat(per_layer).numpy(), torch.cat(final).numpy()


def gram_geometry(z):
    """Gram matrix, squared-distance matrix, and cosine matrix, all NxN.

    The 98304-d DINO-WM embedding forbids any (N, N, D) broadcast -- that array
    would be terabytes -- so every pairwise panel reads from these instead."""
    z = z.astype(np.float64)
    G = z @ z.T
    sq = np.diag(G).copy()
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0)
    nrm = np.sqrt(np.maximum(sq, 1e-30))
    cos = G / (nrm[:, None] * nrm[None, :])
    return G, d2, cos


def gram_spectrum(G):
    """Covariance eigenvalues from the double-centered Gram: the N nonzero
    eigenvalues of Xc Xc^T / (N-1) equal those of the D x D covariance."""
    n = G.shape[0]
    J = np.eye(n) - 1.0 / n
    ev = np.linalg.eigvalsh(J @ G @ J)[::-1] / (n - 1)
    ev = np.clip(ev, 0.0, None)
    return ev[ev > ev.max() * 1e-12]


def main():
    task = sys.argv[1]
    spec = TASKS[task]
    models = spec["models"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"task={task} device={device}", flush=True)

    dataset = swm.data.load_dataset(spec["lance"], keys_to_load=["pixels", spec["qcol"]])
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    test_eps = perm[: int(n_ep * TEST_EPISODE_FRAC)]
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)
    pool = np.concatenate([offsets[e] + np.arange(lengths[e]) for e in test_eps])
    rows = np.sort(g.choice(pool, N, replace=False))
    print(f"held-out episodes={len(test_eps)} frames sampled={N}", flush=True)

    pix, cols = load_frames(dataset, rows, device, cols=(spec["qcol"],))
    q = cols[spec["qcol"]][:, :2]
    qs = (q - q.mean(0)) / q.std(0)
    print(f"q dim={q.shape[1]}", flush=True)

    layers, finals, geom = {}, {}, {}
    for key, (label, ckpt, _) in models.items():
        m = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        m.requires_grad_(False)
        enc = encode_layers_dinowm if key == "dw" else encode_layers
        layers[key], finals[key] = enc(m, pix, device)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        geom[key] = gram_geometry(finals[key])
        print(f"encoded {key}: layers {layers[key].shape} final {finals[key].shape}",
              flush=True)

    # ---- RSA vs the physical q ----
    sub = g.choice(N, N_RDM, replace=False)
    _, ideal_full, _ = gram_geometry(qs)
    ideal = ideal_full[np.ix_(sub, sub)]
    iu = np.triu_indices(N_RDM, 1)
    rsa = {}
    print(f"RSA vs {spec['qname']} ideal (Spearman):")
    for key in models:
        rdm = geom[key][1][np.ix_(sub, sub)]
        rsa[key] = spearmanr(rdm[iu], ideal[iu]).statistic
        print(f"  {key:4s} RSA={rsa[key]:.3f}", flush=True)

    fig = plt.figure(figsize=(17, 9.5), dpi=150)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.15], hspace=0.42, wspace=0.3)

    axA = fig.add_subplot(gs[0, 0])
    for key, (label, _, color) in models.items():
        ev = gram_spectrum(geom[key][0])
        axA.plot(np.arange(1, len(ev) + 1), ev / ev.sum(), color=color, lw=1.8, label=label)
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_title("A. covariance spectrum", fontsize=9, color=TEXT)
    axA.legend(frameon=False, fontsize=7, labelcolor=TEXT)

    axB1 = fig.add_subplot(gs[0, 1]); axB2 = fig.add_subplot(gs[0, 2])
    idx = g.integers(0, N, size=(20000, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    for key, (label, _, color) in models.items():
        _, d2, cosm = geom[key]
        d = np.sqrt(d2[idx[:, 0], idx[:, 1]])
        axB1.hist(d / d.mean(), bins=60, density=True, histtype="step", color=color,
                  lw=1.6, label=label)
        axB2.hist(cosm[idx[:, 0], idx[:, 1]], bins=60, density=True, histtype="step",
                  color=color, lw=1.6)
    axB1.set_title("B1. pairwise distance (/mean)", fontsize=9, color=TEXT)
    axB2.set_title("B2. pairwise cosine similarity", fontsize=9, color=TEXT)
    axB1.legend(frameon=False, fontsize=7, labelcolor=TEXT)

    axE = fig.add_subplot(gs[0, 3])
    keys = list(models)
    axE.bar(range(len(keys)), [rsa[k] for k in keys],
            color=[models[k][2] for k in keys], width=0.6)
    axE.set_xticks(range(len(keys)))
    axE.set_xticklabels([models[k][0] for k in keys], fontsize=7, color=TEXT)
    axE.set_ylim(0, 1)
    axE.set_title(f"E. RSA vs {spec['qname']} (Spearman)", fontsize=9, color=TEXT)
    for i, k in enumerate(keys):
        axE.text(i, rsa[k] + 0.02, f"{rsa[k]:.2f}", ha="center", fontsize=7, color=TEXT)

    for j, (a, b) in enumerate(CKA_PAIRS):
        ax = fig.add_subplot(gs[1, j])
        La, Lb = layers[a].shape[1], layers[b].shape[1]
        M = np.zeros((La, Lb))
        for i in range(La):
            for k in range(Lb):
                M[i, k] = cka(layers[a][:, i], layers[b][:, k])
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.set_title(f"C. layer CKA: {a} vs {b}", fontsize=8, color=TEXT)
        ax.set_xlabel(b, fontsize=7, color=MUTED); ax.set_ylabel(a, fontsize=7, color=MUTED)
        if j == len(CKA_PAIRS) - 1:
            fig.colorbar(im, ax=ax, fraction=0.046)

    order = leaves_list(linkage(finals["base"][sub], method="ward"))
    for j, (key, (label, _, _)) in enumerate(models.items()):
        ax = fig.add_subplot(gs[2, j])
        rdm = geom[key][1][np.ix_(sub[order], sub[order])]
        rdm = rdm / np.median(rdm)
        ax.imshow(np.clip(rdm, 0, 3), cmap="magma")
        ax.set_title(f"D. RDM ({label}), base-cluster order", fontsize=9, color=TEXT)
        ax.set_xticks([]); ax.set_yticks([])

    out = Path(f"eval_results/viz_general_{task}.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
