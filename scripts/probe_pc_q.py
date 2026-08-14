"""Do the TOP principal directions of the embedding carry q? (the reverse arrow)

The viz panels established the forward arrow: L_obj lifts RSA-to-q and reshapes the
head of the covariance spectrum. This probe tests the reverse direction: project the
embeddings onto their own top-k principal directions and regress q from those k
coordinates alone. If a small head (k of a few) already recovers q with high held-out
R^2 for the obj arm but not for baseline, the lifted head IS the q subspace and the
chain closes bidirectionally. If R^2 stays low, the spectrum lift has another cause.

Control: the same regression from the BOTTOM-k principal directions (the tail of the
nonzero spectrum). "q lives in the head" is only meaningful if the tail of equal size
does markedly worse -- with SIGReg pushing toward isotropy, information could plausibly
be spread everywhere.

Mechanics: PC scores come from the double-centered Gram matrix (eigvecs * sqrt(eigvals)
= the N x k projection coordinates), so no D x D covariance is ever formed and the
DINO-WM arm's 98304-d patch space costs the same as LeWM's 192-d. Ridge regression,
first half fit / second half held-out (probe.py's convention), q standardized per-dim;
reported R^2 is uniform-averaged over q dims, per-dim values go to the JSON.

    usage: probe_pc_q.py {tworoom|pointmaze|reacher}
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402
from utils import build_q_cube_effector, build_q_raw, build_q_reacher_joints  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, load_frames  # noqa: E402
from scripts.visualize_general import encode_layers  # noqa: E402
from scripts.visualize_general_dw import encode_layers_dinowm  # noqa: E402

CUBE_QCOLS = ("proprio_effector_pos", "proprio_effector_yaw",
              "proprio_gripper_opening", "privileged_block_0_pos")

TASKS = {
    "pusht": {
        "lance": "pusht_expert_train.lance", "qcols": ("state",),
        "build_q": lambda c: build_q_raw(torch.from_numpy(c["state"])).numpy(),
        "qdims": {"agent x": 0, "agent y": 1, "block x": 2, "block y": 3,
                  "cos th": 4, "sin th": 5},
        "models": {
            "base": ("baseline", "lewm_c1_s3072/weights_epoch_10.pt", "#6b7280"),
            "obj": ("L_obj", "lewm_c3_sig_obj0.1_s3072/weights_epoch_10.pt", "#4f46e5"),
            "aux": ("aux", "lewm_c5_qhead0.3_s3072/weights_epoch_10.pt", "#d97706"),
            "dw": ("DINO-WM", "dinowm_pusht_s3072/weights_epoch_10.pt", "#0d9488"),
        },
    },
    "cube": {
        "lance": "ogbench/cube_single_expert.lance", "qcols": CUBE_QCOLS,
        "build_q": lambda c: build_q_cube_effector(
            *[torch.from_numpy(c[k]) for k in CUBE_QCOLS]).numpy(),
        "qdims": {"eff x": 0, "eff y": 1, "eff z": 2, "cos 2psi": 3, "sin 2psi": 4,
                  "gripper": 5, "block x": 6, "block y": 7, "block z": 8},
        "models": {
            "base": ("baseline", "lewm_k1_cube_s3072/weights_epoch_10.pt", "#6b7280"),
            "obj": ("L_obj", "lewm_k2_cube_obj_eff0.1_s3072/weights_epoch_10.pt", "#4f46e5"),
            "aux": ("aux", "lewm_k4_cube_qhead_eff0.1_s3072/weights_epoch_10.pt", "#d97706"),
            "dw": ("DINO-WM", "dinowm_cube_s3072/weights_epoch_10.pt", "#0d9488"),
        },
    },
    "tworoom": {
        "lance": "tworoom.lance", "qcols": ("pos_agent",),
        "build_q": lambda c: c["pos_agent"][:, :2],
        "qdims": {"x": 0, "y": 1},
        "models": {
            "base": ("baseline", "lewm_t1_tworoom_s3072/weights_epoch_10.pt", "#6b7280"),
            "obj": ("L_obj", "lewm_t2_tworoom_obj0.1_s3072/weights_epoch_10.pt", "#4f46e5"),
            "aux": ("aux", "lewm_t5_tworoom_qhead0.1_s3072/weights_epoch_10.pt", "#d97706"),
            "dw": ("DINO-WM", "dinowm_tworoom_s3072/weights_epoch_10.pt", "#0d9488"),
        },
    },
    "pointmaze": {
        "lance": "pointmaze.lance", "qcols": ("pos",),
        "build_q": lambda c: c["pos"][:, :2],
        "qdims": {"x": 0, "y": 1},
        "models": {
            "base": ("baseline", "lewm_p1_pointmaze_s3072/weights_epoch_10.pt", "#6b7280"),
            "obj": ("L_obj", "lewm_p2_pointmaze_s3072/weights_epoch_10.pt", "#4f46e5"),
            "aux": ("aux", "lewm_p5_pointmaze_s3072/weights_epoch_10.pt", "#d97706"),
            "dw": ("DINO-WM", "dinowm_pointmaze_s3072/weights_epoch_10.pt", "#0d9488"),
        },
    },
    "reacher": {
        "lance": "reacher.lance", "qcols": ("qpos",),
        "build_q": lambda c: build_q_reacher_joints(torch.from_numpy(c["qpos"])).numpy(),
        "qdims": {"cos q0": 0, "sin q0": 1, "cos q1": 2, "sin q1": 3},
        "models": {
            "base": ("baseline", "lewm_r1_reacher_s3072/weights_epoch_10.pt", "#6b7280"),
            "obj": ("L_obj full q", "lewm_r2_reacher_paep_l015_s3072/weights_epoch_10.pt", "#4f46e5"),
            "obj_h": ("L_obj half q", "lewm_hq_obj_reacher_s3072/weights_epoch_10.pt", "#8a87ec"),
            "aux": ("aux", "lewm_r5_qhead0.4_s3072/weights_epoch_10.pt", "#d97706"),
            "dw": ("DINO-WM", "dinowm_reacher_s3072/weights_epoch_10.pt", "#0d9488"),
        },
    },
}
N = 1500
KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
TEXT, MUTED = "#3d3d3c", "#6f6e66"


def pc_scores(z):
    """N x r PC-score matrix (descending eigenvalue order) from the double-centered
    Gram -- never forms the D x D covariance, so 98304-d costs the same as 192-d."""
    z = z.astype(np.float64)
    G = z @ z.T
    n = G.shape[0]
    J = np.eye(n) - 1.0 / n
    w, V = np.linalg.eigh(J @ G @ J)
    w, V = w[::-1], V[:, ::-1]
    keep = w > w[0] * 1e-10
    return V[:, keep] * np.sqrt(w[keep])


def ridge_r2_per_dim(S, y):
    """First half fit, second half held-out; per-dim R^2 on standardized q."""
    n = len(S)
    tr, va = slice(0, n // 2), slice(n // 2, n)
    Sc = S[tr] - S[tr].mean(0)
    yc = y[tr] - y[tr].mean(0)
    W = np.linalg.solve(Sc.T @ Sc + 1e-2 * np.eye(S.shape[1]), Sc.T @ yc)
    pred = (S[va] - S[tr].mean(0)) @ W + y[tr].mean(0)
    ss_res = ((y[va] - pred) ** 2).sum(0)
    ss_tot = ((y[va] - y[va].mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / ss_tot


def main():
    task = sys.argv[1]
    spec = TASKS[task]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"task={task} device={device}", flush=True)

    dataset = swm.data.load_dataset(spec["lance"], keys_to_load=["pixels", *spec["qcols"]])
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    test_eps = perm[: int(n_ep * TEST_EPISODE_FRAC)]
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)
    pool = np.concatenate([offsets[e] + np.arange(lengths[e]) for e in test_eps])
    rows = np.sort(g.choice(pool, N, replace=False))
    pix, cols = load_frames(dataset, rows, device, cols=spec["qcols"])
    q = np.asarray(spec["build_q"](cols))
    qs = (q - q.mean(0)) / q.std(0)
    # shuffle the fit/holdout halves so they are not episode-ordered
    sh = g.permutation(N)
    qs = qs[sh]
    print(f"q dim={q.shape[1]}", flush=True)

    out = {"task": task, "n": N, "ks": [], "models": {}}
    for key, (label, ckpt, color) in spec["models"].items():
        m = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        m.requires_grad_(False)
        enc = encode_layers_dinowm if key == "dw" else encode_layers
        _, finals = enc(m, pix, device)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        S = pc_scores(finals)[sh]
        r = S.shape[1]
        ks = [k for k in KS if k <= r]
        res = {"rank": int(r), "top": {}, "bottom": {}}
        for k in ks:
            res["top"][k] = ridge_r2_per_dim(S[:, :k], qs).tolist()
            res["bottom"][k] = ridge_r2_per_dim(S[:, r - k:], qs).tolist()
        out["models"][key] = res
        out["ks"] = sorted(set(out["ks"]) | set(ks))
        top_line = "  ".join(f"k={k}:{np.mean(res['top'][k]):.2f}" for k in ks[:8])
        print(f"{key:6s} rank={r:4d}  top-k R2  {top_line}", flush=True)

    Path("eval_results").mkdir(exist_ok=True)
    Path(f"eval_results/pcq_{task}.json").write_text(json.dumps(out, indent=1))

    # ---- figure: mean R^2 vs k, solid = top-k, dashed = bottom-k control ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    for key, (label, _, color) in spec["models"].items():
        res = out["models"][key]
        ks = sorted(int(k) for k in res["top"])
        axes[0].plot(ks, [np.mean(res["top"][k]) for k in ks], "-o", ms=3.5,
                     color=color, lw=1.8, label=label)
        axes[1].plot(ks, [np.mean(res["top"][k]) for k in ks], "-o", ms=3.5,
                     color=color, lw=1.8, label=f"{label} top-k")
        axes[1].plot(ks, [np.mean(res["bottom"][k]) for k in ks], "--", ms=3,
                     color=color, lw=1.2, alpha=0.55)
    for ax, title in zip(axes, [f"held-out R² of q from top-k PCs ({task})",
                                "solid = top-k · dashed = bottom-k control"]):
        ax.set_xscale("log")
        ax.set_xlabel("k (principal directions)", fontsize=8, color=MUTED)
        ax.set_ylim(-0.05, 1.0)
        ax.set_title(title, fontsize=9, color=TEXT)
        ax.tick_params(labelsize=7, colors=MUTED)
    axes[0].set_ylabel("mean held-out R²", fontsize=8, color=MUTED)
    axes[0].legend(frameon=False, fontsize=7, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(f"eval_results/pcq_{task}.png", facecolor="white", bbox_inches="tight")
    print(f"wrote eval_results/pcq_{task}.json / .png")


if __name__ == "__main__":
    main()
