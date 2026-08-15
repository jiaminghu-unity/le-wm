"""Camera-ready probe figures and statistics, four models x five tasks.

Model names are the paper's: LeWM (the baseline), SCALE (L_obj), Aux (q-head),
DINO-WM. No hyperparameters appear in any figure.

Per task this produces:
  fig_spectrum_<task>.png   covariance spectrum, log-log, 4 curves
  fig_topk_<task>.png       grouped bars: held-out R^2 of q from top-k PCs, k in {2,4,8}
  paper_stats_<task>.json   everything the markdown needs:
      spectrum   normalized eigenvalue curves (for reproducibility)
      topk       per-dim R^2 at k = 2/4/8
      full       per-dim R^2 from the FULL embedding (all nonzero PCs) -- the
                 complete-q probing table
      ratio      distribution of ||dz|| / ||dq|| over sampled frame pairs
                 (median, p10, p90, p90/p10). The median is in arbitrary
                 embedding units and is NOT comparable across models; the
                 p90/p10 dispersion is scale-free and is.

Reuses probe_pc_q's task registry, sampling split, Gram-based PC scores and the
ridge probe, so every number is computed under the identical protocol.

    usage: paper_figs.py {pusht|reacher|cube|tworoom|pointmaze}
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
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, load_frames  # noqa: E402
from scripts.probe_pc_q import TASKS, pc_scores, ridge_r2_per_dim  # noqa: E402
from scripts.visualize_general import encode_layers  # noqa: E402
from scripts.visualize_general_dw import encode_layers_dinowm  # noqa: E402

# paper names + the validated arm palette; reacher's SCALE is the full-q arm
NAME = {"base": "LeWM", "obj": "SCALE", "aux": "Aux", "dw": "DINO-WM"}
COLOR = {"base": "#6b7280", "obj": "#4f46e5", "aux": "#d97706", "dw": "#0d9488"}
ARMS = ["base", "obj", "aux", "dw"]
KS_BAR = [2, 4, 8]
N = 1500
N_PAIRS = 20000
TEXT, MUTED = "#3d3d3c", "#6f6e66"
TASK_TITLE = {"pusht": "Push-T", "reacher": "Reacher", "cube": "Cube",
              "tworoom": "Two-Room", "pointmaze": "PointMaze"}


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
    sh = g.permutation(N)
    qs_sh = qs[sh]

    idx = g.integers(0, N, size=(N_PAIRS, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    dq = np.linalg.norm(qs[idx[:, 0]] - qs[idx[:, 1]], axis=1)
    keep = dq > 1e-6  # identical-pose pairs would put a 0 in the denominator

    out = {"task": task, "qdims": list(spec["qdims"]), "n": N,
           "spectrum": {}, "topk": {}, "full": {}, "ratio": {}}
    for key in ARMS:
        label, ckpt, _ = spec["models"][key]
        m = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        m.requires_grad_(False)
        enc = encode_layers_dinowm if key == "dw" else encode_layers
        _, finals = enc(m, pix, device)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()

        S = pc_scores(finals)
        r = S.shape[1]
        # spectrum from the same decomposition the scores came from
        ev = (S ** 2).sum(0) / (N - 1)
        ev = ev / ev.sum()
        out["spectrum"][key] = ev.tolist()

        S_sh = S[sh]
        out["topk"][key] = {k: ridge_r2_per_dim(S_sh[:, :k], qs_sh).tolist() for k in KS_BAR}
        out["full"][key] = ridge_r2_per_dim(S_sh, qs_sh).tolist()

        z = finals.astype(np.float64)
        dz = np.linalg.norm(z[idx[keep, 0]] - z[idx[keep, 1]], axis=1)
        ratio = dz / dq[keep]
        p10, med, p90 = np.percentile(ratio, [10, 50, 90])
        out["ratio"][key] = {"median": float(med), "p10": float(p10), "p90": float(p90),
                             "p90_over_p10": float(p90 / p10)}
        print(f"{NAME[key]:8s} rank={r} full-R2={np.mean(out['full'][key]):.3f} "
              f"ratio med={med:.3g} disp={p90/p10:.2f}", flush=True)

    Path("eval_results").mkdir(exist_ok=True)
    Path(f"eval_results/paper_stats_{task}.json").write_text(json.dumps(out, indent=1))

    # ---- figure 1: covariance spectrum ----
    fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=200)
    for key in ARMS:
        ev = np.asarray(out["spectrum"][key])
        ax.plot(np.arange(1, len(ev) + 1), ev, color=COLOR[key], lw=1.8, label=NAME[key])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("eigenvalue rank", fontsize=9, color=MUTED)
    ax.set_ylabel("normalized variance", fontsize=9, color=MUTED)
    ax.set_title(TASK_TITLE[task], fontsize=11, color=TEXT)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"eval_results/fig_spectrum_{task}.png", facecolor="white",
                bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: grouped bars, R^2 of q from top-k PCs ----
    fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=200)
    W = 0.2
    for mi, key in enumerate(ARMS):
        vals = [float(np.mean(out["topk"][key][k])) for k in KS_BAR]
        xs = np.arange(len(KS_BAR)) + (mi - 1.5) * W
        ax.bar(xs, vals, width=W * 0.9, color=COLOR[key], label=NAME[key])
        for x, v in zip(xs, vals):
            ax.text(x, max(v, 0) + 0.015, f"{v:.2f}", ha="center", fontsize=5.6, color=TEXT)
    ax.set_xticks(range(len(KS_BAR)))
    ax.set_xticklabels([f"k = {k}" for k in KS_BAR], fontsize=9, color=TEXT)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("held-out R² of q from top-k PCs", fontsize=9, color=MUTED)
    ax.set_title(TASK_TITLE[task], fontsize=11, color=TEXT)
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"eval_results/fig_topk_{task}.png", facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote eval_results/fig_*_{task}.png and paper_stats_{task}.json")


if __name__ == "__main__":
    main()
