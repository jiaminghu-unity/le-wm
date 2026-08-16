"""Held-out R^2 of q from the top principal subspace, indexed by CUMULATIVE
EXPLAINED VARIANCE instead of component count.

Top-k comparisons across models are unfair when ambient dimensionality differs
(DINO-WM's spectrum has ~1499 nonzero components, the LeWM arms ~192): k=8 is a
different fraction of each model's variance. This probe fixes the x-axis at
variance fractions v in {5,10,...,99}%: for each model, k(v) = the smallest k
whose top-k eigenvalues sum to >= v of total variance, then the usual ridge probe
from those k(v) PC scores. Same data protocol as every other probe (1500 held-out
frames, shared shuffle, q standardized). k(v) per model is stored in the JSON so
the mapping is auditable.

    usage: probe_cumvar.py {pusht|reacher|cube|tworoom|pointmaze}
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

NAME = {"base": "LeWM", "obj": "SCALE", "aux": "Aux", "dw": "DINO-WM"}
COLOR = {"base": "#6b7280", "obj": "#4f46e5", "aux": "#d97706", "dw": "#0d9488"}
ARMS = ["base", "obj", "aux", "dw"]
VFRACS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
N = 1500
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

    out = {"task": task, "n": N, "vfracs": VFRACS, "models": {}}
    for key in ARMS:
        label, ckpt, _ = spec["models"][key]
        m = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        m.requires_grad_(False)
        enc = encode_layers_dinowm if key == "dw" else encode_layers
        _, finals = enc(m, pix, device)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        S = pc_scores(finals)          # descending-eigenvalue order
        ev = (S ** 2).sum(0)
        cum = np.cumsum(ev) / ev.sum()
        S_sh = S[sh]
        res = {"rank": int(S.shape[1]), "k": {}, "r2": {}}
        for v in VFRACS:
            k = int(np.searchsorted(cum, v / 100.0) + 1)
            res["k"][v] = k
            res["r2"][v] = float(np.mean(ridge_r2_per_dim(S_sh[:, :k], qs_sh)))
        out["models"][key] = res
        line = "  ".join(f"{v}%:k={res['k'][v]},R2={res['r2'][v]:.2f}" for v in (10, 30, 50, 90))
        print(f"{key:5s} rank={res['rank']:5d}  {line}", flush=True)

    Path("eval_results").mkdir(exist_ok=True)
    Path(f"eval_results/cumvar_{task}.json").write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=200)
    for key in ARMS:
        res = out["models"][key]
        ax.plot(VFRACS, [res["r2"][v] for v in VFRACS], "-o", ms=3.5,
                color=COLOR[key], lw=1.8, label=NAME[key])
    ax.set_xlabel("cumulative explained variance (%)", fontsize=9, color=MUTED)
    ax.set_ylabel("held-out R² of q", fontsize=9, color=MUTED)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(TASK_TITLE[task], fontsize=11, color=TEXT)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"eval_results/fig_cumvar_{task}.png", facecolor="white",
                bbox_inches="tight")
    print(f"wrote eval_results/fig_cumvar_{task}.png and cumvar_{task}.json")


if __name__ == "__main__":
    main()
