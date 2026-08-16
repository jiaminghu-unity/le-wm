"""Pearson correlation between squared latent and squared q pair distances --
the exact quantity L_obj trains (lobj.py's rho) -- measured on held-out frames
for every current model.

Same data protocol as the other probes (1500 held-out frames, q standardized with
dataset-wide stats). Unlike training, which samples 4096 stratified pairs per step,
this computes the correlation over ALL N(N-1)/2 pairs exactly, via the Gram matrix
(no (N,N,D) broadcast, so DINO-WM's 98304-d space costs the same as 192-d).
Spearman over the same pairs is reported alongside (rank version, robust to the
heavy right tail of squared distances). Reacher additionally includes the half-q
SCALE arm; its rho is measured against BOTH the full 4-d q and its own kept 2-d q.

    usage: probe_rho.py {pusht|reacher|cube|tworoom|pointmaze}
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, load_frames  # noqa: E402
from scripts.probe_pc_q import TASKS  # noqa: E402
from scripts.visualize_general import encode_layers  # noqa: E402
from scripts.visualize_general_dw import encode_layers_dinowm  # noqa: E402

N = 1500


def sq_dists(z):
    """Upper-triangle vector of squared pairwise distances, via the Gram matrix."""
    z = np.asarray(z, dtype=np.float64)
    G = z @ z.T
    sq = np.diag(G).copy()
    d2 = sq[:, None] + sq[None, :] - 2.0 * G
    iu = np.triu_indices(len(z), 1)
    return np.maximum(d2[iu], 0.0)


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
    y_full = sq_dists(qs)
    refs = {"full_q": y_full}
    if task == "reacher":
        refs["kept_shoulder_q"] = sq_dists(qs[:, :2])  # half-q arm's own target

    arms = list(spec["models"])  # includes obj_h on reacher
    out = {"task": task, "n": N, "pairs": int(len(y_full)), "rho": {}}
    for key in arms:
        label, ckpt, _ = spec["models"][key]
        m = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        m.requires_grad_(False)
        enc = encode_layers_dinowm if key == "dw" else encode_layers
        _, finals = enc(m, pix, device)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        x = sq_dists(finals)
        out["rho"][key] = {}
        for rname, y in refs.items():
            pear = float(np.corrcoef(x, y)[0, 1])
            spear = float(spearmanr(x, y).statistic)
            out["rho"][key][rname] = {"pearson": pear, "spearman": spear}
            print(f"{key:6s} vs {rname:16s} pearson={pear:.4f} spearman={spear:.4f}", flush=True)

    Path("eval_results").mkdir(exist_ok=True)
    Path(f"eval_results/rho_{task}.json").write_text(json.dumps(out, indent=1))
    print(f"wrote eval_results/rho_{task}.json")


if __name__ == "__main__":
    main()
