"""MLP counterpart of the full-q probing table (4 models x 5 tasks).

Same protocol as the linear table in RESULTS_scale_probes.md: identical 1500
held-out frames, identical fit/holdout halves, q standardized per-dim. The probe
follows scripts/probe.py's existing MLP convention exactly -- 2-layer, hidden 256,
ReLU, Adam lr 1e-3, 30 epochs, batch 1024, seed 0 -- but is device-agnostic (the
original hard-codes .cuda(); this runs on whatever is available, the fit is 30
tiny steps either way). Inputs are the raw embeddings, as in probe.py.

A linear ridge R^2 is recomputed alongside as a consistency check against
paper_stats_<task>.json (same numbers expected up to float noise).

    usage: probe_mlp_q.py {pusht|reacher|cube|tworoom|pointmaze}
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402
from scripts.probe import MLP_BS, MLP_EPOCHS, MLP_HIDDEN, MLP_LR  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, load_frames  # noqa: E402
from scripts.probe_pc_q import TASKS, pc_scores, ridge_r2_per_dim  # noqa: E402
from scripts.visualize_general import encode_layers  # noqa: E402
from scripts.visualize_general_dw import encode_layers_dinowm  # noqa: E402

ARMS = ["base", "obj", "aux", "dw"]
N = 1500


def mlp_r2_per_dim(z, y, device, seed=0):
    """probe.py's fit_mlp, device-agnostic; first half fit, second half held-out."""
    n = len(z)
    tr, va = slice(0, n // 2), slice(n // 2, n)
    z_tr = torch.as_tensor(z[tr], dtype=torch.float32, device=device)
    y_tr = torch.as_tensor(y[tr], dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(z_tr.size(1), MLP_HIDDEN), nn.ReLU(),
                        nn.Linear(MLP_HIDDEN, y_tr.size(1))).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=MLP_LR)
    for _ in range(MLP_EPOCHS):
        perm = torch.randperm(len(z_tr), device=device)
        for j in range(0, len(perm), MLP_BS):
            idx = perm[j: j + MLP_BS]
            opt.zero_grad()
            loss = (net(z_tr[idx]) - y_tr[idx]).pow(2).mean()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.as_tensor(z[va], dtype=torch.float32, device=device)).cpu().numpy()
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
    sh = g.permutation(N)
    qs_sh = qs[sh]

    out = {"task": task, "qdims": list(spec["qdims"]), "n": N, "mlp": {}, "linear_check": {}}
    for key in ARMS:
        label, ckpt, _ = spec["models"][key]
        m = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        m.requires_grad_(False)
        enc = encode_layers_dinowm if key == "dw" else encode_layers
        _, finals = enc(m, pix, device)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        # MLP input = the FULL set of PC coordinates (a lossless rotation; rank
        # ~1499 for DINO-WM, ~192 for the LeWM arms), scaled by ONE global scalar
        # from the fit half. Two failed variants documented so nobody retries them:
        # raw 98304-d features (25M-param first layer vs 750 samples: R^2 -2..-11)
        # and per-dim standardized scores (unit-scales the numerical-noise tail of
        # the spectrum; the MLP then overfits noise and every arm degrades).
        S = pc_scores(finals)[sh]
        S = S / S[: N // 2].std()
        out["mlp"][key] = mlp_r2_per_dim(S, qs_sh, device).tolist()
        # cross-check: ridge via PC scores, NEVER in feature space. Ridge is
        # rotation-invariant, so this equals full-space ridge -- while solving the
        # D x D system directly on DINO-WM's 98304-d features allocates a 77 GB
        # matrix (it OOM-killed this node on 08-15 and took a neighbouring GPU
        # job down with it).
        out["linear_check"][key] = ridge_r2_per_dim(S, qs_sh).tolist()
        print(f"{key:5s} mlp {np.mean(out['mlp'][key]):.4f}  "
              f"linear {np.mean(out['linear_check'][key]):.4f}", flush=True)

    Path("eval_results").mkdir(exist_ok=True)
    Path(f"eval_results/probe_mlp_{task}.json").write_text(json.dumps(out, indent=1))
    print(f"wrote eval_results/probe_mlp_{task}.json")


if __name__ == "__main__":
    main()
