"""Rebuttal diagnostics for arbitrary checkpoints, under the paper's own protocols.

For each run it reports, on the same 1500 held-out frames the paper's probes use:
  full_r2      per-component held-out R^2 of q from the FULL embedding   (paper Tables 6-10, linear)
  topk_r2      per-component R^2 from the leading k = 2/4/8 PCs          (paper Fig. 3)
  spectrum     normalized eigenvalues; top2/4/8 = their cumulative share (paper Fig. 2)
  rho          Pearson and Spearman between squared latent and squared q distances over all
               N(N-1)/2 pairs                                            (paper Tables 1, 12)
               against the full q, and on Push-T / Cube also against the high- and
               low-recoverability subsets of q used by the split arms (q_recov.RECOV_OF).

Code path is paper_figs.py + probe_rho.py (same sampling, PC scores, ridge probe, Gram-based
distances); the only additions are that models are given on the command line and that an LpWM
checkpoint's embedding is passed through its RepReLU, as JEPA.encode does at planning time.

    usage: rebuttal_diag.py <task> <run> [<run> ...]
    writes eval_results/rebuttal_diag/<task>__<run>.json (one file per run)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import q_recov  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, load_frames  # noqa: E402
from scripts.probe_pc_q import TASKS, pc_scores, ridge_r2_per_dim  # noqa: E402
from scripts.probe_rho import sq_dists  # noqa: E402
from scripts.visualize_general import encode_layers  # noqa: E402

N = 1500
KS = [2, 4, 8]
GROUPS = {  # task -> {group: indices of the paper's q}
    "pusht": {"high": q_recov.RECOV_OF["pusht_recov_high"][1], "low": q_recov.RECOV_OF["pusht_recov_low"][1]},
    "cube": {"high": q_recov.RECOV_OF["cube_recov_high"][1], "low": q_recov.RECOV_OF["cube_recov_low"][1]},
}


def main():
    task, runs = sys.argv[1], sys.argv[2:]
    spec = TASKS[task]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"task={task} device={device} runs={len(runs)}", flush=True)

    # identical frame sample and probe split to paper_figs.py (same generator, same call order)
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

    refs = {"full_q": sq_dists(qs)}
    for name, idx in GROUPS.get(task, {}).items():
        refs[f"{name}_q"] = sq_dists(qs[:, list(idx)])

    outdir = Path("eval_results/rebuttal_diag")
    outdir.mkdir(parents=True, exist_ok=True)
    qnames = [k for k, _ in sorted(spec["qdims"].items(), key=lambda kv: kv[1])]
    for run in runs:
        m = swm.wm.utils.load_pretrained(run).to(device).eval()
        m.requires_grad_(False)
        rep_relu = bool(getattr(m, "rep_relu", False))
        _, finals = encode_layers(m, pix, device)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
        if rep_relu:
            finals = np.maximum(finals, 0.0)  # RepReLU forward, as in JEPA.encode

        S = pc_scores(finals)
        ev = (S ** 2).sum(0) / (N - 1)
        ev = ev / ev.sum()
        S_sh = S[sh]
        full = ridge_r2_per_dim(S_sh, qs_sh)
        res = {
            "task": task, "run": run, "n": N, "rep_relu": rep_relu, "qdims": qnames,
            "rank": int(S.shape[1]),
            "full_r2": full.tolist(), "full_r2_mean": float(np.mean(full)),
            "topk_r2": {str(k): ridge_r2_per_dim(S_sh[:, :k], qs_sh).tolist() for k in KS},
            "spectrum_top": {str(k): float(ev[:k].sum()) for k in KS},
            "spectrum": ev.tolist(),
            "rho": {},
        }
        x = sq_dists(finals)
        for rname, y in refs.items():
            res["rho"][rname] = {"pearson": float(np.corrcoef(x, y)[0, 1]),
                                 "spearman": float(spearmanr(x, y).statistic)}
        (outdir / f"{task}__{run}.json").write_text(json.dumps(res, indent=1))
        print(f"DIAG {run} R2={res['full_r2_mean']:.3f} top8={res['spectrum_top']['8']:.3f} "
              + " ".join(f"{k}:S={v['spearman']:.3f}" for k, v in res["rho"].items()), flush=True)


if __name__ == "__main__":
    main()
