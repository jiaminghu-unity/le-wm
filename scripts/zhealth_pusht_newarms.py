"""Latent-health check for the PushT new arms: is the c2p (obj-only, no-SIGReg)
failure a representation collapse?

For each pixel model: encode held-out lance frames, report
  * eff-rank of the embedding covariance (collapse alarm, same formula as training's
    latent_health),
  * ||z|| mean/std,
  * ridge R^2 z -> q (6-d canonical q), fit on train-side frames, scored held-out.

    usage: zhealth_pusht_newarms.py c1ref:lewm_c1_s3072/weights_epoch_10.pt c2p:... c9:...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, load_frames  # noqa: E402
from utils import build_q_raw  # noqa: E402

N_FIT, N_HOLD = 4000, 1500


@torch.no_grad()
def encode(model, pix_list, device):
    zs = []
    for p in pix_list:
        out = model.encoder(p.to(device), interpolate_pos_encoding=True)
        zs.append(model.projector(out.last_hidden_state[:, 0]).float().cpu())
    return torch.cat(zs).double().numpy()


def sample_rows(lengths, offsets, eps, n, g):
    pool = np.concatenate([offsets[e] + np.arange(lengths[e]) for e in eps])
    return np.sort(g.choice(pool, n, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="label:ckpt_path")
    ap.add_argument("--out", default="eval_results/zhealth_pusht_newarms.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path("eval_results").mkdir(exist_ok=True)

    dataset = swm.data.load_dataset("pusht_expert_train.lance", keys_to_load=["pixels", "state"])
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(len(lengths))
    n_test = int(len(lengths) * TEST_EPISODE_FRAC)
    test_eps, train_eps = perm[:n_test], perm[n_test:]

    rows_fit = sample_rows(lengths, offsets, train_eps, N_FIT, g)
    rows_hold = sample_rows(lengths, offsets, test_eps, N_HOLD, g)
    pix_f, cols_f = load_frames(dataset, rows_fit, device, cols=("state",))
    pix_h, cols_h = load_frames(dataset, rows_hold, device, cols=("state",))
    qf = build_q_raw(torch.from_numpy(cols_f["state"])).numpy().astype(np.float64)
    qh = build_q_raw(torch.from_numpy(cols_h["state"])).numpy().astype(np.float64)
    mu, sd = qf.mean(0), qf.std(0)
    qf = (qf - mu) / sd; qh = (qh - mu) / sd

    out = []
    for spec in args.models:
        label, ckpt = spec.split(":", 1)
        model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        model.requires_grad_(False)
        Zf = encode(model, pix_f, device)
        Zh = encode(model, pix_h, device)

        zc = Zh - Zh.mean(0)
        eig = np.clip(np.linalg.eigvalsh(zc.T @ zc / max(len(zc) - 1, 1)), 0, None)
        p = eig / max(eig.sum(), 1e-12)
        eff = float(np.exp(-(p * np.log(np.clip(p, 1e-12, None))).sum()))
        norms = np.linalg.norm(Zh, axis=1)

        Zc = Zf - Zf.mean(0)
        A = np.linalg.solve(Zc.T @ Zc + 1e-2 * np.eye(Zf.shape[1]), Zc.T @ qf)
        pred = (Zh - Zf.mean(0)) @ A
        ss_res = ((qh - pred) ** 2).sum(0)
        ss_tot = ((qh - qh.mean(0)) ** 2).sum(0)
        r2_dim = 1 - ss_res / np.clip(ss_tot, 1e-12, None)

        row = {"label": label, "ckpt": ckpt, "eff_rank": eff,
               "z_norm_mean": float(norms.mean()), "z_norm_std": float(norms.std()),
               "r2_mean": float(r2_dim.mean()), "r2_per_dim": [float(x) for x in r2_dim]}
        out.append(row)
        print(f"{label:8s} eff-rank {eff:7.2f}  |z| {norms.mean():8.3f}±{norms.std():.3f}  "
              f"R2(z->q) {r2_dim.mean():.4f}  per-dim {[round(float(x),3) for x in r2_dim]}", flush=True)
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
