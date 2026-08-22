"""Representation-space analysis of the metric-distilled SCALE (Push-T), against
LeWM baseline / raw-q SCALE / Aux, plus the frozen q-only teacher itself.

Per pixel model, on held-out lance frames (probe split convention):
  * covariance eigenspectrum: top-20 normalized eigenvalues + eff-rank
  * top-k PCA -> q ridge R^2 (k = 2, 4, 8; PCA and ridge fit train-side, scored
    held-out) -- the fig_topk question: is q in the spectral HEAD?
  * full ridge R^2 z -> q (linear readability; 6-d canonical q)
  * rho_q = Pearson(||dz||^2, ||dq||^2)          -- alignment to the RAW q metric
  * rho_T = Pearson(||dz||^2, ||d T(q)||^2)      -- alignment to the TEACHER metric
The teacher T (q-only model, projector output) also gets its own spectrum/rho row.
The distillation-specific question is rho_T vs rho_q: which metric did the
distilled student actually inherit, and did raw-q SCALE accidentally have it too?

    usage: repspace_qdistill_pusht.py base:<ckpt> scale:<ckpt> distill:<ckpt> aux:<ckpt> \
             --teacher lewm_q1_qinput_s3072/weights_epoch_10.pt
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
from qjepa import QJEPA  # noqa: E402
from utils import build_q_raw  # noqa: E402

N_FIT, N_HOLD = 4000, 1500
TOPK = (2, 4, 8)


@torch.no_grad()
def encode(model, pix_list, device):
    zs = []
    for p in pix_list:
        out = model.encoder(p.to(device), interpolate_pos_encoding=True)
        zs.append(model.projector(out.last_hidden_state[:, 0]).float().cpu())
    return torch.cat(zs).double().numpy()


def ridge_r2(Xf, yf, Xh, yh, lam=1e-2):
    Xc = Xf - Xf.mean(0)
    A = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xf.shape[1]), Xc.T @ yf)
    pred = (Xh - Xf.mean(0)) @ A
    ss_res = ((yh - pred) ** 2).sum(0)
    ss_tot = ((yh - yh.mean(0)) ** 2).sum(0)
    return float((1 - ss_res / np.clip(ss_tot, 1e-12, None)).mean())


def spectrum(Zh):
    zc = Zh - Zh.mean(0)
    eig = np.clip(np.linalg.eigvalsh(zc.T @ zc / max(len(zc) - 1, 1)), 0, None)[::-1]
    p = eig / max(eig.sum(), 1e-12)
    eff = float(np.exp(-(p * np.log(np.clip(p, 1e-12, None))).sum()))
    return eig, eff


def pair_rho(X, Y, n_pairs=200000, seed=0):
    g = np.random.default_rng(seed)
    i = g.integers(0, len(X), n_pairs); j = g.integers(0, len(X), n_pairs)
    keep = i != j
    dx = ((X[i[keep]] - X[j[keep]]) ** 2).sum(-1)
    dy = ((Y[i[keep]] - Y[j[keep]]) ** 2).sum(-1)
    return float(np.corrcoef(dx, dy)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="label:ckpt (pixel models)")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", default="eval_results/repspace_qdistill_pusht.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path("eval_results").mkdir(exist_ok=True)

    dataset = swm.data.load_dataset("pusht_expert_train.lance", keys_to_load=["pixels", "state"])
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(len(lengths))
    n_test = int(len(lengths) * TEST_EPISODE_FRAC)
    test_eps, train_eps = perm[:n_test], perm[n_test:]

    def rows_of(eps, n):
        pool = np.concatenate([offsets[e] + np.arange(lengths[e]) for e in eps])
        return np.sort(g.choice(pool, n, replace=False))

    pix_f, cols_f = load_frames(dataset, rows_of(train_eps, N_FIT), device, cols=("state",))
    pix_h, cols_h = load_frames(dataset, rows_of(test_eps, N_HOLD), device, cols=("state",))
    qf_raw = build_q_raw(torch.from_numpy(cols_f["state"]))
    qh_raw = build_q_raw(torch.from_numpy(cols_h["state"]))
    mu, sd = qf_raw.mean(0), qf_raw.std(0)
    qf = ((qf_raw - mu) / sd).numpy().astype(np.float64)
    qh = ((qh_raw - mu) / sd).numpy().astype(np.float64)

    # teacher embedding of held-out q (its own persisted normalization)
    teacher = swm.wm.utils.load_pretrained(args.teacher).to(device).eval()
    teacher.requires_grad_(False)
    assert isinstance(teacher, QJEPA), type(teacher)
    with torch.no_grad():
        tq_h = teacher.projector(teacher.encoder(
            ((qh_raw.float().to(device) - teacher.q_mean) / teacher.q_std))).cpu().double().numpy()
        tq_f = teacher.projector(teacher.encoder(
            ((qf_raw.float().to(device) - teacher.q_mean) / teacher.q_std))).cpu().double().numpy()

    out = []
    # teacher's own row (representation over q, not pixels)
    eig, eff = spectrum(tq_h)
    out.append({"label": "teacher(q-only)", "eff_rank": eff,
                "eigs_top20": (eig[:20] / eig.sum()).tolist(),
                "rho_q": pair_rho(tq_h, qh), "rho_T": 1.0,
                "r2_full": ridge_r2(tq_f, qf, tq_h, qh)})
    # top-k PCA for the teacher, same recipe as the pixel models below
    zc = tq_f - tq_f.mean(0)
    _, _, Vt = np.linalg.svd(zc, full_matrices=False)
    sf = zc @ Vt.T; sh = (tq_h - tq_f.mean(0)) @ Vt.T
    out[-1]["topk_r2"] = {str(k): ridge_r2(sf[:, :k], qf, sh[:, :k], qh) for k in TOPK}
    print(f"teacher(q-only): eff {eff:.1f} rho_q {out[-1]['rho_q']:.3f} "
          f"R2 {out[-1]['r2_full']:.3f} topk {out[-1]['topk_r2']}", flush=True)

    for spec in args.models:
        label, ckpt = spec.split(":", 1)
        model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        model.requires_grad_(False)
        Zf, Zh = encode(model, pix_f, device), encode(model, pix_h, device)
        eig, eff = spectrum(Zh)
        zc = Zf - Zf.mean(0)
        _, _, Vt = np.linalg.svd(zc, full_matrices=False)
        sf = zc @ Vt.T; sh = (Zh - Zf.mean(0)) @ Vt.T
        row = {
            "label": label, "ckpt": ckpt, "eff_rank": eff,
            "eigs_top20": (eig[:20] / eig.sum()).tolist(),
            "r2_full": ridge_r2(Zf, qf, Zh, qh),
            "topk_r2": {str(k): ridge_r2(sf[:, :k], qf, sh[:, :k], qh) for k in TOPK},
            "rho_q": pair_rho(Zh, qh),
            "rho_T": pair_rho(Zh, tq_h),
        }
        out.append(row)
        print(f"{label:10s} eff {eff:6.1f}  R2 {row['r2_full']:.3f}  "
              f"topk {row['topk_r2']}  rho_q {row['rho_q']:.3f}  rho_T {row['rho_T']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
