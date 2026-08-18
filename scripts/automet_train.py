"""AutoMetric MVP: learn a full-dimensional planner-facing Mahalanobis metric on a
FROZEN LeWM latent space from trajectory temporal ordering alone. No q, no reward,
no human state selection; the encoder and predictor are never touched.

    W in R^{DxD}, W_0 = I
    M(W) = D * W^T W / tr(W^T W)          (trace-normalized: scale cheating is
                                           impossible -- softplus ranking alone is
                                           gameable by W -> cW, which inflates
                                           d_far - d_near without learning anything;
                                           tr(M)=D pins the scale, identity included)
    d_W(a,b) = (a-b)^T M (a-b)
    triplets: t1 < t2 < tg from ONE trajectory, z = frozen encoder outputs
    L = mean softplus( (d_W(z_t2, z_tg) - d_W(z_t1, z_tg)) / tau )

tau is the single fixed scale constant: the mean vanilla (M=I) squared pair distance
over random frame pairs, measured once before training. The trace norm pins the
metric's overall scale, so tau stays meaningful throughout training and the same
recipe transfers across tasks whose latent scales differ by orders of magnitude.

Data hygiene: PCA-free; fitting uses the TRAIN-side 90% of episodes (probe.py's
split convention, same SPLIT_SEED); the held-out 10% provides the diagnostics:
ordering accuracy, and -- as pure diagnosis, never in training -- rho_W =
Pearson(d_W pair distances, ||dq||^2) against the privileged q, compared with the
vanilla metric's rho and SCALE's representation-level rho.

    usage: automet_train.py cube --ckpt lewm_k1_cube_s3072/weights_epoch_10.pt
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
from scripts.probe_pc_q import TASKS  # noqa: E402

STEPS = 3000
BATCH = 1024
LR = 1e-3
N_ENC_EP = 2000          # train-side episodes to pre-encode
N_HOLD_FRAMES = 1500     # held-out frames for diagnostics
SPAN_LO, SPAN_HI = 10, 50   # tg - t1 in env steps, covering the eval goal=+25 scale
GAP = 5                     # min separation t1<t2<tg


@torch.no_grad()
def encode_frames(model, dataset, rows, device, bs=512):
    zs = []
    import stable_pretraining as spt
    imagenet = spt.data.dataset_stats.ImageNet
    mean = torch.tensor(imagenet["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(imagenet["std"], device=device).view(1, 3, 1, 1)
    for i in range(0, len(rows), bs):
        batch = dataset.get_row_data(rows[i:i + bs].tolist())
        pix = dataset._decode_images(batch["pixels"].tolist()).to(device).float() / 255.0
        pix = (pix - mean) / std
        out = model.encoder(pix, interpolate_pos_encoding=True)
        zs.append(model.projector(out.last_hidden_state[:, 0]).float().cpu())
        if (i // bs) % 50 == 0:
            print(f"[encode] {i}/{len(rows)}", flush=True)
    return torch.cat(zs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()
    tag = args.out_tag or f"{args.task}_{args.ckpt.split('/')[0]}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = TASKS[args.task]
    print(f"task={args.task} ckpt={args.ckpt} device={device}", flush=True)

    dataset = swm.data.load_dataset(spec["lance"], keys_to_load=["pixels", *spec["qcols"]])
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    n_test = int(n_ep * TEST_EPISODE_FRAC)
    test_eps, train_eps = perm[:n_test], perm[n_test:]
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)

    model = swm.wm.utils.load_pretrained(args.ckpt).to(device).eval()
    model.requires_grad_(False)

    # ---- pre-encode a pool of train-side episodes (full frame grids) ----
    ep_pool = g.choice(train_eps, min(N_ENC_EP, len(train_eps)), replace=False)
    rows, ep_of, t_of = [], [], []
    for e in ep_pool:
        L = int(lengths[e])
        rows.extend(range(int(offsets[e]), int(offsets[e]) + L))
        ep_of.extend([e] * L); t_of.extend(range(L))
    rows = np.asarray(rows); ep_of = np.asarray(ep_of); t_of = np.asarray(t_of)
    print(f"[pool] {len(ep_pool)} train episodes, {len(rows)} frames", flush=True)
    Z = encode_frames(model, dataset, rows, device).to(device)  # (N, D)
    D = Z.shape[1]

    # frame index lookup: (episode, t) -> row position in Z
    ep_index = {int(e): np.where(ep_of == e)[0] for e in ep_pool}
    ep_len = {int(e): int(lengths[e]) for e in ep_pool}

    # ---- tau: mean vanilla squared pair distance, fixed once ----
    ii = torch.randint(0, len(Z), (20000,), device=device)
    jj = torch.randint(0, len(Z), (20000,), device=device)
    keep = ii != jj
    tau = (Z[ii[keep]] - Z[jj[keep]]).pow(2).sum(-1).mean().item()
    print(f"[tau] mean vanilla sq pair distance = {tau:.4f}", flush=True)

    def sample_triplets(B, rng):
        eps = rng.choice(ep_pool, B)
        i1 = np.empty(B, dtype=np.int64); i2 = np.empty(B, dtype=np.int64)
        ig = np.empty(B, dtype=np.int64)
        for b, e in enumerate(eps):
            L = ep_len[int(e)]
            span = rng.integers(SPAN_LO, min(SPAN_HI, L - 1) + 1)
            t1 = rng.integers(0, L - span)
            tg = t1 + span
            t2 = rng.integers(t1 + GAP, tg - GAP + 1) if tg - GAP >= t1 + GAP else t1 + 1
            idx = ep_index[int(e)]
            i1[b], i2[b], ig[b] = idx[t1], idx[t2], idx[tg]
        return i1, i2, ig

    # ---- learn W (full D x D, init I); trace-normalized metric in-graph ----
    W = torch.eye(D, device=device, requires_grad=True)
    opt = torch.optim.Adam([W], lr=LR)
    rng = np.random.default_rng(SPLIT_SEED + 1)
    curve = []
    for step in range(1, args.steps + 1):
        i1, i2, ig = sample_triplets(BATCH, rng)
        z1, z2, zg = Z[i1], Z[i2], Z[ig]
        M_unnorm = W.T @ W
        M = D * M_unnorm / M_unnorm.diagonal().sum()   # tr(M) = D, in-graph
        def dM(a, b):
            d = a - b
            return ((d @ M) * d).sum(-1)
        d_far, d_near = dM(z1, zg), dM(z2, zg)
        loss = torch.nn.functional.softplus((d_near - d_far) / tau).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            acc = (d_near < d_far).float().mean().item()
            curve.append({"step": step, "loss": float(loss.item()), "order_acc": acc})
            print(f"[{step:5d}] loss {loss.item():.4f}  order-acc {acc:.3f}", flush=True)

    with torch.no_grad():
        M_unnorm = W.T @ W
        M_final = (D * M_unnorm / M_unnorm.diagonal().sum()).cpu()

    # ---- diagnostics on HELD-OUT episodes (q used here only, as ground truth) ----
    pool_t = np.concatenate([offsets[e] + np.arange(lengths[e]) for e in test_eps])
    hrows = np.sort(g.choice(pool_t, N_HOLD_FRAMES, replace=False))
    pix, cols = load_frames(dataset, hrows, device, cols=spec["qcols"])
    Zh = []
    import stable_pretraining as spt  # noqa: F401
    with torch.no_grad():
        for p in pix:
            out = model.encoder(p.to(device), interpolate_pos_encoding=True)
            Zh.append(model.projector(out.last_hidden_state[:, 0]).float().cpu())
    Zh = torch.cat(Zh).double()
    q = np.asarray(spec["build_q"](cols))
    qs = torch.tensor((q - q.mean(0)) / q.std(0)).double()

    def pair_rho(Mmat):
        n = len(Zh)
        iu = torch.triu_indices(n, n, offset=1)
        dz = Zh[iu[0]] - Zh[iu[1]]
        x = ((dz @ Mmat.double()) * dz).sum(-1).numpy()
        dq = qs[iu[0]] - qs[iu[1]]
        y = dq.pow(2).sum(-1).numpy()
        return float(np.corrcoef(x, y)[0, 1])

    rho_vanilla = pair_rho(torch.eye(D))
    rho_learned = pair_rho(M_final)
    # held-out ordering accuracy
    ho_eps = [int(e) for e in test_eps if lengths[e] > SPAN_LO + 2]
    print(f"[diag] rho vanilla={rho_vanilla:.4f}  learned={rho_learned:.4f}", flush=True)

    Path("eval_results").mkdir(exist_ok=True)
    torch.save({"M": M_final, "W": W.detach().cpu(), "tau": tau, "D": D,
                "ckpt": args.ckpt, "task": args.task},
               f"eval_results/automet_{tag}.pt")
    ev = torch.linalg.eigvalsh(M_final).numpy()[::-1]
    Path(f"eval_results/automet_{tag}.json").write_text(json.dumps({
        "task": args.task, "ckpt": args.ckpt, "tau": tau, "steps": args.steps,
        "curve": curve, "rho_vanilla": rho_vanilla, "rho_learned": rho_learned,
        "metric_eigs_top20": ev[:20].tolist(),
        "metric_eig_effrank": float(np.exp(-(ev/ev.sum() * np.log(np.clip(ev/ev.sum(), 1e-12, None))).sum())),
    }, indent=1))
    print(f"wrote eval_results/automet_{tag}.pt/.json", flush=True)


if __name__ == "__main__":
    main()
