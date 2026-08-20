"""NONLINEAR AutoMetric: learn an embedding-based metric on a frozen LeWM latent
space from trajectory temporal ordering.

    phi(z) = z + g(z),  g = MLP(D -> 256 -> D) with ZERO-initialised last layer,
                        so phi_0 = identity and d_0 is exactly the vanilla planner
                        cost -- the same start-at-baseline property W_0 = I gave
                        the linear version.
    d_phi(a, b) = ||phi(a) - phi(b)||^2      (a valid pseudo-metric by construction)
    L = mean softplus( (d_near - d_far) / tau_b ),  tau_b = mean(d_far) IN-GRAPH,
        so the loss is invariant to global rescaling of phi -- the nonlinear
        analogue of the linear version's trace normalisation.

Everything else (triplet sampling, spans, data split, diagnostics) is identical to
automet_fit_W.py, so linear-vs-nonlinear differences are attributable to the metric
class alone. Requested for pusht/reacher, where the linear W hurt; the pre-registered
prediction (for the record) is that extra capacity amplifies the misaligned temporal
signal rather than fixing it.

    usage: automet_fit_phi_nonlinear.py pusht --ckpt lewm_c1_s3072/weights_epoch_10.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stable_worldmodel as swm  # noqa: E402
from scripts.probe import SPLIT_SEED, TEST_EPISODE_FRAC, load_frames  # noqa: E402
from scripts.probe_pc_q import TASKS  # noqa: E402
from scripts.automet_fit_W import encode_frames  # noqa: E402

STEPS = 3000
BATCH = 1024
LR = 1e-3
N_ENC_EP = 2000
N_HOLD_FRAMES = 1500
SPAN_LO, SPAN_HI = 10, 50
GAP = 5
HIDDEN = 256


class Phi(nn.Module):
    def __init__(self, D, hidden=HIDDEN):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, D))
        nn.init.zeros_(self.g[-1].weight)
        nn.init.zeros_(self.g[-1].bias)

    def forward(self, z):
        return z + self.g(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()
    tag = args.out_tag or f"{args.task}_nl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = TASKS[args.task]
    print(f"[nl] task={args.task} ckpt={args.ckpt} device={device}", flush=True)

    dataset = swm.data.load_dataset(spec["lance"], keys_to_load=["pixels", *spec["qcols"]])
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    n_test = int(n_ep * TEST_EPISODE_FRAC)
    test_eps, train_eps = perm[:n_test], perm[n_test:]
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)

    model = swm.wm.utils.load_pretrained(args.ckpt).to(device).eval()
    model.requires_grad_(False)

    ep_pool = g.choice(train_eps, min(N_ENC_EP, len(train_eps)), replace=False)
    rows, ep_of = [], []
    for e in ep_pool:
        L = int(lengths[e])
        rows.extend(range(int(offsets[e]), int(offsets[e]) + L))
        ep_of.extend([e] * L)
    rows = np.asarray(rows); ep_of = np.asarray(ep_of)
    print(f"[pool] {len(ep_pool)} episodes, {len(rows)} frames", flush=True)
    Z = encode_frames(model, dataset, rows, device).to(device)
    D = Z.shape[1]
    ep_index = {int(e): np.where(ep_of == e)[0] for e in ep_pool}
    ep_len = {int(e): int(lengths[e]) for e in ep_pool}

    def sample_triplets(B, rng):
        eps = rng.choice(ep_pool, B)
        i1 = np.empty(B, np.int64); i2 = np.empty(B, np.int64); ig = np.empty(B, np.int64)
        for b, e in enumerate(eps):
            L = ep_len[int(e)]
            span = rng.integers(SPAN_LO, min(SPAN_HI, L - 1) + 1)
            t1 = rng.integers(0, L - span); tg = t1 + span
            t2 = rng.integers(t1 + GAP, tg - GAP + 1) if tg - GAP >= t1 + GAP else t1 + 1
            idx = ep_index[int(e)]
            i1[b], i2[b], ig[b] = idx[t1], idx[t2], idx[tg]
        return i1, i2, ig

    phi = Phi(D).to(device)
    opt = torch.optim.Adam(phi.parameters(), lr=LR)
    rng = np.random.default_rng(SPLIT_SEED + 1)
    curve = []
    for step in range(1, args.steps + 1):
        i1, i2, ig = sample_triplets(BATCH, rng)
        e1, e2, eg = phi(Z[i1]), phi(Z[i2]), phi(Z[ig])
        d_far = (e1 - eg).pow(2).sum(-1)
        d_near = (e2 - eg).pow(2).sum(-1)
        tau_b = d_far.mean()                       # IN-GRAPH: scale invariance
        loss = torch.nn.functional.softplus((d_near - d_far) / tau_b.clamp_min(1e-8)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            acc = (d_near < d_far).float().mean().item()
            curve.append({"step": step, "loss": float(loss.item()), "order_acc": acc})
            print(f"[{step:5d}] loss {loss.item():.4f}  order-acc {acc:.3f}", flush=True)

    # ---- held-out diagnostics ----
    pool_t = np.concatenate([offsets[e] + np.arange(lengths[e]) for e in test_eps])
    hrows = np.sort(g.choice(pool_t, N_HOLD_FRAMES, replace=False))
    pix, cols = load_frames(dataset, hrows, device, cols=spec["qcols"])
    Zh = []
    with torch.no_grad():
        for p in pix:
            out = model.encoder(p.to(device), interpolate_pos_encoding=True)
            Zh.append(model.projector(out.last_hidden_state[:, 0]).float())
    Zh = torch.cat(Zh).to(device)
    q = np.asarray(spec["build_q"](cols))
    qs = (q - q.mean(0)) / q.std(0)
    with torch.no_grad():
        Eh = phi(Zh).cpu().double().numpy()
    Zc = Zh.cpu().double().numpy()
    iu = np.triu_indices(len(Zc), 1)
    y = ((qs[iu[0]] - qs[iu[1]]) ** 2).sum(-1)
    x_nl = ((Eh[iu[0]] - Eh[iu[1]]) ** 2).sum(-1)
    x_v = ((Zc[iu[0]] - Zc[iu[1]]) ** 2).sum(-1)
    rho_v = float(np.corrcoef(x_v, y)[0, 1])
    rho_nl = float(np.corrcoef(x_nl, y)[0, 1])
    print(f"[diag] rho vanilla={rho_v:.4f}  nonlinear={rho_nl:.4f}", flush=True)

    Path("eval_results").mkdir(exist_ok=True)
    torch.save({"phi_state": phi.state_dict(), "D": D, "hidden": HIDDEN,
                "ckpt": args.ckpt, "task": args.task, "kind": "nonlinear"},
               f"eval_results/automet_{tag}.pt")
    Path(f"eval_results/automet_{tag}.json").write_text(json.dumps({
        "task": args.task, "ckpt": args.ckpt, "curve": curve,
        "rho_vanilla": rho_v, "rho_nl": rho_nl}, indent=1))
    print(f"wrote eval_results/automet_{tag}.pt/.json", flush=True)


if __name__ == "__main__":
    main()
