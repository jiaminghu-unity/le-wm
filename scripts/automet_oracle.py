"""In-class ORACLE ceiling for AutoMetric: how good can a full-dimensional
Mahalanobis metric on this frozen representation possibly be, if supervision were
perfect?

Construction: fit the ridge linear probe z -> q_hat on TRAIN-side frames (the same
probe family every diagnostic uses), stack its coefficient matrix A (d_q x D) as the
metric square root, i.e.

    W_oracle = A          M_oracle = A^T A  (trace-normalized like the learned one)

so d_oracle(z_i, z_j) = ||q_hat_i - q_hat_j||^2: the best quadratic-form proxy for
physical distance this representation admits. This uses privileged q -- it is a
DIAGNOSTIC BOUND, never a method.

Decision rule it feeds:
  * rho_oracle high  -> the function class suffices; v1's weak result means the
    temporal-ordering supervision is the bottleneck -> upgrade supervision.
  * rho_oracle low   -> no quadratic metric can fix this frozen geometry ->
    the representation itself is the bottleneck; move task or touch representation.

    usage: automet_oracle.py cube --ckpt lewm_k1_cube_s3072/weights_epoch_10.pt
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

N_TRAIN = 20000
N_HOLD = 1500


@torch.no_grad()
def encode(model, pix, device):
    zs = []
    for p in pix:
        out = model.encoder(p.to(device), interpolate_pos_encoding=True)
        zs.append(model.projector(out.last_hidden_state[:, 0]).float().cpu())
    return torch.cat(zs).double().numpy()


def sample_rows(dataset, eps, n, g):
    lengths, offsets = np.asarray(dataset.lengths), np.asarray(dataset.offsets)
    pool = np.concatenate([offsets[e] + np.arange(lengths[e]) for e in eps])
    return np.sort(g.choice(pool, n, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--q-variant", default="canonical",
                    choices=["canonical", "cube_full", "reacher_full"],
                    help="cube_full: 22-d full-config q; reacher_full: joints cos/sin + finger xy (6-d)")
    args = ap.parse_args()
    tag = args.out_tag or f"{args.task}_oracle"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = dict(TASKS[args.task])
    if args.q_variant == "cube_full":
        assert args.task == "cube"
        import torch as _t
        from q_cube_full import Q_VARIANTS_CUBE_FULL
        builder, sources, _chk = Q_VARIANTS_CUBE_FULL["cube_full_config"]
        spec["qcols"] = tuple(sources)
        spec["build_q"] = lambda c: builder(*[_t.from_numpy(c[k]) for k in sources]).numpy()
        print("[oracle] using 22-d cube_full_config q", flush=True)
    elif args.q_variant == "reacher_full":
        assert args.task == "reacher"
        import torch as _t
        from utils import build_q_reacher_joints_finger
        spec["qcols"] = ("qpos", "finger_pos")
        spec["build_q"] = lambda c: build_q_reacher_joints_finger(
            _t.from_numpy(c["qpos"]), _t.from_numpy(c["finger_pos"])).numpy()
        print("[oracle] using 6-d reacher joints+finger q", flush=True)

    dataset = swm.data.load_dataset(spec["lance"], keys_to_load=["pixels", *spec["qcols"]])
    n_ep = len(dataset.lengths)
    g = np.random.default_rng(SPLIT_SEED)
    perm = g.permutation(n_ep)
    n_test = int(n_ep * TEST_EPISODE_FRAC)
    test_eps, train_eps = perm[:n_test], perm[n_test:]

    model = swm.wm.utils.load_pretrained(args.ckpt).to(device).eval()
    model.requires_grad_(False)

    # ---- fit the probe on train side ----
    rows_tr = sample_rows(dataset, train_eps, N_TRAIN, g)
    pix, cols = load_frames(dataset, rows_tr, device, cols=spec["qcols"])
    Z = encode(model, pix, device)
    q = np.asarray(spec["build_q"](cols)).astype(np.float64)
    qs = (q - q.mean(0)) / q.std(0)
    Zc = Z - Z.mean(0)
    A = np.linalg.solve(Zc.T @ Zc + 1e-2 * np.eye(Z.shape[1]), Zc.T @ qs).T  # (dq, D)
    D = Z.shape[1]
    M = A.T @ A
    M = D * M / np.trace(M)

    # ---- held-out rho under the oracle metric ----
    rows_ho = sample_rows(dataset, test_eps, N_HOLD, g)
    pix, cols = load_frames(dataset, rows_ho, device, cols=spec["qcols"])
    Zh = encode(model, pix, device)
    qh = np.asarray(spec["build_q"](cols)).astype(np.float64)
    qhs = (qh - q.mean(0)) / q.std(0)   # train-side stats, like everything else
    iu = np.triu_indices(N_HOLD, 1)

    def rho(Mm):
        dz = Zh[iu[0]] - Zh[iu[1]]
        x = np.einsum("nd,dk,nk->n", dz, Mm, dz)
        dq = qhs[iu[0]] - qhs[iu[1]]
        y = (dq ** 2).sum(-1)
        return float(np.corrcoef(x, y)[0, 1])

    rho_van = rho(np.eye(D))
    rho_orc = rho(M)
    print(f"[oracle] rho vanilla={rho_van:.4f}  oracle={rho_orc:.4f}", flush=True)

    Path("eval_results").mkdir(exist_ok=True)
    torch.save({"M": torch.tensor(M).float(), "tau": None, "D": D,
                "ckpt": args.ckpt, "task": args.task, "kind": "oracle"},
               f"eval_results/automet_{tag}.pt")
    Path(f"eval_results/automet_{tag}.json").write_text(json.dumps({
        "task": args.task, "ckpt": args.ckpt,
        "rho_vanilla": rho_van, "rho_oracle": rho_orc}, indent=1))
    print(f"wrote eval_results/automet_{tag}.pt/.json", flush=True)


if __name__ == "__main__":
    main()
