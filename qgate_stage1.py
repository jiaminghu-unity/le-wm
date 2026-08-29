"""Stage 1 of automatic planning-state discovery for SCALE: learn a sparse gate g
over simulator state variables from behavior alone.

    [ Q_t (ungated, full context) ; g ⊙ (Q_{t+H} − Q_t) ]  →  MLP  →  a_t chunk

Deliberately DECOUPLED from JEPA/LeWM: no images, no encoder, no z anywhere.
Q_t is not gated (a controller legitimately needs the full current state); only
the desired displacement is gated, so g answers "which dimensions of the desired
change matter for deciding how the plan proceeds" -- exactly what SCALE's metric
wants. The action MLP is a discovery tool and is discarded after g* is read out.

Anti-shortcut machinery (the fixed-goal trap: if Q_t alone predicts a_t, sparsity
kills every gate without meaning "no Q matters"):
  * multi-horizon goals H ∈ {1,2,4,8} world-model steps (× frameskip raw frames);
  * negative-goal ranking: log p(a_t | Q_t, gΔQ⁺) must beat an in-batch shuffled
    goal's log p by margin m (hinge), weight γ;
  * a goal-blind diagnostic NLL (ΔQ zeroed) is always reported -- if blind ≈ full,
    the dataset lacks goal variation and g is not interpretable. Fail loudly.

Outputs one JSON per run: per-dimension gates over training, final g*, NLL /
blind-NLL / rank-loss curves, normalization stats (needed by Stage 2's weighted
metric d_{Q,g*}), and the exact config.

    usage: qgate_stage1.py --task pusht --h5 <path> --lambda-sparse 0.01
           [--gamma 1.0] [--margin 1.0] [--steps 8000] [--out out.json]
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

FRAMESKIP = 5          # raw frames per world-model step (matches LeWM training)
HORIZONS = (1, 2, 4, 8)  # goal offsets in world-model steps

# task -> (build_q(state np array) -> Q, gate groups [(name, [q dims])], action_dim)
def _build_q_pusht(state):
    pos = state[..., :4]
    th = state[..., 4:5]
    vel = state[..., 5:7]
    return np.concatenate([pos, np.cos(th), np.sin(th), vel], axis=-1)


_CUBE_ARM_JOINTS = [0, 1, 2, 3, 5]  # = utils.CUBE_ARM_JOINTS (joint 4 frozen)


def _build_q_cube_full(cols):
    """22-d cube full-config q, numpy mirror of q_cube_full.build_q_cube_full."""
    eff, yaw = cols["proprio_effector_pos"], cols["proprio_effector_yaw"]
    parts = [eff[..., :3],
             np.cos(2.0 * yaw[..., :1]), np.sin(2.0 * yaw[..., :1]),
             cols["proprio_gripper_opening"][..., :1],
             cols["proprio_gripper_contact"][..., :1]]
    jp = cols["proprio_joint_pos"]
    for i in _CUBE_ARM_JOINTS:
        parts += [np.cos(jp[..., i:i + 1]), np.sin(jp[..., i:i + 1])]
    parts += [cols["privileged_block_0_pos"][..., :3],
              np.cos(4.0 * cols["privileged_block_0_yaw"][..., :1]),
              np.sin(4.0 * cols["privileged_block_0_yaw"][..., :1])]
    q = np.concatenate(parts, axis=-1)
    assert q.shape[-1] == 22, q.shape
    return q


_CUBE_COLS = ["proprio_effector_pos", "proprio_effector_yaw", "proprio_gripper_opening",
              "proprio_gripper_contact", "proprio_joint_pos",
              "privileged_block_0_pos", "privileged_block_0_yaw"]

TASKS = {
    "pusht": dict(
        build_q=_build_q_pusht,
        state_cols=["state"],
        action_col="action",
        dim_names=["pusher_x", "pusher_y", "tblock_x", "tblock_y",
                   "cos_theta", "sin_theta", "vx", "vy"],
    ),
    "cube": dict(
        build_q=_build_q_cube_full,
        state_cols=_CUBE_COLS,
        action_col="action",
        dim_names=["eff_x", "eff_y", "eff_z", "cos2psi", "sin2psi",
                   "grip_open", "grip_contact",
                   "cos_j0", "sin_j0", "cos_j1", "sin_j1", "cos_j2", "sin_j2",
                   "cos_j3", "sin_j3", "cos_j5", "sin_j5",
                   "block_x", "block_y", "block_z", "cos4th", "sin4th"],
    ),
}


class GatedActor(nn.Module):
    """[Q_t ; g⊙ΔQ] -> 3-layer MLP (LayerNorm+GELU, width 1024) -> diag Gaussian.
    One independent gate per q DIMENSION (no manual grouping)."""

    def __init__(self, q_dim, act_chunk_dim, width=1024):
        super().__init__()
        self.gate_logit = nn.Parameter(torch.full((q_dim,), 2.0))  # sigmoid(2)≈0.88, start open
        self.mlp = nn.Sequential(
            nn.Linear(2 * q_dim, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(),
            nn.Linear(width, 2 * act_chunk_dim),
        )
        self.act_chunk_dim = act_chunk_dim

    def gates(self):
        return torch.sigmoid(self.gate_logit)

    def log_prob(self, q_t, dq, a):
        g = self.gates()                             # (q_dim,)
        h = torch.cat([q_t, g * dq], dim=-1)
        mu, log_std = self.mlp(h).chunk(2, dim=-1)
        log_std = log_std.clamp(-5.0, 2.0)
        nll = 0.5 * ((a - mu) / log_std.exp()) ** 2 + log_std
        return -(nll + 0.5 * np.log(2 * np.pi)).sum(-1)  # (B,) log p(a|·)


def load_samples(h5_path, spec):
    with h5py.File(h5_path, "r") as f:
        cols = {c: np.asarray(f[c], dtype=np.float64) for c in spec["state_cols"]}
        action = np.asarray(f[spec["action_col"]], dtype=np.float64)
        ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
    q = spec["build_q"](cols if len(spec["state_cols"]) > 1 else cols[spec["state_cols"][0]])
    max_h = max(HORIZONS) * FRAMESKIP
    rows, goals = [], {h: [] for h in HORIZONS}
    for L, O in zip(ep_len, ep_off):
        n = int(L) - max_h - FRAMESKIP
        if n <= 0:
            continue
        idx = O + np.arange(n)
        rows.append(idx)
    rows = np.concatenate(rows)
    return q, action, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--h5", required=True)
    ap.add_argument("--lambda-sparse", type=float, required=True)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = TASKS[args.task]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    q, action, rows = load_samples(args.h5, spec)
    for name, arr in (("q", q), ("action", action)):
        n_bad = int(np.size(arr) - np.isfinite(arr).sum())
        if n_bad:
            print(f"[data] WARNING: {n_bad} non-finite values in {name}, zero-filled", flush=True)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
    # std 下限:近似常数维不许把标准化值炸上天(NaN 事故 2026-08-29 的根因候选)
    q_mean, q_std = q.mean(0), np.maximum(q.std(0), 1e-3)
    qn = np.clip((q - q_mean) / q_std, -10.0, 10.0)
    print("[data] q per-dim std:", np.round(q.std(0), 5).tolist(), flush=True)
    a_dim = action.shape[-1] * FRAMESKIP  # action-chunk dim inferred from data
    a_mean = action.mean(0)
    a_std = np.maximum(action.std(0), 1e-3)
    an = np.clip((action - a_mean) / a_std, -10.0, 10.0)

    q_dim = qn.shape[-1]
    names = spec["dim_names"]
    assert len(names) == q_dim, (len(names), q_dim)

    qn_t = torch.tensor(qn, dtype=torch.float32, device=dev)
    an_t = torch.tensor(an, dtype=torch.float32, device=dev)
    rows_t = torch.tensor(rows, device=dev)

    model = GatedActor(q_dim, a_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    hist = {"step": [], "nll": [], "nll_blind": [], "rank": [], "gates": []}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        sel = rows_t[torch.randint(0, len(rows_t), (args.batch,), device=dev)]
        h = torch.tensor(rng.choice(HORIZONS, size=args.batch), device=dev)
        q_t = qn_t[sel]
        q_g = qn_t[sel + h * FRAMESKIP]
        # action chunk: FRAMESKIP consecutive low-level actions from t
        a = torch.stack([an_t[sel + i] for i in range(FRAMESKIP)], dim=1).flatten(1)
        dq_pos = q_g - q_t
        dq_neg = torch.roll(q_g, 1, dims=0) - q_t   # in-batch shuffled goal (other trajectory)

        lp_pos = model.log_prob(q_t, dq_pos, a)
        lp_neg = model.log_prob(q_t, dq_neg, a)
        nll = -lp_pos.mean()
        rank = F.relu(args.margin - (lp_pos - lp_neg)).mean()
        g = model.gates()
        loss = nll + args.gamma * rank + args.lambda_sparse * g.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step <= 5 or step % 200 == 0:
            if not torch.isfinite(loss):
                print(f"FATAL: non-finite loss at step {step} "
                      f"(nll={float(nll)}, rank={float(rank)}); "
                      f"batch |q| max={float(q_t.abs().max())}, |a| max={float(a.abs().max())}",
                      flush=True)
                raise SystemExit(3)

        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                nll_blind = -model.log_prob(q_t, torch.zeros_like(dq_pos), a).mean()
            hist["step"].append(step)
            hist["nll"].append(round(float(nll), 4))
            hist["nll_blind"].append(round(float(nll_blind), 4))
            hist["rank"].append(round(float(rank), 4))
            hist["gates"].append([round(float(x), 4) for x in g.detach().cpu()])
            if step % 1000 == 0:
                gs = ", ".join(f"{n}={v:.2f}" for n, v in zip(names, hist["gates"][-1]))
                print(f"[{step}] nll={nll:.3f} blind={nll_blind:.3f} rank={rank:.3f} | {gs}",
                      flush=True)

    g_star = {n: round(float(v), 4) for n, v in zip(names, model.gates().detach().cpu())}
    blind_gap = hist["nll_blind"][-1] - hist["nll"][-1]
    verdict = ("OK" if blind_gap > 0.1 else
               "WARNING: goal-blind NLL ~= full NLL -- dataset may lack goal variation; "
               "g is NOT interpretable at this lambda")
    out = {
        "task": args.task, "lambda_sparse": args.lambda_sparse, "gamma": args.gamma,
        "margin": args.margin, "steps": args.steps, "seed": args.seed,
        "horizons_wm_steps": list(HORIZONS), "frameskip": FRAMESKIP,
        "dim_names": names,
        "g_star": g_star,
        "goal_blind_gap_nats": round(float(blind_gap), 4),
        "verdict": verdict,
        "q_mean": [round(float(x), 6) for x in q_mean],
        "q_std": [round(float(x), 6) for x in q_std],
        "history": hist,
        "wallclock_s": round(time.time() - t0, 1),
        "n_samples": int(len(rows)),
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[done] g* = {g_star}")
    print(f"[done] goal-blind gap = {blind_gap:.3f} nats ({verdict})")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
