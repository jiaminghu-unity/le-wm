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

def _build_q_reacher(cols):
    """joints cos/sin (4) + finger xy (2) [+ qvel (2) if the h5 carries it]."""
    qp = cols["qpos"]
    parts = [np.cos(qp[..., :1]), np.sin(qp[..., :1]),
             np.cos(qp[..., 1:2]), np.sin(qp[..., 1:2]),
             cols["finger_pos"][..., :2]]
    if "qvel" in cols:
        parts.append(cols["qvel"][..., :2])
    return np.concatenate(parts, axis=-1)


def _arm17_np(cols):
    yaw = cols["proprio/effector_yaw"]
    psi2 = 2.0 * yaw.reshape(*yaw.shape[:-1], -1)[..., :1]
    jp = cols["proprio/joint_pos"]
    joints = jp.reshape(*jp.shape[:-1], -1)[..., :5]
    flat = lambda k: cols[k].reshape(*cols[k].shape[:-1], -1)[..., :1]
    return np.concatenate([
        cols["proprio/effector_pos"][..., :3],
        np.cos(psi2), np.sin(psi2),
        flat("proprio/gripper_opening"), flat("proprio/gripper_contact"),
        np.cos(joints), np.sin(joints),
    ], axis=-1)


def _block5_np(cols, i):
    yaw = cols[f"privileged/block_{i}_yaw"]
    th4 = 4.0 * yaw.reshape(*yaw.shape[:-1], -1)[..., :1]
    return np.concatenate([cols[f"privileged/block_{i}_pos"][..., :3],
                           np.cos(th4), np.sin(th4)], axis=-1)


def _build_q_cube_double(cols):
    q = np.concatenate([_arm17_np(cols), _block5_np(cols, 0), _block5_np(cols, 1)], axis=-1)
    assert q.shape[-1] == 27, q.shape
    return q


def _make_build_q_cube_n(n):
    def build(cols):
        q = np.concatenate([_arm17_np(cols)] + [_block5_np(cols, i) for i in range(n)], axis=-1)
        assert q.shape[-1] == 17 + 5 * n, q.shape
        return q
    return build


def _build_q_puzzle(cols):
    flat = lambda k: cols[k].reshape(*cols[k].shape[:-1], -1)[..., :1]
    q = np.concatenate([_arm17_np(cols)] + [flat(f"privileged/button_{i}_state") for i in range(9)], axis=-1)
    assert q.shape[-1] == 26, q.shape
    return q


def _build_q_scene(cols):
    flat = lambda k: cols[k].reshape(*cols[k].shape[:-1], -1)[..., :1]
    q = np.concatenate([_arm17_np(cols), _block5_np(cols, 0),
                        flat("privileged/drawer_pos"), flat("privileged/window_pos"),
                        flat("privileged/button_0_state"), flat("privileged/button_1_state")], axis=-1)
    assert q.shape[-1] == 26, q.shape
    return q


_ARM_SRC_SLASH = ["proprio/effector_pos", "proprio/effector_yaw", "proprio/gripper_opening",
                  "proprio/gripper_contact", "proprio/joint_pos"]
_ARM_NAMES = (["eff_x", "eff_y", "eff_z", "cos2psi", "sin2psi", "grip_open", "grip_contact"]
              + [f"cos_j{i}" for i in range(5)] + [f"sin_j{i}" for i in range(5)])
_BLOCK_NAMES = lambda p: [f"{p}_x", f"{p}_y", f"{p}_z", f"{p}_cos4th", f"{p}_sin4th"]

TASKS = {
    "pusht": dict(
        build_q=_build_q_pusht,
        state_cols=["state"],
        action_col="action",
        dim_names=["pusher_x", "pusher_y", "tblock_x", "tblock_y",
                   "cos_theta", "sin_theta", "vx", "vy"],
    ),
    "reacher": dict(
        build_q=_build_q_reacher,
        state_cols=["qpos", "finger_pos"],
        optional_cols=["qvel"],
        action_col="action",
        dim_names=["cos_j0", "sin_j0", "cos_j1", "sin_j1", "finger_x", "finger_y"],
        optional_dim_names=["qvel_0", "qvel_1"],
    ),
    "reacher_novel": dict(  # 对照:Q_t 不含 qvel(检验"速度冗余"解释)
        build_q=_build_q_reacher,
        state_cols=["qpos", "finger_pos"],
        action_col="action",
        dim_names=["cos_j0", "sin_j0", "cos_j1", "sin_j1", "finger_x", "finger_y"],
    ),
    "cube_double": dict(
        build_q=_build_q_cube_double, loader="lance",
        state_cols=_ARM_SRC_SLASH + ["privileged/block_0_pos", "privileged/block_0_yaw",
                                     "privileged/block_1_pos", "privileged/block_1_yaw"],
        action_col="action",
        dim_names=_ARM_NAMES + _BLOCK_NAMES("b0") + _BLOCK_NAMES("b1"),
    ),
    "scene": dict(
        build_q=_build_q_scene, loader="lance",
        state_cols=_ARM_SRC_SLASH + ["privileged/block_0_pos", "privileged/block_0_yaw",
                                     "privileged/drawer_pos", "privileged/window_pos",
                                     "privileged/button_0_state", "privileged/button_1_state"],
        action_col="action",
        dim_names=_ARM_NAMES + _BLOCK_NAMES("b0") + ["drawer", "window", "btn0", "btn1"],
    ),
    "cube_triple": dict(
        build_q=_make_build_q_cube_n(3), loader="lance",
        state_cols=_ARM_SRC_SLASH + ["privileged/block_0_pos", "privileged/block_0_yaw"] + ["privileged/block_1_pos", "privileged/block_1_yaw"] + ["privileged/block_2_pos", "privileged/block_2_yaw"],
        action_col="action",
        dim_names=_ARM_NAMES + _BLOCK_NAMES("b0") + _BLOCK_NAMES("b1") + _BLOCK_NAMES("b2"),
    ),
    "cube_quadruple": dict(
        build_q=_make_build_q_cube_n(4), loader="lance",
        state_cols=_ARM_SRC_SLASH + ["privileged/block_0_pos", "privileged/block_0_yaw"] + ["privileged/block_1_pos", "privileged/block_1_yaw"] + ["privileged/block_2_pos", "privileged/block_2_yaw"] + ["privileged/block_3_pos", "privileged/block_3_yaw"],
        action_col="action",
        dim_names=_ARM_NAMES + _BLOCK_NAMES("b0") + _BLOCK_NAMES("b1") + _BLOCK_NAMES("b2") + _BLOCK_NAMES("b3"),
    ),
    "puzzle_3x3": dict(
        build_q=_build_q_puzzle, loader="lance",
        state_cols=_ARM_SRC_SLASH + [f"privileged/button_{i}_state" for i in range(9)],
        action_col="action",
        dim_names=_ARM_NAMES + [f"btn{i}" for i in range(9)],
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


def load_samples_lance(path, spec):
    from stable_worldmodel.data.formats.lance import LanceDataset
    need = spec["state_cols"] + [spec["action_col"]]
    ds = LanceDataset(path=path, keys_to_load=need)
    ep = np.asarray(ds.get_col_data("episode_idx")).reshape(-1)
    st = np.asarray(ds.get_col_data("step_idx")).reshape(-1)
    order = np.lexsort((st, ep))
    assert (order == np.arange(len(ep))).all(), "lance rows not episode-major"
    cols = {c: np.asarray(ds.get_col_data(c), dtype=np.float64) for c in spec["state_cols"]}
    for c in list(cols):
        if cols[c].ndim == 1:
            cols[c] = cols[c][:, None]
    action = np.asarray(ds.get_col_data(spec["action_col"]), dtype=np.float64)
    q = spec["build_q"](cols)
    max_h = max(HORIZONS) * FRAMESKIP
    ok = np.zeros(len(ep), bool)
    n = len(ep) - max_h - FRAMESKIP
    ok[:n] = ep[:n] == ep[max_h + FRAMESKIP:]
    return q, action, np.nonzero(ok)[0]


def load_samples(h5_path, spec):
    if spec.get("loader") == "lance":
        return load_samples_lance(h5_path, spec)
    with h5py.File(h5_path, "r") as f:
        cols = {c: np.asarray(f[c], dtype=np.float64) for c in spec["state_cols"]}
        for c in spec.get("optional_cols", []):
            if c in f:
                cols[c] = np.asarray(f[c], dtype=np.float64)
                print(f"[data] optional column present: {c}", flush=True)
        action = np.asarray(f[spec["action_col"]], dtype=np.float64)
        ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
    q = spec["build_q"](cols if len(spec["state_cols"]) > 1 else cols[spec["state_cols"][0]])
    max_h = max(HORIZONS) * FRAMESKIP
    if getattr(load_samples, "_max_eps", 0):
        rng_ep = np.random.default_rng(load_samples._data_seed)
        keep = rng_ep.choice(len(ep_len), size=min(load_samples._max_eps, len(ep_len)), replace=False)
        ep_len, ep_off = ep_len[sorted(keep)], ep_off[sorted(keep)]
        print(f"[data] episode subsample: {len(ep_len)} episodes (seed {load_samples._data_seed})", flush=True)
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
    ap.add_argument("--reg", choices=["l1", "l2"], default="l1",
                    help="gate sparsity penalty: l1 = lambda*sum(g) (drives gates to zero), "
                         "l2 = lambda*sum(g^2) (shrinks without zeroing)")
    ap.add_argument("--rank", choices=["hinge", "infonce"], default="hinge",
                    help="goal-identifiability loss: hinge = single-negative margin (K=1, m nats), "
                         "infonce = 1 positive vs --neg-k in-batch negatives at tau=1")
    ap.add_argument("--neg-k", type=int, default=255)
    ap.add_argument("--max-episodes", type=int, default=0,
                    help="subsample this many WHOLE episodes (0 = all); for the data-efficiency study")
    ap.add_argument("--data-seed", type=int, default=0, help="episode-subsample RNG seed")
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

    load_samples._max_eps = args.max_episodes
    load_samples._data_seed = args.data_seed
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
    names = list(spec["dim_names"])
    extra = spec.get("optional_dim_names", [])
    while len(names) < q_dim and extra:
        names.append(extra[len(names) - len(spec["dim_names"])])
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
        nll = -lp_pos.mean()
        if args.rank == "hinge":
            lp_neg = model.log_prob(q_t, dq_neg, a)
            rank = F.relu(args.margin - (lp_pos - lp_neg)).mean()
        else:
            # InfoNCE at tau=1 (logp is already in nats): 1 positive vs K in-batch
            # negatives (goal displacements of K other samples, rolled indices),
            # scored in ONE batched forward -- run with --batch <= 1024 so the
            # (B*(K+1)) activation footprint stays a few GB.
            B = q_t.shape[0]
            K = min(args.neg_k, B - 1)
            idx = (torch.arange(B, device=dev)[:, None]
                   + torch.arange(1, K + 1, device=dev)[None, :]) % B      # (B, K)
            dq_all = torch.cat([dq_pos[:, None, :], q_g[idx] - q_t[:, None, :]], dim=1)  # (B, 1+K, d)
            q_rep = q_t[:, None, :].expand(-1, K + 1, -1)
            a_rep = a[:, None, :].expand(-1, K + 1, -1)
            logits = model.log_prob(q_rep.reshape(B * (K + 1), -1),
                                    dq_all.reshape(B * (K + 1), -1),
                                    a_rep.reshape(B * (K + 1), -1)).view(B, K + 1)
            rank = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=dev))
        g = model.gates()
        reg = g.sum() if args.reg == "l1" else (g * g).sum()
        loss = nll + args.gamma * rank + args.lambda_sparse * reg
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
        "reg": args.reg, "rank_loss": args.rank, "neg_k": (args.neg_k if args.rank == "infonce" else 1),
        "max_episodes": args.max_episodes, "data_seed": args.data_seed,
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
