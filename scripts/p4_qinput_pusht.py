"""P4 probe for the q-input model (QJEPA) on PushT: prediction vs planning
decomposition with the encoder fed STATES instead of pixels.

Protocol is transcribed from p4_bottleneck.py (same episodes file, same CAND_SEED so
candidate action blocks are bit-equal, same env rollout and physical cost, same
rank_metrics), so every number is mergeable with p4_pusht*.json rows. The only
substitution: z comes from q = build_q_raw(state) normalized with the model's own
persisted q_stats buffers -- for the start (dataset init state), the goal (dataset
goal_state), and each candidate's TRUE terminal state (env infos['state'], the same
interface budget_sweep_qinput.py plans through, empirically validated by q1's SR).

Channels, as in P4:
  rollerr : ||z_hat - z(true terminal state)||^2 / scale     -> prediction quality
  (a)     : rank(c_imag) vs rank(c_enc)                      -> predictor's share
  (b)     : rank(c_enc)  vs rank(c_phys)                     -> repr/cost's share
  (t)     : rank(c_imag) vs rank(c_phys)                     -> what the planner sees

    usage: p4_qinput_pusht.py q1:lewm_q1_qinput_s3072/weights_epoch_10.pt [--starts N --cands N]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MUJOCO_GL", "egl")

import hdf5plugin  # noqa: F401,E402
import stable_worldmodel as swm  # noqa: E402
from sklearn import preprocessing  # noqa: E402
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal  # noqa: E402

from scripts.budget_sweep import ENV_PRESETS  # noqa: E402
from scripts.p4_bottleneck import (  # noqa: E402
    ACTION_BLOCK,
    CAND_SEED,
    EPISODES,
    GOAL_OFFSET,
    HORIZON,
    imagined_terminal,
    infos_val,
    rank_metrics,
    wrap,
)
from qjepa import QJEPA  # noqa: E402
from utils import build_q_raw  # noqa: E402


@torch.no_grad()
def encode_states(model, states, device, bs=4096):
    """states: (N, 7) raw PushT state rows -> (N, D) embeddings via the q path."""
    x = build_q_raw(torch.as_tensor(np.asarray(states), dtype=torch.float32))
    x = (x.to(device) - model.q_mean) / model.q_std
    zs = []
    for i in range(0, len(x), bs):
        zs.append(model.projector(model.encoder(x[i:i + bs])).float().cpu())
    return torch.cat(zs)


def last_row(arr):
    a = np.asarray(arr, dtype=np.float64)
    return a[-1] if a.ndim > 1 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="label:ckpt_path (QJEPA checkpoints)")
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--cands", type=int, default=64)
    ap.add_argument("--out", default="eval_results/p4_pusht_qinput.json")
    args = ap.parse_args()
    Path("eval_results").mkdir(parents=True, exist_ok=True)

    task = "pusht"
    preset = ENV_PRESETS[task]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k_elite = max(round(0.1 * args.cands), 2)

    episodes = json.loads(Path(EPISODES[task]).read_text())["episodes"][: args.starts]
    ds_kwargs = {}
    if preset.get("keys_to_load"):
        ds_kwargs["keys_to_load"] = preset["keys_to_load"]
    dataset = swm.data.HDF5Dataset(preset["dataset"], keys_to_cache=preset["process_cols"],
                                   cache_dir=Path(swm.data.utils.get_cache_dir()), **ds_kwargs)
    act = dataset.get_col_data("action")
    scaler = preprocessing.StandardScaler()
    scaler.fit(act[~np.isnan(act).any(axis=1)])
    adim = act.shape[1]

    world = swm.World(env_name=preset["env_name"], num_envs=1, image_shape=(224, 224),
                      max_episode_steps=10_000, **preset["env_kwargs"])
    env = world.envs.envs[0].unwrapped

    g = torch.Generator().manual_seed(CAND_SEED)
    cands = torch.randn(args.starts, args.cands, HORIZON, ACTION_BLOCK * adim, generator=g)

    start_states, goal_states, final_states = [], [], []
    c_phys = np.zeros((args.starts, args.cands))

    for si, ep in enumerate(episodes):
        init_state, goal_state, _ = _extract_init_goal(
            dataset, [ep["traj_id"]], [ep["start_idx"]], GOAL_OFFSET)
        start_states.append(np.asarray(init_state["state"][0], dtype=np.float64).ravel()[:7])
        goal_states.append(np.asarray(goal_state["goal_state"][0], dtype=np.float64).ravel()[:7])
        merged = {**init_state, **goal_state}
        env_init = {k: v[0] for k, v in merged.items() if hasattr(v, "__len__")}

        per_cand = []
        for ci in range(args.cands):
            world.reset(seed=[ep["env_seed"]])
            _apply_callables(env, preset["callables"], env_init)
            raw = scaler.inverse_transform(
                cands[si, ci].reshape(HORIZON * ACTION_BLOCK, adim).numpy())
            for a in raw:
                world.envs.step(a[None].astype(np.float32))
            st = last_row(world.infos["state"][0]).ravel()
            assert st.size >= 7 and np.isfinite(st[:7]).all(), f"bad terminal state {st}"
            per_cand.append(st[:7])
            # physical cost: identical to p4_bottleneck's pusht branch
            bp = infos_val(world, "block_pose", 3)
            gt = np.asarray(goal_state["goal_state"][0]).ravel()
            c_phys[si, ci] = (np.linalg.norm(bp[:2] - gt[2:4]) / 512.0
                              + abs(wrap(bp[2] - gt[4])) / np.pi)
        final_states.append(np.stack(per_cand))
        print(f"[q-probe] env rollouts: start {si+1}/{args.starts} "
              f"c_phys range {c_phys[si].min():.4f}..{c_phys[si].max():.4f}", flush=True)
    world.close()

    spread = c_phys.max(1) - c_phys.min(1)
    live = spread > 1e-9
    print(f"[q-probe] starts with non-degenerate spread: {live.sum()}/{len(live)}")

    start_states = np.stack(start_states)
    goal_states = np.stack(goal_states)
    final_states = np.stack(final_states)  # (starts, cands, 7)

    rows = []
    for spec in args.models:
        label, ckpt = spec.split(":", 1)
        model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        model.requires_grad_(False)
        assert isinstance(model, QJEPA), type(model)
        assert not bool((model.q_std == 1).all()), "q_std buffer untrained"

        z_start = encode_states(model, start_states, device)
        z_goal = encode_states(model, goal_states, device)
        z_final = encode_states(model, final_states.reshape(-1, 7), device)
        z_final = z_final.reshape(args.starts, args.cands, -1)

        flat = z_final.reshape(-1, z_final.size(-1)).double()
        ii = torch.randint(0, flat.size(0), (20000,)); jj = torch.randint(0, flat.size(0), (20000,))
        keep = ii != jj
        scale = (flat[ii[keep]] - flat[jj[keep]]).pow(2).sum(-1).mean().item()
        zc = flat - flat.mean(0, keepdim=True)
        eig = torch.linalg.eigvalsh(zc.T @ zc / max(len(zc) - 1, 1)).clamp_min(0)
        p = (eig / eig.sum().clamp_min(1e-12)).numpy()
        eff_rank = float(np.exp(-(p * np.log(np.clip(p, 1e-12, None))).sum()))

        acc = {f"{pfx}_{m}": [] for pfx in "abt" for m in ("tau", "reg", "ovl", "ereg", "vreg")}
        acc["roll"] = []
        for si in range(args.starts):
            if not live[si]:
                continue
            z_hat = imagined_terminal(model, z_start[si:si + 1].to(device),
                                      cands[si].to(device), device).cpu()
            c_imag = (z_hat - z_goal[si]).pow(2).sum(-1).numpy() / scale
            c_enc = (z_final[si] - z_goal[si]).pow(2).sum(-1).numpy() / scale
            cp = c_phys[si]
            for pre, (x, y) in [("a", (c_imag, c_enc)), ("b", (c_enc, cp)), ("t", (c_imag, cp))]:
                tau, reg, ovl, ereg, vreg = rank_metrics(x, y, k_elite)
                acc[f"{pre}_tau"].append(tau); acc[f"{pre}_reg"].append(reg)
                acc[f"{pre}_ovl"].append(ovl); acc[f"{pre}_ereg"].append(ereg)
                acc[f"{pre}_vreg"].append(vreg)
            acc["roll"].append(float((z_hat - z_final[si]).pow(2).sum(-1).mean().item() / scale))
        m = {k: float(np.mean(v)) for k, v in acc.items() if v}
        m["_per_start"] = {k: list(map(float, v)) for k, v in acc.items() if v}
        m["label"] = label
        m["z_eff_rank"] = eff_rank
        rows.append(m)
        print(f"  {label:8s} rollerr {m['roll']:.4f}  (a)tau {m['a_tau']:.3f}  "
              f"(b)tau {m['b_tau']:.3f}  (t)tau {m['t_tau']:.3f}  eff-rank {eff_rank:.1f}", flush=True)

    Path(args.out).write_text(json.dumps(
        {"task": task, "encoder_input": "state->q", "starts": int(live.sum()),
         "cands": args.cands, "k_elite": k_elite, "rows": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
