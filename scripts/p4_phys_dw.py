"""DINO-WM arm of the P4 probe on the physics tasks (pusht / reacher / cube).

The original p4_bottleneck.py run predates the DINO-WM baseline, so its physics
tables have three arms. This fills the fourth under the IDENTICAL protocol --
same episodes files, same CAND_SEED so the candidate action blocks are bit-equal,
same env rollout code (transcribed from p4_bottleneck.py, not re-derived), same
rank metrics -- which makes the result mergeable with the existing p4_<task>.json:
rollerr and (a)-channel tau are per-model quantities over the same candidates.

Model side is the DINO-WM path of p4_bottleneck_nav.py, imported unchanged.

    usage: p4_phys_dw.py {pusht|reacher|cube} <label>:<ckpt> [--starts N] [--cands N]
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
    img_tf,
    infos_val,
    rank_metrics,
    wrap,
)
from scripts.p4_bottleneck_nav import dw_imagined_terminal, dw_patch_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["pusht", "reacher", "cube"])
    ap.add_argument("models", nargs="+", help="label:ckpt_path")
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--cands", type=int, default=64)
    args = ap.parse_args()
    Path("eval_results").mkdir(parents=True, exist_ok=True)

    task = args.task
    preset = ENV_PRESETS[task]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = img_tf()
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

    # ---- env rollouts + physical cost: transcribed from p4_bottleneck.py main() ----
    start_frames, goal_frames, final_frames = [], [], []
    c_phys = np.zeros((args.starts, args.cands))
    for si, ep in enumerate(episodes):
        init_state, goal_state, _ = _extract_init_goal(
            dataset, [ep["traj_id"]], [ep["start_idx"]], GOAL_OFFSET)
        start_frames.append(tf(np.asarray(init_state["pixels"][0]).astype(np.uint8)))
        gk = next((k for k in ("goal", "goal_pixels") if k in goal_state),
                  next((k for k in goal_state if "pixel" in k), None))
        goal_frames.append(tf(np.asarray(goal_state[gk][0]).astype(np.uint8)))
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
            px = np.asarray(world.infos["pixels"][0])
            per_cand.append(tf(px[-1] if px.ndim > 3 else px))
            if task == "cube":
                pos = infos_val(world, "privileged/block_0_pos", 3)
                tgt = np.asarray(goal_state["goal_privileged_block_0_pos"][0]).ravel()[:3]
                c_phys[si, ci] = np.linalg.norm(pos - tgt)
            elif task == "reacher":
                qp = infos_val(world, "qpos", 2)
                tgt = np.asarray(goal_state["goal_qpos"][0]).ravel()[:2]
                c_phys[si, ci] = np.abs(wrap(qp - tgt)).sum()
            else:  # pusht
                bp = infos_val(world, "block_pose", 3)
                gt = np.asarray(goal_state["goal_state"][0]).ravel()
                c_phys[si, ci] = (np.linalg.norm(bp[:2] - gt[2:4]) / 512.0
                                  + abs(wrap(bp[2] - gt[4])) / np.pi)
        final_frames.append(torch.stack(per_cand))
        print(f"[{task}] env rollouts: start {si+1}/{args.starts} "
              f"c_phys range {c_phys[si].min():.4f}..{c_phys[si].max():.4f}", flush=True)
    world.close()

    spread = c_phys.max(1) - c_phys.min(1)
    live = spread > 1e-9
    print(f"[{task}] non-degenerate starts: {live.sum()}/{len(live)}")

    starts = torch.stack(start_frames)
    goals = torch.stack(goal_frames)
    finals = torch.stack(final_frames)

    rows = []
    for spec in args.models:
        label, ckpt = spec.split(":", 1)
        model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        model.requires_grad_(False)
        assert hasattr(model, "backbone") and not hasattr(model, "projector"), (
            "this filler script is for the DINO-WM arm; LeWM arms already have "
            "numbers in p4_<task>.json")

        z_start = dw_patch_features(model, starts, device)          # (S,P,384)
        z_goal = dw_patch_features(model, goals, device).flatten(1)
        z_final = dw_patch_features(model, finals.reshape(-1, *finals.shape[2:]), device)
        z_final = z_final.reshape(args.starts, args.cands, -1)

        flat = z_final.reshape(-1, z_final.size(-1))
        ii = torch.randint(0, flat.size(0), (20000,)); jj = torch.randint(0, flat.size(0), (20000,))
        keep = ii != jj
        scale = (flat[ii[keep]] - flat[jj[keep]]).pow(2).sum(-1).mean().item()

        acc = {f"{p}_{m}": [] for p in "abt" for m in ("tau", "reg", "ovl", "ereg")}
        acc["roll"] = []
        for si in range(args.starts):
            if not live[si]:
                continue
            z_hat = dw_imagined_terminal(model, z_start[si:si+1], cands[si], device)
            z_hat = z_hat.flatten(1).cpu()
            c_imag = (z_hat - z_goal[si]).pow(2).sum(-1).numpy() / scale
            c_enc = (z_final[si] - z_goal[si]).pow(2).sum(-1).numpy() / scale
            cp = c_phys[si]
            for pre, (x, y) in [("a", (c_imag, c_enc)), ("b", (c_enc, cp)), ("t", (c_imag, cp))]:
                tau, reg, ovl, ereg, _ = rank_metrics(x, y, k_elite)
                acc[f"{pre}_tau"].append(tau); acc[f"{pre}_reg"].append(reg)
                acc[f"{pre}_ovl"].append(ovl); acc[f"{pre}_ereg"].append(ereg)
            acc["roll"].append(float(
                (z_hat - z_final[si]).pow(2).sum(-1).mean().item() / scale))
        m = {k: float(np.mean(v)) for k, v in acc.items() if v}
        m["_per_start"] = {k: list(map(float, v)) for k, v in acc.items() if v}
        m["label"] = label
        rows.append(m)
        print(f"{label:6s} rollerr {m['roll']:.4f}  (a)tau {m['a_tau']:.3f}  "
              f"(b)tau {m['b_tau']:.3f}  tot tau {m['t_tau']:.3f}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    out = f"eval_results/p4dw_{task}.json"
    Path(out).write_text(json.dumps(
        {"task": task, "starts": int(live.sum()), "cands": args.cands,
         "k_elite": k_elite, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
