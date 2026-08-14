"""P4 for the navigation tasks (two-room / PointMaze), with the DINO-WM arm.

Same question and same channels as p4_bottleneck.py, whose helpers this imports
unchanged (that file and its results stay untouched):

    c_imag  = model's cost of its own imagined terminal state   (encoder + predictor)
    c_enc   = same cost computed on the REAL terminal frame     (encoder only)
    c_phys  = the task's physical cost from the simulator       (neither)

    (a) rollout  : rank(c_imag) vs rank(c_enc)   predictor channel
    (b) geometry : rank(c_enc)  vs rank(c_phys)  encoder-geometry channel
    rollerr      = ||z_hat - z_true||^2 / scale  (scale = mean sq distance between
                   random pairs of real terminal embeddings, per model)

DINO-WM is not scored in someone else's latent: its three costs live in ITS
planning space -- flattened DINOv2 patch features, squared distance to the goal's
patch features (rank-identical to the patch-MSE its criterion() uses) -- and its
imagined terminal comes from its own CausalPredictor, stepped exactly as
PreJEPA.rollout steps it (action embedding tiled per patch, sliding history_size
window, actions replaced between predictions, one final action-free prediction).
The LeWM arms go through the identical code path p4_bottleneck.py used.

c_phys mirrors each task's success criterion, position-only in both:
    tworoom   : ||agent - goal||        (env success: < 16 px, from infos['proprio'])
    pointmaze : ||pos - goal_xy||       (DINO-WM eval_state: < 0.5, velocities
                                         ignored; read from the adapter's _state,
                                         the same field its termination check uses)

Candidates are model-independent and shared across all four arms: the same z-scored
action blocks are fed to every model and inverse-transformed once for the simulator.

    usage: p4_bottleneck_nav.py {tworoom|pointmaze} <label>:<ckpt> [...]
                                [--starts N] [--cands N] [--episodes PATH]
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
from scipy.stats import wilcoxon  # noqa: E402
from sklearn import preprocessing  # noqa: E402
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal  # noqa: E402

from scripts.budget_sweep import ENV_PRESETS  # noqa: E402
from scripts.p4_bottleneck import (  # noqa: E402
    ACTION_BLOCK,
    CAND_SEED,
    GOAL_OFFSET,
    HORIZON,
    encode_frames,
    imagined_terminal,
    img_tf,
    infos_val,
    rank_metrics,
)
from scripts.tworoom_preset import TWOROOM_PRESET  # noqa: E402


# ------------------------------------------------------------------ DINO-WM side
@torch.no_grad()
def dw_patch_features(model, frames, device, bs=48):
    """Flattened-per-frame DINOv2 patch grid, the space DINO-WM's criterion costs in.
    CLS is dropped exactly as PreJEPA._encode_image drops it."""
    feats = []
    for i in range(0, len(frames), bs):
        out = model.backbone(frames[i : i + bs].to(device))
        feats.append(out.last_hidden_state[:, 1:, :].float().cpu())
    return torch.cat(feats)  # (B, P, 384)


@torch.no_grad()
def dw_imagined_terminal(model, p0, act_blocks, device):
    """Autoregressive imagination in DINO-WM's own scheme.

    p0: (1, P, 384) start-frame patch features; act_blocks: (S, HORIZON, 10) z-scored
    action blocks, the same candidates the LeWM arms consume. Each context frame is
    [patch features | action embedding tiled over patches]; predictions slide over the
    last history_size frames; the action dims of each prediction are overwritten with
    the next candidate block (PreJEPA.rollout's replace_action_in_embedding, inlined
    for a single already-aligned action slot); the final prediction keeps no action.
    Returns (S, P, 384): the pixels part of the terminal state after HORIZON blocks.
    """
    S = act_blocks.size(0)
    h = int(model.history_size)
    pdim = p0.shape[-1]
    z_act = model.extra_encoders["action"](act_blocks.to(device))  # (S, HORIZON, 10)

    def with_act(pix_emb, a):  # (S, 1, P, pdim) + (S, adim) -> (S, 1, P, pdim+adim)
        at = a[:, None, None, :].expand(-1, 1, pix_emb.shape[2], -1)
        return torch.cat([pix_emb, at], dim=-1)

    ctx = with_act(p0.to(device).expand(S, -1, -1).unsqueeze(1), z_act[:, 0])
    for k in range(1, HORIZON):
        pred = model.predict(ctx[:, -h:])[:, -1:]  # (S, 1, P, pdim+adim)
        ctx = torch.cat([ctx, with_act(pred[..., :pdim], z_act[:, k])], dim=1)
    final = model.predict(ctx[:, -h:])[:, -1:]
    return final[:, 0, :, :pdim]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["tworoom", "pointmaze"])
    ap.add_argument("models", nargs="+", help="label:ckpt_path")
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--cands", type=int, default=64)
    ap.add_argument("--episodes", default=None,
                    help="episodes json (default: eval_sets convention, seed 101)")
    args = ap.parse_args()
    Path("eval_results").mkdir(parents=True, exist_ok=True)

    task = args.task
    presets = dict(ENV_PRESETS)
    if task == "tworoom":
        presets["tworoom"] = TWOROOM_PRESET
    else:
        # imports mujoco_py via pointmaze_env, so only when actually needed
        from scripts.pointmaze_preset import register as _reg_pm
        _reg_pm(presets)
    preset = presets[task]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = img_tf()
    k_elite = max(round(0.1 * args.cands), 2)

    eps_path = args.episodes or f"eval_sets/episodes_{task}_s101_100.json"
    episodes = json.loads(Path(eps_path).read_text())["episodes"][: args.starts]
    dataset = swm.data.HDF5Dataset(preset["dataset"], keys_to_cache=preset["process_cols"],
                                   cache_dir=Path(swm.data.utils.get_cache_dir()),
                                   keys_to_load=preset["keys_to_load"])
    act = dataset.get_col_data("action")
    scaler = preprocessing.StandardScaler()
    scaler.fit(act[~np.isnan(act).any(axis=1)])
    adim = act.shape[1]
    assert ACTION_BLOCK * adim == 10, (adim, "both nav tasks have 2-d actions")

    world = swm.World(env_name=preset["env_name"], num_envs=1, image_shape=(224, 224),
                      max_episode_steps=10_000, **preset["env_kwargs"])
    env = world.envs.envs[0].unwrapped

    g = torch.Generator().manual_seed(CAND_SEED)
    cands = torch.randn(args.starts, args.cands, HORIZON, ACTION_BLOCK * adim, generator=g)

    start_frames, goal_frames, final_frames = [], [], []
    c_phys = np.zeros((args.starts, args.cands))
    raw_phys = np.full((args.starts, args.cands, 4), np.nan)
    accessor_note = None

    for si, ep in enumerate(episodes):
        init_state, goal_state, _ = _extract_init_goal(
            dataset, [ep["traj_id"]], [ep["start_idx"]], GOAL_OFFSET)
        start_frames.append(tf(np.asarray(init_state["pixels"][0]).astype(np.uint8)))
        gk = next((k for k in ("goal", "goal_pixels") if k in goal_state),
                  next((k for k in goal_state if "pixel" in k), None))
        if gk is None:
            raise KeyError(f"no goal image key in {sorted(goal_state)}")
        goal_frames.append(tf(np.asarray(goal_state[gk][0]).astype(np.uint8)))
        merged = {**init_state, **goal_state}
        env_init = {k: v[0] for k, v in merged.items() if hasattr(v, "__len__")}

        if task == "tworoom":
            goal_pos = np.asarray(goal_state["goal_proprio"][0]).ravel()[:2]
        else:
            goal_pos = np.asarray(goal_state["goal_state"][0]).ravel()[:2]

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
            if task == "tworoom":
                # success = ||agent - target|| < 16 px; agent from infos['proprio']
                pos = infos_val(world, "proprio", 2)
                accessor_note = "infos['proprio'] vs dataset goal_proprio"
            else:
                # success = ||pos - goal_xy|| < 0.5; _state is the adapter field its
                # own termination check reads, velocities ignored by the criterion
                pos = np.asarray(env._state, dtype=np.float64).ravel()[:2]
                accessor_note = "adapter _state[:2] vs dataset goal_state[:2]"
            c_phys[si, ci] = np.linalg.norm(pos - goal_pos)
            raw_phys[si, ci, :2] = pos
            raw_phys[si, ci, 2:4] = goal_pos
        final_frames.append(torch.stack(per_cand))
        print(f"[{task}] env rollouts: start {si+1}/{args.starts} "
              f"c_phys range {c_phys[si].min():.4f}..{c_phys[si].max():.4f}", flush=True)
    world.close()
    print(f"[{task}] physical-cost source: {accessor_note}", flush=True)

    spread = c_phys.max(1) - c_phys.min(1)
    live = spread > 1e-9
    print(f"[{task}] starts with non-degenerate physical spread: {live.sum()}/{len(live)}")

    starts = torch.stack(start_frames)
    goals = torch.stack(goal_frames)
    finals = torch.stack(final_frames)

    rows = []
    for spec in args.models:
        label, ckpt = spec.split(":", 1)
        model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        model.requires_grad_(False)
        # PreJEPA/DinoWM carries .backbone and no .projector; LeWM carries .encoder+.projector
        is_dw = hasattr(model, "backbone") and not hasattr(model, "projector")

        if is_dw:
            z_start = dw_patch_features(model, starts, device).flatten(1)
            z_goal = dw_patch_features(model, goals, device).flatten(1)
            z_final = dw_patch_features(
                model, finals.reshape(-1, *finals.shape[2:]), device)
            z_final = z_final.reshape(args.starts, args.cands, -1)
        else:
            z_start = encode_frames(model, starts, device)
            z_goal = encode_frames(model, goals, device)
            z_final = encode_frames(model, finals.reshape(-1, *finals.shape[2:]), device)
            z_final = z_final.reshape(args.starts, args.cands, -1)

        flat = z_final.reshape(-1, z_final.size(-1))
        ii = torch.randint(0, flat.size(0), (20000,)); jj = torch.randint(0, flat.size(0), (20000,))
        keep = ii != jj
        scale = (flat[ii[keep]] - flat[jj[keep]]).pow(2).sum(-1).mean().item()

        acc = {f"{p}_{m}": [] for p in "abt" for m in ("tau", "reg", "ovl", "ereg")}
        acc["roll"] = []
        cost_dump = {"c_imag": {}, "c_enc": {}}
        for si in range(args.starts):
            if not live[si]:
                continue
            if is_dw:
                p0 = dw_patch_features(model, starts[si:si+1], device)
                z_hat = dw_imagined_terminal(model, p0, cands[si], device).flatten(1).cpu()
            else:
                z_hat = imagined_terminal(model, z_start[si:si+1].to(device),
                                          cands[si].to(device), device).cpu()
            c_imag = (z_hat - z_goal[si]).pow(2).sum(-1).numpy() / scale
            c_enc = (z_final[si] - z_goal[si]).pow(2).sum(-1).numpy() / scale
            cp = c_phys[si]
            cost_dump["c_imag"][si] = c_imag; cost_dump["c_enc"][si] = c_enc
            for pre, (x, y) in [("a", (c_imag, c_enc)), ("b", (c_enc, cp)), ("t", (c_imag, cp))]:
                tau, reg, ovl, ereg, _ = rank_metrics(x, y, k_elite)
                acc[f"{pre}_tau"].append(tau); acc[f"{pre}_reg"].append(reg)
                acc[f"{pre}_ovl"].append(ovl); acc[f"{pre}_ereg"].append(ereg)
            acc["roll"].append(float(
                (z_hat - z_final[si]).pow(2).sum(-1).mean().item() / scale))
        m = {k: float(np.mean(v)) for k, v in acc.items() if v}
        m["_per_start"] = {k: list(map(float, v)) for k, v in acc.items() if v}
        m["label"] = label
        np.savez(f"eval_results/p4nav_costs_{task}_{label}.npz",
                 c_imag=np.stack([cost_dump["c_imag"][k] for k in sorted(cost_dump["c_imag"])]),
                 c_enc=np.stack([cost_dump["c_enc"][k] for k in sorted(cost_dump["c_enc"])]),
                 starts=np.array(sorted(cost_dump["c_imag"])))
        rows.append(m)
        print(f"  {label:14s} done", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*104}\nP4-nav {task}  ({int(live.sum())} starts x {args.cands} candidates, "
          f"elite k={k_elite})\n{'='*104}")
    print("regret = rank percentile of the model's pick in the true ordering "
          "(0 best, 0.5 random); ereg = same for the whole elite set\n")
    print(f"{'model':8s}{'rollerr':>9s}"
          f"{'(a)tau':>8s}{'(a)reg':>8s}{'(a)ereg':>9s}{'(a)elite':>9s}"
          f"{'(b)tau':>8s}{'(b)reg':>8s}{'(b)ereg':>9s}{'(b)elite':>9s}"
          f"{'tot tau':>9s}{'tot reg':>9s}")
    for m in rows:
        print(f"{m['label']:8s}{m['roll']:9.4f}"
              f"{m['a_tau']:8.3f}{m['a_reg']:8.3f}{m['a_ereg']:9.3f}{m['a_ovl']:9.2f}"
              f"{m['b_tau']:8.3f}{m['b_reg']:8.3f}{m['b_ereg']:9.3f}{m['b_ovl']:9.2f}"
              f"{m['t_tau']:9.3f}{m['t_reg']:9.3f}")

    if len(rows) > 1:
        base = rows[0]
        print(f"\n--- paired over the {len(base['_per_start']['b_tau'])} starts "
              f"(Wilcoxon signed-rank vs {base['label']}) ---")
        print(f"  {'model':6s}{'metric':10s}{'base':>8s}{'model':>8s}{'delta':>9s}"
              f"{'SD(delta)':>11s}{'SE':>8s}{'p':>9s}")
        for m in rows[1:]:
            for key, lab in [("roll", "roll err"), ("a_tau", "(a) tau"), ("b_tau", "(b) tau"),
                             ("b_ereg", "(b) ereg"), ("t_tau", "tot tau"), ("t_ereg", "tot ereg")]:
                x = np.array(base["_per_start"][key]); y = np.array(m["_per_start"][key])
                d = y - x
                try:
                    p_ = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                except ValueError:
                    p_ = 1.0
                print(f"  {m['label']:6s}{lab:10s}{x.mean():8.4f}{y.mean():8.4f}{d.mean():+9.4f}"
                      f"{d.std(ddof=1):11.4f}{d.std(ddof=1)/np.sqrt(len(d)):8.4f}{p_:9.4f}"
                      + ("  *" if p_ < 0.05 else ""))

    out = f"eval_results/p4nav_{task}.json"
    Path(out).write_text(json.dumps(
        {"task": task, "starts": int(live.sum()), "cands": args.cands,
         "k_elite": k_elite, "rows": rows}, indent=2))
    np.savez(f"eval_results/p4nav_cache_{task}.npz", c_phys=c_phys, live=live,
             raw_phys=raw_phys)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
