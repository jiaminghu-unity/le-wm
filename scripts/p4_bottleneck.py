"""P4: which bottleneck bites — rollout error or cost-ranking error?

P1 measured only the first: its cost_model and cost_true both pass through the SAME
encoder, so its eps isolates the predictor. Nothing in the existing pipeline ever
compares the latent cost against a physical ground truth, so the geometry channel
has never been measured.

Adding the simulator's own cost closes that. For each start state we sample N
candidate action sequences, execute every one in the simulator, and record three
costs per candidate:

    c_imag  = ||z_hat    - z_goal||^2     predictor's guess          (encoder + predictor)
    c_enc   = ||z_true   - z_goal||^2     encoder on the real future (encoder only)
    c_phys  = task's own physical cost    from simulator state       (neither)

Two channels then separate cleanly, both as rank agreement over the candidates:

    (a) rollout   : rank(c_imag) vs rank(c_enc)    same encoder both sides,
                                                   only the predictor differs
    (b) geometry  : rank(c_enc)  vs rank(c_phys)   no predictor involved at all
    total         : rank(c_imag) vs rank(c_phys)

Pre-registration (fixed before running):
  1. aux reduces (a) more than L_obj does; L_obj reduces (b) more than aux does.
     Falsified if either ordering reverses.
  2. Reacher's total error is dominated by (a), Cube's by (b). Push-T is contact-rich
     and expected to sit on the (a) side despite its q being larger than Reacher's,
     which is the observation the q-dimension story cannot accommodate.
  3. regret (the decision-relevant number: how much true cost you pay for picking
     the model's argmin instead of the real best candidate) follows (a)+(b).

c_phys mirrors each task's success criterion so the ranking target is the thing the
episode is actually scored on:
    reacher : sum_j |wrap(qpos_j - target_qpos_j)|   (success = per-joint threshold)
    cube    : ||block_0_pos - target_pos||           (success = <= 0.04 m)
    pusht   : -(coverage reward)                     (success = coverage threshold)

    usage: p4_bottleneck.py <task> <ckpt_label>:<ckpt_path> [...] [--starts N] [--cands N]
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
import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from scipy.stats import kendalltau, wilcoxon  # noqa: E402
from sklearn import preprocessing  # noqa: E402
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal  # noqa: E402
from torchvision.transforms import v2 as transforms  # noqa: E402

from scripts.budget_sweep import ENV_PRESETS  # noqa: E402

HORIZON = 5
ACTION_BLOCK = 5
GOAL_OFFSET = 25
CAND_SEED = 7          # candidates are model-independent and shared, as in P1

EPISODES = {
    "pusht": "scripts/episodes_pusht_50.json",
    "reacher": "scripts/episodes_reacher_250.json",
    "cube": "scripts/episodes_cube_s101_100.json",
}


# ---------------------------------------------------------------- physical cost
def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def infos_val(world, key, want):
    """Physical state comes from world.infos, the documented interface — verified by
    scripts/introspect_env.py, which also confirmed each of these actually moves when
    the env steps. Reading env internals instead was the original mistake: the guess
    was wrong, and a wrong guess yields a CONSTANT cost, which makes every ranking
    metric look perfect (tau=1, regret=0) rather than looking like a bug.

    Note the infos keys use slashes ('privileged/block_0_pos') while the dataset
    columns use underscores ('privileged_block_0_pos') — they are not interchangeable.
    """
    if key not in world.infos:
        raise KeyError(f"infos has no {key!r}; available: {sorted(world.infos)}")
    v = np.asarray(world.infos[key], dtype=np.float64).ravel()
    if v.size < want or not np.isfinite(v[:want]).all():
        raise ValueError(f"infos[{key!r}] = {v[:want]} (size {v.size}, want {want} finite)")
    return v[:want]


# ---------------------------------------------------------------- model helpers
def img_tf():
    return transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet), transforms.Resize(224)])


@torch.no_grad()
def encode_frames(model, frames, device, bs=96):
    zs = []
    for i in range(0, len(frames), bs):
        out = model.encoder(frames[i:i + bs].to(device), interpolate_pos_encoding=True)
        zs.append(model.projector(out.last_hidden_state[:, 0]).float().cpu())
    return torch.cat(zs)


@torch.no_grad()
def imagined_terminal(model, z0, act_blocks, device):
    S = act_blocks.size(0)
    ctx = z0.expand(S, 1, -1).clone().to(device)
    for k in range(HORIZON):
        c = ctx[:, -3:]
        blocks = act_blocks[:, max(0, k + 1 - c.size(1)): k + 1][:, -c.size(1):]
        pred = model.predict(c, model.action_encoder(blocks))[:, -1:]
        ctx = torch.cat([ctx, pred], dim=1)
    return ctx[:, -1]


# ---------------------------------------------------------------- rank metrics
def rank_metrics(a, b, k_elite):
    """Agreement between two cost vectors over one start's candidates (lower = better).

    Value-scaled regret, (b[argmin a] - min b) / (mean b - min b), is unusable here:
    most random action sequences fling the arm far off, so mean-min is huge and every
    model's regret collapses towards 0 (Reacher gave 0.003/0.003/0.007 while tau was
    only 0.54-0.68). The denominator, not the model, set the number.

    So regret is measured in RANKS: the percentile the model's pick occupies in the
    true ordering. 0 = truly best, ~0.5 = no better than random, 1 = truly worst.
    Unit-free, immune to outlier candidates, and comparable across the (a)/(b)/total
    channels even though their reference costs differ.
    """
    n = len(a)
    tau = kendalltau(a, b).statistic
    ai, bi = np.argsort(a), np.argsort(b)
    true_rank = np.empty(n, dtype=np.float64)
    true_rank[bi] = np.arange(n)
    regret = true_rank[ai[0]] / (n - 1)                       # the argmin the planner takes
    elite_regret = true_rank[ai[:k_elite]].mean() / (n - 1)   # what CEM actually keeps
    overlap = len(set(ai[:k_elite]) & set(bi[:k_elite])) / k_elite
    # kept for continuity with the first run, but see the docstring before using it
    best, mean = b.min(), b.mean()
    val_regret = (b[ai[0]] - best) / max(mean - best, 1e-12)
    return tau, regret, overlap, elite_regret, val_regret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(ENV_PRESETS))
    ap.add_argument("models", nargs="+", help="label:ckpt_path")
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--cands", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # create the output dir FIRST: the per-model cost dumps are written inside the
    # model loop, long before the summary json, and np.savez does not create parents
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

    start_frames, goal_frames, final_frames = [], [], []
    c_phys = np.zeros((args.starts, args.cands))
    # raw physical quantities, so a different cost definition never needs a re-run
    raw_phys = np.full((args.starts, args.cands, 6), np.nan)
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

        per_cand = []
        for ci in range(args.cands):
            world.reset(seed=[ep["env_seed"]])
            _apply_callables(env, preset["callables"], env_init)
            raw = scaler.inverse_transform(
                cands[si, ci].reshape(HORIZON * ACTION_BLOCK, adim).numpy())
            reward = 0.0
            for a in raw:
                out = world.envs.step(a[None].astype(np.float32))
                if isinstance(out, tuple) and len(out) >= 3:
                    reward = float(np.asarray(out[1]).ravel()[0])
            px = np.asarray(world.infos["pixels"][0])
            per_cand.append(tf(px[-1] if px.ndim > 3 else px))
            if task == "cube":
                # success = ||block - target|| <= 0.04 m
                pos = infos_val(world, "privileged/block_0_pos", 3)
                tgt = np.asarray(goal_state["goal_privileged_block_0_pos"][0]).ravel()[:3]
                c_phys[si, ci] = np.linalg.norm(pos - tgt)
                raw_phys[si, ci, :3] = pos; raw_phys[si, ci, 3:6] = tgt
                accessor_note = "infos['privileged/block_0_pos']"
            elif task == "reacher":
                # success = per-joint |qpos - target_qpos| < threshold, velocity-free
                qp = infos_val(world, "qpos", 2)
                tgt = np.asarray(goal_state["goal_qpos"][0]).ravel()[:2]
                c_phys[si, ci] = np.abs(wrap(qp - tgt)).sum()
                raw_phys[si, ci, :2] = qp; raw_phys[si, ci, 3:5] = tgt
                accessor_note = "infos['qpos'] vs dataset goal_qpos"
            else:  # pusht
                # infos['reward'] stays nan even after stepping (verified by
                # introspect_env.py), and infos['goal_state'] is clobbered by the env
                # every step — which is why budget_sweep re-injects the goal snapshot.
                # So: block pose from infos, target block pose from the DATASET.
                # Success here is coverage of the target T, not a pose distance; this
                # normalised pose distance is a proxy, so the raw components are saved
                # to the npz and alternative weightings stay recomputable.
                bp = infos_val(world, "block_pose", 3)
                gt = np.asarray(goal_state["goal_state"][0]).ravel()
                raw_phys[si, ci, :3] = bp
                raw_phys[si, ci, 3:6] = gt[2:5]
                c_phys[si, ci] = (np.linalg.norm(bp[:2] - gt[2:4]) / 512.0
                                  + abs(wrap(bp[2] - gt[4])) / np.pi)
                accessor_note = "infos['block_pose'] vs dataset goal_state[2:5]"
        final_frames.append(torch.stack(per_cand))
        print(f"[{task}] env rollouts: start {si+1}/{args.starts} "
              f"c_phys range {c_phys[si].min():.4f}..{c_phys[si].max():.4f}", flush=True)
    world.close()
    if accessor_note:
        print(f"[{task}] physical-cost source: {accessor_note}", flush=True)

    # a start whose candidates are all equally (un)successful carries no ranking
    # information and would inject 0/0 into regret
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
        z_start = encode_frames(model, starts, device)
        z_goal = encode_frames(model, goals, device)
        z_final = encode_frames(model, finals.reshape(-1, *finals.shape[2:]), device)
        z_final = z_final.reshape(args.starts, args.cands, -1)

        flat = z_final.reshape(-1, z_final.size(-1))
        ii = torch.randint(0, flat.size(0), (20000,)); jj = torch.randint(0, flat.size(0), (20000,))
        keep = ii != jj
        scale = (flat[ii[keep]] - flat[jj[keep]]).pow(2).sum(-1).mean().item()

        acc = {f"{p}_{m}": [] for p in "abt"
               for m in ("tau", "reg", "ovl", "ereg", "vreg")}
        acc["roll"] = []
        cost_dump = {"c_imag": {}, "c_enc": {}}
        for si in range(args.starts):
            if not live[si]:
                continue
            z_hat = imagined_terminal(model, z_start[si:si+1].to(device),
                                      cands[si].to(device), device).cpu()
            c_imag = (z_hat - z_goal[si]).pow(2).sum(-1).numpy() / scale
            c_enc = (z_final[si] - z_goal[si]).pow(2).sum(-1).numpy() / scale
            cp = c_phys[si]
            cost_dump["c_imag"][si] = c_imag; cost_dump["c_enc"][si] = c_enc
            for pre, (x, y) in [("a", (c_imag, c_enc)), ("b", (c_enc, cp)), ("t", (c_imag, cp))]:
                tau, reg, ovl, ereg, vreg = rank_metrics(x, y, k_elite)
                acc[f"{pre}_tau"].append(tau); acc[f"{pre}_reg"].append(reg)
                acc[f"{pre}_ovl"].append(ovl); acc[f"{pre}_ereg"].append(ereg)
                acc[f"{pre}_vreg"].append(vreg)
            acc["roll"].append(float(
                (z_hat - z_final[si]).pow(2).sum(-1).mean().item() / scale))
        m = {k: float(np.mean(v)) for k, v in acc.items() if v}
        # keep the per-start vectors: point estimates over 20 starts say nothing about
        # whether a +0.08 tau shift is larger than the start-to-start spread
        m["_per_start"] = {k: list(map(float, v)) for k, v in acc.items() if v}
        m["label"] = label
        # every metric stays recomputable from these without touching a GPU again
        np.savez(f"eval_results/p4_costs_{task}_{label}.npz",
                 c_imag=np.stack([cost_dump["c_imag"][k] for k in sorted(cost_dump["c_imag"])]),
                 c_enc=np.stack([cost_dump["c_enc"][k] for k in sorted(cost_dump["c_enc"])]),
                 starts=np.array(sorted(cost_dump["c_imag"])))
        rows.append(m)
        print(f"  {label:14s} done", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*104}\nP4 {task}  ({int(live.sum())} starts x {args.cands} candidates, "
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
        b = rows[0]
        print(f"\n相对 {b['label']} 的变化 (regret 越低越好):")
        for m in rows[1:]:
            print(f"  {m['label']:8s} rollerr {100*(m['roll']/b['roll']-1):+6.1f}%"
                  f"  | (a) tau {m['a_tau']-b['a_tau']:+.3f} reg {m['a_reg']-b['a_reg']:+.3f}"
                  f"  | (b) tau {m['b_tau']-b['b_tau']:+.3f} reg {m['b_reg']-b['b_reg']:+.3f}"
                  f"  | tot tau {m['t_tau']-b['t_tau']:+.3f} reg {m['t_reg']-b['t_reg']:+.3f}")

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

    out = args.out or f"eval_results/p4_{task}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"task": task, "starts": int(live.sum()), "cands": args.cands,
         "k_elite": k_elite, "rows": rows}, indent=2))
    np.savez(f"eval_results/p4_cache_{task}.npz", c_phys=c_phys, live=live,
             raw_phys=raw_phys)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
