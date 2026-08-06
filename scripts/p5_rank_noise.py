"""P5: how much does imagination noise scramble the candidate ranking?

No physical ground truth here, deliberately. The reference is the encoder's own
verdict on the real future, so everything measured is attributable to the predictor:

    c_imag = ||z_hat  - z_goal||^2     what the planner scores candidates with
    c_enc  = ||z_true - z_goal||^2     what it would have scored them with had it
                                       seen the actual outcome, same encoder

eps = c_imag - c_enc is therefore pure imagination error expressed in cost units,
and the question is how much of the ranking survives it.

Three views, because they answer different things:

  magnitude      ||z_hat - z_true||^2 / scale, and P1's decomposition of eps into
                 marginal noise sigma, common-mode share (ICC), and comparison
                 noise sqrt(2 sigma^2 (1 - rho)). A start-wide offset cancels in
                 every pairwise comparison, so only the comparison term can hurt.

  rank agreement Kendall tau over all pairs. Reported for continuity, but it
                 weights all C(N,2) pairs equally and most of those are two bad
                 candidates being compared to each other, which no planner cares
                 about.

  elite survival of the k candidates the planner keeps, how many belong in the true
                 top k, and where its picks actually sit in the true order. Swept
                 over k, since that is the whole question: a ranking can be far too
                 noisy to sort 300 candidates yet still reliably surface the best 30.

N_CAND defaults to 300 and the headline k is 30, which is exactly CEM's T1 tier
(300 candidates, 30 elites), so the numbers describe the search the planner really
runs rather than an arbitrary sample size.

Env rollouts are model-independent, so all models share them and adding models is
nearly free; the expensive part is the simulator, not the encoders.

    usage: p5_rank_noise.py <task> <label>:<ckpt> [...] [--starts N] [--cands N]
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
from scipy.stats import kendalltau, wilcoxon  # noqa: E402
from sklearn import preprocessing  # noqa: E402
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal  # noqa: E402

from scripts.budget_sweep import ENV_PRESETS  # noqa: E402
from scripts.p4_bottleneck import (  # noqa: E402
    ACTION_BLOCK, CAND_SEED, EPISODES, GOAL_OFFSET, HORIZON,
    encode_frames, imagined_terminal, img_tf,
)

K_SWEEP = (1, 3, 10, 30, 60)
K_HEAD = 30            # CEM T1 keeps 30 of 300


def elite_stats(c_model, c_ref, k):
    """Of the k candidates the planner would keep, how many belong in the true top k,
    and where do its picks sit in the true ordering (0 = best, 1 = worst)."""
    n = len(c_model)
    mi, ri = np.argsort(c_model), np.argsort(c_ref)
    true_rank = np.empty(n, dtype=np.float64)
    true_rank[ri] = np.arange(n)
    overlap = len(set(mi[:k]) & set(ri[:k])) / k
    elite_rank = true_rank[mi[:k]].mean() / (n - 1)
    argmin_rank = true_rank[mi[0]] / (n - 1)
    return overlap, elite_rank, argmin_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(ENV_PRESETS))
    ap.add_argument("models", nargs="+", help="label:ckpt")
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--cands", type=int, default=300)
    # a second candidate draw is the check that the numbers are not an artefact of
    # one particular set of 300 random action sequences
    ap.add_argument("--cand-seed", type=int, default=CAND_SEED)
    args = ap.parse_args()
    Path("eval_results").mkdir(parents=True, exist_ok=True)

    task, preset = args.task, ENV_PRESETS[args.task]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = img_tf()

    episodes = json.loads(Path(EPISODES[task]).read_text())["episodes"][: args.starts]
    ds_kwargs = {"keys_to_load": preset["keys_to_load"]} if preset.get("keys_to_load") else {}
    dataset = swm.data.HDF5Dataset(preset["dataset"], keys_to_cache=preset["process_cols"],
                                   cache_dir=Path(swm.data.utils.get_cache_dir()), **ds_kwargs)
    act = dataset.get_col_data("action")
    scaler = preprocessing.StandardScaler()
    scaler.fit(act[~np.isnan(act).any(axis=1)])
    adim = act.shape[1]

    world = swm.World(env_name=preset["env_name"], num_envs=1, image_shape=(224, 224),
                      max_episode_steps=10_000, **preset["env_kwargs"])
    env = world.envs.envs[0].unwrapped

    g = torch.Generator().manual_seed(args.cand_seed)
    cands = torch.randn(args.starts, args.cands, HORIZON, ACTION_BLOCK * adim, generator=g)

    start_frames, goal_frames, final_frames = [], [], []
    for si, ep in enumerate(episodes):
        init_state, goal_state, _ = _extract_init_goal(
            dataset, [ep["traj_id"]], [ep["start_idx"]], GOAL_OFFSET)
        start_frames.append(tf(np.asarray(init_state["pixels"][0]).astype(np.uint8)))
        gk = next((k for k in ("goal", "goal_pixels") if k in goal_state),
                  next((k for k in goal_state if "pixel" in k), None))
        goal_frames.append(tf(np.asarray(goal_state[gk][0]).astype(np.uint8)))
        env_init = {k: v[0] for k, v in {**init_state, **goal_state}.items()
                    if hasattr(v, "__len__")}
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
        final_frames.append(torch.stack(per_cand))
        print(f"[{task}] env rollouts: start {si+1}/{args.starts}", flush=True)
    world.close()

    starts = torch.stack(start_frames)
    goals = torch.stack(goal_frames)
    finals = torch.stack(final_frames)

    rows, per_start = [], {}
    for spec in args.models:
        label, ckpt = spec.split(":", 1)
        model = swm.wm.utils.load_pretrained(ckpt).to(device).eval()
        model.requires_grad_(False)
        z_start = encode_frames(model, starts, device)
        z_goal = encode_frames(model, goals, device)
        z_final = encode_frames(model, finals.reshape(-1, *finals.shape[2:]), device)
        z_final = z_final.reshape(args.starts, args.cands, -1)

        # every cost is divided by this model's own mean pairwise distance, so the
        # noise numbers are comparable between models with different latent scales
        flat = z_final.reshape(-1, z_final.size(-1))
        ii = torch.randint(0, flat.size(0), (20000,)); jj = torch.randint(0, flat.size(0), (20000,))
        keep = ii != jj
        scale = (flat[ii[keep]] - flat[jj[keep]]).pow(2).sum(-1).mean().item()

        eps = np.zeros((args.starts, args.cands))
        roll, taus = [], []
        ov = {k: [] for k in K_SWEEP}
        er = {k: [] for k in K_SWEEP}
        amr = []
        for si in range(args.starts):
            z_hat = imagined_terminal(model, z_start[si:si+1].to(device),
                                      cands[si].to(device), device).cpu()
            c_imag = (z_hat - z_goal[si]).pow(2).sum(-1).numpy() / scale
            c_enc = (z_final[si] - z_goal[si]).pow(2).sum(-1).numpy() / scale
            eps[si] = c_imag - c_enc
            roll.append(float((z_hat - z_final[si]).pow(2).sum(-1).mean().item() / scale))
            taus.append(kendalltau(c_imag, c_enc).statistic)
            for k in K_SWEEP:
                o, e, a = elite_stats(c_imag, c_enc, k)
                ov[k].append(o); er[k].append(e)
            amr.append(a)

        sigma = float(eps.std())
        icc = float(eps.mean(axis=1).var() / eps.var())
        cmp_noise = float(np.sqrt(2 * sigma ** 2 * (1 - icc)))
        row = {"label": label, "roll": float(np.mean(roll)), "tau": float(np.mean(taus)),
               "sigma": sigma, "icc": icc, "cmp_noise": cmp_noise,
               "argmin_rank": float(np.mean(amr))}
        for k in K_SWEEP:
            row[f"ovl{k}"] = float(np.mean(ov[k]))
            row[f"erank{k}"] = float(np.mean(er[k]))
        rows.append(row)
        per_start[label] = {"roll": roll, "tau": taus,
                            **{f"ovl{k}": ov[k] for k in K_SWEEP},
                            **{f"erank{k}": er[k] for k in K_SWEEP}}
        print(f"  {label:8s} done", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*108}\nP5 {task}  ({args.starts} starts x {args.cands} candidates)  "
          f"headline k={K_HEAD} (= CEM T1's 300/30)  cand_seed={args.cand_seed}\n{'='*108}")
    print(f"{'model':8s}{'rollerr':>9s}{'sigma':>8s}{'ICC':>7s}{'cmp_noise':>11s}"
          f"{'tau':>7s}" + "".join(f"{'ovl@'+str(k):>9s}" for k in K_SWEEP)
          + f"{'erank@30':>10s}")
    for r in rows:
        print(f"{r['label']:8s}{r['roll']:9.4f}{r['sigma']:8.4f}{r['icc']:7.3f}"
              f"{r['cmp_noise']:11.4f}{r['tau']:7.3f}"
              + "".join(f"{r['ovl'+str(k)]:9.3f}" for k in K_SWEEP)
              + f"{r['erank30']:10.3f}")

    if len(rows) > 1:
        b = rows[0]
        print(f"\n相对 {b['label']}（rollerr / cmp_noise / erank 越低越好，ovl 越高越好）:")
        for r in rows[1:]:
            print(f"  {r['label']:8s} rollerr {100*(r['roll']/b['roll']-1):+7.1f}%"
                  f"   cmp_noise {100*(r['cmp_noise']/b['cmp_noise']-1):+7.1f}%"
                  f"   ovl@{K_HEAD} {r['ovl'+str(K_HEAD)]-b['ovl'+str(K_HEAD)]:+.3f}"
                  f"   erank@{K_HEAD} {100*(r['erank'+str(K_HEAD)]/b['erank'+str(K_HEAD)]-1):+7.1f}%")
        print(f"\n配对 Wilcoxon vs {b['label']}（每个起点一个观测，n={args.starts}）:")
        for r in rows[1:]:
            out = []
            for m in ("roll", f"ovl{K_HEAD}", f"erank{K_HEAD}"):
                x, y = np.array(per_start[b["label"]][m]), np.array(per_start[r["label"]][m])
                p = wilcoxon(y, x).pvalue if np.any(y != x) else 1.0
                out.append(f"{m}: p={p:.4f}{'*' if p < 0.05 else ' '}")
            print(f"  {r['label']:8s} " + "   ".join(out))

    tag = "" if args.cand_seed == CAND_SEED else f"_cs{args.cand_seed}"
    out = f"eval_results/p5_{task}{tag}.json"
    Path(out).write_text(json.dumps(
        {"task": task, "starts": args.starts, "cands": args.cands,
         "cand_seed": args.cand_seed, "k_sweep": list(K_SWEEP),
         "rows": rows, "per_start": per_start}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
