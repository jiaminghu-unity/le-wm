"""Budget-sweep evaluation: Push-T success rate vs CEM planning compute.

EVAL-ONLY. Reuses the repo's components unchanged (PushT env stack, dataset
init/goal extraction, WorldModelPolicy, CEMSolver); only the OUTER loop differs
from eval.py — episodes run one at a time so that:

  * every (config, tier) cell replays the exact episodes in episodes_pusht_50.json,
  * the CEM generator is re-seeded per (episode, tier): crc32("ep|tier") —
    identical noise across configs, different across tiers (paired design),
  * per-episode metrics are recorded (success, env steps, replans, wallclock).

Everything about planning stays at repo defaults except num_samples/n_steps/topk,
which define the budget tiers (topk scales as 10% of candidates, min 2).

Usage:
  python scripts/budget_sweep.py --config c1 lewm_c1_s3072/weights_epoch_10.pt \
      [--tiers T1 T5] [--out results.csv]
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import zlib
from collections import deque
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # dm_control envs render headless

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # jepa/module importable for load_pretrained

import numpy as np
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal

# --- pinned protocol (repo defaults — do not alter) ---
HORIZON = 5
RECEDING_HORIZON = 5
ACTION_BLOCK = 5
EVAL_BUDGET = 50
GOAL_OFFSET = 25
VAR_SCALE = 1.0

TIERS = {
    "T1": (300, 30),
    "T2": (150, 15),
    "T3": (50, 10),
    "T4": (20, 5),
    "T5": (10, 3),
}

# Gradient-descent budget ladder: (parallel restarts, gradient iterations),
# chosen so that rollout evaluations per replan (samples x steps) match the
# sampling tiers' candidates x iterations exactly: 9000 / 2250 / 500 / 100 / 30.
# Optimizer follows the repo's own adam.yaml anchor (AdamW, lr=0.1).
GD_TIERS = {
    "T1": (100, 90),
    "T2": (75, 30),
    "T3": (50, 10),
    "T4": (20, 5),
    "T5": (10, 3),
}

# per-environment wiring, each mirroring the repo's own eval config verbatim
# (config/eval/pusht.yaml and config/eval/reacher.yaml)
ENV_PRESETS = {
    "pusht": {
        "env_name": "swm/PushT-v1",
        "env_kwargs": {},
        "dataset": "pusht_expert_train",
        "process_cols": ["action", "proprio", "state"],
        "callables": [
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
    },
    "reacher": {
        "env_name": "swm/ReacherDMControl-v0",
        "env_kwargs": {"task": "qpos_match"},
        "dataset": "reacher",
        "process_cols": ["action"],
        "callables": [
            {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
            {"method": "set_target_qpos", "args": {"target_qpos": {"value": "goal_qpos"}}},
        ],
    },
}


class GDSolverPatched(swm.solver.GradientSolver):
    """Upstream 0.1.1 bug workaround: GradientSolver.init_action only moves the
    action tensor to the solver device in its zero-padding branch, so a
    full-horizon warm-start arrives on CPU and collides with CUDA noise."""

    def init_action(self, n_envs, actions=None):
        if actions is not None:
            actions = actions.to(self.device)
        return super().init_action(n_envs, actions)


class CloneActionsCostable:
    """Second 0.1.1 workaround, for gd only: JEPA.get_cost torch.split()s the
    action tensor into views; GradientSolver steps the underlying Parameter
    in-place between iterations, which autograd rejects for stale views.
    Cloning the actions at the model boundary (differentiable) severs the view."""

    def __init__(self, model):
        self._model = model

    def get_cost(self, info_dict, action_candidates):
        return self._model.get_cost(info_dict, action_candidates.clone())

    def __getattr__(self, name):
        if name == "_model":
            raise AttributeError(name)
        return getattr(self._model, name)


def elites(num_candidates):
    return max(round(0.10 * num_candidates), 2)


assert elites(300) == 30, "T1 must reproduce the repo default topk=30"


def cem_seed(episode_id, tier):
    # deterministic across processes (builtin hash() is salted — never use it here)
    return zlib.crc32(f"{episode_id}|{tier}".encode()) & 0x7FFFFFFF


def img_transform():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ]
    )


def build_process(dataset, cols):
    process = {}
    for col in cols:
        scaler = preprocessing.StandardScaler()
        data = dataset.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        scaler.fit(data)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler
    return process


def run_episode(world, policy, solver, dataset, ep, tier, plan_times, callables):
    init_state, goal_state, _ = _extract_init_goal(
        dataset, [ep["traj_id"]], [ep["start_idx"]], GOAL_OFFSET
    )

    world.reset(seed=[ep["env_seed"]])
    merged = {**init_state, **goal_state}
    env_init = {k: v[0] for k, v in merged.items()}
    _apply_callables(world.envs.envs[0].unwrapped, callables, env_init)

    # first observation comes from the dataset frame, as in the stock eval
    shape_prefix = world.infos["pixels"].shape[:2]
    for src in (init_state, goal_state):
        for k, v in src.items():
            if k in world.infos or k in goal_state:
                world.infos[k] = np.broadcast_to(
                    v[:, None, ...], shape_prefix + v.shape[1:]
                ).copy()
    goal_snapshot = {k: world.infos[k].copy() for k in goal_state}

    # fresh planner state + paired CEM seed
    policy._action_buffer = [deque(maxlen=policy.flatten_receding_horizon)]
    policy._next_init = None
    seed = cem_seed(ep["episode_id"], tier)
    solver.torch_gen.manual_seed(seed)
    plan_times.clear()

    counters = {"steps": 0, "success": False}

    def on_step(w):
        w.infos.update(deepcopy(goal_snapshot))
        if not counters["success"]:
            counters["steps"] += 1
            if bool(w.terminateds[0]):
                counters["success"] = True

    t0 = time.perf_counter()
    world._run(max_steps=EVAL_BUDGET, mode="wait", on_step=on_step)
    episode_s = time.perf_counter() - t0

    return {
        "success": int(counters["success"]),
        "env_steps_used": counters["steps"],
        "num_replans": len(plan_times),
        "wallclock_per_plan_ms": round(1e3 * np.mean(plan_times), 1) if plan_times else 0.0,
        "wallclock_episode_s": round(episode_s, 2),
        "cem_seed": seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs=2, required=True, metavar=("NAME", "CKPT"))
    ap.add_argument("--solver", default="cem", choices=["cem", "icem", "mppi", "gd"])
    ap.add_argument("--env", default="pusht", choices=list(ENV_PRESETS))
    ap.add_argument("--tiers", nargs="+", default=list(TIERS), choices=list(TIERS))
    ap.add_argument("--episodes-json", default=str(Path(__file__).parent / "episodes_pusht_50.json"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results.csv"))
    args = ap.parse_args()
    config_name, ckpt = args.config
    preset = ENV_PRESETS[args.env]

    payload = Path(args.episodes_json).read_text()
    episodes_hash = hashlib.sha256(payload.encode()).hexdigest()[:12]
    episodes = json.loads(payload)["episodes"]

    dataset = swm.data.HDF5Dataset(
        preset["dataset"],
        keys_to_cache=preset["process_cols"],
        cache_dir=Path(swm.data.utils.get_cache_dir()),
    )
    process = build_process(dataset, preset["process_cols"])
    tfm = img_transform()

    model = swm.wm.utils.load_pretrained(ckpt)
    model = model.to("cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    world = swm.World(
        env_name=preset["env_name"],
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=2 * EVAL_BUDGET,
        **preset["env_kwargs"],
    )

    out_path = Path(args.out)
    fields = [
        "config", "solver", "checkpoint_path", "tier", "candidates", "iterations", "elites",
        "episode_id", "success", "env_steps_used", "num_replans",
        "predictor_forwards_total", "wallclock_per_plan_ms", "wallclock_episode_s",
        "episodes_hash", "traj_id", "start_idx", "cem_seed",
    ]
    write_header = not out_path.exists()
    if not write_header:
        # appending to a pre-existing file: adopt its header so columns never shift
        existing = out_path.open().readline().strip().split(",")
        fields = existing
    fout = out_path.open("a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fields, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    for tier in args.tiers:
        if args.solver == "gd":
            candidates, iterations = GD_TIERS[tier]
            topk = 0  # n/a for gradient descent
            solver = GDSolverPatched(
                model=CloneActionsCostable(model), batch_size=1, num_samples=candidates,
                var_scale=VAR_SCALE, n_steps=iterations, device="cuda", seed=0,
                optimizer_cls=torch.optim.AdamW, optimizer_kwargs={"lr": 0.1},
            )
        else:
            candidates, iterations = TIERS[tier]
            topk = elites(candidates)
            cls = {"cem": swm.solver.CEMSolver, "icem": swm.solver.ICEMSolver,
                   "mppi": swm.solver.MPPISolver}[args.solver]
            solver = cls(  # icem/mppi extras (noise_beta, alpha, temperature) stay at repo defaults
                model=model, batch_size=1, num_samples=candidates,
                var_scale=VAR_SCALE, n_steps=iterations, topk=topk,
                device="cuda", seed=0,  # re-seeded per episode
            )
        plan_times = []
        orig_solve = solver.solve

        def timed_solve(info_dict, init_action=None, _orig=orig_solve, _times=plan_times):
            t0 = time.perf_counter()
            out = _orig(info_dict, init_action=init_action)
            _times.append(time.perf_counter() - t0)
            return out

        solver.solve = timed_solve

        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=swm.PlanConfig(
                horizon=HORIZON,
                receding_horizon=RECEDING_HORIZON,
                action_block=ACTION_BLOCK,
            ),
            process=process,
            transform={"pixels": tfm, "goal": tfm},
        )
        world.set_policy(policy)

        n_success = 0
        for ep in episodes:
            res = run_episode(world, policy, solver, dataset, ep, tier, plan_times, preset["callables"])
            n_success += res["success"]
            writer.writerow(
                {
                    "config": config_name,
                    "solver": args.solver,
                    "checkpoint_path": ckpt,
                    "tier": tier,
                    "candidates": candidates,
                    "iterations": iterations,
                    "elites": topk,
                    "episode_id": ep["episode_id"],
                    "episodes_hash": episodes_hash,
                    "traj_id": ep["traj_id"],
                    "start_idx": ep["start_idx"],
                    "predictor_forwards_total": candidates * iterations * HORIZON * res["num_replans"],
                    **res,
                }
            )
            fout.flush()
            print(
                f"[{config_name}|{tier}] ep{ep['episode_id']:02d} "
                f"success={res['success']} steps={res['env_steps_used']} "
                f"replans={res['num_replans']} plan={res['wallclock_per_plan_ms']}ms",
                flush=True,
            )
        print(
            f"=== {config_name} @ {tier}: SR={n_success}/{len(episodes)} "
            f"({100.0 * n_success / len(episodes):.1f}%) ===",
            flush=True,
        )

    fout.close()
    world.close()


if __name__ == "__main__":
    main()
