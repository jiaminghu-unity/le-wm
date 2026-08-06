"""Are the frames P4/P5 actually feed to the encoder correct renders?

The gate now proves the renderer is accurate at a pinned state (cube MAE 0.19). It does
NOT prove that the frame P5 reads — world.infos["pixels"] after 25 env steps — is that
same accurate render. My argument for it was indirect: one step drops the error from
8.59 to 0.67, so infos does refresh and does not lag; the residual 0.67 is real physics
because a zero action still commands a position-controlled arm. Plausible, unverified.

So replay P5's exact loop and, at the moment P5 grabs the frame, also render explicitly
and compare. Agreement means the encoder saw correct pixels and the existing rollerr/tau
stand; disagreement means they were computed on stale or degraded frames and must be
re-run.

    usage: diag_p5_frames.py <task> [n_starts] [n_cands]
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import preprocessing

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402

from scripts.check_render_fidelity import _apply_callables, _extract_init_goal, swm  # noqa: E402
from scripts.p4_bottleneck import ACTION_BLOCK, CAND_SEED, EPISODES, GOAL_OFFSET, HORIZON  # noqa: E402
from scripts.budget_sweep import ENV_PRESETS  # noqa: E402

task = sys.argv[1]
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NC = int(sys.argv[3]) if len(sys.argv) > 3 else 4
preset = ENV_PRESETS[task]

ds_kw = {"keys_to_load": preset["keys_to_load"]} if preset.get("keys_to_load") else {}
dataset = swm.data.HDF5Dataset(preset["dataset"], keys_to_cache=preset["process_cols"],
                               cache_dir=Path(swm.data.utils.get_cache_dir()), **ds_kw)
act = dataset.get_col_data("action")
scaler = preprocessing.StandardScaler(); scaler.fit(act[~np.isnan(act).any(axis=1)])
adim = act.shape[1]
world = swm.World(env_name=preset["env_name"], num_envs=1, image_shape=(224, 224),
                  max_episode_steps=10_000, **preset["env_kwargs"])
env = world.envs.envs[0].unwrapped

g = torch.Generator().manual_seed(CAND_SEED)
cands = torch.randn(NS, NC, HORIZON, ACTION_BLOCK * adim, generator=g)
episodes = json.loads(Path(EPISODES[task]).read_text())["episodes"][:NS]

print(f"[{task}] replaying P5's loop; comparing infos['pixels'] against env.render() "
      f"at the exact moment P5 reads the frame\n")
print(f"{'start':6s}{'cand':5s}{'MAE(infos vs render)':>22s}{'|infos−render| max':>20s}"
      f"{'infos vs 25步前':>17s}")
diffs = []
for si, ep in enumerate(episodes):
    init, goal, _ = _extract_init_goal(dataset, [ep["traj_id"]], [ep["start_idx"]], GOAL_OFFSET)
    env_init = {k: v[0] for k, v in {**init, **goal}.items() if hasattr(v, "__len__")}
    for ci in range(NC):
        world.reset(seed=[ep["env_seed"]])
        _apply_callables(env, preset["callables"], env_init)
        px0 = np.asarray(world.infos["pixels"][0]).astype(np.int32)
        px0 = px0[-1] if px0.ndim > 3 else px0
        raw = scaler.inverse_transform(cands[si, ci].reshape(HORIZON * ACTION_BLOCK, adim).numpy())
        for a in raw:
            world.envs.step(a[None].astype(np.float32))
        # exactly what P5 reads
        px = np.asarray(world.infos["pixels"][0]).astype(np.int32)
        px = px[-1] if px.ndim > 3 else px
        rd = np.asarray(env.render())
        if rd.size == px.size:
            rd = rd.reshape(px.shape).astype(np.int32)
            d = float(np.abs(px - rd).mean()); dm = float(np.abs(px - rd).max())
        else:
            d = dm = float("nan")
        moved = float(np.abs(px - px0).mean())
        diffs.append(d)
        print(f"{si:<6d}{ci:<5d}{d:22.4f}{dm:20.1f}{moved:17.2f}")
d = np.array(diffs)
print(f"\n均值 |infos − render| = {np.nanmean(d):.4f}   最大 {np.nanmax(d):.4f}")
print("解读: ~0 => P5 喂给编码器的帧就是正确渲染，现有 rollerr/tau 有效。")
print("      显著>0 => 帧是过期或降级的，两列必须重跑。")
world.close()
