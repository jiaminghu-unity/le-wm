"""Read the LIVE simulator state, not world.infos.

The previous probe compared world.infos["qpos"] before and after _apply_callables and
got a bit-identical result, which I read as "the callables do nothing". That conclusion
was wrong: infos is a snapshot refreshed on reset/step, so a direct mutation of the sim
does not show up in it. The tell was TEST 5 — it reported reacher as equally "broken",
yet reacher reproduces the published table episode-for-episode (250/250), which is
impossible if every episode started from the same reset state.

So compare against mujoco's own MjData, which cube exposes as env._data, and check the
pixels separately. The montage evidence stands on its own: the rendered arm pose does
not match the dataset's, and the block sits on the table (z=0.02) where the dataset has
it held aloft (z=0.11-0.27).

    usage: diag_cube_livestate.py [n]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_render_fidelity import TASKS  # noqa: E402
from scripts.check_render_fidelity import _apply_callables, _extract_init_goal, swm  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
spec = TASKS["cube"]

ds = swm.data.HDF5Dataset(spec["dataset"], keys_to_cache=["action"],
                          cache_dir=Path(swm.data.utils.get_cache_dir()),
                          keys_to_load=spec["keys_to_load"])
world = swm.World(env_name=spec["env"], num_envs=1, image_shape=(224, 224),
                  max_episode_steps=100, **spec["env_kwargs"])
env = world.envs.envs[0].unwrapped


def live_qpos():
    return np.asarray(env._data.qpos, dtype=np.float64).ravel().copy()


def infos_qpos():
    return np.asarray(world.infos["qpos"], dtype=np.float64).ravel().copy()


def px():
    return np.asarray(world.infos["pixels"][0, 0]).astype(np.int32)


eps = json.loads(Path(spec["episodes"]).read_text())["episodes"][:N]
print(f"{'ep':4s}{'|live−ds|':>11s}{'|live−reset|':>13s}{'|infos−live|':>13s}"
      f"{'px MAE':>9s}{'px changed':>12s}   verdict")
for i, e in enumerate(eps):
    init, goal, _ = _extract_init_goal(ds, [e["traj_id"]], [e["start_idx"]], 25)
    ds_qpos = np.asarray(init["qpos"][0], dtype=np.float64).ravel()
    stored = np.asarray(init["pixels"][0]).astype(np.int32)

    world.reset(seed=[e["env_seed"]])
    q_reset, px_reset = live_qpos(), px()
    _apply_callables(env, spec["callables"],
                     {k: v[0] for k, v in {**init, **goal}.items() if hasattr(v, "__len__")})
    q_live, px_after = live_qpos(), px()
    n = min(len(q_live), len(ds_qpos))

    d_ds = np.abs(q_live[:n] - ds_qpos[:n]).max()
    d_reset = np.abs(q_live - q_reset).max()
    d_infos = np.abs(infos_qpos()[:len(q_live)] - q_live).max()
    mae = float(np.abs(px_after - stored).mean())
    px_ch = float(np.abs(px_after - px_reset).mean())
    verdict = ("state SET, pixels match" if d_ds < 1e-3 and mae < 3
               else "state SET but pixels differ" if d_ds < 1e-3
               else "state NOT set" if d_reset < 1e-6
               else "state set to something else")
    print(f"{i:<4d}{d_ds:11.5f}{d_reset:13.5f}{d_infos:13.5f}{mae:9.2f}{px_ch:12.2f}   {verdict}")

print("\n解读：")
print("  |live−ds|     0 => set_state 生效且等于数据集帧")
print("  |live−reset|  0 => callables 完全没动仿真状态")
print("  |infos−live|  >0 => infos 是过期快照（上一版诊断的错因）")
print("  px changed    >0 => 渲染确实跟着状态变了")

# where does the pixel error live? split the frame into the arm region vs the rest
init, goal, _ = _extract_init_goal(ds, [eps[0]["traj_id"]], [eps[0]["start_idx"]], 25)
world.reset(seed=[eps[0]["env_seed"]])
_apply_callables(env, spec["callables"],
                 {k: v[0] for k, v in {**init, **goal}.items() if hasattr(v, "__len__")})
r, s = px(), np.asarray(init["pixels"][0]).astype(np.int32)
err = np.abs(r - s).mean(axis=2)
h, w = err.shape
print(f"\nep0 误差的空间分布（{h}x{w}，按四分之一块）:")
for a in range(2):
    row = "  "
    for b in range(2):
        blk = err[a * h // 2:(a + 1) * h // 2, b * w // 2:(b + 1) * w // 2]
        row += f"{blk.mean():8.2f}"
    print(row)
print(f"  全帧 {err.mean():.2f}   误差>20 的像素占比 {100 * (err > 20).mean():.1f}%")
world.close()
