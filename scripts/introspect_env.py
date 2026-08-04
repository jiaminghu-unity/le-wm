"""Find how each env exposes its physical state, so P4 reads the real accessor
instead of a guessed one.

A wrong guess is the dangerous failure mode, not a loud one: it returns a constant
cost, and a constant ground truth makes every ranking metric look perfect
(tau = 1, regret = 0). That would read as "the geometry channel is fine" — the exact
opposite of a bug report. Hence: enumerate, then step the env and check which
readout actually moved.

    usage: introspect_env.py <task>
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

import hdf5plugin  # noqa: F401,E402
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal  # noqa: E402

from scripts.budget_sweep import ENV_PRESETS  # noqa: E402

task = sys.argv[1]
preset = ENV_PRESETS[task]
EPS = {"pusht": "scripts/episodes_pusht_50.json",
       "reacher": "scripts/episodes_reacher_250.json",
       "cube": "scripts/episodes_cube_s101_100.json"}[task]

ds_kwargs = {"keys_to_load": preset["keys_to_load"]} if preset.get("keys_to_load") else {}
ds = swm.data.HDF5Dataset(preset["dataset"], keys_to_cache=["action"],
                          cache_dir=Path(swm.data.utils.get_cache_dir()), **ds_kwargs)
world = swm.World(env_name=preset["env_name"], num_envs=1, image_shape=(224, 224),
                  max_episode_steps=10_000, **preset["env_kwargs"])
env = world.envs.envs[0].unwrapped

ep = json.loads(Path(EPS).read_text())["episodes"][0]
init, goal, _ = _extract_init_goal(ds, [ep["traj_id"]], [ep["start_idx"]], 25)
world.reset(seed=[ep["env_seed"]])
merged = {**init, **goal}
_apply_callables(env, preset["callables"],
                 {k: v[0] for k, v in merged.items() if hasattr(v, "__len__")})

print(f"\n=== {task}: env class {type(env).__module__}.{type(env).__name__}")

print("\n--- world.infos (the documented interface; preferred if it carries state) ---")
for k, v in world.infos.items():
    a = np.asarray(v)
    extra = f"  value={a.ravel()[:6]}" if a.size <= 32 and a.dtype.kind in "fiu" else ""
    print(f"   {k:36s} shape={str(a.shape):18s} dtype={a.dtype}{extra}")

print("\n--- dir(env) public ---")
print("   " + ", ".join(a for a in dir(env) if not a.startswith("_")))
print("\n--- dir(env) private ---")
print("   " + ", ".join(a for a in dir(env) if a.startswith("_") and not a.startswith("__")))
print("\n--- vars(env) ---")
for k, v in vars(env).items():
    print(f"   {k:30s} {type(v).__module__}.{type(v).__name__}")

print("\n--- one level down (anything that looks like a simulator handle) ---")
INNER = ("physics", "data", "qpos", "sim", "model", "_data", "_physics",
         "unwrapped", "get_state", "_env", "env", "task", "_task")
for k, v in list(vars(env).items()):
    hits = [a for a in dir(v) if a in INNER]
    if not hits:
        continue
    print(f"   env.{k} ({type(v).__name__}) exposes: {hits}")
    for a in hits:
        try:
            got = getattr(v, a)
        except Exception as e:
            print(f"      .{a} raised {type(e).__name__}: {e}")
            continue
        for path, obj in [(f".{a}", got), (f".{a}.data", getattr(got, "data", None))]:
            if obj is None:
                continue
            if hasattr(obj, "qpos"):
                q = np.asarray(obj.qpos).ravel()
                print(f"      env.{k}{path}.qpos  size={q.size}  {q[:8]}")

print("\n--- step 5 times, report which numeric readout actually moved ---")
def snap():
    out = {}
    for k, v in world.infos.items():
        a = np.asarray(v)
        if a.dtype.kind in "fiu" and a.size <= 128:
            out[f"infos[{k}]"] = a.ravel().astype(np.float64).copy()
    return out

before = snap()
adim = ds.get_col_data("action").shape[1]
for _ in range(5):
    world.envs.step(np.full((1, adim), 0.3, dtype=np.float32))
after = snap()
for k in before:
    if k in after and before[k].shape == after[k].shape:
        d = float(np.abs(after[k] - before[k]).max())
        print(f"   {k:36s} max|delta| = {d:.6g}{'   <-- MOVES' if d > 1e-8 else ''}")
world.close()
print("\nINTROSPECT DONE")
