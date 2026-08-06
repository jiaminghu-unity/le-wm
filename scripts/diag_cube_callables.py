"""Why do the cube callables do nothing, and does it also break budget_sweep?

diag_cube_setstate.py established that after _apply_callables the env's qpos is bit
identical to its post-reset qpos on every episode, even though hasattr(env, 'set_state')
is True. Two very different causes, with very different blast radius:

  A. env.set_state is a no-op / needs something extra   -> reacher would break too
  B. _apply_callables fails to dispatch for THIS spec   -> only cube breaks, and it
     breaks budget_sweep as well, since ENV_PRESETS['cube']['callables'] is the same
     dict — meaning every cube episode started from the env's reset state rather than
     the dataset frame.

So: print the resolver's source, call env.set_state directly, and replay the exact dict
budget_sweep builds (which, unlike check_render_fidelity, does NOT filter out scalars).

    usage: diag_cube_callables.py
"""

import inspect
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_render_fidelity import TASKS  # noqa: E402
from scripts.check_render_fidelity import _apply_callables, _extract_init_goal, swm  # noqa: E402

spec = TASKS["cube"]


def qpos(world):
    return np.asarray(world.infos["qpos"], dtype=np.float64).ravel().copy()


print("=" * 78)
print("_apply_callables source")
print("=" * 78)
try:
    print(inspect.getsource(_apply_callables))
except Exception as exc:
    print("unavailable:", exc)

print("=" * 78)
print("env.set_state signature")
print("=" * 78)
ds = swm.data.HDF5Dataset(spec["dataset"], keys_to_cache=["action"],
                          cache_dir=Path(swm.data.utils.get_cache_dir()),
                          keys_to_load=spec["keys_to_load"])
world = swm.World(env_name=spec["env"], num_envs=1, image_shape=(224, 224),
                  max_episode_steps=100, **spec["env_kwargs"])
env = world.envs.envs[0].unwrapped
for name in ("set_state", "set_target_pos"):
    try:
        print(f"  {name}{inspect.signature(getattr(env, name))}")
    except Exception as exc:
        print(f"  {name}: signature unavailable ({exc})")

e = json.loads(Path(spec["episodes"]).read_text())["episodes"][0]
init, goal, _ = _extract_init_goal(ds, [e["traj_id"]], [e["start_idx"]], 25)
ds_qpos = np.asarray(init["qpos"][0], dtype=np.float64).ravel()
ds_qvel = np.asarray(init["qvel"][0], dtype=np.float64).ravel()

print("\n" + "=" * 78)
print("TEST 1 — direct call: env.set_state(qpos, qvel)")
print("=" * 78)
world.reset(seed=[e["env_seed"]])
before = qpos(world)
try:
    env.set_state(ds_qpos, ds_qvel)
    after = qpos(world)
    print(f"  changed: {np.abs(after - before).max():.6f}  "
          f"matches dataset: {np.abs(after - ds_qpos[:len(after)]).max():.6f}")
except Exception as exc:
    print(f"  RAISED {type(exc).__name__}: {exc}")

print("\n" + "=" * 78)
print("TEST 2 — direct call with keywords, as the callable spec names them")
print("=" * 78)
world.reset(seed=[e["env_seed"]])
before = qpos(world)
try:
    env.set_state(qpos=ds_qpos, qvel=ds_qvel)
    after = qpos(world)
    print(f"  changed: {np.abs(after - before).max():.6f}")
except Exception as exc:
    print(f"  RAISED {type(exc).__name__}: {exc}")

print("\n" + "=" * 78)
print("TEST 3 — _apply_callables with budget_sweep's dict (NO scalar filtering)")
print("=" * 78)
world.reset(seed=[e["env_seed"]])
before = qpos(world)
merged = {**init, **goal}
env_init_bs = {k: v[0] for k, v in merged.items()}          # exactly budget_sweep
print(f"  keys passed: {sorted(env_init_bs)}")
try:
    _apply_callables(env, spec["callables"], env_init_bs)
    after = qpos(world)
    print(f"  changed: {np.abs(after - before).max():.6f}")
except Exception as exc:
    print(f"  RAISED {type(exc).__name__}: {exc}")

print("\n" + "=" * 78)
print("TEST 4 — same, but with check_render_fidelity's filtered dict")
print("=" * 78)
world.reset(seed=[e["env_seed"]])
before = qpos(world)
env_init_f = {k: v[0] for k, v in merged.items() if hasattr(v, "__len__")}
print(f"  keys passed: {sorted(env_init_f)}")
print(f"  keys dropped by the filter: {sorted(set(env_init_bs) - set(env_init_f))}")
try:
    _apply_callables(env, spec["callables"], env_init_f)
    after = qpos(world)
    print(f"  changed: {np.abs(after - before).max():.6f}")
except Exception as exc:
    print(f"  RAISED {type(exc).__name__}: {exc}")

print("\n" + "=" * 78)
print("TEST 5 — reacher, same resolver, as the working control")
print("=" * 78)
world.close()
rspec = TASKS["reacher"]
rds = swm.data.HDF5Dataset(rspec["dataset"], keys_to_cache=["action"],
                           cache_dir=Path(swm.data.utils.get_cache_dir()))
rworld = swm.World(env_name=rspec["env"], num_envs=1, image_shape=(224, 224),
                   max_episode_steps=100, **rspec["env_kwargs"])
renv = rworld.envs.envs[0].unwrapped
re_ = json.loads(Path(rspec["episodes"]).read_text())["episodes"][0]
rinit, rgoal, _ = _extract_init_goal(rds, [re_["traj_id"]], [re_["start_idx"]], 25)
rworld.reset(seed=[re_["env_seed"]])
rb = np.asarray(rworld.infos["qpos"], dtype=np.float64).ravel().copy()
_apply_callables(renv, rspec["callables"],
                 {k: v[0] for k, v in {**rinit, **rgoal}.items() if hasattr(v, "__len__")})
ra = np.asarray(rworld.infos["qpos"], dtype=np.float64).ravel()
print(f"  reacher changed: {np.abs(ra - rb).max():.6f}  "
      f"-> {'callables WORK on reacher' if np.abs(ra - rb).max() > 1e-6 else 'broken there too'}")
rworld.close()
