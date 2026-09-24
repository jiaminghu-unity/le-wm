"""Render-fidelity audit for the 4 multi-object OGBench tasks (never gated until
now): set_state from the lance dataset, env.render(), MAE vs the stored frame.
Mirrors check_render_fidelity.measure() with the ogbmulti presets + lance."""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import numpy as np
from PIL import Image

import hdf5plugin  # noqa: F401
import stable_worldmodel as swm
from stable_worldmodel.data.formats.lance import LanceDataset
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal
import ogbmulti_preset

N = 4
for task in ["cube_double", "cube_triple", "cube_quadruple", "scene"]:
    spec = ogbmulti_preset.PRESETS[task]
    root = Path(swm.data.utils.get_cache_dir(sub_folder="datasets"))
    ds = LanceDataset(path=str(root / spec["dataset"]),
                      keys_to_load=spec["keys_to_load"], keys_to_cache=["action"])
    world = swm.World(env_name=spec["env_name"], num_envs=1, image_shape=(224, 224),
                      max_episode_steps=100, **spec["env_kwargs"])
    env = world.envs.envs[0].unwrapped
    eps = json.loads(Path(f"/tmp/eps_{task}.json").read_text())["episodes"][:N]
    maes = []
    for e in eps:
        init, goal, _ = _extract_init_goal(ds, [e["traj_id"]], [e["start_idx"]], 25)
        world.reset(seed=[e["env_seed"]])
        merged = {**init, **goal}
        vals = {k: v[0] for k, v in merged.items() if hasattr(v, "__len__")}
        if hasattr(ogbmulti_preset, "coerce_scalars"):
            vals = ogbmulti_preset.coerce_scalars(vals)
        _apply_callables(env, spec["callables"], vals)
        raw = init["pixels"][0]
        st = (np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.int32)
              if isinstance(raw, (bytes, bytearray)) else np.asarray(raw, dtype=np.int32))
        r = np.asarray(env.render()).astype(np.int32)
        if r.shape != st.shape:
            r = np.asarray(Image.fromarray(r.astype(np.uint8)).resize(st.shape[1::-1])).astype(np.int32)
        maes.append(float(np.abs(st - r).mean()))
    print(f"FIDELITY {task}: MAE per-ep {[round(m,2) for m in maes]} mean={np.mean(maes):.2f}", flush=True)
    world.close()
