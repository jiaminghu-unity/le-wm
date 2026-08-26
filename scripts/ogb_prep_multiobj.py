"""Prepare OGBench multi-object manipulation datasets for this pipeline:
cube-double / cube-triple / cube-quadruple / scene.

OGBench's released state-version npz carries per-step qpos/qvel (their own
generator saves them), so 224x224 pixels are REPLAY-RENDERED here through the same
swm env + set_state + env.render() path our render-fidelity gate validated on
cube-single (MAE 0.19 vs stored frames). Column values come from the env's own
compute_ob_info() after each set_state, with slash-namespaced keys sanitized to
underscore column names -- the exact convention cube_single_expert.h5 uses
(proprio_effector_pos, privileged_block_0_pos, ...). Output is written with swm's
HDF5Writer (standard ep_len/ep_offset layout), then converted to lance elsewhere.

Modes:
  --smoke: 2 episodes x 8 frames; prints npz fields, ob_info keys, render shape,
           per-frame render sanity (non-constant pixels), then exits. The launcher
           gates the full run on this.
  full:    --episodes/--max-frames capped replay-render into the h5.

    usage: ogb_prep_multiobj.py cube_double --npz-dir <dir> --out <path.h5> [--smoke]
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.data.formats.hdf5 import HDF5Writer  # noqa: E402

OGB_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"

TASKS = {
    "cube_double": dict(
        npz="cube-double-play-v0", env="swm/OGBCube-v0",
        env_kwargs=dict(env_type="double", ob_type="states", multiview=False,
                        visualize_info=False, terminate_at_goal=False)),
    "cube_triple": dict(
        npz="cube-triple-play-v0", env="swm/OGBCube-v0",
        env_kwargs=dict(env_type="triple", ob_type="states", multiview=False,
                        visualize_info=False, terminate_at_goal=False)),
    "cube_quadruple": dict(
        npz="cube-quadruple-play-v0", env="swm/OGBCube-v0",
        env_kwargs=dict(env_type="quadruple", ob_type="states", multiview=False,
                        visualize_info=False, terminate_at_goal=False)),
    "scene": dict(
        npz="scene-play-v0", env="swm/OGBScene-v0",
        env_kwargs=dict(ob_type="states", multiview=False,
                        visualize_info=False, terminate_at_goal=False)),
}


def fetch_npz(name, npz_dir):
    path = Path(npz_dir) / f"{name}.npz"
    if not path.exists():
        url = f"{OGB_URL}/{name}.npz"
        print(f"[npz] downloading {url}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(path))
    print(f"[npz] {path} ({path.stat().st_size/1e9:.2f} GB)", flush=True)
    return path


def episode_bounds(terminals):
    ends = np.flatnonzero(terminals) + 1
    starts = np.concatenate([[0], ends[:-1]])
    return list(zip(starts, ends))


def sanitize(key):
    return key.replace("/", "_")


def harvest_columns(env, t, npz, want_keys=None):
    """One frame's worth of columns: render + env ob_info + raw npz fields."""
    ob = env.compute_ob_info()
    row = {}
    for k, v in ob.items():
        v = np.asarray(v)
        if v.dtype.kind in "OUS":       # strings (target_task etc.) -- skip
            continue
        row[sanitize(k)] = v.astype(np.float32).reshape(-1)
    row["pixels"] = np.asarray(env.render()).astype(np.uint8)
    row["action"] = npz["actions"][t].astype(np.float32)
    row["qpos"] = npz["qpos"][t].astype(np.float32)
    row["qvel"] = npz["qvel"][t].astype(np.float32)
    if "button_states" in npz.files:
        row["button_states"] = npz["button_states"][t].astype(np.float32)
    if want_keys is not None:
        row = {k: row[k] for k in want_keys}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("--npz-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max-frames", type=int, default=400_000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    spec = TASKS[args.task]

    npz_path = fetch_npz(spec["npz"], args.npz_dir)
    npz = np.load(npz_path)
    print(f"[npz] fields: {[(k, npz[k].shape, str(npz[k].dtype)) for k in npz.files]}", flush=True)
    for need in ("actions", "terminals", "qpos", "qvel"):
        assert need in npz.files, f"npz missing {need}; cannot replay-render"
    eps = episode_bounds(npz["terminals"])
    print(f"[npz] {len(eps)} episodes, lengths {eps[0][1]-eps[0][0]}(first)"
          f" .. total {npz['terminals'].shape[0]} frames", flush=True)

    world = swm.World(env_name=spec["env"], num_envs=1, image_shape=(224, 224),
                      max_episode_steps=10_000, **spec["env_kwargs"])
    env = world.envs.envs[0].unwrapped
    assert hasattr(env, "set_state") and hasattr(env, "compute_ob_info"), \
        f"env lacks set_state/compute_ob_info: {type(env)}"

    if args.smoke:
        world.reset(seed=[0])
        for ei, (s, e) in enumerate(eps[:2]):
            for t in range(s, min(s + 8, e)):
                env.set_state(npz["qpos"][t], npz["qvel"][t])
                row = harvest_columns(env, t, npz)
                if t == s:
                    print(f"[smoke ep{ei}] columns: "
                          f"{[(k, row[k].shape if hasattr(row[k],'shape') else type(row[k])) for k in sorted(row)]}",
                          flush=True)
                px = row["pixels"]
                assert px.shape == (224, 224, 3), px.shape
                assert px.std() > 5, f"render looks constant (std={px.std():.2f})"
        print("[smoke] OK", flush=True)
        world.close()
        return

    n_ep = min(args.episodes, len(eps))
    want_keys = None
    frames = 0
    with HDF5Writer(args.out, mode="overwrite") as w:
        world.reset(seed=[0])
        for ei, (s, e) in enumerate(eps[:n_ep]):
            if frames >= args.max_frames:
                break
            e = min(e, s + (args.max_frames - frames))
            cols = None
            for t in range(s, e):
                env.set_state(npz["qpos"][t], npz["qvel"][t])
                row = harvest_columns(env, t, npz, want_keys)
                if want_keys is None:
                    want_keys = sorted(row)
                    print(f"[schema] {want_keys}", flush=True)
                if cols is None:
                    cols = {k: [] for k in row}
                for k, v in row.items():
                    cols[k].append(v)
            w.write_episode({k: np.stack(v) for k, v in cols.items()})
            frames += e - s
            if ei % 25 == 0:
                print(f"[render] ep {ei+1}/{n_ep}  frames {frames}", flush=True)
    world.close()
    print(f"[done] wrote {args.out}: {frames} frames", flush=True)


if __name__ == "__main__":
    main()
