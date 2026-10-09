#!/usr/bin/env python3
"""Stage 1 driver: episode split + tactile consequence descriptors (proposal §6, §21).

Outputs
  artifacts/usb/splits.json             fixed episode-level split, reused by every stage
  artifacts/usb/tactile_descriptors.npz r_tau (transition), r_cur (state-only), ids, stats
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import zarr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from touch_guided.tactile_descriptor import (  # noqa: E402
    fit_descriptors, make_episode_split, valid_transitions,
)

p = argparse.ArgumentParser()
p.add_argument("--data-root", default="data/demo_data/insertion_usb")
p.add_argument("--out-dir", default="artifacts/usb")
p.add_argument("--d-tau", type=int, default=32)
p.add_argument("--train-ratio", type=float, default=0.9)
p.add_argument("--seed", type=int, default=3072)
args = p.parse_args()

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)

root = zarr.open(args.data_root, mode="r")
episode_ends = np.array(root["meta/episode_ends"])
n_eps = len(episode_ends)

split_path = out / "splits.json"
if split_path.exists():
    split = json.loads(split_path.read_text())
    assert split["seed"] == args.seed and split["n_episodes"] == n_eps, "split mismatch"
else:
    train_eps, val_eps = make_episode_split(n_eps, args.train_ratio, args.seed)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=Path(__file__).parents[1]).stdout.strip()
    except Exception:
        commit = "unknown"
    split = {"seed": args.seed, "train_ratio": args.train_ratio, "n_episodes": n_eps,
             "train_episodes": train_eps, "val_episodes": val_eps, "git_commit": commit,
             "level": "episode"}
    split_path.write_text(json.dumps(split, indent=1))
print(f"[split] episodes train={len(split['train_episodes'])} val={len(split['val_episodes'])} seed={args.seed}")

rows, ep_ids, progress = valid_transitions(episode_ends)
print(f"[transitions] valid={len(rows)} (frames={episode_ends[-1]}, episodes={n_eps})")

tacff = np.array(root["data/tactile_force_field_right"])
action = np.array(root["data/action"])[rows].astype(np.float32)
r_tau, r_cur, _, fit = fit_descriptors(
    tacff, rows, ep_ids, np.array(split["train_episodes"]), args.d_tau, args.seed)

evr = np.array(fit["evr_full"])
print(f"[pca] d_tau={args.d_tau} 累计解释方差: 8维={evr[:8].sum():.3f} 16维={evr[:16].sum():.3f} 32维={evr.sum():.3f}")

dT_norm = np.linalg.norm(
    (tacff[rows + 1] - tacff[rows]).reshape(len(rows), -1), axis=1).astype(np.float32)

np.savez_compressed(
    out / "tactile_descriptors.npz",
    r_tau=r_tau, r_cur=r_cur, rows=rows, ep_ids=ep_ids,
    progress=progress.astype(np.float32), action=action, dT_norm=dT_norm,
    evr_full=np.array(fit["evr_full"], np.float32),
    evr_cur=np.array(fit["evr_cur"], np.float32))
(out / "tactile_fit_stats.json").write_text(json.dumps(
    {k: fit[k] for k in ("evr_full", "evr_cur")} |
    {"d_tau": args.d_tau, "n_transitions": int(len(rows)),
     "dT_norm_quantiles": {q: float(np.quantile(dT_norm, float(q)))
                           for q in ("0.25", "0.5", "0.75", "0.9", "0.99")}}, indent=1))
print("[saved]", out / "tactile_descriptors.npz")
