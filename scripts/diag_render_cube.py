"""Find out why cube's render fidelity is MAE 9.04 while reacher's is 2.34.

Reacher's failure was the missing NVIDIA GL driver; cube's is not — the driver is
installed and EGL is selected, yet the error is twice as large and swings from 1.27 to
15.52 frame to frame. A constant rasteriser difference does not look like that. Two
things in the scene are set by our protocol rather than by the dataset, and either would
paint a localised, frame-dependent difference:

  * set_target_pos draws the goal cube where OUR protocol put it (the block's position
    at start+25), which need not be where the dataset's own target sat when the frame
    was captured.
  * visualize_info toggles the overlay entirely.

So sweep the variants and correlate the error with the block-to-target distance. If the
target marker is the culprit, MAE should collapse when target and cube coincide — which
is exactly what an already-solved episode looks like, and would explain the 1.27 frame.

Writes a montage (dataset | rendered | 10x abs-diff) so the difference is visible
rather than inferred.

    usage: diag_render_cube.py [n_frames]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_render_fidelity import TASKS, measure  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
spec = TASKS["cube"]
SET_STATE = [c for c in spec["callables"] if c["method"] == "set_state"]

VARIANTS = [
    ("both callables, visualize_info=False", None, {"visualize_info": False}),
    ("state only, NO set_target_pos", SET_STATE, {"visualize_info": False}),
    ("both callables, visualize_info=True", None, {"visualize_info": True}),
]

print(f"{'variant':42s}{'MAE':>8s}   per-frame")
results = {}
for name, cbs, kw in VARIANTS:
    maes, missing, extra, pairs = measure("cube", N, callables=cbs,
                                         env_kwargs_override=kw, want_pairs=True)
    results[name] = (maes, extra, pairs)
    if missing:
        print(f"  !! callables missing on env: {missing}")
    print(f"{name:42s}{np.mean(maes):8.3f}   {[round(m, 1) for m in maes]}")

maes, extra, pairs = results[VARIANTS[0][0]]
dists = []
for init, goal in extra:
    b = np.asarray(init["privileged_block_0_pos"][0]).ravel()[:3]
    t = np.asarray(goal["goal_privileged_block_0_pos"][0]).ravel()[:3]
    dists.append(float(np.linalg.norm(b - t)))

print("\n每帧 MAE vs |block − target|（若目标标记是元凶，两者应强正相关）")
print(f"{'ep':<4s}{'MAE':>8s}{'dist_m':>10s}")
for i, (m, d) in enumerate(zip(maes, dists)):
    print(f"{i:<4d}{m:8.2f}{d:10.4f}")
if len(maes) > 2 and np.std(dists) > 0:
    print(f"\nPearson r(MAE, dist) = {np.corrcoef(maes, dists)[0, 1]:+.3f}")
    solved = [m for m, d in zip(maes, dists) if d <= 0.04]
    other = [m for m, d in zip(maes, dists) if d > 0.04]
    if solved and other:
        print(f"  目标已达标 (dist<=0.04 m): MAE {np.mean(solved):.2f}  (n={len(solved)})")
        print(f"  其余帧:                    MAE {np.mean(other):.2f}  (n={len(other)})")

try:
    from PIL import Image
    rows = []
    for st, r in pairs[:6]:
        d = np.clip(np.abs(st.astype(np.int32) - r.astype(np.int32)) * 10, 0, 255).astype(np.uint8)
        rows.append(np.concatenate([st, r, d], axis=1))
    Path("eval_results").mkdir(exist_ok=True)
    out = "eval_results/diag_render_cube.png"
    Image.fromarray(np.concatenate(rows, axis=0)).save(out)
    print(f"\nwrote {out}  (每行: 数据集帧 | 渲染帧 | 10x 差值)")
except Exception as exc:
    print(f"montage skipped: {exc}")
