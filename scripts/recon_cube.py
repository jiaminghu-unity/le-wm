"""Cube dataset recon: schema, per-component q stats, and the effector-yaw
symmetry question (does psi vs psi+pi differ in pixels / in the task?).

Writes recon_cube.json + recon_cube_yaw.png next to the h5.
"""

import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  -- registers blosc; h5 is unreadable without it
import numpy as np

H5 = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

rep = {}
f = h5py.File(H5, "r", swmr=True)

# ---------------- schema ----------------
rep["schema"] = {
    k: {"shape": list(f[k].shape), "dtype": str(f[k].dtype)} for k in sorted(f.keys())
}
ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
rep["episodes"] = {
    "n_episodes": int(len(ep_len)),
    "total_frames": int(ep_len.sum()),
    "len_min": int(ep_len.min()),
    "len_max": int(ep_len.max()),
}
print(json.dumps(rep, indent=2)[:4000], flush=True)

# ---------------- q candidate columns ----------------
COLS = [
    "proprio_effector_pos", "proprio_effector_yaw", "proprio_gripper_opening",
    "proprio_gripper_contact", "proprio_joint_pos", "proprio_joint_vel",
    "privileged_block_0_pos", "privileged_block_0_yaw", "privileged_block_0_quat",
]
raw = {}
for c in COLS:
    if c not in f:
        print(f"MISSING {c}", flush=True)
        continue
    a = np.asarray(f[c][:], dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    raw[c] = a

stats = {}
for c, a in raw.items():
    stats[c] = {
        "dim": a.shape[1],
        "mean": np.round(a.mean(0), 5).tolist(),
        "std": np.round(a.std(0), 5).tolist(),
        "min": np.round(a.min(0), 5).tolist(),
        "max": np.round(a.max(0), 5).tolist(),
        "frac_nan": float(np.isnan(a).any(1).mean()),
    }
rep["raw_stats"] = stats
print("\n--- raw stats ---", flush=True)
for c, s in stats.items():
    print(f"{c:28s} dim={s['dim']} std={s['std']} min={s['min']} max={s['max']}", flush=True)

# ---------------- Q1: effector_yaw symmetry ----------------
yaw = raw["proprio_effector_yaw"][:, 0]
q1 = {
    "yaw_min": float(yaw.min()), "yaw_max": float(yaw.max()),
    "yaw_hist_32": np.histogram(yaw, bins=32, range=(-np.pi, np.pi))[0].tolist(),
}
# is the arm's joint config for psi vs psi+pi actually different?
# wrist-3 (last joint) is the yaw DOF; correlate it with effector_yaw.
jp = raw["proprio_joint_pos"]
q1["corr_joint_vs_yaw"] = [
    float(np.corrcoef(np.cos(jp[:, i]), np.cos(yaw))[0, 1]) for i in range(jp.shape[1])
]

# grasp moments: contact & closed gripper
contact = raw["proprio_gripper_contact"][:, 0]
opening = raw["proprio_gripper_opening"][:, 0]
byaw = raw["privileged_block_0_yaw"][:, 0]
grasp = (contact > 0.5) & (opening < np.quantile(opening, 0.25))
q1["n_grasp_frames"] = int(grasp.sum())
if grasp.sum() > 100:
    d = (yaw[grasp] - byaw[grasp])
    for period, name in [(2 * np.pi, "mod_2pi"), (np.pi, "mod_pi"), (np.pi / 2, "mod_halfpi")]:
        w = np.mod(d + period / 2, period) - period / 2
        q1[f"rel_yaw_{name}"] = {
            "std": float(w.std()),
            "hist_16": np.histogram(w, bins=16, range=(-period / 2, period / 2))[0].tolist(),
        }
rep["q1_yaw"] = q1
print("\n--- Q1 yaw ---\n" + json.dumps(q1, indent=2), flush=True)

# ---------------- Q4: near-constant components ----------------
bz = raw["privileged_block_0_pos"][:, 2]
rep["q4"] = {
    "block_z": {"std": float(bz.std()), "q": np.round(np.quantile(bz, [0, .01, .5, .99, 1]), 5).tolist(),
                "frac_above_3cm": float((bz > 0.03).mean())},
    "gripper_opening": {"std": float(opening.std()),
                        "q": np.round(np.quantile(opening, [0, .01, .5, .99, 1]), 5).tolist()},
    "contact_rate": float((contact > 0.5).mean()),
}
print("\n--- Q4 ---\n" + json.dumps(rep["q4"], indent=2), flush=True)

# ---------------- pixel panel: psi vs psi+pi ----------------
# pick pairs of frames matched on effector_pos and block_pos but ~pi apart in yaw
try:
    from PIL import Image

    ep = raw["proprio_effector_pos"]
    bp = raw["privileged_block_0_pos"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(yaw), size=min(200000, len(yaw)), replace=False)
    pairs = []
    for i in idx[:20000]:
        d = np.abs(np.mod(yaw[idx] - yaw[i] - np.pi, 2 * np.pi) - np.pi)
        near_pi = np.abs(d - np.pi) < 0.12
        close = (np.linalg.norm(ep[idx] - ep[i], axis=1) < 0.02) & \
                (np.linalg.norm(bp[idx] - bp[i], axis=1) < 0.02) & near_pi
        if close.any():
            pairs.append((int(i), int(idx[np.argmax(close)])))
        if len(pairs) >= 4:
            break
    rep["n_yaw_pairs_found"] = len(pairs)
    print(f"\nmatched psi/psi+pi pairs: {len(pairs)}", flush=True)

    def frame(i):
        px = f["pixels"][i]
        px = np.asarray(px)
        if px.ndim == 1:  # encoded bytes
            import io
            px = np.array(Image.open(io.BytesIO(px.tobytes())))
        return px.astype(np.uint8)

    if pairs:
        rows = [np.concatenate([frame(a), frame(b)], axis=1) for a, b in pairs]
        h = min(r.shape[0] for r in rows)
        sheet = np.concatenate([r[:h] for r in rows], axis=0)
        Image.fromarray(sheet).save(OUT / "recon_cube_yaw.png")
        rep["yaw_pairs"] = [{"i": a, "j": b, "yaw_i": float(yaw[a]), "yaw_j": float(yaw[b])}
                            for a, b in pairs]
    # also a plain yaw sweep strip
    order = np.argsort(yaw[idx[:5000]])
    sweep = [int(idx[:5000][order[int(k)]]) for k in np.linspace(0, 4999, 8)]
    strip = np.concatenate([frame(i) for i in sweep], axis=1)
    Image.fromarray(strip).save(OUT / "recon_cube_sweep.png")
    rep["sweep_yaws"] = [round(float(yaw[i]), 3) for i in sweep]
except Exception as e:  # noqa: BLE001
    rep["pixel_panel_error"] = repr(e)
    print("pixel panel failed:", e, flush=True)

(OUT / "recon_cube.json").write_text(json.dumps(rep, indent=2))
print("\nWROTE", OUT / "recon_cube.json", flush=True)
