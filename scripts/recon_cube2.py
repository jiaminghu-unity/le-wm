"""Cube recon round 2 — settles Q1 (yaw period) quantitatively.

1. Pixel discriminability of a yaw flip: among frames tightly matched on
   effector_pos / block_pos / block_yaw, does pixel distance grow with |dpsi|?
   If frames at dpsi ~ pi are no more different than dpsi ~ 0, the flip is
   invisible and q must fold yaw mod pi.
2. Grasp geometry: at lifted-cube frames, the effector-vs-block yaw residual
   folded mod 2pi / pi / pi/2 -- which period the task actually respects.
3. Variant-B hygiene: per-joint spread, flagging near-frozen joints whose
   cos/sin would be amplified to unit variance by standardization.
"""

import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

H5 = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
rep = {}

f = h5py.File(H5, "r", swmr=True)
eff_pos = np.asarray(f["proprio_effector_pos"][:], dtype=np.float64)
eff_yaw = np.asarray(f["proprio_effector_yaw"][:], dtype=np.float64)[:, 0]
blk_pos = np.asarray(f["privileged_block_0_pos"][:], dtype=np.float64)
blk_yaw = np.asarray(f["privileged_block_0_yaw"][:], dtype=np.float64)[:, 0]
grip_op = np.asarray(f["proprio_gripper_opening"][:], dtype=np.float64)[:, 0]
contact = np.asarray(f["proprio_gripper_contact"][:], dtype=np.float64)[:, 0]
jpos = np.asarray(f["proprio_joint_pos"][:], dtype=np.float64)
N = len(eff_yaw)


def wrap(x, period):
    return np.mod(x + period / 2, period) - period / 2


# ---------- 1. pixel discriminability of a yaw flip ----------
POOL = 300_000
pool = np.sort(rng.choice(N, size=POOL, replace=False))
P_eff, P_blk = eff_pos[pool], blk_pos[pool]
P_byaw, P_eyaw = blk_yaw[pool], eff_yaw[pool]

BUCKETS = [(0.0, 0.15), (0.7, 0.9), (1.4, 1.75), (2.9, 3.15)]  # |dpsi| radians
res = {f"{lo:.2f}-{hi:.2f}": [] for lo, hi in BUCKETS}
n_anchor, found = 0, {k: 0 for k in res}

for a in rng.choice(POOL, size=4000, replace=False):
    d_eff = np.linalg.norm(P_eff - P_eff[a], axis=1)
    d_blk = np.linalg.norm(P_blk - P_blk[a], axis=1)
    d_byaw = np.abs(wrap(P_byaw - P_byaw[a], np.pi / 2))  # cube is 4-fold
    ok = (d_eff < 0.008) & (d_blk < 0.008) & (d_byaw < 0.05)
    if ok.sum() < 2:
        continue
    dpsi = np.abs(wrap(P_eyaw - P_eyaw[a], 2 * np.pi))
    n_anchor += 1
    for lo, hi in BUCKETS:
        key = f"{lo:.2f}-{hi:.2f}"
        if len(res[key]) >= 120:
            continue
        cand = np.nonzero(ok & (dpsi >= lo) & (dpsi < hi))[0]
        cand = cand[cand != a]
        if len(cand) == 0:
            continue
        b = cand[rng.integers(len(cand))]
        ia, ib = int(pool[a]), int(pool[b])
        pa = np.asarray(f["pixels"][ia], dtype=np.int16)
        pb = np.asarray(f["pixels"][ib], dtype=np.int16)
        res[key].append(float(np.abs(pa - pb).mean()))
        found[key] += 1
    if all(len(v) >= 120 for v in res.values()):
        break

rep["pixel_vs_dpsi"] = {
    "n_anchors_used": n_anchor,
    "match_tol": {"eff_pos_m": 0.008, "blk_pos_m": 0.008, "blk_yaw_rad_mod_halfpi": 0.05},
    "buckets": {
        k: {"n": len(v), "pixel_mae_mean": round(float(np.mean(v)), 4),
            "pixel_mae_std": round(float(np.std(v)), 4)}
        for k, v in res.items() if v
    },
}
print("\n--- Q1a pixel MAE vs |dpsi| ---")
print(json.dumps(rep["pixel_vs_dpsi"], indent=2), flush=True)

# ---------- 2. grasp geometry ----------
lifted = (blk_pos[:, 2] > 0.05) & (contact > 0.5)
rep["grasp"] = {"n_lifted_contact": int(lifted.sum()),
                "frac": float(lifted.mean()),
                "grip_opening_when_lifted": np.round(
                    np.quantile(grip_op[lifted], [0, .25, .5, .75, 1]), 4).tolist()
                if lifted.sum() else None}
if lifted.sum() > 1000:
    d = eff_yaw[lifted] - blk_yaw[lifted]
    for period, name in [(2 * np.pi, "mod_2pi"), (np.pi, "mod_pi"), (np.pi / 2, "mod_halfpi")]:
        w = wrap(d, period)
        # circular concentration: 1 = perfectly aligned, 0 = uniform
        ang = w * (2 * np.pi / period)
        R = float(np.abs(np.exp(1j * ang).mean()))
        rep["grasp"][name] = {
            "resultant_length_R": round(R, 4),
            "std_rad": round(float(w.std()), 4),
            "hist_16": np.histogram(w, bins=16, range=(-period / 2, period / 2))[0].tolist(),
        }
print("\n--- Q1b grasp yaw residual ---")
print(json.dumps(rep["grasp"], indent=2), flush=True)

# ---------- 3. variant-B hygiene ----------
jb = {}
for i in range(jpos.shape[1]):
    t = jpos[:, i]
    c, s = np.cos(t), np.sin(t)
    jb[f"joint_{i}"] = {
        "range_rad": [round(float(t.min()), 5), round(float(t.max()), 5)],
        "span_rad": round(float(t.max() - t.min()), 5),
        "std_rad": round(float(t.std()), 6),
        "cos_std": float(f"{c.std():.3e}"),
        "sin_std": float(f"{s.std():.3e}"),
        "wraps_past_pi": bool(t.min() < -np.pi or t.max() > np.pi),
    }
rep["variant_b_joints"] = jb
print("\n--- Q3 per-joint (variant B) ---")
for k, v in jb.items():
    print(f"{k}: span={v['span_rad']:.4f} std={v['std_rad']:.6f} "
          f"cos_std={v['cos_std']:.3e} sin_std={v['sin_std']:.3e} wraps={v['wraps_past_pi']}", flush=True)

(OUT / "recon_cube2.json").write_text(json.dumps(rep, indent=2))
print("\nWROTE", OUT / "recon_cube2.json", flush=True)
