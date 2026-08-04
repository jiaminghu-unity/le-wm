"""Pre-flight for the cube round. Three things in one pass over the real data:

1. q builders: run utils.Q_VARIANTS['cube_effector'|'cube_plus_joints'] on real
   columns, confirm 9 / 19 dims and print per-component raw mean/std (the numbers
   that get frozen into q_stats.json).
2. IK null-space excitation: among frames matched tightly on (effector_pos, psi),
   how much does joint_pos still vary? If ~0 the IK is deterministic, the null
   space is never exercised, and K3/K5 carry no information over K2/K4.
3. The money metric: among pairs that q_effector calls (nearly) identical, what is
   the q_plus_joints distance? That difference IS the whole K2-vs-K3 contrast.
"""

import json
import sys
import types
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

# utils.py imports stable_pretraining / lightning at module level for the training
# helpers; the q builders need neither. Stub them so this stays a test of the REAL
# utils.py rather than a copy of the builders.
for _m in ("stable_pretraining", "stable_pretraining.data", "lightning",
           "lightning.pytorch", "lightning.pytorch.callbacks"):
    sys.modules.setdefault(_m, types.ModuleType(_m))
sys.modules["stable_pretraining"].data = sys.modules["stable_pretraining.data"]
sys.modules["lightning.pytorch.callbacks"].Callback = object

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import CUBE_ARM_JOINTS, Q_VARIANTS  # noqa: E402

H5 = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
rep = {}

f = h5py.File(H5, "r", swmr=True)
cols = {}
for c in ["proprio_effector_pos", "proprio_effector_yaw", "proprio_gripper_opening",
          "privileged_block_0_pos", "proprio_joint_pos"]:
    a = np.asarray(f[c][:], dtype=np.float64)
    cols[c] = a[:, None] if a.ndim == 1 else a
N = len(cols["proprio_effector_yaw"])

# ---------- 1. q builders ----------
qs = {}
for variant in ["cube_effector", "cube_plus_joints"]:
    builder, sources, (acol, aidx, lo, hi) = Q_VARIANTS[variant]
    ang = cols[acol][:, aidx]
    assert ang.min() >= lo and ang.max() <= hi, f"{variant}: angle out of range"
    q = builder(*[torch.from_numpy(cols[s]) for s in sources]).numpy()
    qs[variant] = q
    rep[variant] = {
        "dim": int(q.shape[1]),
        "sources": sources,
        "mean": np.round(q.mean(0), 6).tolist(),
        "std": np.round(q.std(0), 6).tolist(),
        "std_min": float(q.std(0).min()),
        "std_argmin": int(q.std(0).argmin()),
    }
    print(f"\n{variant}: dim={q.shape[1]}  min_component_std={q.std(0).min():.6f} "
          f"(component {int(q.std(0).argmin())})", flush=True)
    print("  std:", np.round(q.std(0), 5).tolist(), flush=True)
assert qs["cube_effector"].shape[1] == 9
assert qs["cube_plus_joints"].shape[1] == 9 + 2 * len(CUBE_ARM_JOINTS)

# standardized (what L_obj actually sees)
Z = {k: (v - v.mean(0)) / v.std(0) for k, v in qs.items()}
rep["standardized_absmax"] = {k: float(np.abs(v).max()) for k, v in Z.items()}
print("\nstandardized |q| max:", rep["standardized_absmax"], flush=True)

# ---------- 2. null-space excitation ----------
eff = cols["proprio_effector_pos"]
psi = cols["proprio_effector_yaw"][:, 0]
jp = cols["proprio_joint_pos"][:, list(CUBE_ARM_JOINTS)]

POOL = 300_000
pool = np.sort(rng.choice(N, size=POOL, replace=False))
P_eff, P_psi, P_jp = eff[pool], psi[pool], jp[pool]
P_ZA, P_ZB = Z["cube_effector"][pool], Z["cube_plus_joints"][pool]


def circ_std(a):
    return float(np.sqrt(-2 * np.log(np.abs(np.exp(1j * a).mean()) + 1e-12)))


cond_std, dA, dB, nnb = [], [], [], []
for a in rng.choice(POOL, size=3000, replace=False):
    dpsi = np.abs(np.mod(P_psi - P_psi[a] + np.pi, 2 * np.pi) - np.pi)
    m = (np.linalg.norm(P_eff - P_eff[a], axis=1) < 0.005) & (dpsi < 0.02)
    m[a] = False
    k = int(m.sum())
    if k < 10:
        continue
    nnb.append(k)
    cond_std.append([circ_std(P_jp[m][:, i] - P_jp[a, i]) for i in range(P_jp.shape[1])])
    dA.append(float(np.linalg.norm(P_ZA[m] - P_ZA[a], axis=1).mean()))
    dB.append(float(np.linalg.norm(P_ZB[m] - P_ZB[a], axis=1).mean()))
    if len(cond_std) >= 400:
        break

cond_std = np.asarray(cond_std)
marg = np.array([circ_std(jp[:, i]) for i in range(jp.shape[1])])
rep["null_space"] = {
    "n_anchors": len(cond_std),
    "median_neighbours": float(np.median(nnb)) if nnb else 0,
    "match_tol": {"eff_pos_m": 0.005, "psi_rad": 0.02},
    "joints": list(CUBE_ARM_JOINTS),
    "conditional_circstd_rad": np.round(cond_std.mean(0), 5).tolist(),
    "marginal_circstd_rad": np.round(marg, 5).tolist(),
    "ratio_cond_over_marg": np.round(cond_std.mean(0) / marg, 5).tolist(),
}
print("\n--- null-space excitation ---")
print(json.dumps(rep["null_space"], indent=2), flush=True)

rep["pair_distance"] = {
    "mean_qA_dist_among_effector_matched": round(float(np.mean(dA)), 5),
    "mean_qB_dist_among_effector_matched": round(float(np.mean(dB)), 5),
    "ratio_B_over_A": round(float(np.mean(dB) / max(np.mean(dA), 1e-9)), 4),
}
print("\n--- pair distance among effector-matched frames ---")
print(json.dumps(rep["pair_distance"], indent=2), flush=True)

(OUT / "check_cube_q.json").write_text(json.dumps(rep, indent=2))
print("\nWROTE", OUT / "check_cube_q.json", flush=True)
