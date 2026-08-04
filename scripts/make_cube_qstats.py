"""Compute the cube q_stats ONCE and freeze them, so all 8 runs share byte-identical
normalization (instructions §2: computed once over the whole training set, persisted).

Calls the real utils.get_q_normalizer through a thin shim so the output is exactly
what train.py would have produced on its own -- training then finds the files and
skips recomputation entirely.
"""

import json
import sys
import types
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

for _m in ("stable_pretraining", "stable_pretraining.data", "lightning",
           "lightning.pytorch", "lightning.pytorch.callbacks"):
    sys.modules.setdefault(_m, types.ModuleType(_m))
sys.modules["stable_pretraining"].data = sys.modules["stable_pretraining.data"]
sys.modules["lightning.pytorch.callbacks"].Callback = object

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import get_q_normalizer  # noqa: E402

H5 = Path(sys.argv[1])
DATASET_NAME = sys.argv[2]          # e.g. ogbench/cube_single_expert.h5
OUT_DIR = Path(sys.argv[3])         # $STABLEWM_HOME/datasets
OUT_DIR.mkdir(parents=True, exist_ok=True)


class ColShim:
    """Minimal stand-in for the swm Dataset: get_q_normalizer only calls get_col_data."""

    def __init__(self, path):
        self.f = h5py.File(path, "r", swmr=True)

    def get_col_data(self, col):
        a = np.asarray(self.f[col][:], dtype=np.float64)
        return a[:, None] if a.ndim == 1 else a


ds = ColShim(H5)
for variant in ("cube_effector", "cube_plus_joints"):
    p = OUT_DIR / f"{Path(DATASET_NAME).name}.q_stats.{variant}.json"
    if p.exists():
        p.unlink()  # always regenerate, never inherit a stale file
    get_q_normalizer(ds, p, variant)
    s = json.loads(p.read_text())
    print(f"{p.name}: dim={len(s['mean'])} n_frames={s['n_frames']} "
          f"angle_range={[round(x, 5) for x in s['angle_range']]}", flush=True)
    print(f"  mean={[round(x, 5) for x in s['mean']]}", flush=True)
    print(f"  std ={[round(x, 5) for x in s['std']]}", flush=True)
print("QSTATS OK", flush=True)
