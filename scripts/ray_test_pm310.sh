#!/usr/bin/env bash
# Can mujoco_py live inside the python-3.10 evaluation venv?
#   usage: ray_test_pm310.sh
#
# The evaluation stack (budget_sweep + swm.World + CEM) lives in the standard py3.10 venv;
# the maze env was proven to work in a py3.9 venv built from DINO-WM's pins. One process
# cannot span two venvs, so either mujoco_py joins the 3.10 venv (single process, simplest)
# or a cross-process env server is needed. This job answers which, by doing the whole thing
# for real in one process:
#   import stable_worldmodel AND mujoco_py, build the maze, set_init_state from the h5,
#   render, and compare against the dataset frame -- the same gate as before, in 3.10.
#
# Known risks this probes: mujoco-py's cython extension under py3.10 (supported upstream,
# but built against whatever numpy/cython this venv carries), and old gym 0.23.1 coexisting
# with gymnasium (different packages; both importable in principle).
set -euo pipefail

BUCKET=gs://prism-training-us/le-wm
TS_REPO=https://raw.githubusercontent.com/agentic-learning-ai-lab/temporal-straightening/main

SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets"; mkdir -p "$DS"
LOG="$SSD/test_pm310.log"; : > "$LOG"

sudo apt-get update -q
sudo apt-get install -y -q build-essential patchelf libosmesa6-dev libglew-dev \
  libgl1-mesa-dev libglfw3 swig zstd curl libgl1 libglib2.0-0

if [ ! -d "$HOME/.mujoco/mujoco210" ]; then
  mkdir -p "$HOME/.mujoco"
  curl -fsSL https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz | tar -xz -C "$HOME/.mujoco"
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

# ---- the STANDARD py3.10 venv, exactly as every eval job builds it ----
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets scikit-learn
python -c "import numpy, sys; print('[venv]', sys.version.split()[0], 'numpy', numpy.__version__)" | tee -a "$LOG"

# ---- add mujoco_py + old gym INTO this venv ----
echo "[pip] mujoco-py into py3.10" | tee -a "$LOG"
uv pip install 'cython==0.29.37' 2>&1 | tail -2 | tee -a "$LOG"
uv pip install 'gym==0.23.1' 'glfw==2.7.0' 2>&1 | tail -2 | tee -a "$LOG"
uv pip install --no-build-isolation 'mujoco-py==2.1.2.14' 2>&1 | tail -6 | tee -a "$LOG"

# ---- vendored env + shims, same as the 3.9 gate ----
V="$SSD/pmenv"; mkdir -p "$V/env/pointmaze" "$V/d4rl"
for f in __init__.py maze_model.py point_maze_wrapper.py dynamic_mjc.py; do
  [ -f "$V/env/pointmaze/$f" ] || curl -fsSL "$TS_REPO/env/pointmaze/$f" -o "$V/env/pointmaze/$f"
done
touch "$V/env/__init__.py" "$V/d4rl/__init__.py"
cat > "$V/utils.py" <<'PY'
def aggregate_dct(dcts):
    import numpy as np
    if not dcts:
        return {}
    if not isinstance(dcts[0], dict):
        return np.stack(dcts)
    return {k: np.stack([d[k] for d in dcts]) for k in dcts[0]}
PY
cat > "$V/d4rl/offline_env.py" <<'PY'
class OfflineEnv:
    """Stand-in: MazeEnv only inherits this and forwards kwargs; get_dataset is never called."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
PY

H5="$DS/pointmaze.h5"
[ -f "$H5" ] || { echo "[data] fetching pointmaze.h5 (30 GB)"; time gcloud storage cp "$BUCKET/datasets/pointmaze.h5" "$H5"; }

# ---- one process, both stacks ----
python - "$V" "$H5" <<'PY' 2>&1 | tee -a "$LOG"
import sys

sys.path.insert(0, sys.argv[1])

# order matters for the test's honesty: import the eval stack FIRST, exactly as
# budget_sweep does, then mujoco_py on top of it
import hdf5plugin  # noqa: F401
import stable_worldmodel as swm  # noqa: F401
import torch

import mujoco_py

print(f"[both] stable_worldmodel + mujoco_py {mujoco_py.__version__} + torch "
      f"{torch.__version__} in one interpreter")

import h5py
import numpy as np

from env.pointmaze.maze_model import U_MAZE
from env.pointmaze.point_maze_wrapper import PointMazeWrapper

env = PointMazeWrapper(maze_spec=U_MAZE)
rng = np.random.default_rng(1)
with h5py.File(sys.argv[2], "r") as f:
    idx = np.sort(rng.choice(f["state"].shape[0], 6, replace=False))
    states = f["state"][:][idx]
    ref = np.stack([f["pixels"][i] for i in idx]).astype(np.int32)

maes = []
for st, r in zip(states, ref):
    obs, state = env.prepare(seed=0, init_state=st.astype(np.float64))
    img = np.asarray(obs["visual"] if isinstance(obs, dict) and "visual" in obs else obs)
    maes.append(float(np.abs(img.astype(np.int32) - r).mean()))
print(f"[gate-310] frame MAE mean {np.mean(maes):.3f} max {max(maes):.3f} (n={len(maes)})")
assert np.mean(maes) < 3.0

# and a CUDA forward while mujoco_py is loaded, since the real eval interleaves both
t = torch.randn(64, 192, device="cuda") @ torch.randn(192, 192, device="cuda")
print(f"[cuda] ok, norm {t.norm().item():.1f}")
print("PM310 OK")
PY
rc=$?
gcloud storage cp "$LOG" "$BUCKET/eval/test_pm310.log" || true
exit $rc
