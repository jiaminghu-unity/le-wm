#!/usr/bin/env bash
# Install the DINO-WM maze environment and run the scene-reconstruction gate.
#   usage: ray_gate_pointmaze.sh
#
# THE POINT. Evaluating pointmaze — under our protocol or DINO-WM's own — requires
# stepping their environment: success is decided by the env after executing planned
# actions, so there is no simulator-free evaluation. That environment is the hard part:
# maze_model.py is mujoco_py + d4rl + old gym, a notoriously brittle install. This job
# does the install FROM THEIR OWN PINS (dino_wm environment.yaml: python 3.9.19,
# gym 0.23.1, mujoco-py 2.1.2.14, d4rl 1.1, cython 0.29.37, numpy 1.26.4) and then
# answers the only question that matters before any evaluation is built:
#
#   does set_init_state(state) + render reproduce the dataset's own pixels?
#
# Measured as MAE per pixel on the same scale as the other gates (two-room 0.000,
# reacher 0.0001, cube 0.175, pusht 0.474; threshold 3.0). If this fails, no pointmaze
# number would be trustworthy and nothing further is built.
#
# The env files are vendored from temporal-straightening's repo (their adaptation of
# DINO-WM's, with prepare()/set_init_state()), pinned to a commit. Only four files plus
# a one-function shim for `from utils import aggregate_dct`.
#
# Nothing existing is touched; this job writes only a log and a verdict file to eval/.
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
LOG="$SSD/gate_pointmaze.log"; : > "$LOG"
echo "[env] $(hostname), free=$(df -h --output=avail "$SSD" | tail -1 | tr -d ' ')" | tee -a "$LOG"

# ---- system deps for mujoco_py: compile chain + headless GL ----
sudo apt-get update -q
sudo apt-get install -y -q build-essential patchelf libosmesa6-dev libglew-dev \
  libgl1-mesa-dev libglfw3 zstd curl

# ---- mujoco210 binaries, exactly where mujoco_py looks ----
if [ ! -d "$HOME/.mujoco/mujoco210" ]; then
  echo "[mujoco] fetching mujoco210" | tee -a "$LOG"
  mkdir -p "$HOME/.mujoco"
  curl -fsSL https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz | tar -xz -C "$HOME/.mujoco"
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

# ---- python 3.9 venv with DINO-WM's pins (separate from the training venv) ----
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
VENV="$SSD/.venv_dwm"
if [ ! -x "$VENV/bin/python" ]; then uv venv --python=3.9 "$VENV"; fi
source "$VENV/bin/activate"
# cython/numpy first: mujoco-py compiles against them at install time
uv pip install -q 'cython==0.29.37' 'numpy==1.26.4'
uv pip install -q 'gym==0.23.1' 'glfw==2.7.0' h5py imageio
echo "[pip] installing mujoco-py 2.1.2.14 (compiles)" | tee -a "$LOG"
uv pip install --no-build-isolation 'mujoco-py==2.1.2.14' 2>&1 | tail -15 | tee -a "$LOG"
# d4rl is NOT installed. maze_model.py uses it only as `offline_env.OfflineEnv`, a mixin
# whose __init__ stores dataset-download kwargs we never touch (our data is our own h5).
# Installing d4rl==1.1 drags in dm-control -> mujoco 3.5.0, which has no python-3.9 wheel
# and fails to build -- that is what killed the first attempt. A four-line shim replaces it.

# mujoco_py compiles its extension on first import; do it now, loudly
python - <<'PY' 2>&1 | tee -a "$LOG"
import mujoco_py  # noqa: F401  (first import triggers the cython build)
print("[mujoco_py] import + build OK:", mujoco_py.__version__)
PY
grep -q "\[mujoco_py\] import + build OK" "$LOG" || { echo "FATAL: mujoco_py build failed" >&2
  gcloud storage cp "$LOG" "$BUCKET/eval/gate_pointmaze.log" || true; exit 1; }

# ---- vendor the env files (pinned upstream source, not rewritten) ----
V="$SSD/pmenv"; mkdir -p "$V/env/pointmaze"
for f in __init__.py maze_model.py point_maze_wrapper.py dynamic_mjc.py; do
  curl -fsSL "$TS_REPO/env/pointmaze/$f" -o "$V/env/pointmaze/$f"
done
# the wrapper does `from utils import aggregate_dct`; shim just that function
cat > "$V/utils.py" <<'PY'
def aggregate_dct(dcts):
    """Stack a list of per-step dicts (or arrays) into arrays, as the upstream helper does."""
    import numpy as np
    if not dcts:
        return {}
    if not isinstance(dcts[0], dict):
        return np.stack(dcts)
    return {k: np.stack([d[k] for d in dcts]) for k in dcts[0]}
PY
touch "$V/env/__init__.py"
mkdir -p "$V/d4rl"
cat > "$V/d4rl/__init__.py" <<'PY'
PY
cat > "$V/d4rl/offline_env.py" <<'PY'
class OfflineEnv:
    """Stand-in for d4rl.offline_env.OfflineEnv. The real class manages dataset download
    URLs and normalisation scores; MazeEnv only inherits it and forwards constructor
    kwargs, and this evaluation never calls get_dataset -- the data is our own h5."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
PY

# ---- the dataset frames to check against ----
H5="$DS/pointmaze.h5"
[ -f "$H5" ] || { echo "[data] fetching pointmaze.h5 (30 GB)"; time gcloud storage cp "$BUCKET/datasets/pointmaze.h5" "$H5"; }

# ---- the gate ----
python - "$V" "$H5" <<'PY' 2>&1 | tee -a "$LOG"
import sys

sys.path.insert(0, sys.argv[1])

import h5py
import numpy as np

from env.pointmaze.point_maze_wrapper import PointMazeWrapper
from env.pointmaze.maze_model import U_MAZE

# constructor args mirror temporal-straightening's conf/env/point_maze.yaml (empty
# args/kwargs -> the wrapper's defaults, which are the U-maze this dataset came from)
env = PointMazeWrapper(maze_spec=U_MAZE)

rng = np.random.default_rng(0)
with h5py.File(sys.argv[2], "r") as f:
    n = f["state"].shape[0]
    idx = np.sort(rng.choice(n, 12, replace=False))
    states = f["state"][:][idx]
    ref = f["pixels"][:][idx].astype(np.int32) if False else None
    # per-frame reads: the file chunks one frame per chunk, full-column read is 30 GB
    ref = np.stack([f["pixels"][i] for i in idx]).astype(np.int32)

maes = []
for st, r in zip(states, ref):
    obs, state = env.prepare(seed=0, init_state=st.astype(np.float64))
    img = obs["visual"] if isinstance(obs, dict) and "visual" in obs else obs
    img = np.asarray(img).astype(np.int32)
    if img.shape != r.shape:
        img = img.reshape(r.shape)
    maes.append(float(np.abs(img - r).mean()))
    back = np.asarray(state, dtype=np.float64).reshape(-1)[:2]
    want = st[:2].astype(np.float64)
    print(f"  state restore |diff|={np.abs(back - want).max():.6f}  frame MAE={maes[-1]:.3f}")

mae = float(np.mean(maes))
print(f"[pointmaze] frame MAE vs dataset: mean {mae:.3f}  max {max(maes):.3f}  (n={len(maes)}; "
      f"reference: tworoom 0.000, reacher 0.0001, cube 0.175, pusht 0.474)")
if mae > 3.0:
    print("GATE FAIL: the reconstructed scene does not match the dataset frames",
          file=sys.stderr)
    sys.exit(1)
print("POINTMAZE GATE OK")
PY
rc=$?
gcloud storage cp "$LOG" "$BUCKET/eval/gate_pointmaze.log" || true
exit $rc
