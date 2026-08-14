#!/usr/bin/env bash
# P4 bottleneck probe (rollerr / (a) rollout tau / (b) geometry tau) for the navigation
# tasks, four arms: baseline, L_obj, aux, DINO-WM.
#   usage: ray_p4_nav.sh
#
# One job, both tasks sequentially, ONE GPU -- runs on the card reserved for the user,
# at the user's request. Provisioning is the union of the two eval stacks: the standard
# py3.10 venv plus the mujoco_py chain and vendored env PointMaze needs (both proven by
# the eval sweeps). Reads ckpts_{tworoom,pointmaze}/ read-only; writes only
# eval/p4nav_* artifacts.
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
DS="$STABLEWM_HOME/datasets"
mkdir -p "$DS" "$STABLEWM_HOME/checkpoints" "$SSD/eps"
echo "[env] p4_nav on $(hostname), free=$(df -h --output=avail "$SSD"|tail -1|tr -d ' ')"

sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd curl patchelf \
  libosmesa6-dev libglew-dev libgl1-mesa-dev libglfw3 \
  libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1

if [ ! -d "$HOME/.mujoco/mujoco210" ]; then
  mkdir -p "$HOME/.mujoco"
  curl -fsSL https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz | tar -xz -C "$HOME/.mujoco"
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets scikit-learn scipy
uv pip install -q 'cython==0.29.37' 'gym==0.23.1' 'glfw==2.7.0'
uv pip install -q --no-build-isolation 'mujoco-py==2.1.2.14' 2>&1 | tail -2
python -c "import mujoco_py; print('[mujoco_py] OK:', mujoco_py.__version__)"

export PMENV_DIR="$SSD/pmenv"
mkdir -p "$PMENV_DIR/env/pointmaze" "$PMENV_DIR/d4rl"
for f in __init__.py maze_model.py point_maze_wrapper.py dynamic_mjc.py; do
  [ -f "$PMENV_DIR/env/pointmaze/$f" ] || \
    curl -fsSL "$TS_REPO/env/pointmaze/$f" -o "$PMENV_DIR/env/pointmaze/$f"
done
touch "$PMENV_DIR/env/__init__.py" "$PMENV_DIR/d4rl/__init__.py"
cat > "$PMENV_DIR/utils.py" <<'PY'
def aggregate_dct(dcts):
    import numpy as np
    if not dcts:
        return {}
    if not isinstance(dcts[0], dict):
        return np.stack(dcts)
    return {k: np.stack([d[k] for d in dcts]) for k in dcts[0]}
PY
cat > "$PMENV_DIR/d4rl/offline_env.py" <<'PY'
class OfflineEnv:
    """Stand-in: MazeEnv only inherits this and forwards kwargs; get_dataset is never called."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
PY

for t in tworoom pointmaze; do
  [ -f "$DS/$t.h5" ] || { echo "[data] fetching $t.h5"; time gcloud storage cp "$BUCKET/datasets/$t.h5" "$DS/"; }
  gcloud storage cp "$BUCKET/eval_sets/episodes_${t}_s101_100.json" "$SSD/eps/"
done

for spec in \
  "ckpts_tworoom lewm_t1_tworoom_s3072" \
  "ckpts_tworoom lewm_t2_tworoom_obj0.1_s3072" \
  "ckpts_tworoom lewm_t5_tworoom_qhead0.1_s3072" \
  "ckpts_tworoom dinowm_tworoom_s3072" \
  "ckpts_pointmaze lewm_p1_pointmaze_s3072" \
  "ckpts_pointmaze lewm_p2_pointmaze_s3072" \
  "ckpts_pointmaze lewm_p5_pointmaze_s3072" \
  "ckpts_pointmaze dinowm_pointmaze_s3072" ; do
  set -- $spec
  mkdir -p "$STABLEWM_HOME/checkpoints/$2"
  gcloud storage cp "$BUCKET/$1/$2/weights_epoch_10.pt" "$STABLEWM_HOME/checkpoints/$2/"
  gcloud storage cp "$BUCKET/$1/$2/config.json" "$STABLEWM_HOME/checkpoints/$2/" || true
done

RC=0
run_task() {
  local task=$1; shift
  local log="$SSD/p4nav_$task.log"
  set +e
  python scripts/p4_bottleneck_nav.py "$task" "$@" \
    --episodes "$SSD/eps/episodes_${task}_s101_100.json" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  [ "$rc" -ne 0 ] && RC=$rc
  gcloud storage cp "$log" "$BUCKET/eval/" || true
  for f in eval_results/p4nav_${task}.json eval_results/p4nav_cache_${task}.npz \
           eval_results/p4nav_costs_${task}_*.npz; do
    [ -f "$f" ] && gcloud storage cp "$f" "$BUCKET/eval/" || true
  done
}

run_task tworoom \
  base:lewm_t1_tworoom_s3072/weights_epoch_10.pt \
  obj:lewm_t2_tworoom_obj0.1_s3072/weights_epoch_10.pt \
  aux:lewm_t5_tworoom_qhead0.1_s3072/weights_epoch_10.pt \
  dw:dinowm_tworoom_s3072/weights_epoch_10.pt

run_task pointmaze \
  base:lewm_p1_pointmaze_s3072/weights_epoch_10.pt \
  obj:lewm_p2_pointmaze_s3072/weights_epoch_10.pt \
  aux:lewm_p5_pointmaze_s3072/weights_epoch_10.pt \
  dw:dinowm_pointmaze_s3072/weights_epoch_10.pt

echo "[done] rc=$RC"
exit $RC
