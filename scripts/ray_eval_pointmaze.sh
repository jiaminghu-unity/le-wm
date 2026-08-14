#!/usr/bin/env bash
# PointMaze SR sweep: one job = (config, solver) x the given episode seeds.
#   usage: ray_eval_pointmaze.sh <cfgname> <ckpt dir under ckpts_pointmaze/> <solver> <seed>...
#
# Written fresh, not derived by string replacement (the tworoom->pointmaze derivation
# produced four silent-miss failures). Provisioning is identical to what the smoke run
# validated end to end: standard py3.10 eval venv + mujoco210 + mujoco-py 2.1.2.14 +
# vendored DINO-WM env + d4rl/utils shims, then the eval-path gate (FATAL) before any
# GPU-hours are spent. Skips any seed whose CSV already exists in GCS.
#
# Results go to final_eval_pointmaze/; ckpts_pointmaze/ is read-only.
set -euo pipefail

CFG="${1:?cfgname}"; CKPT_DIR="${2:?ckpt dir}"; SOLVER="${3:?solver}"; shift 3
SEEDS=("$@")
BUCKET=gs://prism-training-us/le-wm
OUTP="$BUCKET/final_eval_pointmaze"
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
DS="$STABLEWM_HOME/datasets"; mkdir -p "$DS" "$SSD/eps"
echo "[env] pointmaze/$CFG/$SOLVER on $(hostname), free=$(df -h --output=avail "$SSD"|tail -1|tr -d ' ')"

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
uv pip install -q hdf5plugin -U datasets scikit-learn
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

H5="$DS/pointmaze.h5"
[ -f "$H5" ] || { echo "[data] fetching pointmaze.h5 (30 GB)"; time gcloud storage cp "$BUCKET/datasets/pointmaze.h5" "$H5"; }

gcloud storage cp "$BUCKET/eval_sets/episodes_pointmaze_s101_100.json" "$SSD/eps/" 2>/dev/null || true

# ---- eval-path gate, FATAL: the verdict lands in this job's own log ----
GATELOG="$SSD/render_gate_pointmaze.log"
if ! python scripts/check_render_pointmaze.py 12 \
     --episodes "$SSD/eps/episodes_pointmaze_s101_100.json" --max-mae 3.0 2>&1 | tee "$GATELOG"; then
  echo "FATAL: eval-path gate failed — refusing to produce SR on a scene the encoder was
not trained on" | tee -a "$GATELOG"
  gcloud storage cp "$GATELOG" "$BUCKET/eval/" || true
  exit 1
fi
gcloud storage cp "$GATELOG" "$BUCKET/eval/" || true

mkdir -p "$STABLEWM_HOME/checkpoints/$CKPT_DIR"
gcloud storage cp "$BUCKET/ckpts_pointmaze/$CKPT_DIR/weights_epoch_10.pt" "$STABLEWM_HOME/checkpoints/$CKPT_DIR/"
gcloud storage cp "$BUCKET/ckpts_pointmaze/$CKPT_DIR/config.json" "$STABLEWM_HOME/checkpoints/$CKPT_DIR/" || true

RC=0
for S in "${SEEDS[@]}"; do
  EPSNAME="episodes_pointmaze_s${S}_100.json"
  OUT="final_pointmaze_${CFG}_${SOLVER}_s${S}.csv"
  if gcloud storage ls "$OUTP/$OUT" >/dev/null 2>&1; then
    echo "[skip] $OUT already in GCS"; continue
  fi
  gcloud storage cp "$BUCKET/eval_sets/$EPSNAME" "$SSD/eps/"
  echo "[run] pointmaze $CFG $SOLVER seed=$S"
  set +e
  python scripts/budget_sweep_pointmaze.py \
    --env pointmaze --solver "$SOLVER" \
    --config "$CFG" "$CKPT_DIR/weights_epoch_10.pt" \
    --tiers T1 T2 T3 T4 T5 \
    --episodes-json "$SSD/eps/$EPSNAME" \
    --out "$SSD/$OUT" 2>&1 | tail -30
  rc=${PIPESTATUS[0]}; set -e
  [ "$rc" -ne 0 ] && RC=$rc
  [ -f "$SSD/$OUT" ] && gcloud storage cp "$SSD/$OUT" "$OUTP/$OUT"
done
echo "[done] rc=$RC -> $OUTP/final_pointmaze_${CFG}_${SOLVER}_s*.csv"
exit $RC
