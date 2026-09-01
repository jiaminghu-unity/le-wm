#!/usr/bin/env bash
# MPPI temperature-matched evaluation for the LeWM family.
#   usage: ray_eval_mppi_t.sh <task> <cfgname> <ckpt_prefix> <ckpt_dir> <T-list> <seed-list>
#     task        : pusht | reacher | cube | tworoom | pointmaze
#     ckpt_prefix : GCS dir the ckpt lives under (ckpts | ckpts_tworoom | ckpts_pointmaze)
#     T-list      : comma-separated temperatures, e.g. "2,8,32,128,512"
#     seed-list   : comma-separated episode seeds, e.g. "101" or "102,103,104,105,106"
#
# Runs budget_sweep_mppi_t.py (solver=mppi, all 5 tiers) for every (T, seed);
# CSVs go to the NEW prefix final_eval_mppi_t/ as final_<task>_<cfg>_mppiT<T>_s<seed>.csv.
# Nothing existing is touched; skip-if-present per output.
set -euo pipefail

TASK="${1:?task}"; CFG="${2:?cfg}"; CKP="${3:?ckpt prefix}"; CKPT_DIR="${4:?ckpt dir}"
TS="${5:?T list}"; SEEDLIST="${6:?seed list}"
BUCKET=gs://prism-training-us/le-wm
OUTP="$BUCKET/final_eval_mppi_t"
TS_REPO=https://raw.githubusercontent.com/agentic-learning-ai-lab/temporal-straightening/main

case "$TASK" in
  pusht)     H5NAME=pusht_expert_train.h5; SRC="$BUCKET/datasets/pusht_expert_train.h5"; SUB="" ;;
  reacher)   H5NAME=reacher.h5;            SRC="$BUCKET/datasets/reacher.h5";            SUB="" ;;
  cube)      H5NAME=cube_single_expert.h5; SRC="$BUCKET/datasets/ogbench/cube_single_expert.tar.zst"; SUB="ogbench" ;;
  tworoom)   H5NAME=tworoom.h5;            SRC="$BUCKET/datasets/tworoom.h5";            SUB="" ;;
  pointmaze) H5NAME=pointmaze.h5;          SRC="$BUCKET/datasets/pointmaze.h5";          SUB="" ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
SWEEPER=scripts/budget_sweep_mppi_t.py

SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets"; mkdir -p "$DS" "$SSD/eps" "$STABLEWM_HOME/checkpoints"
echo "[env] mppi_t/$TASK/$CFG T={$TS} on $(hostname)"

sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd curl patchelf \
  libosmesa6-dev libglew-dev libgl1-mesa-dev libglfw3 \
  libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
  libegl1 libegl-mesa0 libgles2 libglvnd0 libopengl0 libosmesa6
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
sudo apt-get install -y -q "libnvidia-gl-${DRV:-580}-server" || sudo apt-get install -y -q "libnvidia-gl-${DRV:-580}" || true
if [ "$TASK" = pointmaze ] && [ ! -d "$HOME/.mujoco/mujoco210" ]; then
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
if [ "$TASK" = pointmaze ]; then
  uv pip install -q 'cython==0.29.37' 'gym==0.23.1' 'glfw==2.7.0'
  uv pip install -q --no-build-isolation 'mujoco-py==2.1.2.14' 2>&1 | tail -2
  export PMENV_DIR="$SSD/pmenv"
  mkdir -p "$PMENV_DIR/env/pointmaze" "$PMENV_DIR/d4rl"
  for f in __init__.py maze_model.py point_maze_wrapper.py dynamic_mjc.py; do
    [ -f "$PMENV_DIR/env/pointmaze/$f" ] || \
      curl -fsSL "$TS_REPO/env/pointmaze/$f" -o "$PMENV_DIR/env/pointmaze/$f"
  done
  touch "$PMENV_DIR/env/__init__.py" "$PMENV_DIR/d4rl/__init__.py"
  printf 'def aggregate_dct(dcts):\n    import numpy as np\n    if not dcts:\n        return {}\n    if not isinstance(dcts[0], dict):\n        return np.stack(dcts)\n    return {k: np.stack([d[k] for d in dcts]) for k in dcts[0]}\n' > "$PMENV_DIR/utils.py"
  printf 'class OfflineEnv:\n    def __init__(self, **kwargs):\n        for k, v in kwargs.items():\n            setattr(self, k, v)\n' > "$PMENV_DIR/d4rl/offline_env.py"
fi

if [ -n "$SUB" ]; then
  mkdir -p "$DS/$SUB"
  [ -f "$DS/$SUB/$H5NAME" ] || { gcloud storage cp "$SRC" - | zstd -dc --long=31 | tar -xf - -C "$DS/$SUB"; }
else
  [ -f "$DS/$H5NAME" ] || gcloud storage cp "$SRC" "$DS/$H5NAME"
fi

mkdir -p "$STABLEWM_HOME/checkpoints/$CKPT_DIR"
gcloud storage cp "$BUCKET/$CKP/$CKPT_DIR/weights_epoch_10.pt" "$STABLEWM_HOME/checkpoints/$CKPT_DIR/"
gcloud storage cp "$BUCKET/$CKP/$CKPT_DIR/config.json" "$STABLEWM_HOME/checkpoints/$CKPT_DIR/" || true

RC=0
IFS=',' read -ra TARR <<< "$TS"
IFS=',' read -ra SARR <<< "$SEEDLIST"
# render fidelity gate (hard abort: a soft-rendered worker biases SR ~20pp on reacher)
GATELOG="$SSD/render_gate_${TASK}.log"
if ! python scripts/check_render_fidelity.py "$TASK" 8 --max-mae 3.0 2>&1 | tee "$GATELOG"; then
  echo "[FATAL] render fidelity gate FAILED — aborting" | tee -a "$GATELOG"
  gcloud storage cp "$GATELOG" "$BUCKET/eval/" || true
  exit 41
fi

for T in "${TARR[@]}"; do
  for S in "${SARR[@]}"; do
    EPSNAME="episodes_${TASK}_s${S}_100.json"
    gcloud storage cp "$BUCKET/eval_sets/$EPSNAME" "$SSD/eps/" 2>/dev/null || true
    OUT="final_${TASK}_${CFG}_mppiT${T}_s${S}.csv"
    if gcloud storage ls "$OUTP/$OUT" >/dev/null 2>&1; then
      echo "[skip] $OUT"; continue
    fi
    echo "[run] $TASK $CFG mppi T=$T seed=$S"
    set +e
    MPPI_T="$T" python "$SWEEPER" \
      --env "$TASK" --solver mppi \
      --config "$CFG" "$CKPT_DIR/weights_epoch_10.pt" \
      --tiers T1 T2 T3 T4 T5 \
      --episodes-json "$SSD/eps/$EPSNAME" \
      --out "$SSD/$OUT" 2>&1 | tail -12
    rc=${PIPESTATUS[0]}; set -e
    [ "$rc" -ne 0 ] && RC=$rc
    [ -f "$SSD/$OUT" ] && gcloud storage cp "$SSD/$OUT" "$OUTP/$OUT"
  done
done
echo "[done] rc=$RC"
exit $RC
