#!/usr/bin/env bash
# q-input (QJEPA) eval for reacher/cube: state-key sweeper (budget_sweep_qinput_any).
#   usage: ray_eval_final.sh <task> <cfgname> <ckpt_dir> <solver> <seed> [seed...]
#     task    : pusht | reacher | cube
#     cfgname : short label written into the CSV 'config' column
#     ckpt_dir: directory name under gs://.../le-wm/ckpts/
#     solver  : cem | icem | mppi | gd
# Skips any (seed) whose result CSV is already in GCS, so partially completed
# work is reused rather than recomputed.
set -euo pipefail

TASK="${1:?task}"; CFG="${2:?cfgname}"; CKPT_DIR="${3:?ckpt dir}"; SOLVER="${4:?solver}"; shift 4
SEEDS=("$@")
BUCKET=gs://prism-training-us/le-wm
OUTP="$BUCKET/final_eval"

case "$TASK" in
  pusht)   H5NAME=pusht_expert_train.h5; SRC="$BUCKET/datasets/pusht_expert_train.h5"; SUB="" ;;
  reacher) H5NAME=reacher.h5;            SRC="$BUCKET/datasets/reacher.h5";            SUB="" ;;
  cube)    H5NAME=cube_single_expert.h5; SRC="$BUCKET/datasets/ogbench/cube_single_expert.tar.zst"; SUB="ogbench" ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac

SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets${SUB:+/$SUB}"
mkdir -p "$DS" "$STABLEWM_HOME/checkpoints/$CKPT_DIR"
echo "[env] $TASK/$CFG/$SOLVER on $(hostname), free=$(df -h --output=avail "$SSD"|tail -1|tr -d ' ')"

sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd \
  libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
  libegl1 libegl-mesa0 libgles2 libglvnd0 libopengl0 libosmesa6 libosmesa6-dev
# The DL image ships the NVIDIA COMPUTE driver only: no libEGL_nvidia, no
# 10_nvidia.json, so MUJOCO_GL=egl silently degrades to software rendering and the
# env's pixels stop matching the dataset the model was trained on (MAE 4.83 vs 2.34).
sudo apt-get install -y -q libnvidia-gl-580-server || true
sudo usermod -aG render "$(id -un)" 2>/dev/null || true
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

# ---- dataset ----
H5="$DS/$H5NAME"
if [ ! -f "$H5" ]; then
  echo "[data] fetching $TASK"
  if [ "$TASK" = cube ]; then
    time gcloud storage cat "$SRC" | zstd -dc --long=31 | tar -xf - -C "$DS"
  else
    time gcloud storage cp "$SRC" "$H5"
  fi
fi
ls -la "$H5"

# ---- checkpoint ----
gcloud storage cp "$BUCKET/ckpts/$CKPT_DIR/weights_epoch_10.pt" "$STABLEWM_HOME/checkpoints/$CKPT_DIR/"
gcloud storage cp "$BUCKET/ckpts/$CKPT_DIR/config.json" "$STABLEWM_HOME/checkpoints/$CKPT_DIR/" || true

# ---- renderer (cube/reacher need a working mujoco GL backend) ----
GL=egl
for g in egl osmesa; do
  if MUJOCO_GL=$g PYOPENGL_PLATFORM=$g python - <<'PY' >/dev/null 2>&1
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 64, 64); r.update_scene(mujoco.MjData(m)); assert r.render().shape == (64, 64, 3)
PY
  then GL=$g; break; fi
done
export MUJOCO_GL="$GL" PYOPENGL_PLATFORM="$GL"
echo "[gl] $GL"

# Gate on render fidelity BEFORE spending GPU-hours: if the env's pixels disagree
# with the stored dataset frames, the absolute SR is biased and not comparable to
# the paper. Non-fatal (paired comparisons stay valid) but always reported.
# Non-fatal for SR: paired comparisons survive a renderer offset, only absolute SR
# shifts. But the verdict is recorded next to the results instead of vanishing into the
# job log, which is what let a failed cube gate go unnoticed.
GATELOG="$SSD/render_gate_${TASK}.log"
python scripts/check_render_fidelity.py "$TASK" 8 --max-mae 3.0 2>&1 | tee "$GATELOG" || \
  echo "[warn] render fidelity gate FAILED — absolute SR not comparable to published numbers" \
    | tee -a "$GATELOG"
gcloud storage cp "$GATELOG" "$BUCKET/eval/" || true

mkdir -p "$SSD/eps"
RC=0
for S in "${SEEDS[@]}"; do
  # EPS_N lets a run point at a set that is not the usual 100 episodes
  EPSNAME="episodes_${TASK}_s${S}_${EPS_N:-100}.json"
  OUT="final_${TASK}_${CFG}_${SOLVER}_s${S}${EPS_N:+_$EPS_N}.csv"
  if gcloud storage ls "$OUTP/$OUT" >/dev/null 2>&1; then
    echo "[skip] $OUT already in GCS"; continue
  fi
  gcloud storage cp "$BUCKET/eval_sets/$EPSNAME" "$SSD/eps/"
  echo "[run] $TASK $CFG $SOLVER seed=$S"
  set +e
  python scripts/budget_sweep_qinput_any.py \
    --env "$TASK" --solver "$SOLVER" \
    --config "$CFG" "$CKPT_DIR/weights_epoch_10.pt" \
    --tiers T1 T2 T3 T4 T5 \
    --episodes-json "$SSD/eps/$EPSNAME" \
    --out "$SSD/$OUT" 2>&1 | tail -40
  rc=${PIPESTATUS[0]}; set -e
  [ "$rc" -ne 0 ] && RC=$rc
  [ -f "$SSD/$OUT" ] && gcloud storage cp "$SSD/$OUT" "$OUTP/$OUT"
done
echo "[done] rc=$RC -> $OUTP/final_${TASK}_${CFG}_${SOLVER}_s*.csv"
exit $RC
