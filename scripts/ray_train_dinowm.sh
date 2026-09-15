#!/usr/bin/env bash
# DINO-WM baseline training, one task per job.
#   usage: ray_train_dinowm.sh <pusht|reacher|cube|tworoom|pointmaze>
# Written fresh (no string-derivation). Checkpoints go to ckpts_dinowm/<RUN>/, with RUN
# resolved from the composed hydra config, never hardcoded.
set -euo pipefail
TASK="${1:?usage: ray_train_dinowm.sh <task> [seed]}"
SEED="${2:-3072}"
case "$TASK" in
  pusht)     DSNAME=pusht_expert_train.lance ;;
  reacher)   DSNAME=reacher.lance ;;
  cube)      DSNAME=ogbench/cube_single_expert.lance ;;
  tworoom)   DSNAME=tworoom.lance ;;
  pointmaze) DSNAME=pointmaze.lance ;;
  cube_double)    DSNAME=ogbench/cube_double_play.lance ;;
  cube_triple)    DSNAME=ogbench/cube_triple_play.lance ;;
  cube_quadruple) DSNAME=ogbench/cube_quadruple_play.lance ;;
  scene)          DSNAME=ogbench/scene_play.lance ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
EXP="dw_${TASK}"
BUCKET=gs://prism-training-us/le-wm
OUTP="$BUCKET/ckpts_dinowm"

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
echo "[env] dinowm/$TASK on $(hostname), free=$(df -h --output=avail "$SSD"|tail -1|tr -d ' ')"

# NVML health: unattended-upgrades can replace the userspace NVIDIA libs under a
# live node -> "Driver/library version mismatch" and every GPU step dies. Try a
# kernel-module reload; if the GPU is busy or reload fails, abort loudly so the
# retry lands on a healthy node. Also hold nvidia packages against auto-upgrade.
if ! nvidia-smi >/dev/null 2>&1; then
  echo "[gpu] NVML broken ($(nvidia-smi 2>&1 | head -1)) -> reloading modules"
  sudo systemctl stop unattended-upgrades 2>/dev/null || true
  sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true
  sudo modprobe nvidia 2>/dev/null || true
  nvidia-smi >/dev/null 2>&1 || { echo "[gpu] FATAL: NVML still broken after reload"; exit 43; }
fi
dpkg -l | awk '/^ii +(libnvidia|nvidia-)/{print $2}' | xargs -r sudo apt-mark hold >/dev/null 2>&1 || true
sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd libgl1 libglib2.0-0 libxcb1 \
  libsm6 libxext6 libxrender1
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets transformers
# shared-venv poison guard: another job type may have installed a torch build
# that cannot see CUDA on this node -> rebuild the venv once and reinstall.
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "[env] torch cannot see CUDA -> rebuilding venv"
  deactivate || true
  rm -rf "$SSD/.venv"
  uv venv --python=3.10 "$SSD/.venv"
  source "$SSD/.venv/bin/activate"
  uv pip install -q 'stable-worldmodel[train,env,format]'
  uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
  uv pip install -q hdf5plugin -U datasets transformers
  python -c "import torch; assert torch.cuda.is_available()" || { echo "FATAL: still no CUDA"; exit 1; }
fi

if [ ! -d "$DS/$DSNAME" ]; then
  echo "[data] fetching $DSNAME"; mkdir -p "$(dirname "$DS/$DSNAME")"
  time gcloud storage rsync -r "$BUCKET/datasets/$DSNAME" "$DS/$DSNAME"
fi
du -sh "$DS/$DSNAME"

RUN=$(python train_dinowm.py --cfg job --resolve "experiment=$EXP" "seed=$SEED" 2>/dev/null \
      | grep -E "^output_model_name:" | awk '{print $2}' | tr -d '\r')
[ -n "$RUN" ] || { echo "FATAL: could not resolve output_model_name for $EXP" >&2; exit 1; }
LOG="$SSD/train_$RUN.log"; : > "$LOG"
echo "[train] experiment=$EXP -> $RUN" | tee -a "$LOG"
export HYDRA_FULL_ERROR=1
set +e
python train_dinowm.py "experiment=$EXP" "seed=$SEED" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "[train] exit $rc" | tee -a "$LOG"
# absence of this line means the backbone was NOT frozen and the run is not DINO-WM
grep -q "\[dinowm\] backbone frozen" "$LOG" || { echo "FATAL: backbone freeze never ran" >&2; rc=1; }

CKDIR="$STABLEWM_HOME/checkpoints/$RUN"
if [ "$rc" = 0 ] && [ -f "$CKDIR/weights_epoch_10.pt" ]; then
  gcloud storage cp "$CKDIR/weights_epoch_10.pt" "$OUTP/$RUN/"
  gcloud storage cp "$CKDIR/config.json" "$OUTP/$RUN/" || true
fi
gcloud storage cp "$LOG" "$OUTP/logs/" || true
echo "[done] rc=$rc  $RUN"
exit $rc
