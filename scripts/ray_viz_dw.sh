#!/usr/bin/env bash
# Representation-diagnostics panel for the navigation tasks, four arms including the
# DINO-WM baseline.
#   usage: ray_viz_dw.sh {tworoom|pointmaze}
#
# CPU-only on purpose, same as ray_viz_cube.sh was: every GPU is either running the
# DINO-WM eval sweeps or reserved for the user's interactive use, and encoding 1500
# frames through ViT-Tiny + DINOv2-small is minutes on CPU.
# Reads ckpts_<task>/ (read-only) and datasets/; writes only eval/viz_general_<task>.png.
set -euo pipefail

TASK="${1:?tworoom|pointmaze}"
BUCKET=gs://prism-training-us/le-wm

case "$TASK" in
  tworoom)
    CKP="ckpts_tworoom"
    CKS="lewm_t1_tworoom_s3072 lewm_t2_tworoom_obj0.1_s3072 lewm_t5_tworoom_qhead0.1_s3072 dinowm_tworoom_s3072" ;;
  pointmaze)
    CKP="ckpts_pointmaze"
    CKS="lewm_p1_pointmaze_s3072 lewm_p2_pointmaze_s3072 lewm_p5_pointmaze_s3072 dinowm_pointmaze_s3072" ;;
  *) echo "FATAL: unknown task $TASK" >&2; exit 1 ;;
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
DS="$STABLEWM_HOME/datasets"
mkdir -p "$DS" "$STABLEWM_HOME/checkpoints"
echo "[env] viz_dw/$TASK on $(hostname), free=$(df -h --output=avail "$SSD"|tail -1|tr -d ' ')"

sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd \
  libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1

if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets scikit-learn scipy matplotlib

LANCE="$DS/$TASK.lance"
if [ ! -d "$LANCE" ]; then
  echo "[data] pulling $TASK.lance"
  time gcloud storage rsync -r "$BUCKET/datasets/$TASK.lance" "$LANCE"
fi

for ck in $CKS; do
  mkdir -p "$STABLEWM_HOME/checkpoints/$ck"
  gcloud storage cp "$BUCKET/$CKP/$ck/weights_epoch_10.pt" "$STABLEWM_HOME/checkpoints/$ck/"
  gcloud storage cp "$BUCKET/$CKP/$ck/config.json" "$STABLEWM_HOME/checkpoints/$ck/" || true
done

python scripts/visualize_general_dw.py "$TASK" 2>&1 | tee "$SSD/viz_dw_$TASK.log"

gcloud storage cp "eval_results/viz_general_$TASK.png" "$BUCKET/eval/"
gcloud storage cp "$SSD/viz_dw_$TASK.log" "$BUCKET/eval/"
echo "VIZ DW $TASK DONE"
