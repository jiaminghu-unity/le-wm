#!/usr/bin/env bash
# Representation-diagnostics panel for the Reacher reduced-q question
# (baseline / full-q L_obj / half-q L_obj / aux).
#   usage: ray_viz_reacher_half.sh
#
# CPU-only, same rationale as ray_viz_dw.sh: encoding 1500 frames through four
# ViT-Tiny models is minutes on CPU and every GPU is spoken for.
# Reads ckpts/ and ckpts_half/ read-only; writes only eval/viz_general_reacher_half.png.
set -euo pipefail

BUCKET=gs://prism-training-us/le-wm

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
echo "[env] viz_reacher_half on $(hostname), free=$(df -h --output=avail "$SSD"|tail -1|tr -d ' ')"

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

LANCE="$DS/reacher.lance"
if [ ! -d "$LANCE" ]; then
  echo "[data] pulling reacher.lance"
  time gcloud storage rsync -r "$BUCKET/datasets/reacher.lance" "$LANCE"
fi

for spec in \
  "ckpts lewm_r1_reacher_s3072" \
  "ckpts lewm_r2_reacher_paep_l015_s3072" \
  "ckpts_half lewm_hq_obj_reacher_s3072" \
  "ckpts lewm_r5_qhead0.4_s3072" ; do
  set -- $spec
  mkdir -p "$STABLEWM_HOME/checkpoints/$2"
  gcloud storage cp "$BUCKET/$1/$2/weights_epoch_10.pt" "$STABLEWM_HOME/checkpoints/$2/"
  gcloud storage cp "$BUCKET/$1/$2/config.json" "$STABLEWM_HOME/checkpoints/$2/" || true
done

python scripts/visualize_general_reacher_half.py 2>&1 | tee "$SSD/viz_reacher_half.log"

gcloud storage cp eval_results/viz_general_reacher_half.png "$BUCKET/eval/"
gcloud storage cp "$SSD/viz_reacher_half.log" "$BUCKET/eval/"
echo "VIZ REACHER HALF DONE"
