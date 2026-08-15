#!/usr/bin/env bash
# DINO-WM arm of the P4 probe on pusht/reacher/cube, one GPU, sequential.
# Reads ckpts/ read-only; writes eval/p4dw_*.{json,log}.
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
DS="$STABLEWM_HOME/datasets"; mkdir -p "$DS" "$STABLEWM_HOME/checkpoints"

sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd libgl1 libglib2.0-0 libxcb1 \
  libsm6 libxext6 libxrender1 libosmesa6-dev libglew-dev libgl1-mesa-dev
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

for spec in "pusht_expert_train pusht" "reacher reacher" "ogbench/cube_single_expert cube"; do
  set -- $spec
  [ -f "$DS/$1.h5" ] || { mkdir -p "$(dirname "$DS/$1")"; gcloud storage cp "$BUCKET/datasets/$1.h5" "$DS/$1.h5"; }
done
for ck in dinowm_pusht_s3072 dinowm_reacher_s3072 dinowm_cube_s3072; do
  mkdir -p "$STABLEWM_HOME/checkpoints/$ck"
  gcloud storage cp "$BUCKET/ckpts/$ck/weights_epoch_10.pt" "$STABLEWM_HOME/checkpoints/$ck/"
  gcloud storage cp "$BUCKET/ckpts/$ck/config.json" "$STABLEWM_HOME/checkpoints/$ck/" || true
done

RC=0
for t in pusht reacher cube; do
  if gcloud storage ls "$BUCKET/eval/p4dw_$t.json" >/dev/null 2>&1; then
    echo "[skip] p4dw_$t.json already in GCS"; continue
  fi
  set +e
  python scripts/p4_phys_dw.py "$t" dw:dinowm_${t}_s3072/weights_epoch_10.pt \
    2>&1 | tee "$SSD/p4dw_$t.log"
  rc=${PIPESTATUS[0]}; set -e
  [ "$rc" -ne 0 ] && RC=$rc
  gcloud storage cp "$SSD/p4dw_$t.log" "$BUCKET/eval/" || true
  [ -f "eval_results/p4dw_$t.json" ] && gcloud storage cp "eval_results/p4dw_$t.json" "$BUCKET/eval/"
done
echo "[done] rc=$RC"; exit $RC
