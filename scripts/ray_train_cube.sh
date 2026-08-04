#!/usr/bin/env bash
# Cube (data=ogb) single-GPU train entrypoint for a Ray worker.
# REUSES: NVMe-mount block from scripts/ray_stage_cube.sh, dataset fetch from
#         scripts/fetch_cube_worker.sh, dep pins from REPRODUCE.md (torch cu126 +
#         stable-worldmodel[train,env]). New file only because no cube train
#         entrypoint existed (the pusht ray_train_task.sh was never created).
# usage: ray_train_cube.sh <hydra-overrides...>
#   e.g. ray_train_cube.sh experiment=c1_baseline data=ogb
set -euo pipefail

# ---- 0. mount local NVMe (a2-ultragpu ships local SSD; DL image does not auto-mount) ----
SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe found" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="${STABLEWM_HOME:-$SSD/stable-wm}"
mkdir -p "$STABLEWM_HOME/datasets"
echo "== scratch $SSD: $(df -h --output=avail "$SSD" | tail -1 | tr -d ' ') free; STABLEWM_HOME=$STABLEWM_HOME"

# ---- 1. deps FIRST (fail fast before the long 46/102 GB fetch) ----
echo "== [deps] building venv on NVMe"
command -v uv >/dev/null || pip install -q uv
uv venv --python=3.10 "$SSD/.venv"
# shellcheck disable=SC1091
source "$SSD/.venv/bin/activate"
uv pip install -q "stable-worldmodel[train,env]"
uv pip install -q "torch==2.12.1+cu126" torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets
sudo apt-get update -q && sudo apt-get install -y -q \
  swig build-essential libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 zstd || true
python -c "import torch; print('== torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"

# ---- 2. dataset (GCS -> NVMe h5, ~46 GB in / 101.9 GB out) ----
echo "== [data] fetching cube h5"
bash scripts/fetch_cube_worker.sh

# ---- 3. train ----
echo "== [train] python train.py $*"
python train.py "$@"
echo "== TRAIN EXIT $?"
