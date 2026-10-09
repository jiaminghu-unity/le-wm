#!/usr/bin/env bash
# ContactWorld B3/Ours 集群训练启动器。用法: bash ray_train_cw.sh <b3|ours>
set -euo pipefail
V="${1:?variant}"
SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=""; for d in $(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1}'); do [ -z "$(lsblk -no MOUNTPOINT "$d" | tr -d '[:space:]')" ] || continue; dev="$d"; break; done
  if [ -n "$dev" ]; then sudo mkfs.ext4 -F -q -m 0 "$dev"; sudo mkdir -p "$SSD"; sudo mount -o discard,defaults "$dev" "$SSD"; else sudo mkdir -p "$SSD"; fi
  sudo chmod a+w "$SSD"
fi
command -v uv >/dev/null || { pip install -q uv; PATH="$(python3 -m site --user-base)/bin:$PATH"; export PATH; hash -r; }
[ -x "$SSD/.cwvenv/bin/python" ] || uv venv --python=3.10 "$SSD/.cwvenv"
source "$SSD/.cwvenv/bin/activate"
uv pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -q 'zarr<3' numcodecs numpy einops pillow
python -c 'import torch,zarr,einops; assert torch.cuda.is_available()' || { echo FATAL_ENV; exit 43; }
mkdir -p data/demo_data
echo "[data] rsync insertion_usb"
gcloud storage rsync -r gs://prism-training-us/contactworld/datasets/insertion_usb data/demo_data/insertion_usb >/dev/null
# 后台每 15 分钟同步 ckpt/日志(崩溃可续)
( while true; do sleep 900; gcloud storage rsync -r "outputs/$V" "gs://prism-training-us/contactworld/outputs/${V}_cluster" >/dev/null 2>&1 || true; done ) &
SYNC=$!
# 断点续训:拉回已有 ckpt
mkdir -p "outputs/$V"
gcloud storage rsync -r "gs://prism-training-us/contactworld/outputs/${V}_cluster" "outputs/$V" >/dev/null 2>&1 || true
python train_b3_ours.py --variant "$V"
kill $SYNC 2>/dev/null || true
gcloud storage rsync -r "outputs/$V" "gs://prism-training-us/contactworld/outputs/${V}_cluster"
echo done | gcloud storage cp - "gs://prism-training-us/contactworld/outputs/${V}_cluster/COMPLETE"
echo "CW $V TRAINING COMPLETE"
