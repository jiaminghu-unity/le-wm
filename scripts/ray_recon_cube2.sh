#!/usr/bin/env bash
# Pull the cube dataset onto a worker's local NVMe, expand it, and run recon.
# CPU-only job. Uploads the recon artifacts to GCS.
set -euo pipefail

SRC="gs://prism-training-us/le-wm/datasets/ogbench/cube_single_expert.tar.zst"
DST_GCS="gs://prism-training-us/le-wm/recon2/"
EXPECT_H5_SIZE=101942558720

SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
DS="$SSD/stable-wm/datasets/ogbench"
mkdir -p "$DS"
echo "scratch $SSD: $(df -h --output=avail "$SSD" | tail -1 | tr -d ' ') free"

command -v zstd >/dev/null || { sudo apt-get update -q && sudo apt-get install -y -q zstd; }

OUT_H5="$DS/cube_single_expert.h5"
if [ ! -f "$OUT_H5" ] || [ "$(stat -c%s "$OUT_H5")" != "$EXPECT_H5_SIZE" ]; then
  echo "streaming GCS -> zstd -> tar (46 GB in, 101.9 GB out)"
  time gcloud storage cat "$SRC" | zstd -dc --long=31 | tar -xf - -C "$DS"
fi
got=$(stat -c%s "$OUT_H5")
[ "$got" = "$EXPECT_H5_SIZE" ] || { echo "SIZE MISMATCH: $got" >&2; exit 1; }
echo "h5 ready: $OUT_H5"

python3 -m pip install -q --user h5py hdf5plugin pillow numpy 2>&1 | tail -2
python3 scripts/recon_cube2.py "$OUT_H5" "$SSD/recon2"

gcloud storage cp "$SSD/recon2/"* "$DST_GCS"
echo "RECON UPLOADED -> $DST_GCS"
