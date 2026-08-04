#!/usr/bin/env bash
# Freeze the cube q_stats to GCS so every training job downloads the same file.
set -euo pipefail

SRC="gs://prism-training-us/le-wm/datasets/ogbench/cube_single_expert.tar.zst"
DST_GCS="gs://prism-training-us/le-wm/qstats/"
EXPECT_H5_SIZE=101942558720
DATASET_NAME="ogbench/cube_single_expert.h5"

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
command -v zstd >/dev/null || { sudo apt-get update -q && sudo apt-get install -y -q zstd; }

OUT_H5="$DS/cube_single_expert.h5"
if [ ! -f "$OUT_H5" ] || [ "$(stat -c%s "$OUT_H5")" != "$EXPECT_H5_SIZE" ]; then
  gcloud storage cat "$SRC" | zstd -dc --long=31 | tar -xf - -C "$DS"
fi
[ "$(stat -c%s "$OUT_H5")" = "$EXPECT_H5_SIZE" ] || { echo "SIZE MISMATCH" >&2; exit 1; }

python3 -m pip install -q --user h5py hdf5plugin numpy torch \
  --extra-index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -2
python3 scripts/make_cube_qstats.py "$OUT_H5" "$DATASET_NAME" "$SSD/stable-wm/datasets"

gcloud storage cp "$SSD/stable-wm/datasets/"*.q_stats.*.json "$DST_GCS"
gcloud storage ls -l "$DST_GCS"
echo "QSTATS STAGED -> $DST_GCS"
