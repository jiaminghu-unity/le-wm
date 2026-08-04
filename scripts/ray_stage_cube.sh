#!/usr/bin/env bash
# One-time: stage the OGBench-Cube dataset to GCS from a Ray worker.
#   HuggingFace (46 GB tar.zst) -> worker local NVMe -> verify sha256 -> GCS
# Kept COMPRESSED in GCS; each training job expands it locally (fetch_cube_worker.sh).
# CPU-only job — do not give it a GPU.
set -euo pipefail

SRC_URL="https://huggingface.co/datasets/quentinll/lewm-cube/resolve/main/cube_single_expert.tar.zst"
DST_GCS="gs://prism-training-us/le-wm/datasets/ogbench/cube_single_expert.tar.zst"
EXPECT_SIZE=46184624478
EXPECT_SHA=3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600

# ---- local NVMe (375 GB, unmounted on the DL image; boot disk is too small) ----
SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe found" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD"
  sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
echo "scratch: $SSD ($(df -h --output=avail "$SSD" | tail -1 | tr -d ' ') free)"

FILE="$SSD/cube_single_expert.tar.zst"

if gcloud storage ls "$DST_GCS" >/dev/null 2>&1; then
  echo "already staged: $DST_GCS — nothing to do"; exit 0
fi

echo "[1/4] download 46 GB from HuggingFace"
time curl -fL -C - --retry 5 --retry-delay 10 --retry-all-errors -o "$FILE" "$SRC_URL"

echo "[2/4] verify size"
size=$(stat -c%s "$FILE")
[ "$size" = "$EXPECT_SIZE" ] || { echo "SIZE MISMATCH: $size != $EXPECT_SIZE" >&2; exit 1; }

echo "[3/4] verify sha256"
sha=$(sha256sum "$FILE" | cut -d' ' -f1)
[ "$sha" = "$EXPECT_SHA" ] || { echo "SHA MISMATCH: $sha != $EXPECT_SHA" >&2; exit 1; }
echo "      ok $sha"

echo "[4/4] upload to GCS"
time gcloud storage cp "$FILE" "$DST_GCS"
gcloud storage ls -l "$DST_GCS"
echo "STAGED OK -> $DST_GCS"
