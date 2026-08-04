#!/usr/bin/env bash
# Worker-side: materialize the OGBench-Cube dataset on the ephemeral worker.
# Streams GCS -> zstd -> tar so the 46 GB archive never lands on disk; only the
# 101.9 GB h5 does. Expects $STABLEWM_HOME to be set by the caller.
#
# Result: $STABLEWM_HOME/datasets/ogbench/cube_single_expert.h5
#   (matches config/train/data/ogb.yaml -> name: ogbench/cube_single_expert.h5)
set -euo pipefail

SRC="gs://prism-training-us/le-wm/datasets/ogbench/cube_single_expert.tar.zst"
DS="${STABLEWM_HOME:?STABLEWM_HOME not set}/datasets"
OUT="$DS/ogbench/cube_single_expert.h5"
EXPECT_H5_SIZE=101942558720

if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT")" = "$EXPECT_H5_SIZE" ]; then
  echo "cube h5 already present and correctly sized — skipping fetch"
  exit 0
fi

avail=$(df -B1 --output=avail "$DS" | tail -1)
if [ "$avail" -lt $((EXPECT_H5_SIZE + 5000000000)) ]; then
  echo "FATAL: need ~107 GB free under $DS, have $((avail / 1000000000)) GB." >&2
  echo "Point SCRATCH/STABLEWM_HOME at a bigger disk, or raise worker diskSizeGb." >&2
  exit 1
fi

mkdir -p "$DS/ogbench"
command -v zstd >/dev/null || { sudo apt-get update -q && sudo apt-get install -y -q zstd; }

echo "streaming $SRC -> $OUT (46 GB compressed / 101.9 GB extracted)"
gcloud storage cat "$SRC" | zstd -dc --long=31 | tar -xf - -C "$DS/ogbench"

got=$(stat -c%s "$OUT")
[ "$got" = "$EXPECT_H5_SIZE" ] || { echo "SIZE MISMATCH: $got != $EXPECT_H5_SIZE" >&2; exit 1; }
echo "cube dataset ready: $OUT"
