#!/bin/bash
# Restore Push-T dataset: download h5.zst from HF -> decompress -> convert to lance
set -euo pipefail

export STABLEWM_HOME=/mnt/data/stable-wm
export HF_HOME=/mnt/data/cache/huggingface
DATASETS=$STABLEWM_HOME/datasets
VENV=/mnt/data/code/le-wm/.venv
mkdir -p "$DATASETS"

echo "=== [1/3] download $(date) ==="
"$VENV/bin/hf" download quentinll/lewm-pusht pusht_expert_train.h5.zst \
    --repo-type dataset --local-dir "$DATASETS"

echo "=== [2/3] decompress $(date) ==="
zstd -d --rm -f "$DATASETS/pusht_expert_train.h5.zst" -o "$DATASETS/pusht_expert_train.h5"

echo "=== [3/3] convert to lance $(date) ==="
"$VENV/bin/python" - <<'EOF'
from stable_worldmodel.data import convert
convert(
    '/mnt/data/stable-wm/datasets/pusht_expert_train.h5',
    '/mnt/data/stable-wm/datasets/pusht_expert_train.lance',
    dest_format='lance',
)
EOF

echo "=== DONE $(date) ==="
ls -la "$DATASETS"
