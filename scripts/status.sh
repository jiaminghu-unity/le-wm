#!/bin/bash
# One-shot training status: progress line per run + GPU load.
# Live single-run view:  tail -f /mnt/data/stable-wm/train_logs/<run>.log
L=/mnt/data/stable-wm/train_logs
echo "=== progress ==="
for f in "$L"/*.log; do
    printf "%-14s %s\n" "$(basename "$f" .log):" "$(tail -1 "$f" | tr -d '\r')"
done
echo "=== gpus ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "=== checkpoints ==="
ls -t /mnt/data/stable-wm/checkpoints/*/weights_epoch_*.pt 2>/dev/null | head -4
