#!/bin/bash
# Phase 1 (instructions §5.1-5.2): C1 baseline + C2p lambda sweep, one run per GPU.
# Detached with setsid so runs survive terminal/session exit.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export STABLEWM_HOME=/mnt/data/stable-wm
LOGDIR=$STABLEWM_HOME/train_logs
mkdir -p "$LOGDIR"

launch() {
    local gpu=$1 name=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu setsid nohup python train.py "$@" \
        > "$LOGDIR/$name.log" 2>&1 < /dev/null &
    echo "GPU$gpu $name pid=$!"
}

launch 0 c1        experiment=c1_baseline
launch 1 c2p_l0.01 experiment=c2p_obj_projector loss.obj.weight=0.01
launch 2 c2p_l0.1  experiment=c2p_obj_projector loss.obj.weight=0.1
launch 3 c2p_l1.0  experiment=c2p_obj_projector loss.obj.weight=1.0
