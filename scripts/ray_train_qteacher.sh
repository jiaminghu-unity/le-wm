#!/usr/bin/env bash
# q-only teacher trainer: replica launcher + reacher.lance staging (the q-input
# configs read the lance dataset directly).
#   usage: ray_train_replica.sh <pusht|reacher|cube> <hydra overrides...>
#     e.g. ray_train_replica.sh pusht experiment=c1_baseline data=pusht seed=3073
#
# The original pusht/reacher rounds had no reusable Ray launcher (cube's fetches only
# the h5). This one stages the exact dataset each canonical config trains from
# (pusht: pusht_expert_train.lance; reacher: reacher.h5 via data=dmc; cube:
# ogbench/cube_single_expert.lance via data=ogb_cube), fetches persisted q-stats when
# GCS has them (computed deterministically from the dataset otherwise), resolves the
# run name WITH the given overrides so seed-suffixed names land in new directories,
# and uploads to ckpts/<run>/. Nothing existing is overwritten: replication seeds
# produce *_s<seed> names that cannot collide with the canonical *_s3072 ones.
set -euo pipefail

TASK="${1:?pusht|reacher|cube}"; shift
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
DS="$STABLEWM_HOME/datasets"; mkdir -p "$DS"

sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd libgl1 libglib2.0-0 libxcb1 \
  libsm6 libxext6 libxrender1
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets

case "$TASK" in
  pusht)
    [ -d "$DS/pusht_expert_train.lance" ] || \
      gcloud storage rsync -r "$BUCKET/datasets/pusht_expert_train.lance" "$DS/pusht_expert_train.lance" ;;
  reacher)
    [ -f "$DS/reacher.h5" ] || gcloud storage cp "$BUCKET/datasets/reacher.h5" "$DS/"
    [ -d "$DS/reacher.lance" ] || \
      gcloud storage rsync -r "$BUCKET/datasets/reacher.lance" "$DS/reacher.lance" ;;
  cube)
    mkdir -p "$DS/ogbench"
    [ -d "$DS/ogbench/cube_single_expert.lance" ] || \
      gcloud storage rsync -r "$BUCKET/datasets/ogbench/cube_single_expert.lance" "$DS/ogbench/cube_single_expert.lance" ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
# persisted q-stats, so replicas share the canonical normalizer bytes when available
gcloud storage cp "$BUCKET/datasets/*.q_stats.*.json" "$DS/" 2>/dev/null || true
gcloud storage cp "$BUCKET/datasets/ogbench/*.q_stats.*.json" "$DS/ogbench/" 2>/dev/null || true

RUN=$(python train.py --cfg job --resolve "$@" 2>/dev/null \
      | grep -E "^output_model_name:" | awk '{print $2}' | tr -d '\r')
[ -n "$RUN" ] || { echo "FATAL: could not resolve output_model_name for: $*" >&2; exit 1; }
echo "[replica] $TASK -> $RUN"
if gcloud storage ls "$BUCKET/ckpts/$RUN/weights_epoch_10.pt" >/dev/null 2>&1; then
  echo "[skip] $RUN already trained"; exit 0
fi

LOG="$SSD/train_$RUN.log"
python train.py "$@" 2>&1 | tee "$LOG"

CKDIR=$(find "$STABLEWM_HOME" outputs -type d -name "$RUN" 2>/dev/null | head -1)
[ -n "$CKDIR" ] && [ -f "$CKDIR/weights_epoch_10.pt" ] || \
  CKDIR=$(dirname "$(find "$STABLEWM_HOME" outputs -name weights_epoch_10.pt -path "*$RUN*" 2>/dev/null | head -1)")
[ -f "$CKDIR/weights_epoch_10.pt" ] || { echo "FATAL: no weights_epoch_10.pt for $RUN" >&2; exit 1; }
gcloud storage cp "$CKDIR/weights_epoch_10.pt" "$BUCKET/ckpts/$RUN/"
gcloud storage cp "$CKDIR/config.json" "$BUCKET/ckpts/$RUN/" || true
gcloud storage cp "$LOG" "$BUCKET/ckpts/logs/" || true
echo "REPLICA TRAIN DONE $RUN"
