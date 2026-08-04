#!/usr/bin/env bash
# Stage a Push-T or Reacher dataset from HuggingFace to GCS, and generate the
# three 100-episode evaluation sets for the final cross-task sweep.
#   usage: ray_stage_task_data.sh pusht|reacher
set -euo pipefail

TASK="${1:?pusht|reacher}"
BUCKET=gs://prism-training-us/le-wm

case "$TASK" in
  pusht)
    HF=quentinll/lewm-pusht; ARCHIVE=pusht_expert_train.h5.zst
    H5NAME=pusht_expert_train.h5; DSNAME=pusht_expert_train; KIND=zst ;;
  reacher)
    HF=quentinll/lewm-reacher; ARCHIVE=reacher.tar.zst
    H5NAME=reacher.h5; DSNAME=reacher; KIND=tar ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac

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
echo "[env] $TASK on $(hostname), $(df -h --output=avail "$SSD" | tail -1 | tr -d ' ') free"

sudo apt-get update -q
sudo apt-get install -y -q zstd swig build-essential \
  libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets 'huggingface_hub[cli]'

H5="$DS/$H5NAME"
if [ ! -f "$H5" ]; then
  echo "[1/3] download $ARCHIVE from $HF"
  hf download "$HF" "$ARCHIVE" --repo-type dataset --local-dir "$DS"
  echo "[2/3] decompress"
  if [ "$KIND" = zst ]; then
    zstd -d --rm -f "$DS/$ARCHIVE" -o "$H5"
  else
    zstd -dc --long=31 "$DS/$ARCHIVE" | tar -xf - -C "$DS" && rm -f "$DS/$ARCHIVE"
  fi
fi
ls -la "$H5"

echo "[3/3] generate 3 x 100-episode sets"
mkdir -p "$SSD/eps"
i=0
for seed in 101 102 103; do
  base=$((40000 + i * 10000)); i=$((i + 1))
  python scripts/gen_episodes.py --num 100 --seed "$seed" --dataset "$DSNAME" \
    --env-seed-base "$base" --out "$SSD/eps/episodes_${TASK}_s${seed}_100.json"
done

gcloud storage cp "$SSD/eps/"*.json "$BUCKET/eval_sets/"
gcloud storage cp "$H5" "$BUCKET/datasets/$H5NAME"
echo "STAGED $TASK -> $BUCKET/datasets/$H5NAME  +  3 episode sets"
