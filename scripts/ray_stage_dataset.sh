#!/usr/bin/env bash
# Stage the LeWM Push-T dataset to GCS. Runs as a Ray job on an ephemeral
# a2-ultragpu-1g worker (150G boot disk). EXTENDS scripts/restore_data.sh
# (the proven 2026-07-21 local recovery path): download h5.zst from HF ->
# decompress -> convert to lance -> upload lance+h5 to GCS via the worker's
# attached SA (prism-training-sa, cloud-platform scope; no user OAuth needed).
# Result:
#   gs://prism-training-us/le-wm/datasets/pusht_expert_train.lance/  (train reads lance)
#   gs://prism-training-us/le-wm/datasets/pusht_expert_train.h5      (eval reads h5)
set -euo pipefail

GCS_BASE="gs://prism-training-us/le-wm/datasets"
HF_REPO="quentinll/lewm-pusht"
HF_FILE="pusht_expert_train.h5.zst"

echo "=== host/env $(date -u) ==="
hostname; whoami; nvidia-smi -L 2>/dev/null || true
echo "--- disks ---"; df -h | sort -k4 -h

# ---- pick the mount with the most free space (need ~100G peak) as SCRATCH ----
pick_scratch() {
  local best_mnt="/" best_avail=0 mnt avail
  for mnt in / /mnt /mnt/disks /home /var/tmp; do
    [ -d "$mnt" ] || continue
    avail=$(df -B1 --output=avail "$mnt" 2>/dev/null | tail -1 | tr -d ' ') || continue
    [ -n "$avail" ] || continue
    if [ "$avail" -gt "$best_avail" ]; then best_avail=$avail; best_mnt=$mnt; fi
  done
  echo "$best_mnt $best_avail"
}
read -r SCR_MNT SCR_AVAIL < <(pick_scratch)
NEED=$((100*1000*1000*1000))
echo "scratch mount=$SCR_MNT avail=$((SCR_AVAIL/1000000000))G (need ~100G peak)"
if [ "$SCR_AVAIL" -lt "$NEED" ]; then
  echo "FATAL: no mount with >=100G free (best=$SCR_MNT $((SCR_AVAIL/1000000000))G)" >&2
  exit 1
fi

SCRATCH="$SCR_MNT/lewm-stage"
sudo mkdir -p "$SCRATCH" && sudo chown "$(whoami)" "$SCRATCH"
export STABLEWM_HOME="$SCRATCH/stable-wm"
export HF_HOME="$SCRATCH/hf"
DS="$STABLEWM_HOME/datasets"; mkdir -p "$DS"
H5="$DS/pusht_expert_train.h5"
ZST="$DS/$HF_FILE"
LANCE="$DS/pusht_expert_train.lance"

# ---- [0] deps: fresh uv venv, mirror the proven restore_data.sh env ----
echo "=== [0/5] deps $(date -u) ==="
sudo apt-get update -q && sudo apt-get install -y -q \
  zstd swig build-essential libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 || true
command -v uv >/dev/null 2>&1 || pip install -q uv || python3 -m pip install -q uv
export PATH="$HOME/.local/bin:$PATH"
uv venv --python=3.10 "$SCRATCH/.venv"
# shellcheck disable=SC1091
source "$SCRATCH/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets

echo "=== [1/5] download $HF_FILE from HF $(date -u) ==="
hf download "$HF_REPO" "$HF_FILE" --repo-type dataset --local-dir "$DS"
ls -la "$ZST"

echo "=== [2/5] decompress -> h5 (44G) $(date -u) ==="
zstd -d --rm -f "$ZST" -o "$H5"
ls -la "$H5"

echo "=== [3/5] convert -> lance $(date -u) ==="
python - <<PYEOF
from stable_worldmodel.data import convert
convert("$H5", "$LANCE", dest_format="lance")
print("converted -> $LANCE")
PYEOF
echo "--- lance dir ---"; ls -la "$LANCE" | head; du -sh "$LANCE" || true

echo "=== [4/5] upload lance + h5 to GCS $(date -u) ==="
gcloud storage rsync -r "$LANCE" "$GCS_BASE/pusht_expert_train.lance"
gcloud storage cp "$H5" "$GCS_BASE/pusht_expert_train.h5"

echo "=== [5/5] verify on GCS $(date -u) ==="
gcloud storage ls -r "$GCS_BASE/pusht_expert_train.lance" | head
gcloud storage ls -l "$GCS_BASE/pusht_expert_train.h5"
gcloud storage du -s "$GCS_BASE/pusht_expert_train.lance" "$GCS_BASE/pusht_expert_train.h5"
echo "STAGED_OK $(date -u)"
