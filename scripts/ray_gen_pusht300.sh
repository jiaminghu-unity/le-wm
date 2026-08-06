#!/usr/bin/env bash
# Push-T episode set s107 with 300 episodes (the existing sets are 100 each).
# env_seed_base continues the convention 40000 + (seed-101)*10000 -> s107 = 100000.
set -euo pipefail
BUCKET=gs://prism-training-us/le-wm
SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"; sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"; mkdir -p "$STABLEWM_HOME/datasets"
sudo apt-get update -q && sudo apt-get install -y -q build-essential
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
[ -x "$SSD/.venv/bin/python" ] || uv venv --python=3.10 "$SSD/.venv"
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]' hdf5plugin -U
# gen_episodes.py imports stable_worldmodel, which imports torch, so the same cu126
# pin every other job uses is required here too — the default wheel lands on a torch
# whose CUDA libs mismatch this image (undefined symbol: ncclCommResume).
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
H5="$STABLEWM_HOME/datasets/pusht_expert_train.h5"
[ -f "$H5" ] || gcloud storage cp "$BUCKET/datasets/pusht_expert_train.h5" "$H5"
mkdir -p "$SSD/eps"
python scripts/gen_episodes.py --num 300 --seed 107 --dataset pusht_expert_train \
  --env-seed-base 100000 --out "$SSD/eps/episodes_pusht_s107_300.json"
gcloud storage cp "$SSD/eps/episodes_pusht_s107_300.json" "$BUCKET/eval_sets/"
echo "PUSHT300 DONE"
