#!/usr/bin/env bash
# SEEDS_OVERRIDE variant for the five-task retest (new file; original untouched).
# Generate the pre-registered episode sets s104..s113 for Push-T and Reacher.
#
# The count is fixed BEFORE any of them is evaluated and every one of them is
# reported: adding seeds until the answer looks good is optional stopping, and at
# Reacher's effect size (~+0.7pp against a 3.6pp per-seed SD) a "3 of 3 favour
# L_obj" outcome happens 10% of the time under a zero effect.
#
# env_seed_base follows the existing convention 40000 + (seed-101)*10000, so the
# env resets of a new set never collide with an old one. budget_sweep derives the
# planner noise from crc32("episode_id|tier"), and episode_id is the index within
# the file, so each set also gets its own independent CEM draws.
set -euo pipefail

BUCKET=gs://prism-training-us/le-wm
SEEDS=(${SEEDS_OVERRIDE:-104 105 106 107 108 109 110 111 112 113})

SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
mkdir -p "$STABLEWM_HOME/datasets"

sudo apt-get update -q
sudo apt-get install -y -q build-essential
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q hdf5plugin -U

for pair in "pusht:pusht_expert_train.h5" "reacher:reacher.h5"; do
  H5NAME="${pair#*:}"
  H5="$STABLEWM_HOME/datasets/$H5NAME"
  [ -f "$H5" ] || { echo "[data] fetching $H5NAME"; time gcloud storage cp "$BUCKET/datasets/$H5NAME" "$H5"; }
done

mkdir -p "$SSD/eps"
for S in "${SEEDS[@]}"; do
  BASE=$(( 40000 + (S - 101) * 10000 ))
  for pair in "pusht:pusht_expert_train" "reacher:reacher"; do
    TASK="${pair%%:*}"; DSNAME="${pair#*:}"
    OUT="$SSD/eps/episodes_${TASK}_s${S}_100.json"
    python scripts/gen_episodes.py --num 100 --seed "$S" --dataset "$DSNAME" \
      --env-seed-base "$BASE" --out "$OUT"
  done
done

for f in "$SSD/eps/"episodes_*_100.json; do gcloud storage cp "$f" "$BUCKET/eval_sets/"; done
echo "GEN DONE: $(ls "$SSD/eps" | wc -l) sets -> $BUCKET/eval_sets/"
