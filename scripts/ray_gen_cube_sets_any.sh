#!/usr/bin/env bash
# SEEDS_OVERRIDE variant for the five-task retest (new file; original untouched).
# Cube episode sets s104/s105/s106, matching the pre-registered pusht/reacher batch.
# env_seed_base follows the same convention: 40000 + (seed-101)*10000.
# The non-trivial-goal filter in gen_episodes_cube.py stays on — without it ~32% of
# uniformly sampled cube episodes are already solved at t=0.
set -euo pipefail
BUCKET=gs://prism-training-us/le-wm
SEEDS=(${SEEDS_OVERRIDE:-104 105 106})
SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"; sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets/ogbench"; mkdir -p "$DS"
sudo apt-get update -q && sudo apt-get install -y -q build-essential zstd
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
[ -x "$SSD/.venv/bin/python" ] || uv venv --python=3.10 "$SSD/.venv"
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]' hdf5plugin -U
H5="$DS/cube_single_expert.h5"
[ -f "$H5" ] || { echo "[data] extracting cube h5"; time gcloud storage cat \
  "$BUCKET/datasets/ogbench/cube_single_expert.tar.zst" | zstd -dc --long=31 | tar -xf - -C "$DS"; }
ls -la "$H5"
mkdir -p "$SSD/eps"
for S in "${SEEDS[@]}"; do
  BASE=$(( 40000 + (S - 101) * 10000 ))
  python scripts/gen_episodes_cube.py --h5 "$H5" --num 100 --seed "$S" \
    --env-seed-base "$BASE" --out "$SSD/eps/episodes_cube_s${S}_100.json"
done
for f in "$SSD/eps/"episodes_cube_s*_100.json; do gcloud storage cp "$f" "$BUCKET/eval_sets/"; done
echo "CUBE SEEDS DONE"
