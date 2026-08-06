#!/usr/bin/env bash
# Cube episode sets s104/s105/s106 drawn the SAME way as s101-s103: gen_episodes.py,
# the generic sampler, with NO non-trivial-goal filter.
#
# The first attempt used gen_episodes_cube.py, whose displacement filter is hardcoded.
# That made s104-s106 a different population from s101-s103 — absolute SR came out
# 21-29 pp lower on every arm and every solver, because roughly a third of unfiltered
# cube episodes are already solved at t=0 and the filter removes exactly those. The two
# batches cannot be pooled, so this regenerates the new three to match the old three.
set -euo pipefail
BUCKET=gs://prism-training-us/le-wm
SEEDS=(104 105 106)
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
# gen_episodes.py imports stable_worldmodel -> torch, so the cu126 pin is required
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
H5="$DS/cube_single_expert.h5"
[ -f "$H5" ] || { echo "[data] extracting cube h5"; time gcloud storage cat \
  "$BUCKET/datasets/ogbench/cube_single_expert.tar.zst" | zstd -dc --long=31 | tar -xf - -C "$DS"; }
mkdir -p "$SSD/eps"
for S in "${SEEDS[@]}"; do
  BASE=$(( 40000 + (S - 101) * 10000 ))
  python scripts/gen_episodes.py --num 100 --seed "$S" \
    --dataset ogbench/cube_single_expert --env-seed-base "$BASE" \
    --out "$SSD/eps/episodes_cube_s${S}_100.json"
done
# sanity: the new files must have the same 5-field shape as s101-s103, no filter key
python - <<'PY'
import json, glob, sys
for f in sorted(glob.glob('/mnt/disks/ssd0/eps/episodes_cube_s10[456]_100.json')):
    d = json.load(open(f))
    assert 'filter' not in d, f"{f} still carries a filter key"
    assert set(d['episodes'][0]) == {'env_seed','episode_id','goal_idx','start_idx','traj_id'}, \
        f"{f} field set differs from s101-s103"
    print(f"OK {f.split('/')[-1]} meta={ {k:v for k,v in d.items() if k!='episodes'} }")
PY
gcloud storage cp "$SSD/eps/episodes_cube_s10"[456]"_100.json" "$BUCKET/eval_sets/"
echo "CUBE NOFILTER SEEDS DONE"
