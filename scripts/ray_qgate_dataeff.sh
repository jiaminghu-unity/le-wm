#!/usr/bin/env bash
# Data-efficiency Stage-1 runs (cube, L1+InfoNCE recipe): one job = one (N, data-seed),
# both lambdas {0.01, 0.1} sequentially. Outputs to qgate_dataeff/ with N and seed in
# the name; skip-if-present per output.
#   usage: ray_qgate_dataeff.sh <max_episodes> <data_seed>
set -euo pipefail
N="${1:?max_episodes}"; DS="${2:?data_seed}"
BUCKET=gs://prism-training-us/le-wm
SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
cd "$(dirname "$0")/.." || exit 1
command -v uv >/dev/null || {
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
}
[ -x "$SSD/.venv/bin/python" ] || uv venv --python=3.10 "$SSD/.venv"
source "$SSD/.venv/bin/activate"
uv pip install -q torch numpy h5py hdf5plugin scipy 2>/dev/null || uv pip install -q torch numpy h5py hdf5plugin scipy
H5="$SSD/cube_single_expert.h5"
# node-level lock: several dataeff jobs share one worker NVMe and the same $H5;
# unlocked concurrent stage+untar corrupts the extraction (short writes / ENOSPC)
exec 9>"$SSD/.cube_stage.lock"
flock 9
if [ ! -f "$H5.ok" ]; then
  rm -f "$H5"
  FREE_G=$(df --output=avail -BG "$SSD" | tail -1 | tr -dc 0-9)
  [ "${FREE_G:-0}" -lt 120 ] && { rm -f "$SSD"/*.gstmp; find "$SSD/stable-wm/datasets" -mindepth 1 -maxdepth 2 -print -exec rm -rf {} + 2>/dev/null || true; }
  gcloud storage cp "$BUCKET/datasets/ogbench/cube_single_expert.tar.zst" "$SSD/"
  tar --use-compress-program=unzstd -xf "$SSD/cube_single_expert.tar.zst" -C "$SSD"
  find "$SSD" -name "cube_single_expert.h5" -exec mv {} "$H5" \; 2>/dev/null || true
fi
[ -f "$H5" ] || { echo "FATAL: cube h5 missing" >&2; exit 1; }
touch "$H5.ok"
flock -u 9
for L in 0.01 0.1; do
  OUT="qgate_dataeff_cube_nce_N${N}_r${DS}_lam${L}.json"
  if gcloud storage ls "$BUCKET/qgate_dataeff/$OUT" >/dev/null 2>&1; then echo "[skip] $OUT"; continue; fi
  python qgate_stage1.py --task cube --h5 "$H5" --lambda-sparse "$L" \
    --rank infonce --neg-k 255 --batch 1024 \
    --max-episodes "$N" --data-seed "$DS" --out "$SSD/$OUT"
  gcloud storage cp "$SSD/$OUT" "$BUCKET/qgate_dataeff/$OUT"
done
echo "DATAEFF DONE N=$N seed=$DS"
