#!/usr/bin/env bash
# Convert a task's h5 dataset to lance and stage it in GCS.
#   usage: ray_convert_lance.sh <pusht|reacher>
#
# The cube lance already lives in GCS (scripts/ray_convert_cube_lance.sh put it
# there); Push-T's and Reacher's never did — they were built on a worker's ephemeral
# disk and lost. The q_stats filenames in artifacts/ (pusht_expert_train.lance.*,
# reacher.lance.*) confirm the original encoders were trained on lance, and lance
# stores one JPEG blob per frame while h5 stores raw uint8, so the two formats do NOT
# hold identical pixels. Training the frozen-encoder ablation on h5 would feed those
# encoders inputs from a different distribution than they were trained on, which
# would leave the cross-arm comparison intact but contaminate the per-arm
# delta = orig - frozen term. Hence: same format, same JPEG quality (writer default
# 95, exactly as the cube conversion used).
set -euo pipefail

TASK="${1:?usage: ray_convert_lance.sh <pusht|reacher>}"
BUCKET=gs://prism-training-us/le-wm
case "$TASK" in
  pusht)   H5NAME=pusht_expert_train.h5; LANCE=pusht_expert_train.lance ;;
  reacher) H5NAME=reacher.h5;            LANCE=reacher.lance ;;
  *) echo "unknown task $TASK (pusht|reacher)" >&2; exit 1 ;;
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

H5="$DS/$H5NAME"
[ -f "$H5" ] || { echo "[data] fetching $H5NAME"; time gcloud storage cp "$BUCKET/datasets/$H5NAME" "$H5"; }
ls -la "$H5"

OUT="$DS/$LANCE"
if [ -d "$OUT" ]; then
  echo "[convert] $LANCE already present locally, skipping conversion"
else
  echo "[convert] $H5NAME -> $LANCE (JPEG q95, writer default)"
  time python - <<PY
import hdf5plugin  # noqa: F401  -- blosc filter for the source h5
from stable_worldmodel.data import convert
convert("$H5", "$OUT", dest_format="lance")
PY
fi
du -sh "$OUT"

echo "[verify] load a window through the same path training uses"
python - <<PY
import stable_worldmodel as swm
ds = swm.data.load_dataset("$LANCE", keys_to_load=["pixels", "action"],
                           num_steps=4, frameskip=5)
row = ds.get_row_data([0])
import numpy as np
px = np.asarray(row["pixels"])
print("  rows:", len(ds), " pixels dtype:", px.dtype, " shape:", getattr(px, "shape", None))
assert len(ds) > 0
PY

echo "[upload] $LANCE -> GCS"
time gcloud storage rsync -r "$OUT" "$BUCKET/datasets/$LANCE"
echo "LANCE STAGED -> $BUCKET/datasets/$LANCE"
