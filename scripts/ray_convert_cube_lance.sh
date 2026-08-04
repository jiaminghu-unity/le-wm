#!/usr/bin/env bash
# One-time: convert the cube h5 -> lance (the format upstream uses for Push-T and
# the format this team already used for Push-T and Reacher), then stage to GCS.
#
# Why: the shipped h5 chunks pixels at (100,224,224,3) = 15 MB/chunk, so pulling a
# 4-frame clip decompresses 100 frames — ~25x read amplification that pinned all 6
# dataloader workers at 100% CPU and starved the GPU to ~27% util (2.0 it/s vs the
# 5.7 it/s Push-T reaches on lance). Lance stores one JPEG blob per frame, so there
# is no amplification, and decode goes through torchvision's libjpeg-turbo path.
#
# JPEG quality stays at the writer default (95) — same as the Push-T and Reacher
# rounds, so the three tasks share one pipeline. Declared deviation: upstream
# ogb.yaml specifies h5 for cube.
set -euo pipefail

BUCKET=gs://prism-training-us/le-wm
SRC="$BUCKET/datasets/ogbench/cube_single_expert.tar.zst"
EXPECT_H5_SIZE=101942558720

SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets/ogbench"
mkdir -p "$DS"
echo "[env] host=$(hostname) free=$(df -h --output=avail "$SSD" | tail -1 | tr -d ' ')"

sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd \
  libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1

if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
# reuse an existing venv (a previous job on this worker may have built one)
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets

H5="$DS/cube_single_expert.h5"
if [ ! -f "$H5" ] || [ "$(stat -c%s "$H5")" != "$EXPECT_H5_SIZE" ]; then
  echo "[data] streaming 46 GB -> 101.9 GB h5"
  time gcloud storage cat "$SRC" | zstd -dc --long=31 | tar -xf - -C "$DS"
fi
[ "$(stat -c%s "$H5")" = "$EXPECT_H5_SIZE" ] || { echo "FATAL: h5 size mismatch" >&2; exit 1; }

LANCE="$DS/cube_single_expert.lance"
echo "[convert] h5 -> lance (JPEG q95, writer default)"
time python - <<PY
import hdf5plugin  # noqa: F401  -- blosc filter for the source h5
from stable_worldmodel.data import convert
convert("$H5", "$LANCE", dest_format="lance")
PY

echo "[verify] lance schema + a sample read"
python - <<PY
import stable_worldmodel as swm
ds = swm.data.load_dataset(
    "ogbench/cube_single_expert.lance",
    keys_to_load=["pixels", "action", "proprio_effector_pos", "proprio_effector_yaw",
                  "proprio_gripper_opening", "proprio_joint_pos",
                  "privileged_block_0_pos"],
    num_steps=4, frameskip=5,
)
print("len(clips) =", len(ds))
s = ds[0]
for k, v in s.items():
    print(f"  {k:28s} {tuple(getattr(v, 'shape', ()))} {getattr(v, 'dtype', type(v))}")
PY
du -sh "$LANCE"

# q_stats: the non-pixel columns survive the conversion losslessly, so reuse the
# already-validated frozen values verbatim — only the filename prefix changes
# (train.py derives it from the dataset name, which is now *.lance).
for v in cube_effector cube_plus_joints; do
  gcloud storage cp "$BUCKET/qstats/cube_single_expert.h5.q_stats.$v.json" \
                    "$BUCKET/qstats/cube_single_expert.lance.q_stats.$v.json"
done
gcloud storage ls "$BUCKET/qstats/"

echo "[upload] lance -> GCS"
time gcloud storage rsync -r "$LANCE" "$BUCKET/datasets/ogbench/cube_single_expert.lance"
echo "LANCE STAGED -> $BUCKET/datasets/ogbench/cube_single_expert.lance"
