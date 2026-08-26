#!/usr/bin/env bash
# Prepare one OGBench multi-object dataset end to end on a GPU worker:
# download state npz -> smoke (gates) -> replay-render 224x224 h5 -> lance -> GCS.
#   usage: ray_ogb_prep.sh <cube_double|cube_triple|cube_quadruple|scene>
# New GCS names only (datasets/ogbench/<task>_play.{h5,lance}); nothing existing
# is touched. EGL rendering, same env stack as the eval workers.
set -euo pipefail

TASK="${1:?cube_double|cube_triple|cube_quadruple|scene}"
BUCKET=gs://prism-training-us/le-wm
OUTNAME="${TASK}_play"

SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets/ogbench"; mkdir -p "$DS" "$SSD/ogb_raw"

sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd \
  libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
  libegl1 libegl-mesa0 libgles2 libglvnd0 libopengl0 libosmesa6 libosmesa6-dev
sudo apt-get install -y -q libnvidia-gl-580-server || true
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin ogbench -U datasets

# working GL backend (same probe as the eval launchers)
GL=egl
for g in egl osmesa; do
  if MUJOCO_GL=$g PYOPENGL_PLATFORM=$g python - <<'PY' >/dev/null 2>&1
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 64, 64); r.update_scene(mujoco.MjData(m)); assert r.render().shape == (64, 64, 3)
PY
  then GL=$g; break; fi
done
export MUJOCO_GL="$GL" PYOPENGL_PLATFORM="$GL"
echo "[gl] $GL"

H5="$DS/${OUTNAME}.h5"
LANCE="$DS/${OUTNAME}.lance"

echo "[smoke] $TASK"
python scripts/ogb_prep_multiobj.py "$TASK" --npz-dir "$SSD/ogb_raw" --out "$H5" --smoke

echo "[render] $TASK full"
time python scripts/ogb_prep_multiobj.py "$TASK" --npz-dir "$SSD/ogb_raw" --out "$H5"

echo "[convert] h5 -> lance"
time python - <<PY
import hdf5plugin  # noqa: F401
from stable_worldmodel.data import convert
convert("$H5", "$LANCE", dest_format="lance")
PY

echo "[verify] lance sample read"
python - <<PY
import stable_worldmodel as swm
ds = swm.data.load_dataset("ogbench/${OUTNAME}.lance", keys_to_load=["pixels", "action", "qpos"])
print("episodes:", len(ds.lengths), "frames:", int(sum(ds.lengths)))
row = ds.get_row_data([0, 1])
print("sample ok:", {k: getattr(v, "shape", type(v)) for k, v in row.items()})
PY

echo "[upload]"
gcloud storage cp "$H5" "$BUCKET/datasets/ogbench/${OUTNAME}.h5"
gcloud storage rsync -r "$LANCE" "$BUCKET/datasets/ogbench/${OUTNAME}.lance"
echo "OGB PREP DONE $TASK"
