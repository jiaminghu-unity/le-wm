#!/usr/bin/env bash
# Prepare one OGBench multi-object dataset end to end on a GPU worker:
# SELF-COLLECT at 224x224 with swm World + OGBench's official oracle (the Berkeley
# npz host is down behind an infra-incident redirect, and self-collection is how
# cube_single_expert was evidently produced) -> lance -> h5 -> GCS.
#   usage: ray_ogb_prep.sh <cube_double|cube_triple|cube_quadruple|scene>
# New GCS names only (datasets/ogbench/<task>_play.{h5,lance}); nothing existing
# is touched. EGL rendering, same env stack as the eval workers.
set -euo pipefail

TASK="${1:?cube_double|cube_triple|cube_quadruple|scene|puzzle_3x3}"
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
DS="$STABLEWM_HOME/datasets/ogbench"; mkdir -p "$DS"

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
  if MUJOCO_GL=$g PYOPENGL_PLATFORM=$g python - <<'GLPROBE' >/dev/null 2>&1
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 64, 64); r.update_scene(mujoco.MjData(m)); assert r.render().shape == (64, 64, 3)
GLPROBE
  then GL=$g; break; fi
done
export MUJOCO_GL="$GL" PYOPENGL_PLATFORM="$GL"
echo "[gl] $GL"

H5="$DS/${OUTNAME}.h5"
LANCE="$DS/${OUTNAME}.lance"

# ---- disk hygiene: this job needs ~40 GB headroom at lance finalize ----
echo "[disk] before cleanup:"; df -h "$SSD" | tail -1
du -sh "$STABLEWM_HOME/datasets"/* "$DS"/* 2>/dev/null | sort -h | tail -8 || true
rm -rf "$DS"/smoke_*.lance "$DS"/*_play.h5 "$DS"/*_play.h5.tmp
# stale outputs of OTHER prep tasks that never got uploaded stay (chain will finish
# them); only re-claim big refetchable inputs when tight:
FREE_GB=$(df -BG --output=avail "$SSD" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 80 ]; then
  echo "[disk] tight ($FREE_GB GB) — dropping refetchable datasets not needed here"
  rm -f "$STABLEWM_HOME/datasets/pusht_expert_train.h5" "$STABLEWM_HOME/datasets/reacher.h5"
  rm -rf "$STABLEWM_HOME/datasets/pusht_expert_train.lance" "$STABLEWM_HOME/datasets/reacher.lance" \
         "$DS/cube_single_expert.lance"
fi
FREE_GB=$(df -BG --output=avail "$SSD" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 80 ]; then
  echo "[disk] still tight ($FREE_GB GB) — dropping the 102GB cube_single h5 (refetchable)"
  rm -f "$DS/cube_single_expert.h5"
fi
echo "[disk] after cleanup:"; df -h "$SSD" | tail -1

echo "[smoke] $TASK (self-collection via OGBench oracle)"
rm -rf "$DS/smoke_${TASK}.lance"
python scripts/ogb_collect_multiobj.py "$TASK" --out "$DS/smoke_${TASK}.lance" --smoke

echo "[collect] $TASK full"
rm -rf "$LANCE"
time python scripts/ogb_collect_multiobj.py "$TASK" --out "$LANCE"

# h5 conversion deferred: raw-uint8 h5 is ~60GB/task and blew ENOSPC next to the
# 102GB cube_single h5; training reads lance only. Convert on a clean worker when
# the eval side for these tasks is wired.
rm -f "$H5"

echo "[upload]"
gcloud storage rsync -r "$LANCE" "$BUCKET/datasets/ogbench/${OUTNAME}.lance"
: # h5 upload deferred (see note above)
echo "OGB PREP DONE $TASK"
