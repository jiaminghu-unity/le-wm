#!/usr/bin/env bash
set -euo pipefail
TASK="${1:?task}"
BUCKET=gs://prism-training-us/le-wm
case "$TASK" in
  pusht)   H5=pusht_expert_train.h5; SRC="$BUCKET/datasets/pusht_expert_train.h5"; SUB="" ;;
  reacher) H5=reacher.h5;            SRC="$BUCKET/datasets/reacher.h5";            SUB="" ;;
  cube)    H5=cube_single_expert.h5; SRC="$BUCKET/datasets/ogbench/cube_single_expert.tar.zst"; SUB="ogbench" ;;
esac
SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"; sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets${SUB:+/$SUB}"; mkdir -p "$DS"
sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd libgl1 libglib2.0-0 libxcb1 libsm6 \
  libxext6 libxrender1 libegl1 libegl-mesa0 libgles2 libglvnd0 libopengl0 libosmesa6 libosmesa6-dev
sudo apt-get install -y -q libnvidia-gl-580-server || true
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
[ -x "$SSD/.venv/bin/python" ] || uv venv --python=3.10 "$SSD/.venv"
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets scikit-learn
if [ ! -f "$DS/$H5" ]; then
  if [ "$TASK" = cube ]; then gcloud storage cat "$SRC" | zstd -dc --long=31 | tar -xf - -C "$DS"
  else gcloud storage cp "$SRC" "$DS/$H5"; fi
fi
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python scripts/introspect_env.py "$TASK" 2>&1 | tee "$SSD/introspect_$TASK.log"
gcloud storage cp "$SSD/introspect_$TASK.log" "$BUCKET/eval/"
