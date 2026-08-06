#!/usr/bin/env bash
set -euo pipefail
BUCKET=gs://prism-training-us/le-wm
SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"; sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets/ogbench"; mkdir -p "$DS"
sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd libgl1 libglib2.0-0 libxcb1 libsm6 \
  libxext6 libxrender1 libegl1 libegl-mesa0 libgles2 libglvnd0 libopengl0 libosmesa6 libosmesa6-dev
sudo apt-get install -y -q libnvidia-gl-580-server || true
sudo usermod -aG render "$(id -un)" 2>/dev/null || true
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
[ -x "$SSD/.venv/bin/python" ] || uv venv --python=3.10 "$SSD/.venv"
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U pillow
RH5="$STABLEWM_HOME/datasets/reacher.h5"
[ -f "$RH5" ] || gcloud storage cp "$BUCKET/datasets/reacher.h5" "$RH5"
H5="$DS/cube_single_expert.h5"
[ -f "$H5" ] || { time gcloud storage cat "$BUCKET/datasets/ogbench/cube_single_expert.tar.zst" \
  | zstd -dc --long=31 | tar -xf - -C "$DS"; }
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
OUT="$SSD/diag_p5_frames.log"; : > "$OUT"
for t in cube reacher; do
  echo "########## $t ##########" | tee -a "$OUT"
  python scripts/diag_p5_frames.py "$t" 3 4 2>&1 | tee -a "$OUT"
done
gcloud storage cp "$OUT" "$BUCKET/eval/"
echo "DIAG DONE"
