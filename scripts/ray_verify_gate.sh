#!/usr/bin/env bash
# Re-run the fixed render-fidelity gate on all three tasks and record the verdicts.
set -euo pipefail
BUCKET=gs://prism-training-us/le-wm
SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"; sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
mkdir -p "$STABLEWM_HOME/datasets/ogbench"
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
uv pip install -q hdf5plugin -U
for d in pusht_expert_train reacher; do
  f="$STABLEWM_HOME/datasets/$d.h5"; [ -f "$f" ] || gcloud storage cp "$BUCKET/datasets/$d.h5" "$f"
done
CH="$STABLEWM_HOME/datasets/ogbench/cube_single_expert.h5"
[ -f "$CH" ] || gcloud storage cat "$BUCKET/datasets/ogbench/cube_single_expert.tar.zst" \
  | zstd -dc --long=31 | tar -xf - -C "$STABLEWM_HOME/datasets/ogbench"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
OUT="$SSD/render_gate_all.log"; : > "$OUT"
for t in reacher cube pusht; do
  echo "########## $t ##########" | tee -a "$OUT"
  python scripts/check_render_fidelity.py "$t" 8 --max-mae 3.0 2>&1 | tee -a "$OUT" || \
    echo "  -> GATE FAILED for $t" | tee -a "$OUT"
done
gcloud storage cp "$OUT" "$BUCKET/eval/"
echo "GATE VERIFY DONE"
