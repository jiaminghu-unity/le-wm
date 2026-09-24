#!/usr/bin/env bash
set -e
SSD=/mnt/disks/ssd0
mountpoint -q "$SSD" || { dev=""; for d in $(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1}'); do [ -z "$(lsblk -no MOUNTPOINT "$d" | tr -d '[:space:]')" ] || continue; dev="$d"; break; done; sudo mkfs.ext4 -F -q "$dev"; sudo mkdir -p "$SSD"; sudo mount "$dev" "$SSD"; sudo chmod a+w "$SSD"; }
sudo apt-get install -y -q swig build-essential libgl1 libegl1 >/dev/null 2>&1 || true
command -v uv >/dev/null || { pip install -q uv; PATH="$(python3 -m site --user-base)/bin:$PATH"; }
[ -x "$SSD/.venv/bin/python" ] || uv venv --python=3.10 "$SSD/.venv"
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]' hdf5plugin
export STABLEWM_HOME="$SSD/stable-wm"; mkdir -p "$STABLEWM_HOME/datasets"
[ -f "$STABLEWM_HOME/datasets/reacher.h5" ] || gcloud storage cp gs://prism-training-us/le-wm/datasets/reacher.h5 "$STABLEWM_HOME/datasets/reacher.h5"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
for v in 3.12.0 3.11.0 3.10.0 3.9.0; do
  echo "=== trying mujoco==$v"
  uv pip install -q "mujoco==$v" || { echo "install failed $v"; continue; }
  python scripts/check_render_fidelity.py reacher 8 --max-mae 3.0 && { echo "PASS_AT mujoco==$v"; exit 0; }
done
echo "NO VERSION PASSED"; exit 1
