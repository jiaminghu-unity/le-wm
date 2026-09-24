#!/usr/bin/env bash
set -e
SSD=/mnt/disks/ssd0
mountpoint -q "$SSD" || { dev=""; for d in $(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1}'); do [ -z "$(lsblk -no MOUNTPOINT "$d" | tr -d '[:space:]')" ] || continue; dev="$d"; break; done; sudo mkfs.ext4 -F -q "$dev"; sudo mkdir -p "$SSD"; sudo mount "$dev" "$SSD"; sudo chmod a+w "$SSD"; }
sudo apt-get install -y -q swig build-essential libgl1 libegl1 >/dev/null 2>&1 || true
command -v uv >/dev/null || { pip install -q uv; PATH="$(python3 -m site --user-base)/bin:$PATH"; }
# 独立 venv,避免污染共享环境
uv venv --python=3.10 "$SSD/.venv_r311"
source "$SSD/.venv_r311/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]' hdf5plugin
uv pip install -q 'mujoco==3.11.0' 'dm_control==1.0.44'
export STABLEWM_HOME="$SSD/stable-wm"; mkdir -p "$STABLEWM_HOME/datasets"
[ -f "$STABLEWM_HOME/datasets/reacher.h5" ] || gcloud storage cp gs://prism-training-us/le-wm/datasets/reacher.h5 "$STABLEWM_HOME/datasets/reacher.h5"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python -c "import mujoco,dm_control;print('mujoco',mujoco.__version__)"
python scripts/check_render_fidelity.py reacher 8 --max-mae 3.0 && echo "PASS_AT mj3.11+dmc1.0.44" || echo "FAIL_AT mj3.11+dmc1.0.44"
