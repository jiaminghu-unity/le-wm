#!/usr/bin/env bash
set -e
SSD=/mnt/disks/ssd0
mountpoint -q "$SSD" || { dev=""; for d in $(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1}'); do [ -z "$(lsblk -no MOUNTPOINT "$d" | tr -d '[:space:]')" ] || continue; dev="$d"; break; done; sudo mkfs.ext4 -F -q "$dev"; sudo mkdir -p "$SSD"; sudo mount "$dev" "$SSD"; sudo chmod a+w "$SSD"; }
command -v uv >/dev/null || { pip install -q uv; PATH="$(python3 -m site --user-base)/bin:$PATH"; }
[ -x "$SSD/.venv/bin/python" ] || uv venv --python=3.10 "$SSD/.venv"
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]' hdf5plugin pyopengl
export STABLEWM_HOME="$SSD/stable-wm"; mkdir -p "$STABLEWM_HOME/datasets"
[ -f "$STABLEWM_HOME/datasets/reacher.h5" ] || gcloud storage cp gs://prism-training-us/le-wm/datasets/reacher.h5 "$STABLEWM_HOME/datasets/reacher.h5"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python scripts/probe_reacher_render.py
python scripts/check_render_fidelity.py reacher 8 --max-mae 3.0 || true
