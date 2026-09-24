#!/usr/bin/env bash
set -e
SSD=/mnt/disks/ssd0
mountpoint -q "$SSD" || { dev=""; for d in $(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1}'); do [ -z "$(lsblk -no MOUNTPOINT "$d" | tr -d '[:space:]')" ] || continue; dev="$d"; break; done; sudo mkfs.ext4 -F -q "$dev"; sudo mkdir -p "$SSD"; sudo mount "$dev" "$SSD"; sudo chmod a+w "$SSD"; }
sudo apt-get install -y -q swig build-essential libgl1 libegl1 zstd >/dev/null 2>&1 || true
command -v uv >/dev/null || { pip install -q uv; PATH="$(python3 -m site --user-base)/bin:$PATH"; }
[ -x "$SSD/.venv/bin/python" ] || uv venv --python=3.10 "$SSD/.venv"
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]' hdf5plugin pillow
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets/ogbench"; mkdir -p "$DS"
B=gs://prism-training-us/le-wm
for d in cube_double_play cube_triple_play cube_quadruple_play scene_play; do
  [ -d "$DS/$d.lance" ] || gcloud storage rsync -r "$B/datasets/ogbench/$d.lance" "$DS/$d.lance"
done
for t in cube_double cube_triple cube_quadruple scene; do
  gcloud storage cp "$B/eval_sets/episodes_${t}_s101_100.json" "/tmp/eps_${t}.json"
done
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python scripts/probe_ogbmulti_fidelity.py
