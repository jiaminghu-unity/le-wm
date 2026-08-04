#!/usr/bin/env bash
# One cube budget-sweep evaluation = one 1-GPU Ray job.
#   usage: ray_eval_cube.sh <config-name> <run-dir> <ckpt-subpath> <episodes-json>
# e.g.   ray_eval_cube.sh k2 k2_cube_obj_eff lewm_k2_cube_obj_eff0.1_s3072/weights_epoch_10.pt \
#            scripts/episodes_cube_50.json
#
# Eval needs the raw h5 (swm.data.HDF5Dataset is hardcoded in budget_sweep.py and
# eval.py alike) plus the full env stack (mujoco/ogbench + EGL), unlike training
# which reads lance.
set -euo pipefail

NAME="${1:?config name}"; RUN="${2:?run dir}"; CKPT="${3:?ckpt subpath}"; EPS="${4:?episodes json}"

BUCKET=gs://prism-training-us/le-wm
SRC_H5="$BUCKET/datasets/ogbench/cube_single_expert.tar.zst"
EXPECT_H5_SIZE=101942558720
TAG="$(basename "$EPS" .json)"
OUT_CSV="eval_${NAME}_${TAG}.csv"

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
mkdir -p "$DS" "$STABLEWM_HOME/checkpoints"
echo "[env] host=$(hostname) cfg=$NAME free=$(df -h --output=avail "$SSD" | tail -1 | tr -d ' ')"

sudo apt-get update -q
# The REPRODUCE.md list covers box2d/OpenCV (Push-T) and dm_control. Cube renders
# through mujoco + EGL, which additionally needs the GLVND/EGL stack; osmesa is
# installed too as a software fallback if EGL cannot bind to the NVIDIA vendor.
sudo apt-get install -y -q swig build-essential zstd \
  libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
  libegl1 libegl-mesa0 libgles2 libglvnd0 libopengl0 libosmesa6 libosmesa6-dev

if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets scikit-learn

# ---- data: eval reads the h5, not the lance ----
H5="$DS/cube_single_expert.h5"
if [ ! -f "$H5" ] || [ "$(stat -c%s "$H5")" != "$EXPECT_H5_SIZE" ]; then
  echo "[data] streaming 46 GB -> 101.9 GB h5"
  time gcloud storage cat "$SRC_H5" | zstd -dc --long=31 | tar -xf - -C "$DS"
fi
[ "$(stat -c%s "$H5")" = "$EXPECT_H5_SIZE" ] || { echo "FATAL: h5 size mismatch" >&2; exit 1; }

# ---- checkpoint ----
CKPT_DIR="$STABLEWM_HOME/checkpoints/$(dirname "$CKPT")"
mkdir -p "$CKPT_DIR"
gcloud storage cp "$BUCKET/runs/$RUN/checkpoints/$CKPT" "$CKPT_DIR/"
gcloud storage cp "$BUCKET/runs/$RUN/checkpoints/$(dirname "$CKPT")/config.json" "$CKPT_DIR/" || true
ls -la "$CKPT_DIR"

# Pick a working mujoco renderer: EGL (GPU, fast) if it binds, else osmesa (software).
pick_gl() {
  for gl in egl osmesa; do
    if MUJOCO_GL=$gl PYOPENGL_PLATFORM=$gl python - <<'PY' >/dev/null 2>&1
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 64, 64); r.update_scene(mujoco.MjData(m)); assert r.render().shape == (64, 64, 3)
PY
    then echo "$gl"; return; fi
  done
  echo "none"
}
GL=$(pick_gl)
[ "$GL" = "none" ] && { echo "FATAL: no working mujoco renderer (egl and osmesa both failed)" >&2; exit 1; }
export MUJOCO_GL="$GL" PYOPENGL_PLATFORM="$GL"
echo "[eval] renderer=$GL  budget_sweep --env cube --config $NAME $CKPT --tiers T1..T5"
set +e
python scripts/budget_sweep.py \
  --env cube --solver cem \
  --config "$NAME" "$CKPT" \
  --tiers T1 T2 T3 T4 T5 \
  --episodes-json "$EPS" \
  --out "$SSD/$OUT_CSV" 2>&1 | tee "$SSD/eval_$NAME.log"
RC=${PIPESTATUS[0]}
set -e

gcloud storage cp "$SSD/$OUT_CSV" "$BUCKET/eval/$OUT_CSV" || true
gcloud storage cp "$SSD/eval_$NAME.log" "$BUCKET/eval/eval_${NAME}_${TAG}.log" || true
echo "[done] rc=$RC -> $BUCKET/eval/$OUT_CSV"
exit $RC
