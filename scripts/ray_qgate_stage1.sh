#!/usr/bin/env bash
# Stage-1 q-gate discovery (behavior -> sparse g over simulator variables).
#   usage: ray_qgate_stage1.sh <task> [lambda ...]   (default lambdas: 0.003 0.01 0.03)
# Cheap: no images, no encoder; minutes per lambda on one GPU.
set -euo pipefail
TASK="${1:?task}"; shift || true
LAMBDAS=("${@:-}"); [ -z "${LAMBDAS[0]:-}" ] && LAMBDAS=(0.003 0.01 0.03)
BUCKET=gs://prism-training-us/le-wm
case "$TASK" in
  pusht) H5NAME=pusht_expert_train.h5; SRC="$BUCKET/datasets/pusht_expert_train.h5"; TAR=0 ;;
  cube)  H5NAME=cube_single_expert.h5; SRC="$BUCKET/datasets/ogbench/cube_single_expert.tar.zst"; TAR=1 ;;
  reacher) H5NAME=reacher.h5; SRC="$BUCKET/datasets/reacher.h5"; TAR=0 ;;
  reacher_novel) H5NAME=reacher.h5; SRC="$BUCKET/datasets/reacher.h5"; TAR=0 ;;
  cube_double) LANCE=cube_double_play.lance ;;
  scene)       LANCE=scene_play.lance ;;
  cube_triple)    LANCE=cube_triple_play.lance ;;
  cube_quadruple) LANCE=cube_quadruple_play.lance ;;
  puzzle_3x3)     LANCE=puzzle_3x3_play.lance ;;
  *) echo "task $TASK not wired yet" >&2; exit 1 ;;
esac
SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q torch numpy h5py hdf5plugin 2>/dev/null || uv pip install -q torch numpy h5py hdf5plugin
[ -n "${LANCE:-}" ] && uv pip install -q 'stable-worldmodel[format]' opencv-python-headless 
if [ -n "${LANCE:-}" ]; then
  H5="$SSD/stable-wm/datasets/ogbench/$LANCE"
  mkdir -p "$(dirname "$H5")"
  echo "[data] rsync $LANCE"
  time gcloud storage rsync -r "$BUCKET/datasets/ogbench/$LANCE" "$H5"
fi
H5="${H5:-$SSD/$H5NAME}"
if [ -z "${LANCE:-}" ] && [ ! -f "$H5" ]; then
  echo "[data] fetching $H5NAME"
  if [ "$TAR" = 1 ]; then
    sudo apt-get install -y -q zstd >/dev/null 2>&1 || true
    time gcloud storage cat "$SRC" | zstd -dc --long=31 | tar -xf - -C "$SSD"
    F=$(find "$SSD" -name "$H5NAME" 2>/dev/null | head -1 || true)
    [ -n "$F" ] && [ "$F" != "$H5" ] && mv "$F" "$H5"
    [ -f "$H5" ] || { echo "FATAL: $H5NAME not found after extract" >&2; exit 1; }
  else
    time gcloud storage cp "$SRC" "$H5"
  fi
fi
# QGATE_VARIANT: "" (默认 l1+hinge) | "l2" | "infonce" | "l2_infonce"
SFX=""; EXTRA=()
case "${QGATE_VARIANT:-}" in
  l2)         SFX="_l2";        EXTRA=(--reg l2) ;;
  infonce)    SFX="_nce";       EXTRA=(--rank infonce --neg-k 255 --batch 1024) ;;
  l2_infonce) SFX="_l2nce";     EXTRA=(--reg l2 --rank infonce --neg-k 255 --batch 1024) ;;
esac
for L in "${LAMBDAS[@]}"; do
  OUT="qgate_stage1_${TASK}${SFX}_lam${L}.json"
  if gcloud storage ls "$BUCKET/qgate/$OUT" >/dev/null 2>&1; then echo "[skip] $OUT"; continue; fi
  python qgate_stage1.py --task "$TASK" --h5 "$H5" --lambda-sparse "$L" "${EXTRA[@]}" --out "$SSD/$OUT"
  gcloud storage cp "$SSD/$OUT" "$BUCKET/qgate/$OUT"
done
echo "QGATE STAGE1 DONE $TASK"
