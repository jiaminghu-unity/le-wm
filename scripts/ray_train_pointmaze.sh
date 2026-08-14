#!/usr/bin/env bash
# Train one two-room arm. two-room is the fourth LeWM environment and the only one this
# study had not covered.
#   usage: ray_train_pointmaze.sh <base|obj|aux>
#
# Isolation, same as the reduced-q round:
#   * training runs through train_pointmaze.py, a new file; train.py, utils.py, lewm.yaml
#     and every existing experiment config are untouched on disk
#   * the new q variant lives in q_pointmaze.py and is merged into utils.Q_VARIANTS
#     in-process, adding one key
#   * checkpoints go to $BUCKET/ckpts_pointmaze/ under lewm_p*_pointmaze names
#   * q_stats go to $BUCKET/qstats_pointmaze/; the existing qstats/ files are never touched
#
# q = pos_agent, 2-d. Only the agent moves in two-room; pos_target and the door centres
# are per-episode configuration, and pos_target IS the goal, so including it would hand
# the loss the success criterion. This is the smallest and cleanest q in the study: plain
# Euclidean displacement, no periodic coordinate, no mostly-static object.
#
# Dataset is lance (JPEG q95), matching what the other three tasks trained on.
set -euo pipefail

ARM="${1:?usage: ray_train_pointmaze.sh <base|obj|aux>}"
case "$ARM" in base|obj|aux) ;; *) echo "arm must be base, obj or aux" >&2; exit 1 ;; esac
TASK=pointmaze
BUCKET=gs://prism-training-us/le-wm
OUTP="$BUCKET/ckpts_pointmaze"

DSNAME=pointmaze.lance
QVAR=pointmaze_pos
case "$ARM" in
  base) EXP=p1_pointmaze_baseline ;;
  obj)  EXP=p2_pointmaze_obj ;;
  aux)  EXP=p5_pointmaze_qhead ;;
  *) echo "arm must be base|obj|aux" >&2; exit 1 ;;
esac
# RUN is RESOLVED from the composed config, never hardcoded. Hardcoding it is exactly what
# made the two-room chain resubmit ten trainings where three were needed: the configs were
# renamed to carry the loss weight and the chain kept looking for the old name while every
# job succeeded. Resolved below, after the venv exists.
RUN=""
QSTATS="${DSNAME}.q_stats.${QVAR}.json"
EXPECT_DIM=2

SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets"
mkdir -p "$DS"
echo "[env] pointmaze/$ARM  q=$QVAR (${EXPECT_DIM}d) on $(hostname), free=$(df -h --output=avail "$SSD" | tail -1 | tr -d ' ')"

sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd libgl1 libglib2.0-0 libxcb1 \
  libsm6 libxext6 libxrender1
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets
python -c "import torch; print('[torch]', torch.__version__, 'cuda', torch.cuda.is_available())"

RUN=$(python train_pointmaze.py --cfg job --resolve "experiment=$EXP" 2>/dev/null \
      | grep -E "^output_model_name:" | awk '{print $2}' | tr -d '\r')
[ -n "$RUN" ] || { echo "FATAL: could not resolve output_model_name for $EXP" >&2; exit 1; }
echo "[run] experiment=$EXP  ->  $RUN"

# ---- dataset (lance) ----
if [ ! -d "$DS/$DSNAME" ]; then
  echo "[data] fetching $DSNAME"
  mkdir -p "$(dirname "$DS/$DSNAME")"
  time gcloud storage rsync -r "$BUCKET/datasets/$DSNAME" "$DS/$DSNAME"
fi
du -sh "$DS/$DSNAME"

LOG="$SSD/train_$RUN.log"
: > "$LOG"

# ---- q_stats ----
# Reuse if a sibling arm already published them, so all three arms normalise q with the
# same numbers (the protocol the original rounds followed via make_cube_qstats.py).
# Unlike the reduced-q round there is no full variant to check against, so the guard here
# is the dimension plus a range sanity check: pos_agent lives in the env's pixel
# coordinates and the schema probe measured [14.0, 209.0] on both axes.
mkdir -p "$(dirname "$DS/$QSTATS")"
if gcloud storage cp "$BUCKET/qstats_pointmaze/$QSTATS" "$DS/$QSTATS" 2>/dev/null; then
  echo "[stats] reused $BUCKET/qstats_pointmaze/$QSTATS" | tee -a "$LOG"
else
  echo "[stats] computing $QVAR" | tee -a "$LOG"
  python - "$DSNAME" "$QVAR" <<'PY' 2>&1 | tee -a "$LOG"
import sys
from pathlib import Path

import numpy as np

import stable_worldmodel as swm

sys.path.insert(0, ".")
import q_pointmaze  # noqa: E402
import utils  # noqa: E402

utils.Q_VARIANTS.update(q_pointmaze.Q_VARIANTS_POINTMAZE)
name, variant = sys.argv[1], sys.argv[2]
ds_dir = Path(swm.data.utils.get_cache_dir(None, sub_folder="datasets"))
out = ds_dir / f"{name}.q_stats.{variant}.json"
dataset = swm.data.load_dataset(name, transform=None, cache_dir=None)
utils.get_q_normalizer(dataset, out, variant)
import json  # noqa: E402
st = json.loads(out.read_text())
print(f"[stats] dim={len(st['mean'])} mean={np.round(st['mean'], 2).tolist()} "
      f"std={np.round(st['std'], 2).tolist()}")
assert len(st["mean"]) == 2, f"FATAL: q dim {len(st['mean'])}, expected 2"
# PointMaze coordinates, not two-room pixels. Measured over all 200,000 frames:
# x spans [0.342, 3.250] and y [0.344, 3.257], so the mean must sit well inside
# that box. The bound this replaced was inherited from two-room (10 < m < 215, a
# 224-pixel image) and rejected a perfectly correct pointmaze q.
assert all(0.3 < m < 3.3 for m in st["mean"]), (
    f"FATAL: mean {st['mean']} outside the maze bounds [0.34, 3.26]")
# and the scale must be position, not velocity: velocity spans +-6.1 and would give
# a std near 1.7-1.9 with a mean near zero
assert all(0.5 < sd < 1.5 for sd in st["std"]), (
    f"FATAL: std {st['std']} does not look like position (expected ~0.96-1.01)")
print("QSTATS OK")
PY
  grep -q "^QSTATS OK" "$LOG" || { echo "FATAL: q_stats verification failed" >&2; exit 1; }
  gcloud storage cp "$DS/$QSTATS" "$BUCKET/qstats_pointmaze/$QSTATS"
fi

# ---- train ----
export HYDRA_FULL_ERROR=1
echo "[train] experiment=$EXP -> $RUN" | tee -a "$LOG"
set +e
python train_pointmaze.py "experiment=$EXP" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "[train] exit $rc" | tee -a "$LOG"

# The line train_half.py prints when the reduced variants are in the registry. Its
# absence means the run used a full-q variant and is NOT the ablation it claims to be.
grep -q "\[q_pointmaze\] registered" "$LOG" \
  || { echo "FATAL: the two-room q variant was never registered" >&2; rc=1; }

CKDIR="$STABLEWM_HOME/checkpoints/$RUN"
if [ "$rc" = 0 ] && [ -f "$CKDIR/weights_epoch_10.pt" ]; then
  echo "[upload] $RUN -> $OUTP/$RUN/"
  gcloud storage cp "$CKDIR/weights_epoch_10.pt" "$OUTP/$RUN/"
  gcloud storage cp "$CKDIR/config.json" "$OUTP/$RUN/" || true
fi
gcloud storage cp "$LOG" "$OUTP/logs/" || true
echo "[done] rc=$rc  $RUN"
exit $rc
