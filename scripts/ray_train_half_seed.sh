#!/usr/bin/env bash
# SEED-REPLICATION variant of ray_train_half.sh: s$SEED names + hydra seed override.
# Reduced-q ablation: retrain obj or aux end to end with roughly half of q withheld.
#   usage: ray_train_half.sh <pusht|reacher|cube> <obj|aux>
#
# Isolation, deliberately total:
#   * training runs through train_half.py, a new file; train.py, utils.py, lewm.yaml
#     and every existing experiment config are untouched on disk
#   * the new q variants live in q_half.py and are merged into utils.Q_VARIANTS
#     in-process, adding keys only
#   * checkpoints go to $BUCKET/ckpts_half/ under lewm_hq_* names
#   * q_stats go to $BUCKET/qstats_half/ under new per-variant filenames; the existing
#     qstats/ files are read (for the restriction check) and never written
#   * the baseline arm is absent on purpose: it never consumes q, so the existing
#     baseline checkpoints stay the correct comparison
#
# Datasets are lance, matching the original obj/aux runs. lance stores one JPEG blob
# per frame while h5 stores raw uint8, so the formats do not hold identical pixels and
# using h5 here would add a second difference to an experiment meant to have one.
set -euo pipefail

TASK="${1:?usage: ray_train_half.sh <task> <obj|aux>}"
ARM="${2:?usage: ray_train_half.sh <task> <obj|aux>}"
case "$ARM" in obj|aux) ;; *) echo "arm must be obj or aux (baseline uses no q)" >&2; exit 1 ;; esac
BUCKET=gs://prism-training-us/le-wm
OUTP="$BUCKET/ckpts_half"

case "$TASK" in
  pusht)   DSNAME=pusht_expert_train.lance
           HALFV=pusht_block_only;    FULLV=pusht_state ;;
  reacher) DSNAME=reacher.lance
           HALFV=reacher_joint0_only; FULLV=reacher_joints_only ;;
  cube)    DSNAME=ogbench/cube_single_expert.lance
           HALFV=cube_effector_only;  FULLV=cube_effector ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
# full-variant stats as staged by the original rounds: basename, no subdirectory
FULLSTATS="$(basename "$DSNAME").q_stats.${FULLV}.json"
HALFSTATS="$(basename "$DSNAME").q_stats.${HALFV}.json"
# ...but train.py derives its path from the dataset NAME, so a name with a directory
# component puts the file in that subdirectory. Put it exactly there or training
# silently recomputes instead of reading the verified file.
HALFSTATS_PATH="${DSNAME}.q_stats.${HALFV}.json"
RUN="lewm_hq_${ARM}_${TASK}_s${SEED:?SEED env var required}"
EXPECT_DIM=$(case "$TASK" in pusht) echo 4 ;; reacher) echo 2 ;; cube) echo 5 ;; esac)

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
echo "[env] $TASK/$ARM  q=$HALFV (${EXPECT_DIM}d) on $(hostname), free=$(df -h --output=avail "$SSD" | tail -1 | tr -d ' ')"

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

# ---- dataset (lance) ----
if [ ! -d "$DS/$DSNAME" ]; then
  echo "[data] fetching $DSNAME"
  mkdir -p "$(dirname "$DS/$DSNAME")"
  time gcloud storage rsync -r "$BUCKET/datasets/$DSNAME" "$DS/$DSNAME"
fi
du -sh "$DS/$DSNAME"

LOG="$SSD/train_$RUN.log"
: > "$LOG"

# ---- q_stats for the reduced variant ----
# Reuse if a sibling job already published them, so both arms of a task normalise q
# with byte-identical numbers (the protocol the original rounds followed via
# scripts/make_cube_qstats.py). Otherwise compute and verify, then publish.
mkdir -p "$(dirname "$DS/$HALFSTATS_PATH")"
if gcloud storage cp "$BUCKET/qstats_half/$HALFSTATS" "$DS/$HALFSTATS_PATH" 2>/dev/null; then
  echo "[stats] reused $BUCKET/qstats_half/$HALFSTATS" | tee -a "$LOG"
else
  gcloud storage cp "$BUCKET/qstats/$FULLSTATS" "$DS/$FULLSTATS"
  echo "[stats] computing $HALFV and checking it against $FULLV" | tee -a "$LOG"
  python scripts/prep_half_qstats.py "$DSNAME" "$HALFV" "$DS/$FULLSTATS" 2>&1 | tee -a "$LOG"
  grep -q "^QSTATS OK" "$LOG" || { echo "FATAL: q_stats verification failed" >&2; exit 1; }
  gcloud storage cp "$DS/$HALFSTATS_PATH" "$BUCKET/qstats_half/$HALFSTATS"
fi
# the dimension is the whole point of the ablation, so assert it rather than trust it
python - "$DS/$HALFSTATS_PATH" "$EXPECT_DIM" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
d = len(json.loads(open(sys.argv[1]).read())["mean"])
want = int(sys.argv[2])
assert d == want, f"FATAL: q_stats has dim {d}, expected {want}"
print(f"[check] q dim = {d}")
PY
grep -q "^\[check\] q dim = $EXPECT_DIM" "$LOG" || { echo "FATAL: wrong q dim" >&2; exit 1; }

# ---- train ----
export HYDRA_FULL_ERROR=1
echo "[train] experiment=hq_${ARM}_${TASK}" | tee -a "$LOG"
set +e
python train_half.py "experiment=hq_${ARM}_${TASK}" "seed=$SEED" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "[train] exit $rc" | tee -a "$LOG"

# The line train_half.py prints when the reduced variants are in the registry. Its
# absence means the run used a full-q variant and is NOT the ablation it claims to be.
grep -q "\[q_half\] registered reduced-q variants" "$LOG" \
  || { echo "FATAL: reduced-q variants were never registered" >&2; rc=1; }

CKDIR="$STABLEWM_HOME/checkpoints/$RUN"
if [ "$rc" = 0 ] && [ -f "$CKDIR/weights_epoch_10.pt" ]; then
  echo "[upload] $RUN -> $OUTP/$RUN/"
  gcloud storage cp "$CKDIR/weights_epoch_10.pt" "$OUTP/$RUN/"
  gcloud storage cp "$CKDIR/config.json" "$OUTP/$RUN/" || true
fi
gcloud storage cp "$LOG" "$OUTP/logs/" || true
echo "[done] rc=$rc  $RUN"
exit $rc
