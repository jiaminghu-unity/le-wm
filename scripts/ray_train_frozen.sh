#!/usr/bin/env bash
# Frozen-encoder ablation: load one arm's encoder+projector, hold them fixed, and
# retrain the predictor from scratch on the prediction MSE alone.
#   usage: ray_train_frozen.sh <pusht|reacher|cube> <base|obj|aux>
#
# Training goes through train_frozen.py, a standalone entry point: train.py, its
# config and every existing checkpoint are untouched.
# Nothing here touches the original artefacts. Results land under a separate GCS
# prefix (ckpts_frozen/), the checkpoint names carry an lewm_fz_ prefix, and the
# experiment configs are new files — the existing ckpts/, final_eval/ and eval/
# prefixes are read-only from this script's point of view.
#
# Datasets are lance, matching what the original encoders were trained on. lance
# stores one JPEG blob per frame while h5 stores raw uint8, so the formats do not
# hold identical pixels; feeding a frozen encoder h5 pixels would leave the
# cross-arm comparison valid but contaminate the per-arm delta = orig - frozen term.
set -euo pipefail

TASK="${1:?usage: ray_train_frozen.sh <task> <arm>}"
ARM="${2:?usage: ray_train_frozen.sh <task> <arm>}"
BUCKET=gs://prism-training-us/le-wm
OUTP="$BUCKET/ckpts_frozen"

case "$TASK" in
  pusht)   LANCE=pusht_expert_train.lance; QSTAT=pusht_expert_train.lance.q_stats.pusht_state.json ;;
  reacher) LANCE=reacher.lance;            QSTAT=reacher.lance.q_stats.reacher_joints_only.json ;;
  cube)    LANCE=ogbench/cube_single_expert.lance
           QSTAT=cube_single_expert.lance.q_stats.cube_effector.json ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
esac
case "$TASK/$ARM" in
  pusht/base)   SRC=lewm_c1_s3072 ;;
  pusht/obj)    SRC=lewm_c3_sig_obj0.1_s3072 ;;
  pusht/aux)    SRC=lewm_c5_qhead0.3_s3072 ;;
  reacher/base) SRC=lewm_r1_reacher_s3072 ;;
  reacher/obj)  SRC=lewm_r2_reacher_paep_l015_s3072 ;;
  reacher/aux)  SRC=lewm_r5_qhead0.4_s3072 ;;
  cube/base)    SRC=lewm_k1_cube_s3072 ;;
  cube/obj)     SRC=lewm_k2_cube_obj_eff0.1_s3072 ;;
  cube/aux)     SRC=lewm_k4_cube_qhead_eff0.1_s3072 ;;
  *) echo "unknown arm $ARM for $TASK" >&2; exit 1 ;;
esac
RUN="lewm_fz_${ARM}_${TASK}_s3072"

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
mkdir -p "$DS" "$STABLEWM_HOME/checkpoints/$SRC"
echo "[env] $TASK/$ARM on $(hostname), free=$(df -h --output=avail "$SSD" | tail -1 | tr -d ' ')"

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
if [ ! -d "$DS/$LANCE" ]; then
  echo "[data] fetching $LANCE"
  mkdir -p "$(dirname "$DS/$LANCE")"
  time gcloud storage rsync -r "$BUCKET/datasets/$LANCE" "$DS/$LANCE"
fi
du -sh "$DS/$LANCE"

# ---- q normaliser stats: train.py builds the normaliser unconditionally, even with
# every loss weight at zero, and looks it up as <dataset>.q_stats.<variant>.json ----
[ -f "$DS/$QSTAT" ] || gcloud storage cp "$BUCKET/qstats/$QSTAT" "$DS/$QSTAT"

# ---- source checkpoint, read-only: this is the encoder we freeze ----
gcloud storage cp "$BUCKET/ckpts/$SRC/weights_epoch_10.pt" "$STABLEWM_HOME/checkpoints/$SRC/"
gcloud storage cp "$BUCKET/ckpts/$SRC/config.json" "$STABLEWM_HOME/checkpoints/$SRC/" || true

# ---- train ----
LOG="$SSD/train_$RUN.log"
echo "[train] experiment=fz_${ARM}_${TASK}  encoder<-$SRC" | tee "$LOG"
set +e
python train_frozen.py "experiment=fz_${ARM}_${TASK}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "[train] exit $rc" | tee -a "$LOG"

# The two lines train.py prints when the ablation is active. Their absence means the
# guarded block never ran, i.e. the run silently trained everything end to end and is
# NOT the ablation it claims to be.
grep -q "\[init\] encoder+projector <-" "$LOG" || { echo "FATAL: encoder was never loaded" >&2; rc=1; }
grep -q "\[freeze\] encoder+projector frozen" "$LOG" || { echo "FATAL: encoder was never frozen" >&2; rc=1; }

CKDIR="$STABLEWM_HOME/checkpoints/$RUN"
if [ "$rc" = 0 ] && [ -f "$CKDIR/weights_epoch_10.pt" ]; then
  echo "[upload] $RUN -> $OUTP/$RUN/"
  gcloud storage cp "$CKDIR/weights_epoch_10.pt" "$OUTP/$RUN/"
  gcloud storage cp "$CKDIR/config.json" "$OUTP/$RUN/" || true
fi
gcloud storage cp "$LOG" "$OUTP/logs/" || true
echo "[done] rc=$rc  $RUN"
exit $rc
