#!/usr/bin/env bash
# PC-head -> q regression probe (the reverse arrow of the spectrum story).
#   usage: ray_probe_pcq.sh {tworoom|pointmaze|reacher}
# CPU-only: encoding 1500 frames is minutes, and the GPUs are running the dw sweeps.
# Reads ckpts prefixes read-only; writes only eval/pcq_<task>.{json,png,log}.
set -euo pipefail

TASK="${1:?pusht|cube|tworoom|pointmaze|reacher}"
BUCKET=gs://prism-training-us/le-wm

DATA="$TASK.lance"
case "$TASK" in
  pusht)
    DATA="pusht_expert_train.lance"
    CKS="ckpts:lewm_c1_s3072 ckpts:lewm_c3_sig_obj0.1_s3072 ckpts:lewm_c5_qhead0.3_s3072 ckpts:dinowm_pusht_s3072" ;;
  cube)
    DATA="ogbench/cube_single_expert.lance"
    CKS="ckpts:lewm_k1_cube_s3072 ckpts:lewm_k2_cube_obj_eff0.1_s3072 ckpts:lewm_k4_cube_qhead_eff0.1_s3072 ckpts:dinowm_cube_s3072" ;;
  tworoom)
    CKS="ckpts_tworoom:lewm_t1_tworoom_s3072 ckpts_tworoom:lewm_t2_tworoom_obj0.1_s3072 ckpts_tworoom:lewm_t5_tworoom_qhead0.1_s3072 ckpts_tworoom:dinowm_tworoom_s3072" ;;
  pointmaze)
    CKS="ckpts_pointmaze:lewm_p1_pointmaze_s3072 ckpts_pointmaze:lewm_p2_pointmaze_s3072 ckpts_pointmaze:lewm_p5_pointmaze_s3072 ckpts_pointmaze:dinowm_pointmaze_s3072" ;;
  reacher)
    CKS="ckpts:lewm_r1_reacher_s3072 ckpts:lewm_r2_reacher_paep_l015_s3072 ckpts_half:lewm_hq_obj_reacher_s3072 ckpts:lewm_r5_qhead0.4_s3072 ckpts:dinowm_reacher_s3072" ;;
  *) echo "FATAL: unknown task $TASK" >&2; exit 1 ;;
esac

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
mkdir -p "$DS" "$STABLEWM_HOME/checkpoints"
echo "[env] pcq/$TASK on $(hostname)"

sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd \
  libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets scikit-learn scipy matplotlib

LANCE="$DS/$DATA"
if [ ! -d "$LANCE" ]; then
  echo "[data] pulling $DATA"
  mkdir -p "$(dirname "$LANCE")"
  time gcloud storage rsync -r "$BUCKET/datasets/$DATA" "$LANCE"
fi

for spec in $CKS; do
  pfx="${spec%%:*}"; ck="${spec##*:}"
  mkdir -p "$STABLEWM_HOME/checkpoints/$ck"
  gcloud storage cp "$BUCKET/$pfx/$ck/weights_epoch_10.pt" "$STABLEWM_HOME/checkpoints/$ck/"
  gcloud storage cp "$BUCKET/$pfx/$ck/config.json" "$STABLEWM_HOME/checkpoints/$ck/" || true
done

python scripts/probe_pc_q.py "$TASK" 2>&1 | tee "$SSD/pcq_$TASK.log"

gcloud storage cp "eval_results/pcq_$TASK.json" "eval_results/pcq_$TASK.png" "$BUCKET/eval/"
gcloud storage cp "$SSD/pcq_$TASK.log" "$BUCKET/eval/"
echo "PCQ $TASK DONE"
