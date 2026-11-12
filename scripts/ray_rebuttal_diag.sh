#!/usr/bin/env bash
# Rebuttal diagnostics (rebuttal_diag.py) for a list of checkpoints of one task.
#   usage: ray_rebuttal_diag.sh <task> <ckpt-prefix:run> [<ckpt-prefix:run> ...]
# Skips runs whose JSON is already in GCS. Provisioning as ray_probe_rho.sh.
set -euo pipefail

TASK="${1:?pusht|cube|tworoom|pointmaze|reacher}"; shift
BUCKET=gs://prism-training-us/le-wm
OUT="$BUCKET/eval/rebuttal_diag"
DATA="$TASK.lance"
case "$TASK" in
  pusht) DATA="pusht_expert_train.lance" ;;
  cube)  DATA="ogbench/cube_single_expert.lance" ;;
  tworoom|pointmaze|reacher) ;;
  *) echo "FATAL: unknown task $TASK" >&2; exit 1 ;;
esac
CKS=""
for spec in "$@"; do
  ck="${spec##*:}"
  gcloud storage ls "$OUT/${TASK}__${ck}.json" >/dev/null 2>&1 && { echo "[skip] $ck"; continue; }
  CKS="$CKS $spec"
done
[ -n "$CKS" ] || { echo "nothing to do"; exit 0; }
SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=""; for d in $(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1}'); do [ -z "$(lsblk -no MOUNTPOINT "$d" | tr -d '[:space:]')" ] || continue; dev="$d"; break; done
  if [ -n "$dev" ]; then
    sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
    sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  else
    echo "[env] no free NVMe -> using boot-disk dir $SSD"
    sudo mkdir -p "$SSD"
  fi
  sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets"
mkdir -p "$DS" "$STABLEWM_HOME/checkpoints"
echo "[env] pcq/$TASK on $(hostname)"

sudo apt-get -o DPkg::Lock::Timeout=900 update -q
sudo apt-get -o DPkg::Lock::Timeout=900 install -y -q swig build-essential zstd \
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

RUNS=""; for spec in $CKS; do RUNS="$RUNS ${spec##*:}"; done
# shellcheck disable=SC2086
python scripts/rebuttal_diag.py "$TASK" $RUNS 2>&1 | tee "$SSD/rebuttal_diag_$TASK.log"
gcloud storage cp eval_results/rebuttal_diag/${TASK}__*.json "$OUT/"
echo "REBUTTAL DIAG $TASK DONE"
