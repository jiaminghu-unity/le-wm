#!/usr/bin/env bash
# Representation-diagnostics panel for the cube round (k1/k2/k4/k7).
# CPU-only on purpose: every GPU is busy with the eval sweeps, and encoding 1500
# frames through ViT-Tiny is a couple of minutes on CPU.
set -euo pipefail

BUCKET=gs://prism-training-us/le-wm
SRC_LANCE="$BUCKET/datasets/ogbench/cube_single_expert.lance"

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
uv pip install -q 'torch==2.12.1+cu126' torchvision \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets scikit-learn scipy matplotlib

LANCE="$DS/cube_single_expert.lance"
if [ ! -d "$LANCE" ]; then
  echo "[data] pulling lance (~20 GB)"
  time gcloud storage rsync -r "$SRC_LANCE" "$LANCE"
fi

for spec in \
  "k1_cube_baseline lewm_k1_cube_s3072" \
  "k2_cube_obj_eff lewm_k2_cube_obj_eff0.1_s3072" \
  "k4_cube_qhead_eff lewm_k4_cube_qhead_eff0.1_s3072" \
  "k7_cube_obj_eff_l02 lewm_k7_cube_obj_eff0.2_s3072" ; do
  set -- $spec
  mkdir -p "$STABLEWM_HOME/checkpoints/$2"
  gcloud storage cp "$BUCKET/runs/$1/checkpoints/$2/weights_epoch_10.pt" \
                    "$STABLEWM_HOME/checkpoints/$2/"
  gcloud storage cp "$BUCKET/runs/$1/checkpoints/$2/config.json" \
                    "$STABLEWM_HOME/checkpoints/$2/" || true
done

mkdir -p eval_results
for spec in "k1 lewm_k1_cube_s3072" "k2 lewm_k2_cube_obj_eff0.1_s3072" \
            "k4 lewm_k4_cube_qhead_eff0.1_s3072" "k7 lewm_k7_cube_obj_eff0.2_s3072"; do
  set -- $spec
  echo "== probing $1"
  python scripts/probe.py --env cube --config "$1" "$2/weights_epoch_10.pt" \
    --out eval_results/probing_cube.csv 2>&1 | tail -12
done
gcloud storage cp eval_results/probing_cube.csv "$BUCKET/eval/"
echo "PROBING DONE -> $BUCKET/eval/probing_cube.csv"
