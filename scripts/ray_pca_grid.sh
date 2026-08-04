#!/usr/bin/env bash
# 3x3 PCA panel over the nine finally-selected models. Needs all three datasets:
# Push-T + Reacher as h5, Cube as lance (no conversion just for a figure).
set -euo pipefail

BUCKET=gs://prism-training-us/le-wm
SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets"; mkdir -p "$DS/ogbench" "$STABLEWM_HOME/checkpoints"

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
uv pip install -q hdf5plugin -U datasets scipy matplotlib

echo "[data] pulling three datasets (~165 GB)"
[ -f "$DS/pusht_expert_train.h5" ] || gcloud storage cp "$BUCKET/datasets/pusht_expert_train.h5" "$DS/"
[ -f "$DS/reacher.h5" ]            || gcloud storage cp "$BUCKET/datasets/reacher.h5" "$DS/"
[ -d "$DS/ogbench/cube_single_expert.lance" ] || \
  gcloud storage rsync -r "$BUCKET/datasets/ogbench/cube_single_expert.lance" \
                          "$DS/ogbench/cube_single_expert.lance"
df -h --output=avail "$SSD" | tail -1

for ck in lewm_c1_s3072 lewm_c3_sig_obj0.1_s3072 lewm_c5_qhead0.3_s3072 \
          lewm_r1_reacher_s3072 lewm_r2_reacher_paep_l015_s3072 lewm_r5_qhead0.4_s3072 \
          lewm_k1_cube_s3072 lewm_k2_cube_obj_eff0.1_s3072 lewm_k4_cube_qhead_eff0.1_s3072; do
  mkdir -p "$STABLEWM_HOME/checkpoints/$ck"
  gcloud storage cp "$BUCKET/ckpts/$ck/weights_epoch_10.pt" "$STABLEWM_HOME/checkpoints/$ck/"
  gcloud storage cp "$BUCKET/ckpts/$ck/config.json" "$STABLEWM_HOME/checkpoints/$ck/" || true
done

export MUJOCO_GL=egl
python scripts/visualize_pca_grid.py 2>&1 | tee "$SSD/pca_grid.log"
gcloud storage cp eval_results/pca_cache.npz eval_results/viz_pca_grid_q.png eval_results/viz_pca_grid_progress.png eval_results/viz_latent_norms.png "$BUCKET/eval/"
gcloud storage cp "$SSD/pca_grid.log" "$BUCKET/eval/pca_grid.log"
echo "PCA GRID DONE -> $BUCKET/eval/viz_pca_grid.png"
