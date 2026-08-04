#!/usr/bin/env bash
# viz_pca_angle-style panels for Reacher and Cube. Reuses the cached z, so the GPU
# is only needed for the dataset mount/pull, not for encoding.
set -euo pipefail
BUCKET=gs://prism-training-us/le-wm
SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"; sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"; DS="$STABLEWM_HOME/datasets"; mkdir -p "$DS/ogbench"
sudo apt-get update -q
sudo apt-get install -y -q swig build-essential zstd libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"; export PATH; hash -r
fi
[ -x "$SSD/.venv/bin/python" ] || uv venv --python=3.10 "$SSD/.venv"
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]'
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -q hdf5plugin -U datasets scipy matplotlib
[ -f "$DS/reacher.h5" ] || gcloud storage cp "$BUCKET/datasets/reacher.h5" "$DS/"
[ -d "$DS/ogbench/cube_single_expert.lance" ] || \
  gcloud storage rsync -r "$BUCKET/datasets/ogbench/cube_single_expert.lance" "$DS/ogbench/cube_single_expert.lance"
mkdir -p eval_results
gcloud storage cp "$BUCKET/eval/pca_cache.npz" eval_results/
python scripts/visualize_pca_physical.py 2>&1 | tee "$SSD/pca_phys.log"
gcloud storage cp eval_results/viz_pca_physical_reacher.png eval_results/viz_pca_physical_cube.png "$BUCKET/eval/"
gcloud storage cp "$SSD/pca_phys.log" "$BUCKET/eval/pca_phys.log"
echo "PCA PHYSICAL DONE"
