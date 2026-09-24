#!/usr/bin/env bash
set -e
SSD=/mnt/disks/ssd0
mountpoint -q "$SSD" || { dev=""; for d in $(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1}'); do [ -z "$(lsblk -no MOUNTPOINT "$d" | tr -d '[:space:]')" ] || continue; dev="$d"; break; done; sudo mkfs.ext4 -F -q "$dev"; sudo mkdir -p "$SSD"; sudo mount "$dev" "$SSD"; sudo chmod a+w "$SSD"; }
sudo apt-get install -y -q swig build-essential zstd >/dev/null 2>&1 || true
command -v uv >/dev/null || { pip install -q uv; PATH="$(python3 -m site --user-base)/bin:$PATH"; }
[ -x "$SSD/.venv/bin/python" ] || uv venv --python=3.10 "$SSD/.venv"
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,env,format]' hdf5plugin
uv pip install -q 'torch==2.12.1+cu126' torchvision --index-url https://download.pytorch.org/whl/cu126 >/dev/null 2>&1 || true
export STABLEWM_HOME="$SSD/stable-wm"; DS="$STABLEWM_HOME/datasets/ogbench"; mkdir -p "$DS"
B=gs://prism-training-us/le-wm
[ -d "$DS/cube_single_expert.lance" ] || gcloud storage rsync -r "$B/datasets/ogbench/cube_single_expert.lance" "$DS/cube_single_expert.lance"
OUT=/tmp/timecost.txt; : > "$OUT"
run_one(){ # name entry experiment extra...
  local name=$1 entry=$2; shift 2
  echo "===== TIMING $name" | tee -a "$OUT"
  timeout 2400 python "$entry" "$@" "+trainer.limit_train_batches=300" "trainer.max_epochs=1" 2>&1 \
    | grep -oE "\([0-9.]+ it/s\)" | tail -5 | tee -a "$OUT"
}
run_one lewm_baseline  train_qnative.py experiment=k1_cube_baseline seed=3072
run_one lewm_scale     train_qnative.py experiment=k2_cube_obj_eff  seed=3072 loss.obj.weight=0.1
run_one lewm_aux       train_qnative.py experiment=k4_cube_qhead_eff seed=3072 loss.aux.weight=0.1
run_one dinowm         train_dinowm.py  experiment=dw_cube          seed=3072
gcloud storage cp "$OUT" gs://prism-training-us/le-wm/eval/timecost_3072.txt
echo TIMECOST-DONE
