#!/usr/bin/env bash
# Generate + pre-register the episode sets for cube_double / scene evals.
#   usage: ray_gen_ogbmulti_sets.sh <task> [seeds...]   (default seeds 101-106)
# env_seed_base follows the cube convention 30000 + (S-101)*10000.
set -euo pipefail
TASK="${1:?task}"; shift || true
SEEDS=("${@:-}"); [ -z "${SEEDS[0]:-}" ] && SEEDS=(101 102 103 104 105 106)
BUCKET=gs://prism-training-us/le-wm
case "$TASK" in
  cube_double) LANCE=cube_double_play.lance ;;
  scene)       LANCE=scene_play.lance ;;
  cube_triple)    LANCE=cube_triple_play.lance ;;
  cube_quadruple) LANCE=cube_quadruple_play.lance ;;
  puzzle_3x3)     LANCE=puzzle_3x3_play.lance ;;
  *) echo "unknown task $TASK" >&2; exit 1 ;;
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
DS="$STABLEWM_HOME/datasets/ogbench"
mkdir -p "$DS" "$SSD/eps"
if ! command -v uv >/dev/null; then
  pip install -q uv
  PATH="$(python3 -m site --user-base)/bin:$(python3 -c 'import sysconfig;print(sysconfig.get_path("scripts"))'):$PATH"
  export PATH; hash -r
fi
if [ ! -x "$SSD/.venv/bin/python" ]; then uv venv --python=3.10 "$SSD/.venv"; fi
source "$SSD/.venv/bin/activate"
uv pip install -q 'stable-worldmodel[train,format]'
echo "[data] rsync $LANCE"; time gcloud storage rsync -r "$BUCKET/datasets/ogbench/$LANCE" "$DS/$LANCE"

for S in "${SEEDS[@]}"; do
  OUT="episodes_${TASK}_s${S}_100.json"
  if gcloud storage ls "$BUCKET/eval_sets/$OUT" >/dev/null 2>&1; then
    echo "[skip] $OUT exists"; continue
  fi
  python scripts/gen_episodes_ogbmulti.py --task "$TASK" --lance "$DS/$LANCE" \
    --num 100 --seed "$S" --env-seed-base $(( 30000 + (S - 101) * 10000 )) \
    --out "$SSD/eps/$OUT"
  gcloud storage cp "$SSD/eps/$OUT" "$BUCKET/eval_sets/"
done
echo "GEN DONE $TASK"
