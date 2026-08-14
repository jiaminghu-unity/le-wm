#!/usr/bin/env bash
# two-room evaluation smoke test: generate the episode sets, check that the reconstructed
# scene matches the dataset frame, then run ONE cell.
#   usage: [GATE_ONLY=1] ray_smoke_tworoom.sh
#
# Deliberately one cell, not a sweep. Two-room's eval protocol has a genuine unknown the
# other three tasks do not have: its SCENE varies per episode (wall thickness/axis, door
# count and positions live in variation_space and are drawn by reset(seed)), while
# _set_state restores only the agent position. If the scene is not reproduced, every
# two-room number would be biased -- budget_sweep shows the planner the dataset frame at
# t=0 and then steps whatever room reset() built. check_render_tworoom.py measures exactly
# that and is FATAL here, so a broken protocol cannot quietly produce a full sweep.
set -euo pipefail

BUCKET=gs://prism-training-us/le-wm
SEEDS="101 102 103 104 105 106"

SSD=/mnt/disks/ssd0
if ! mountpoint -q "$SSD"; then
  dev=$(lsblk -dnpo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')
  [ -n "$dev" ] || { echo "FATAL: no local NVMe" >&2; exit 1; }
  sudo mkfs.ext4 -F -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
  sudo mkdir -p "$SSD" && sudo mount -o discard,defaults "$dev" "$SSD"
  sudo chmod a+w "$SSD"
fi
export STABLEWM_HOME="$SSD/stable-wm"
DS="$STABLEWM_HOME/datasets"; mkdir -p "$DS" "$SSD/eps"

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
uv pip install -q hdf5plugin -U datasets scikit-learn

LOG="$SSD/smoke_tworoom.log"; : > "$LOG"

H5="$DS/tworoom.h5"
[ -f "$H5" ] || { echo "[data] fetching tworoom.h5"; time gcloud storage cp "$BUCKET/datasets/tworoom.h5" "$H5"; }

# ---- episode sets, same convention as the other tasks ----
# env_seed_base = 40000 + (seed-101)*10000, so env resets never collide across sets, and
# budget_sweep's planner noise is crc32("episode_id|tier") so each set gets its own draws.
for S in $SEEDS; do
  OUT="$SSD/eps/episodes_tworoom_s${S}_100.json"
  if ! gcloud storage ls "$BUCKET/eval_sets/episodes_tworoom_s${S}_100.json" >/dev/null 2>&1; then
    python scripts/gen_episodes.py --num 100 --seed "$S" --dataset tworoom \
      --env-seed-base $(( 40000 + (S - 101) * 10000 )) --out "$OUT" 2>&1 | tee -a "$LOG"
    gcloud storage cp "$OUT" "$BUCKET/eval_sets/"
  else
    gcloud storage cp "$BUCKET/eval_sets/episodes_tworoom_s${S}_100.json" "$OUT"
  fi
done
cp -f "$SSD/eps/episodes_tworoom_s101_100.json" scripts/ 2>/dev/null || true

# ---- does the reconstructed scene match the dataset frame? FATAL if not ----
echo "[gate] scene reconstruction" | tee -a "$LOG"
set +e
python scripts/check_render_tworoom.py 12 \
  --episodes "$SSD/eps/episodes_tworoom_s101_100.json" --max-mae 3.0 2>&1 | tee -a "$LOG"
grc=${PIPESTATUS[0]}
set -e
gcloud storage cp "$LOG" "$BUCKET/eval/" || true
if [ "$grc" != 0 ]; then
  echo "FATAL: two-room scene is not reproduced; refusing to run the sweep" >&2
  exit 1
fi

# GATE_ONLY runs everything that does NOT need a checkpoint -- episode sets and the scene
# check -- so the protocol's one real unknown is settled immediately instead of after the
# 8-hour training. The eval cell then runs later without repeating any of it.
if [ "${GATE_ONLY:-0}" = "1" ]; then
  echo "TWOROOM GATE ONLY DONE"; exit 0
fi

# ---- one cell: baseline, cem, seed 101 ----
CK=lewm_t1_tworoom_s3072
mkdir -p "$STABLEWM_HOME/checkpoints/$CK"
gcloud storage cp "$BUCKET/ckpts_tworoom/$CK/weights_epoch_10.pt" "$STABLEWM_HOME/checkpoints/$CK/"
gcloud storage cp "$BUCKET/ckpts_tworoom/$CK/config.json" "$STABLEWM_HOME/checkpoints/$CK/" || true

OUT="final_tworoom_t1_cem_s101.csv"
echo "[run] tworoom t1 cem seed=101" | tee -a "$LOG"
set +e
python scripts/budget_sweep_tworoom.py \
  --env tworoom --solver cem --config t1 "$CK/weights_epoch_10.pt" \
  --tiers T1 T2 T3 T4 T5 \
  --episodes-json "$SSD/eps/episodes_tworoom_s101_100.json" \
  --out "$SSD/$OUT" > "$SSD/run_$OUT.log" 2>&1
rc=$?
set -e
grep -E "^\[preset\]|=== t1 @" "$SSD/run_$OUT.log" | tee -a "$LOG"
tail -25 "$SSD/run_$OUT.log" | tee -a "$LOG"
[ -f "$SSD/$OUT" ] && gcloud storage cp "$SSD/$OUT" "$BUCKET/final_eval_tworoom/$OUT"
gcloud storage cp "$LOG" "$BUCKET/eval/" || true
gcloud storage cp "$SSD/run_$OUT.log" "$BUCKET/final_eval_tworoom/logs/" || true
echo "[done] rc=$rc"
exit $rc
