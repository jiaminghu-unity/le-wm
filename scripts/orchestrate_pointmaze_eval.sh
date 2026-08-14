#!/usr/bin/env bash
# PointMaze SR sweep: 3 arms x 4 solvers x 6 episode seeds = 12 cells.
#
# PRE-REGISTRATION. Seeds s101-s106, the same six every other task uses; their sets were
# generated and sha256-recorded by the smoke job before any pointmaze SR existed. All four
# solvers, all five tiers, all six seeds reported -- no subsetting. Hyperparameters were
# not tuned on this task (obj 0.1 = Push-T/Cube/two-room's value; aux 0.1 = Cube's and the
# config default, fixed before any result).
#
# Checkpoint directories are RESOLVED FROM GCS by prefix, never hardcoded -- hardcoding is
# what made the two-room chain resubmit ten trainings where three were needed.
#
# The smoke run validated the full path (eval-path gate MAE 0.043, baseline cem s101 SR
# 76-83 across tiers) before this sweep was allowed to start.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
SEEDS="101 102 103 104 105 106"
SOLVERS="cem icem mppi gd"
LOG=/workspace/le-wm/eval_results/pointmaze_eval.log
mkdir -p "$(dirname "$LOG")"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

declare -A CK
DIRS=$(gcloud storage ls "$BUCKET/ckpts_pointmaze/" 2>/dev/null | sed 's|.*ckpts_pointmaze/||;s|/$||')
for arm in p1 p2 p5; do
  d=$(echo "$DIRS" | grep -E "^lewm_${arm}_pointmaze" | head -1)
  [ -n "$d" ] || { log "FATAL: no checkpoint directory matching lewm_${arm}_pointmaze"; exit 1; }
  gcloud storage ls "$BUCKET/ckpts_pointmaze/$d/weights_epoch_10.pt" >/dev/null 2>&1 \
    || { log "FATAL: $d has no weights_epoch_10.pt"; exit 1; }
  CK[$arm]="$d"
  log "arm $arm -> $d"
done

declare -A ATTEMPTS
nrunning(){ python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
pat=sys.argv[1]
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j.get('type')=='SUBMISSION' and j['status'] in ('RUNNING','PENDING')
          and pat in (j.get('entrypoint') or '')))
PY
}

for round in $(seq 1 400); do
  DONECSV=$(gcloud storage ls "$BUCKET/final_eval_pointmaze/" 2>/dev/null | sed 's|.*/||')
  running=$(nrunning "ray_eval_pointmaze.sh")
  todo=(); complete=0; total=0
  for arm in p1 p2 p5; do
    for slv in $SOLVERS; do
      total=$((total+1)); missing=0
      for s in $SEEDS; do
        echo "$DONECSV" | grep -qx "final_pointmaze_${arm}_${slv}_s${s}.csv" || missing=1
      done
      if [ "$missing" = 0 ]; then complete=$((complete+1))
      else
        key="${arm}_${slv}"
        [ "${ATTEMPTS[$key]:-0}" -lt 5 ] && todo+=("$arm $slv")
      fi
    done
  done
  log "round $round: complete $complete/$total, running $running, todo ${#todo[@]}"
  [ "$complete" -ge "$total" ] && { log "POINTMAZE SR SWEEP COMPLETE"; exit 0; }
  if [ ${#todo[@]} -eq 0 ] && [ "$running" -eq 0 ]; then
    log "nothing left to submit but cells missing (attempt cap hit) - stopping"; exit 1
  fi

  cap=$(ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" | cut -d/ -f2 | cut -d. -f1)
  busy=$(python3 - <<'PY' 2>/dev/null
import json,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j.get('type')=='SUBMISSION' and j['status'] in ('RUNNING','PENDING')))
PY
)
  free=$(( ${cap:-8} - ${busy:-0} )); i=0
  while [ "$free" -gt 0 ] && [ "$i" -lt ${#todo[@]} ]; do
    set -- ${todo[$i]}; arm=$1; slv=$2
    key="${arm}_${slv}"
    if [ "$(nrunning "ray_eval_pointmaze.sh $arm ${CK[$arm]} $slv")" = "0" ]; then
      id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
            --working-dir /workspace/le-wm --runtime-env-json "$EXC" \
            -- bash scripts/ray_eval_pointmaze.sh "$arm" "${CK[$arm]}" "$slv" $SEEDS 2>&1 \
          | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
      if [ -n "$id" ]; then
        ATTEMPTS[$key]=$(( ${ATTEMPTS[$key]:-0} + 1 ))
        log "  submitted $arm $slv -> $id (attempt ${ATTEMPTS[$key]})"
        free=$((free-1))
      else log "  submit FAILED for $arm $slv"; fi
    fi
    i=$((i+1))
  done
  sleep 180
done
log "orchestrator hit round cap"; exit 1
