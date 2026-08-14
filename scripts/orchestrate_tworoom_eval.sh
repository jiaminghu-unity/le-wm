#!/usr/bin/env bash
# two-room SR sweep: 3 arms x 4 solvers x 6 episode seeds = 18 cells.
#
# PRE-REGISTRATION. Seeds s101-s106, the same six every other task uses, fixed before any
# two-room number landed (the sets were generated and their sha256 recorded before training
# finished). All four solvers and all five budget tiers, exactly as the other three tasks --
# no subsetting. Hyperparameters were not tuned on this task: obj 0.1 is the value Push-T and
# Cube used, aux 0.1 the one Cube used and the config default. If aux underperforms, an
# untuned weight is a live explanation and will be reported as one.
#
# Checkpoint directory names are RESOLVED FROM GCS, not hardcoded. The earlier training chain
# hardcoded lewm_t2_tworoom_s3072 while the configs name the run lewm_t2_tworoom_obj0.1_s3072,
# and that mismatch made it resubmit ten trainings where three were needed.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
SEEDS="101 102 103 104 105 106"
SOLVERS="cem icem mppi gd"
LOG=/workspace/le-wm/eval_results/tworoom_eval.log
mkdir -p "$(dirname "$LOG")"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# arm -> checkpoint directory, resolved by prefix from what is actually in GCS
declare -A CK
DIRS=$(gcloud storage ls "$BUCKET/ckpts_tworoom/" 2>/dev/null | sed 's|.*ckpts_tworoom/||;s|/$||')
for arm in t1 t2 t5; do
  d=$(echo "$DIRS" | grep -E "^lewm_${arm}_tworoom" | head -1)
  [ -n "$d" ] || { log "FATAL: no checkpoint directory matching lewm_${arm}_tworoom in GCS"; exit 1; }
  gcloud storage ls "$BUCKET/ckpts_tworoom/$d/weights_epoch_10.pt" >/dev/null 2>&1 \
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
  DONECSV=$(gcloud storage ls "$BUCKET/final_eval_tworoom/" 2>/dev/null | sed 's|.*/||')
  running=$(nrunning "ray_eval_tworoom.sh")
  todo=(); complete=0; total=0
  for arm in t1 t2 t5; do
    for slv in $SOLVERS; do
      total=$((total+1)); missing=0
      for s in $SEEDS; do
        echo "$DONECSV" | grep -qx "final_tworoom_${arm}_${slv}_s${s}.csv" || missing=1
      done
      if [ "$missing" = 0 ]; then complete=$((complete+1))
      else
        key="${arm}_${slv}"
        [ "${ATTEMPTS[$key]:-0}" -lt 5 ] && todo+=("$arm $slv")
      fi
    done
  done
  log "round $round: complete $complete/$total, running $running, todo ${#todo[@]}"
  [ "$complete" -ge "$total" ] && { log "TWOROOM SR SWEEP COMPLETE"; exit 0; }
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
    if [ "$(nrunning "ray_eval_tworoom.sh $arm ${CK[$arm]} $slv")" = "0" ]; then
      id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
            --working-dir /workspace/le-wm --runtime-env-json "$EXC" \
            -- bash scripts/ray_eval_tworoom.sh "$arm" "${CK[$arm]}" "$slv" $SEEDS 2>&1 \
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
