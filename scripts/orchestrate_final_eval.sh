#!/usr/bin/env bash
# Keeps the 8 GPUs saturated with the final cross-task sweep until every
# (task, config, solver) x 3-seed cell exists in GCS.
#
# Ray kills a submitted job that waits >15 min for resources, so jobs can never
# be queued ahead — this loop submits only as many as there are idle GPUs.
# Cells already present in GCS are skipped (ray_eval_final.sh re-checks per seed).
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
SEEDS="101 102 103"
LOG=/workspace/le-wm/eval_results/final/orchestrator.log
mkdir -p "$(dirname "$LOG")"

JOBS=(
 # Push-T is NOT re-run: box2d renders on the CPU, so it was never affected by the
 # EGL-fallback bug, and its cells reproduce the original results episode-for-episode.
 "reacher r1      lewm_r1_reacher_s3072"
 "reacher r2_l015 lewm_r2_reacher_paep_l015_s3072"
 "reacher r5_l04  lewm_r5_qhead0.4_s3072"
 "cube k1 lewm_k1_cube_s3072"
 "cube k2 lewm_k2_cube_obj_eff0.1_s3072"
 "cube k4 lewm_k4_cube_qhead_eff0.1_s3072"
 "cube k6 lewm_k6_cube_combo_o0.1a0.1_s3072"
)
SOLVERS="cem icem mppi gd"
declare -A ATTEMPTS

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

for round in $(seq 1 400); do
  DONECSV=$(gcloud storage ls "$BUCKET/final_eval/" 2>/dev/null | sed 's|.*/||')
  RUNNING=$(python3 - <<'PY' 2>/dev/null
import json,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j.get('type')=='SUBMISSION' and j['status'] in ('RUNNING','PENDING')
          and 'ray_eval_final' in (j.get('entrypoint') or '')))
PY
)
  RUNNING=${RUNNING:-0}
  # count outstanding cells
  todo=(); complete=0; total=0
  for spec in "${JOBS[@]}"; do
    set -- $spec; task=$1; cfg=$2; ck=$3
    for slv in $SOLVERS; do
      total=$((total+1)); missing=0
      for s in $SEEDS; do
        echo "$DONECSV" | grep -qx "final_${task}_${cfg}_${slv}_s${s}.csv" || missing=1
      done
      if [ "$missing" = 0 ]; then complete=$((complete+1))
      else
        key="${task}_${cfg}_${slv}"
        [ "${ATTEMPTS[$key]:-0}" -lt 5 ] && todo+=("$task $cfg $ck $slv")
      fi
    done
  done
  log "round $round: complete $complete/$total, running $RUNNING, todo ${#todo[@]}"
  if [ "$complete" -ge "$total" ]; then log "ALL CELLS COMPLETE"; exit 0; fi
  if [ ${#todo[@]} -eq 0 ] && [ "$RUNNING" -eq 0 ]; then
    log "nothing left to submit but cells missing (attempt cap hit) — stopping"; exit 1
  fi

  CAP=$(ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" | cut -d/ -f2 | cut -d. -f1)
  CAP=${CAP:-8}
  FREE=$((CAP - RUNNING))
  log "  capacity ${CAP} GPU, in-flight ${RUNNING}, free ${FREE}"
  i=0
  while [ "$FREE" -gt 0 ] && [ "$i" -lt ${#todo[@]} ]; do
    set -- ${todo[$i]}; task=$1; cfg=$2; ck=$3; slv=$4
    key="${task}_${cfg}_${slv}"
    # do not resubmit something already in flight
    inflight=$(python3 - "$task" "$cfg" "$slv" <<'PY' 2>/dev/null
import json,sys,urllib.request
t,c,s=sys.argv[1:4]
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j.get('type')=='SUBMISSION' and j['status'] in ('RUNNING','PENDING')
          and f'ray_eval_final.sh {t} {c} ' in (j.get('entrypoint') or '')
          and f' {s} ' in (j.get('entrypoint') or '')))
PY
)
    if [ "${inflight:-0}" = "0" ]; then
      id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
            --working-dir /workspace/le-wm --runtime-env-json "$EXC" \
            -- bash scripts/ray_eval_final.sh "$task" "$cfg" "$ck" "$slv" $SEEDS 2>&1 \
          | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
      if [ -n "$id" ]; then
        ATTEMPTS[$key]=$(( ${ATTEMPTS[$key]:-0} + 1 ))
        log "  submitted $task $cfg $slv -> $id (attempt ${ATTEMPTS[$key]})"
        FREE=$((FREE-1))
      else
        log "  submit FAILED for $task $cfg $slv"
      fi
    fi
    i=$((i+1))
  done
  sleep 240
done
log "orchestrator hit round cap"
