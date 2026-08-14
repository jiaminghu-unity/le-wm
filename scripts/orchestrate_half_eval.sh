#!/usr/bin/env bash
# Unattended evaluation of the reduced-q ablation, run once its six checkpoints exist.
#
# PRE-REGISTRATION. Six evaluation-episode seeds, s101-s106 -- the same six the full-q
# results use, fixed before any reduced-q number lands, and every one is reported. No
# seed is added, dropped or re-drawn on the basis of what the results look like.
#
# What it runs, in this order:
#   1. P5 rank-noise, 3 jobs (one per task), each scoring base/obj/aux/(combo)/obj_h/aux_h
#      on the SAME starts and the SAME candidate actions -- the env rollouts are
#      model-independent, so full-q vs reduced-q is paired per start, not two runs
#      compared by their means. Starts: Push-T 50 (its episode file's maximum),
#      Reacher and Cube 64, matching the runs already in eval/.
#   2. SR sweep, 24 jobs = 3 tasks x {obj_h, aux_h} x 4 solvers, 6 seeds each, through
#      ray_eval_half.sh (writes to final_eval_half/, reads ckpts_half/).
#
# Everything lands under new names and new prefixes; no existing artefact is written.
#
# Ray kills a job whose supervisor waits more than 900s in the queue, so this submits
# only as many jobs as there are idle GPUs and re-checks every 3 minutes. Failures are
# retried up to 5 times per cell, which covers spot preemption.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC_BASE='"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]'
SEEDS="101 102 103 104 105 106"
SOLVERS="cem icem mppi gd"
LOG=/workspace/le-wm/eval_results/half_eval.log
mkdir -p "$(dirname "$LOG")"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

declare -A STARTS=( [pusht]=50 [reacher]=64 [cube]=64 )
declare -A ATTEMPTS

# ---- 0. the checkpoints must be there, and be the reduced-q ones ----------------
for t in pusht reacher cube; do for a in obj aux; do
  p="$BUCKET/ckpts_half/lewm_hq_${a}_${t}_s3072/weights_epoch_10.pt"
  gcloud storage ls "$p" >/dev/null 2>&1 || { log "FATAL: missing $p"; exit 1; }
done; done
log "all 6 reduced-q checkpoints present"

nrunning(){  # in-flight jobs whose entrypoint matches $1
  python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
pat=sys.argv[1]
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j.get('type')=='SUBMISSION'
          and j['status'] in ('RUNNING','PENDING')
          and pat in (j.get('entrypoint') or '')))
PY
}
freegpu(){
  local cap run
  cap=$(ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" | cut -d/ -f2 | cut -d. -f1)
  run=$(python3 - <<'PY' 2>/dev/null
import json,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j.get('type')=='SUBMISSION' and j['status'] in ('RUNNING','PENDING')))
PY
)
  echo $(( ${cap:-8} - ${run:-0} ))
}

# ---- 1. P5, three jobs ----------------------------------------------------------
for round in $(seq 1 400); do
  todo=(); done_n=0
  for t in cube reacher pusht; do
    n=${STARTS[$t]}
    if gcloud storage ls "$BUCKET/eval/p5_${t}_half_s${n}.json" >/dev/null 2>&1; then
      done_n=$((done_n+1))
    elif [ "$(nrunning "ray_p5.sh $t")" = "0" ] && [ "${ATTEMPTS[p5_$t]:-0}" -lt 5 ]; then
      todo+=("$t")
    fi
  done
  log "P5 round $round: done $done_n/3, todo ${#todo[@]}"
  [ "$done_n" -ge 3 ] && { log "P5 HALF COMPLETE"; break; }
  free=$(freegpu); i=0
  while [ "$free" -gt 0 ] && [ "$i" -lt ${#todo[@]} ]; do
    t=${todo[$i]}; n=${STARTS[$t]}
    ENVJ="{$EXC_BASE,\"env_vars\":{\"HALF\":\"1\",\"P5TAG\":\"_half_s${n}\"}}"
    id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
          --working-dir /workspace/le-wm --runtime-env-json "$ENVJ" \
          -- bash scripts/ray_p5.sh "$t" --starts "$n" --cands 300 2>&1 \
        | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
    if [ -n "$id" ]; then
      ATTEMPTS[p5_$t]=$(( ${ATTEMPTS[p5_$t]:-0} + 1 ))
      log "  P5 $t ${n} starts -> $id (attempt ${ATTEMPTS[p5_$t]})"
      free=$((free-1))
    else log "  P5 submit FAILED for $t"; fi
    i=$((i+1))
  done
  # SR can start filling idle GPUs while P5 runs; both phases are independent
  sleep 180
done

# ---- 2. SR sweep, 24 cells ------------------------------------------------------
JOBS=()
for t in pusht reacher cube; do for a in obj aux; do
  JOBS+=("$t hq_${a} lewm_hq_${a}_${t}_s3072")
done; done

for round in $(seq 1 600); do
  DONECSV=$(gcloud storage ls "$BUCKET/final_eval_half/" 2>/dev/null | sed 's|.*/||')
  running=$(nrunning "ray_eval_half.sh")
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
  log "SR round $round: complete $complete/$total, running $running, todo ${#todo[@]}"
  [ "$complete" -ge "$total" ] && { log "SR HALF COMPLETE"; exit 0; }
  if [ ${#todo[@]} -eq 0 ] && [ "$running" -eq 0 ]; then
    log "nothing left to submit but cells missing (attempt cap hit) — stopping"; exit 1
  fi
  free=$(freegpu); i=0
  while [ "$free" -gt 0 ] && [ "$i" -lt ${#todo[@]} ]; do
    set -- ${todo[$i]}; task=$1; cfg=$2; ck=$3; slv=$4
    key="${task}_${cfg}_${slv}"
    if [ "$(nrunning "ray_eval_half.sh $task $cfg $ck $slv")" = "0" ]; then
      id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
            --working-dir /workspace/le-wm --runtime-env-json "{$EXC_BASE}" \
            -- bash scripts/ray_eval_half.sh "$task" "$cfg" "$ck" "$slv" $SEEDS 2>&1 \
          | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
      if [ -n "$id" ]; then
        ATTEMPTS[$key]=$(( ${ATTEMPTS[$key]:-0} + 1 ))
        log "  SR $task $cfg $slv -> $id (attempt ${ATTEMPTS[$key]})"
        free=$((free-1))
      else log "  SR submit FAILED for $task $cfg $slv"; fi
    fi
    i=$((i+1))
  done
  sleep 180
done
log "orchestrator hit round cap"
exit 1
