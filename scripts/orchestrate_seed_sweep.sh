#!/usr/bin/env bash
# Add 3 pre-registered evaluation-episode seeds (s104/105/106) to every Push-T and
# Reacher cell, bringing each to 6 seeds.
#
# PRE-REGISTRATION. The seeds are s104, s105, s106 — fixed before the first result
# lands, and every one is reported. ray_gen_eval_seeds.sh builds ten sets
# (s104..s113) so a later extension needs no new generation, but only these three
# are evaluated in this round; picking the flattering subset out of ten would make
# the numbers meaningless. At Reacher's effect size the aggregate is expected to
# stay below 2 sigma at n=6, and that is reported as "not significant" regardless
# of which direction the new seeds fall.
#
# Cube is excluded: its 16 cells are already complete in the archive, and the
# EGL-render fix was shown not to move them (k1 x 10 tiers x 300 episodes agreed
# 295-300/300, all deltas within +/-1.0pp), unlike Reacher where it moved SR 6-14pp.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
SEEDS="104 105 106"
LOG=/workspace/le-wm/eval_results/final/seed_sweep.log
mkdir -p "$(dirname "$LOG")"

JOBS=(
 "pusht   c1         lewm_c1_s3072"
 "pusht   c3_l01     lewm_c3_sig_obj0.1_s3072"
 "pusht   c5_l03     lewm_c5_qhead0.3_s3072"
 # combo (c6) is out of this round: its 3-seed additivity is already measured and
 # the question here is baseline vs L_obj vs aux.
 "reacher r1         lewm_r1_reacher_s3072"
 "reacher r2_l015    lewm_r2_reacher_paep_l015_s3072"
 "reacher r5_l04     lewm_r5_qhead0.4_s3072"
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
  if [ "$complete" -ge "$total" ]; then log "ALL SEED-SWEEP CELLS COMPLETE"; exit 0; fi
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
  sleep 180
done
log "orchestrator hit round cap"
