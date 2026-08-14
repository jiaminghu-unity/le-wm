#!/usr/bin/env bash
# SR sweep with the planner cost switched to L1 -- 18 cells.
#   3 tasks x {baseline, L_obj, aux q-head} x {cem, icem} x 6 episode seeds
#
# PRE-REGISTRATION. Seeds s101-s106, the same six the squared-L2 results use, fixed
# before any L1 number lands, and all six reported. Solvers are cem and icem only,
# decided before the run and for a stated reason rather than after seeing results:
# both select by rank alone, whereas mppi's fixed softmax temperature and gd's fixed
# learning rate react to the L1/L2 magnitude difference, so a change under those two
# would not be attributable to the norm (see scripts/l1_cost.py).
#
# The comparison is exactly paired: same checkpoints, same episode sets, same
# cem_seed = crc32("episode_id|tier"), same tiers, same render gate, same code path --
# the cost function is the only difference. So per-episode McNemar against the
# existing final_eval/ rows is valid, not just a comparison of means.
#
# Nothing existing is written: results go to final_eval_l1/, checkpoints in ckpts/
# are read-only, and the config column carries an _l1 suffix so no CSV name collides.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
SEEDS="101 102 103 104 105 106"
SOLVERS="cem icem"
LOG=/workspace/le-wm/eval_results/l1_eval.log
mkdir -p "$(dirname "$LOG")"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# task | config label written into the CSV | checkpoint dir under ckpts/
JOBS=(
  "pusht   c1_l1        lewm_c1_s3072"
  "pusht   c3_l01_l1    lewm_c3_sig_obj0.1_s3072"
  "pusht   c5_l03_l1    lewm_c5_qhead0.3_s3072"
  "reacher r1_l1        lewm_r1_reacher_s3072"
  "reacher r2_l015_l1   lewm_r2_reacher_paep_l015_s3072"
  "reacher r5_l04_l1    lewm_r5_qhead0.4_s3072"
  "cube    k1_l1        lewm_k1_cube_s3072"
  "cube    k2_l1        lewm_k2_cube_obj_eff0.1_s3072"
  "cube    k4_l1        lewm_k4_cube_qhead_eff0.1_s3072"
)
declare -A ATTEMPTS

nrunning(){
  python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
pat=sys.argv[1]
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j.get('type')=='SUBMISSION' and j['status'] in ('RUNNING','PENDING')
          and pat in (j.get('entrypoint') or '')))
PY
}

for round in $(seq 1 600); do
  DONECSV=$(gcloud storage ls "$BUCKET/final_eval_l1/" 2>/dev/null | sed 's|.*/||')
  running=$(nrunning "ray_eval_l1.sh")
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
  log "round $round: complete $complete/$total, running $running, todo ${#todo[@]}"
  [ "$complete" -ge "$total" ] && { log "L1 SR SWEEP COMPLETE"; exit 0; }
  if [ ${#todo[@]} -eq 0 ] && [ "$running" -eq 0 ]; then
    log "nothing left to submit but cells missing (attempt cap hit) - stopping"; exit 1
  fi

  # Ray kills a job whose supervisor queues longer than 900s, so submit only as many
  # as there are idle GPUs and come back in three minutes.
  cap=$(ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" | cut -d/ -f2 | cut -d. -f1)
  busy=$(python3 - <<'PY' 2>/dev/null
import json,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j.get('type')=='SUBMISSION' and j['status'] in ('RUNNING','PENDING')))
PY
)
  free=$(( ${cap:-8} - ${busy:-0} )); i=0
  while [ "$free" -gt 0 ] && [ "$i" -lt ${#todo[@]} ]; do
    set -- ${todo[$i]}; task=$1; cfg=$2; ck=$3; slv=$4
    key="${task}_${cfg}_${slv}"
    if [ "$(nrunning "ray_eval_l1.sh $task $cfg $ck $slv")" = "0" ]; then
      id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
            --working-dir /workspace/le-wm --runtime-env-json "$EXC" \
            -- bash scripts/ray_eval_l1.sh "$task" "$cfg" "$ck" "$slv" $SEEDS 2>&1 \
          | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
      if [ -n "$id" ]; then
        ATTEMPTS[$key]=$(( ${ATTEMPTS[$key]:-0} + 1 ))
        log "  submitted $task $cfg $slv -> $id (attempt ${ATTEMPTS[$key]})"
        free=$((free-1))
      else log "  submit FAILED for $task $cfg $slv"; fi
    fi
    i=$((i+1))
  done
  sleep 180
done
log "orchestrator hit round cap"
exit 1
