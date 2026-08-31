#!/usr/bin/env bash
# Third training seed (3074) for pusht/reacher/cube (2026-08-31, user request),
# matching the pointmaze r74 precedent: {LeWM, SCALE full-q, q-head} x 3 tasks.
# Phase 1: 9 trainings via ray_train_replica.sh (done-check on epoch-10 ckpt).
# Phase 2: evals -- cem/icem x 6 eval seeds via ray_eval_final.sh (cfg <arm>r74),
#          plus paper-protocol mppi (per-task T, 5 held-out seeds) via ray_eval_mppi_t.sh.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3074
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/replica3074.log
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
declare -A ATT
free(){ python3 - <<'FREEPY' 2>/dev/null
import json, urllib.request
nodes = json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/v0/nodes?limit=100', timeout=20))
rows = nodes.get('data',{}).get('result',{}).get('result',[])
total = sum(n.get('resources_total',{}).get('GPU',0) for n in rows if n.get('state')=='ALIVE')
jobs = json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/', timeout=20))
used = sum(1 for j in jobs if j.get('status') in ('RUNNING','PENDING')
           and ('scripts/ray_' in (j.get('entrypoint') or '') or j.get('entrypoint_num_gpus')))
print(max(int(total-used), 0))
FREEPY
}
nrun(){ python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j['status'] in ('RUNNING','PENDING') and sys.argv[1] in (j.get('entrypoint') or '')))
PY
}
sub(){ timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
  --working-dir /workspace/le-wm --runtime-env-json "$EXC" -- "$@" 2>&1 \
  | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }
try(){ local key=$1; shift
  [ "$(nrun "$*")" != 0 ] && return 1
  [ "$(free)" -lt 1 ] && return 1
  local n=${ATT[$key]:-0}
  [ "$n" -ge 4 ] && { log "$key attempt cap"; return 1; }
  local id; id=$(sub "$@")
  if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"; else log "$key submit FAILED"; fi
}

# name|run|train-cmd  (mirrors run_seed_replica_chain.sh entries, SEED=3074)
TRAINS=(
"pusht_c1|lewm_c1_s${SEED}|bash scripts/ray_train_replica.sh pusht experiment=c1_baseline data=pusht seed=${SEED}"
"pusht_c3|lewm_c3_sig_obj0.1_s${SEED}|bash scripts/ray_train_replica.sh pusht experiment=c3_sig_plus_obj data=pusht seed=${SEED}"
"pusht_c5|lewm_c5_qhead0.3_s${SEED}|bash scripts/ray_train_replica.sh pusht experiment=c5_qhead data=pusht loss.aux.weight=0.3 seed=${SEED}"
"reacher_r1|lewm_r1_reacher_s${SEED}|bash scripts/ray_train_replica.sh reacher experiment=r1_reacher_baseline seed=${SEED}"
"reacher_r2|lewm_r2_reacher_paep_l015_s${SEED}|bash scripts/ray_train_replica.sh reacher experiment=r2_reacher_paep loss.obj.weight=0.15 output_model_name=lewm_r2_reacher_paep_l015_s${SEED} seed=${SEED}"
"reacher_r5|lewm_r5_qhead0.4_s${SEED}|bash scripts/ray_train_replica.sh reacher experiment=r5_qhead loss.aux.weight=0.4 seed=${SEED}"
"cube_k1|lewm_k1_cube_s${SEED}|bash scripts/ray_train_replica.sh cube experiment=k1_cube_baseline seed=${SEED}"
"cube_k2|lewm_k2_cube_obj_eff0.1_s${SEED}|bash scripts/ray_train_replica.sh cube experiment=k2_cube_obj_eff seed=${SEED}"
"cube_k4|lewm_k4_cube_qhead_eff0.1_s${SEED}|bash scripts/ray_train_replica.sh cube experiment=k4_cube_qhead_eff seed=${SEED}"
)
# task|cfgtag|run|mppi-T
ARMS=(
"pusht|c1r74|lewm_c1_s${SEED}|64"
"pusht|c3r74|lewm_c3_sig_obj0.1_s${SEED}|64"
"pusht|c5r74|lewm_c5_qhead0.3_s${SEED}|64"
"reacher|r1r74|lewm_r1_reacher_s${SEED}|32"
"reacher|r2r74|lewm_r2_reacher_paep_l015_s${SEED}|32"
"reacher|r5r74|lewm_r5_qhead0.4_s${SEED}|32"
"cube|k1r74|lewm_k1_cube_s${SEED}|32"
"cube|k2r74|lewm_k2_cube_obj_eff0.1_s${SEED}|32"
"cube|k4r74|lewm_k4_cube_qhead_eff0.1_s${SEED}|32"
)

log "start: replica seed 3074 (pusht/reacher/cube x {base,SCALE,q-head})"
for round in $(seq 1 9000); do
  left=0
  for spec in "${TRAINS[@]}"; do
    IFS='|' read -r name run cmd <<< "$spec"
    gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 && continue
    left=1
    # shellcheck disable=SC2086
    try "tr_$name" $cmd
  done
  for arm in "${ARMS[@]}"; do
    IFS='|' read -r task cfg run T <<< "$arm"
    gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 || { left=1; continue; }
    for sol in cem icem; do
      for seeds in "101 102 103" "104 105 106"; do
        miss=0
        for s in $seeds; do gcloud storage ls "$BUCKET/final_eval/final_${task}_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1; done
        [ "$miss" = 0 ] && continue
        left=1
        # shellcheck disable=SC2086
        try "ev_${cfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh "$task" "$cfg" "$run" "$sol" $seeds
      done
    done
    miss=0
    for s in 102 103 104 105 106; do
      gcloud storage ls "$BUCKET/final_eval_mppi_t/final_${task}_${cfg}_mppiT${T}_s${s}.csv" >/dev/null 2>&1 || miss=1
    done
    if [ "$miss" = 1 ]; then
      left=1
      try "mp_${cfg}" bash scripts/ray_eval_mppi_t.sh "$task" "$cfg" ckpts "$run" "$T" "102,103,104,105,106"
    fi
  done
  [ "$left" = 0 ] && { log "REPLICA-3074 COMPLETE (9 trains, 108 cem/icem + 45 mppi CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
