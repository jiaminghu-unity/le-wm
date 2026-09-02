#!/usr/bin/env bash
# ALIEN-q experiment chain (2026-09-02): cube q = 22 real + 26 structured-irrelevant
# (real scene trajectories). Phases, all done-checked:
#   0. cube_alien.lance exists (built by the standalone data job)
#   1. Stage-1 InfoNCE gates on the 48-d q, lambda {0.01,0.03,0.05,0.1}
#   2. Stage-2: union-ungated (qalien) + gated lam 0.05/0.1 (qalieng05/10)
#   3. evals: cem/icem/mppi x 6 seeds each (standard cube eval; cfg = arm name)
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/alien.log
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
Q=$BUCKET/qgate
log "start: ALIEN-q chain"
for round in $(seq 1 9000); do
  left=0
  if ! gcloud storage ls "$BUCKET/datasets/ogbench/cube_alien.lance/" >/dev/null 2>&1; then
    left=1; log "waiting for cube_alien.lance"
  else
    # phase 1: gates
    miss=0
    for lam in 0.01 0.03 0.05 0.1; do
      gcloud storage ls "$Q/qgate_stage1_cube_alien_nce_lam${lam}.json" >/dev/null 2>&1 || miss=1
    done
    if [ "$miss" = 1 ]; then
      left=1
      try "gate_alien" env QGATE_VARIANT=infonce bash scripts/ray_qgate_stage1.sh cube_alien 0.01 0.03 0.05 0.1
    else
      # phase 2+3
      for arm in qalien qalieng05 qalieng10; do
        run="lewm_${arm}_scale_cube_s${SEED}"
        case $arm in
          qalien)    gate="" ;;
          qalieng05) gate="$Q/qgate_stage1_cube_alien_nce_lam0.05.json" ;;
          qalieng10) gate="$Q/qgate_stage1_cube_alien_nce_lam0.1.json" ;;
        esac
        if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
          left=1
          if [ -z "$gate" ]; then
            try "tr_${arm}" bash scripts/ray_train_qnative.sh cube_alien experiment="${arm}_scale_cube" seed=$SEED
          else
            try "tr_${arm}" env QGATE_GCS="$gate" bash scripts/ray_train_qgate2.sh cube_alien experiment="${arm}_scale_cube" seed=$SEED
          fi
          continue
        fi
        for sol in cem icem mppi; do
          for seeds in "101 102 103" "104 105 106"; do
            miss=0
            for s in $seeds; do
              gcloud storage ls "$BUCKET/final_eval/final_cube_${arm}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
            done
            [ "$miss" = 0 ] && continue
            left=1
            # shellcheck disable=SC2086
            try "ev_${arm}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh cube "$arm" "$run" "$sol" $seeds
          done
        done
      done
    fi
  fi
  [ "$left" = 0 ] && { log "ALIEN COMPLETE (4 gates, 3 arms, 54 CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
