#!/usr/bin/env bash
# cube_triple / cube_quadruple Stage-2 (2026-08-31, user: 全 q + gated 0.01/0.03/0.05/0.1 全都训).
# Phases per task, all done-checked:
#   0. eval episode sets s101-106 (ray_gen_ogbmulti_sets.sh, needs the play lance)
#   1. 5 trainings: obj0.1 + gated{01,03,05,10} (gate JSONs already in GCS)
#   2. evals: cem+icem x 6 seeds per arm -> final_eval_ogbmulti/final_<task>_<cfg>_*.csv
# 2 tasks x 5 arms x 12 = 120 CSVs.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/tq_scale.log
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
try(){ # try <key> <cmd...>: dedup+cap+free 检查后提交
  local key=$1; shift
  [ "$(nrun "$*")" != 0 ] && return 1
  [ "$(free)" -lt 1 ] && return 1
  local n=${ATT[$key]:-0}
  [ "$n" -ge 4 ] && { log "$key attempt cap"; return 1; }
  local id; id=$(sub "$@")
  if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"; else log "$key submit FAILED"; fi
  return 0
}

Q=$BUCKET/qgate
log "start: cube_triple/cube_quadruple Stage-2, obj + gated{01,03,05,10}"
for round in $(seq 1 9000); do
  left=0
  for spec in "cube_triple cubetriple" "cube_quadruple cubequadruple"; do
    set -- $spec; task=$1; cfg=$2
    # phase 0: eval sets
    if ! gcloud storage ls "$BUCKET/eval_sets/episodes_${task}_s106_100.json" >/dev/null 2>&1; then
      left=1
      try "sets_$task" bash scripts/ray_gen_ogbmulti_sets.sh "$task"
      continue
    fi
    # phase 1: trainings
    for arm in obj g01 g03 g05 g10; do
      case $arm in
        obj) run="lewm_${cfg}_obj0.1_s${SEED}"; exp="${cfg}_obj"; gate="" ;;
        *)   g=${arm#g}; run="lewm_${cfg}_scale_gate${g}_s${SEED}"; exp="${cfg}_scale_gate${g}"
             case $g in 01) lam=0.01;; 03) lam=0.03;; 05) lam=0.05;; 10) lam=0.1;; esac
             gate="$Q/qgate_stage1_${task}_lam${lam}.json" ;;
      esac
      if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
        left=1
        if [ -z "$gate" ]; then
          try "tr_${cfg}_${arm}" bash scripts/ray_train_qnative.sh "$task" experiment="$exp" seed=$SEED
        else
          gcloud storage ls "$gate" >/dev/null 2>&1 || continue
          try "tr_${cfg}_${arm}" env QGATE_GCS="$gate" bash scripts/ray_train_qgate2.sh "$task" experiment="$exp" seed=$SEED
        fi
        continue
      fi
      # phase 2: evals
      for sol in cem icem; do
        for seeds in "101 102 103" "104 105 106"; do
          miss=0
          for s in $seeds; do
            gcloud storage ls "$BUCKET/final_eval_ogbmulti/final_${task}_${arm}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
          done
          [ "$miss" = 0 ] && continue
          left=1
          # shellcheck disable=SC2086
          try "ev_${cfg}_${arm}_${sol}_${seeds%% *}" bash scripts/ray_eval_ogbmulti.sh "$task" "$arm" "$run" "$sol" $seeds
        done
      done
    done
  done
  [ "$left" = 0 ] && { log "TRIPLE/QUAD STAGE-2 COMPLETE (10 arms, 120 CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
