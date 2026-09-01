#!/usr/bin/env bash
# L1+InfoNCE recipe end-to-end on the multi-object OGBench tasks (2026-09-01, user:
# double/triple/quadruple/scene 全流程 gate->train->test, lambda {0.01,0.03,0.05,0.1}).
# Phase 0: Stage-1 InfoNCE gates (one job per task, all 4 lambdas, K=255 tau=1).
# Phase 1: 16 Stage-2 trainings (sqrt-gate-weighted L_obj, train_qgate2).
# Phase 2: cem/icem x 6 eval seeds per arm -> final_<task>_nceg<g>_*.csv (192 CSVs).
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/nce_multiobj.log
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
log "start: L1+InfoNCE multi-object, gate -> train -> eval"
for round in $(seq 1 9000); do
  left=0
  for spec in "cube_double cubedouble" "cube_triple cubetriple" "cube_quadruple cubequadruple" "scene scene"; do
    set -- $spec; task=$1; cfg=$2
    # phase 0: InfoNCE gates
    miss=0
    for lam in 0.01 0.03 0.05 0.1; do
      gcloud storage ls "$Q/qgate_stage1_${task}_nce_lam${lam}.json" >/dev/null 2>&1 || miss=1
    done
    if [ "$miss" = 1 ]; then
      left=1
      try "gate_${cfg}" env QGATE_VARIANT=infonce bash scripts/ray_qgate_stage1.sh "$task" 0.01 0.03 0.05 0.1
      continue
    fi
    # phase 1+2
    for g in 01 03 05 10; do
      case $g in 01) lam=0.01;; 03) lam=0.03;; 05) lam=0.05;; 10) lam=0.1;; esac
      run="lewm_${cfg}_nce_gate${g}_s${SEED}"
      gate="$Q/qgate_stage1_${task}_nce_lam${lam}.json"
      if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
        left=1
        try "tr_${cfg}_n${g}" env QGATE_GCS="$gate" bash scripts/ray_train_qgate2.sh "$task" experiment="${cfg}_nce_gate${g}" seed=$SEED
        continue
      fi
      for sol in cem icem; do
        for seeds in "101 102 103" "104 105 106"; do
          miss=0
          for s in $seeds; do
            gcloud storage ls "$BUCKET/final_eval_ogbmulti/final_${task}_nceg${g}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
          done
          [ "$miss" = 0 ] && continue
          left=1
          # shellcheck disable=SC2086
          try "ev_${cfg}_n${g}_${sol}_${seeds%% *}" bash scripts/ray_eval_ogbmulti.sh "$task" "nceg${g}" "$run" "$sol" $seeds
        done
      done
    done
  done
  [ "$left" = 0 ] && { log "NCE MULTIOBJ COMPLETE (16 gates, 16 arms, 192 CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
