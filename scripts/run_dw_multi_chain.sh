#!/usr/bin/env bash
# DINO-WM on the 4 multi-object OGBench tasks (2026-09-14): 4 trainings + cem/icem
# x 6 seeds -> final_eval_ogbmulti/final_<task>_dw_*.csv (48). Config cfg name "dw"
# matches the cube/pusht dw rows' convention.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/dw_multi.log
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
log "start: DINO-WM multi-object (4 tasks)"
for round in $(seq 1 9000); do
  left=0
  for spec in "cube_double cubedouble" "cube_triple cubetriple" "cube_quadruple cubequadruple" "scene scene"; do
    set -- $spec; task=$1; cfg=$2
    run="dinowm_${task}_s${SEED}"
    if ! gcloud storage ls "$BUCKET/ckpts_dinowm/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
      left=1
      try "tr_dw_${cfg}" bash scripts/ray_train_dinowm.sh "$task"
      continue
    fi
    # eval launchers read ckpts/; mirror the trained ckpt there once
    gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 ||       gcloud storage cp -r "$BUCKET/ckpts_dinowm/$run" "$BUCKET/ckpts/"
    for sol in cem icem; do
      for seeds in "101 102 103" "104 105 106"; do
        miss=0
        for s in $seeds; do
          gcloud storage ls "$BUCKET/final_eval_ogbmulti/final_${task}_dw_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
        done
        [ "$miss" = 0 ] && continue
        left=1
        # shellcheck disable=SC2086
        try "ev_dw_${cfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_ogbmulti.sh "$task" dw "$run" "$sol" $seeds
      done
    done
  done
  [ "$left" = 0 ] && { log "DW MULTI COMPLETE (4 trainings, 48 CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
