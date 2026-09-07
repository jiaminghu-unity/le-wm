#!/usr/bin/env bash
# LpWM reg-weight ablation (2026-09-07): official OGBench recipe with RDMReg
# weight 0.1 instead of the paper's 10. Five OGBench tasks x 3 arms, seed 3072.
#   cube:        evals cem/icem/mppi x 6 seeds -> final_eval/          (54)
#   multi-object evals cem/icem      x 6 seeds -> final_eval_ogbmulti/ (144)
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/lpwm3.log
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
log "start: LpWM reg-0.1 ablation (5 OGBench tasks x 3 arms)"
for round in $(seq 1 9000); do
  left=0
  for spec in \
    "cube cube $Q/qgate_stage1_cube_nce_lam0.1.json final final_eval cem,icem,mppi" \
    "cube_double cubedouble $Q/qgate_stage1_cube_double_nce_lam0.1.json ogbmulti final_eval_ogbmulti cem,icem" \
    "cube_triple cubetriple $Q/qgate_stage1_cube_triple_nce_lam0.1.json ogbmulti final_eval_ogbmulti cem,icem" \
    "cube_quadruple cubequadruple $Q/qgate_stage1_cube_quadruple_nce_lam0.1.json ogbmulti final_eval_ogbmulti cem,icem" \
    "scene scene $Q/qgate_stage1_scene_nce_lam0.1.json ogbmulti final_eval_ogbmulti cem,icem"; do
    set -- $spec; task=$1; cfg=$2; gate=$3; evkind=$4; evdir=$5; sols=$6
    for arm in base q g10; do
      name="lpwm3${arm}_${cfg}"
      run="lewm_${name}_s${SEED}"
      if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
        left=1
        if [ "$arm" = g10 ]; then
          try "tr_${name}" env QGATE_GCS="$gate" bash scripts/ray_train_qgate2.sh "$task" experiment="$name" seed=$SEED
        else
          try "tr_${name}" bash scripts/ray_train_qnative.sh "$task" experiment="$name" seed=$SEED
        fi
        continue
      fi
      IFS=',' read -ra SOLS <<< "$sols"
      for sol in "${SOLS[@]}"; do
        for seeds in "101 102 103" "104 105 106"; do
          miss=0
          for s in $seeds; do
            gcloud storage ls "$BUCKET/$evdir/final_${task}_${name}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
          done
          [ "$miss" = 0 ] && continue
          left=1
          if [ "$evkind" = final ]; then
            # shellcheck disable=SC2086
            try "ev_${name}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh cube "$name" "$run" "$sol" $seeds
          else
            # shellcheck disable=SC2086
            try "ev_${name}_${sol}_${seeds%% *}" bash scripts/ray_eval_ogbmulti.sh "$task" "$name" "$run" "$sol" $seeds
          fi
        done
      done
    done
  done
  [ "$left" = 0 ] && { log "LPWM3 ABLATION COMPLETE (15 arms, 198 CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
