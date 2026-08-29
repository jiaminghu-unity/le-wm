#!/usr/bin/env bash
# LeWM pixel baselines for the OGB multi-object tasks (2026-08-29, user:
# Cube-Double/Scene 分别 train lewm 版本测测) -- the missing comparators for the
# q-only rows. TRAINS lewm_cubedouble_base_s3072 + lewm_scene_base_s3072
# (existing cubedouble_base/scene_base experiments via ray_train_qnative.sh, which
# registers the q variants the configs reference and stages the lance datasets).
# EVALS: pixel budget_sweep (budget_sweep_ogbmulti.py) cem+icem x 6 seeds on the
# same pre-registered episode sets as the q-only runs
#   -> final_eval_ogbmulti/final_{cube_double,scene}_base_{cem,icem}_s10x.csv.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"],"env_vars":{"RAY_JOB_START_TIMEOUT_SECONDS":"14400"}}'
L=/workspace/le-wm/eval_results/ogbmulti_lewm.log
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
declare -A ATT
free(){ python3 - <<'FREEPY' 2>/dev/null
import json, urllib.request
nodes = json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/v0/nodes?limit=100', timeout=20))
rows = nodes.get('data',{}).get('result',{}).get('result',[])
total = sum(n.get('resources_total',{}).get('GPU',0) for n in rows if n.get('state')=='ALIVE')
jobs = json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/', timeout=20))
used = sum(1 for j in jobs if j.get('status')=='RUNNING' and j.get('entrypoint_num_gpus'))
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

# name|run|probe|command
TRAINS=(
"cd_base|lewm_cubedouble_base_s${SEED}|experiment=cubedouble_base|bash scripts/ray_train_qnative.sh cube_double experiment=cubedouble_base seed=${SEED}"
"sc_base|lewm_scene_base_s${SEED}|experiment=scene_base|bash scripts/ray_train_qnative.sh scene experiment=scene_base seed=${SEED}"
)
# task|cfg|run
CELLS=()
for spec in "cube_double|base|lewm_cubedouble_base_s${SEED}" "scene|base|lewm_scene_base_s${SEED}"; do
  IFS='|' read -r task cfg run <<< "$spec"
  for sol in cem icem; do
    CELLS+=("$task|$cfg|$run|$sol|101 102 103")
    CELLS+=("$task|$cfg|$run|$sol|104 105 106")
  done
done

log "start: OGB multi-object LeWM pixel baselines (cube_double + scene) at seed $SEED"
for round in $(seq 1 8000); do
  left=0; submitted=0
  for spec in "${TRAINS[@]}"; do
    IFS='|' read -r name run probe cmd <<< "$spec"
    gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 && continue
    left=1
    [ "$(nrun "$probe")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    n=${ATT[$name]:-0}
    [ "$n" -ge 4 ] && { log "$name attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub $cmd)
    if [ -n "$id" ]; then ATT[$name]=$((n+1)); log "$name attempt $((n+1)) -> $id"
    else log "$name submit FAILED"; fi
    submitted=1; break
  done
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r task cfg run sol seeds <<< "$cell"
    miss=0
    for s in $seeds; do
      gcloud storage ls "$BUCKET/final_eval_ogbmulti/final_${task}_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
    done
    [ "$miss" = 0 ] && continue
    left=1
    gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 || continue
    [ "$submitted" != 0 ] && continue
    [ "$(nrun "ray_eval_ogbmulti.sh $task $cfg $run $sol ${seeds%% *}")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    key="${task}_${cfg}_${sol}_${seeds%% *}"
    n=${ATT[$key]:-0}
    [ "$n" -ge 4 ] && { log "$key attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub bash scripts/ray_eval_ogbmulti.sh "$task" "$cfg" "$run" "$sol" $seeds)
    if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"
    else log "$key submit FAILED"; fi
    submitted=1; break
  done
  [ "$left" = 0 ] && { log "OGBMULTI LEWM BASELINES + EVALS COMPLETE (24 CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
