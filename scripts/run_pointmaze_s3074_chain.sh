#!/usr/bin/env bash
# PointMaze THIRD training seed (2026-08-29, user: 重新挂 pointmaze 新训练种子给
# scale/aux/lewm/dinowm). The replication + fresh-set retests showed pointmaze pixel
# arms are training-seed sensitive (SCALE@3072 +5* on three episode grids,
# SCALE@3073 negative on the fresh ones); s3074 resolves which draw is typical.
# TRAINS: 4 arms at SEED=3074 via the existing seed-parameterized launchers.
# EVALS: each model x cem/icem/mppi x BOTH episode grids (s101-106 protocol grid
# and s201-206 fresh grid) -> final_pointmaze_{p1r74,p2r74,p5r74,dwr74}_*.csv.
# All outputs are new files. nohup babysitter conventions.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3074
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/pointmaze_s3074.log
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

# name|ckpt-prefix|run-dir|probe|launcher args
TRAINS=(
"pm74_base|ckpts_pointmaze|lewm_p1_pointmaze_s${SEED}|ray_train_pointmaze_seed.sh base|bash scripts/ray_train_pointmaze_seed.sh base"
"pm74_obj|ckpts_pointmaze|lewm_p2_pointmaze_s${SEED}|ray_train_pointmaze_seed.sh obj|bash scripts/ray_train_pointmaze_seed.sh obj"
"pm74_aux|ckpts_pointmaze|lewm_p5_pointmaze_s${SEED}|ray_train_pointmaze_seed.sh aux|bash scripts/ray_train_pointmaze_seed.sh aux"
"pm74_dw|ckpts_dinowm|dinowm_pointmaze_s${SEED}|ray_train_dinowm_seed.sh pointmaze|bash scripts/ray_train_dinowm_seed.sh pointmaze"
)
# cfg|ckpt-dir(under ckpts_pointmaze/)
MODELS=(
"p1r74|lewm_p1_pointmaze_s${SEED}"
"p2r74|lewm_p2_pointmaze_s${SEED}"
"p5r74|lewm_p5_pointmaze_s${SEED}"
"dwr74|dinowm_pointmaze_s${SEED}"
)
CELLS=()
for sol in cem icem mppi; do
  for m in "${MODELS[@]}"; do
    for half in "101 102 103" "104 105 106" "201 202 203" "204 205 206"; do
      CELLS+=("$m|$sol|$half")
    done
  done
done

log "start: pointmaze s3074 (4 trainings -> 3 solvers x 2 episode grids)"
for round in $(seq 1 8000); do
  left=0; submitted=0
  for spec in "${TRAINS[@]}"; do
    IFS='|' read -r name pfx run probe cmd <<< "$spec"
    gcloud storage ls "$BUCKET/$pfx/$run/weights_epoch_10.pt" >/dev/null 2>&1 && continue
    left=1
    [ "$(nrun "$probe")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    n=${ATT[$name]:-0}
    [ "$n" -ge 4 ] && { log "$name attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub env SEED=$SEED $cmd)
    if [ -n "$id" ]; then ATT[$name]=$((n+1)); log "$name attempt $((n+1)) -> $id"
    else log "$name submit FAILED"; fi
    submitted=1; break
  done
  # one-time: stage dinowm ckpt into ckpts_pointmaze/ for the eval launcher
  if gcloud storage ls "$BUCKET/ckpts_dinowm/dinowm_pointmaze_s${SEED}/weights_epoch_10.pt" >/dev/null 2>&1 \
     && ! gcloud storage ls "$BUCKET/ckpts_pointmaze/dinowm_pointmaze_s${SEED}/weights_epoch_10.pt" >/dev/null 2>&1; then
    log "staging dinowm_pointmaze_s${SEED} -> ckpts_pointmaze/"
    gcloud storage cp "$BUCKET/ckpts_dinowm/dinowm_pointmaze_s${SEED}/weights_epoch_10.pt" \
                      "$BUCKET/ckpts_dinowm/dinowm_pointmaze_s${SEED}/config.json" \
                      "$BUCKET/ckpts_pointmaze/dinowm_pointmaze_s${SEED}/" || log "WARN dinowm staging failed"
  fi
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r cfg ckpt sol seeds <<< "$cell"
    miss=0
    for s in $seeds; do
      gcloud storage ls "$BUCKET/final_eval_pointmaze/final_pointmaze_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
    done
    [ "$miss" = 0 ] && continue
    left=1
    gcloud storage ls "$BUCKET/ckpts_pointmaze/$ckpt/weights_epoch_10.pt" >/dev/null 2>&1 || continue
    [ "$submitted" != 0 ] && continue
    [ "$(nrun "ray_eval_pointmaze.sh $cfg $ckpt $sol ${seeds%% *}")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    key="${cfg}_${sol}_${seeds%% *}"
    n=${ATT[$key]:-0}
    [ "$n" -ge 4 ] && { log "$key attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub bash scripts/ray_eval_pointmaze.sh "$cfg" "$ckpt" "$sol" $seeds)
    if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"
    else log "$key submit FAILED"; fi
    submitted=1; break
  done
  [ "$left" = 0 ] && { log "POINTMAZE S3074 TRAININGS + EVALS COMPLETE (96 CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
