#!/usr/bin/env bash
# FULL-native-q SCALE arms (2026-08-29, user: pusht/reacher/cube_double/scene
# 全做完整维度 q 的 SCALE 训练,有 ckpt 的跳过). Ckpt done-checks make re-runs free.
# TRAINS (seed 3072):
#   pusht   lewm_pusht_scale_native_s3072    obj 0.1  on 8-d native (incl velocities)
#   reacher lewm_reacher_scale_native_s3072  obj 0.15 on 8-d native (joints+finger+qvel)
#   cdouble lewm_cubedouble_obj_s3072        obj 0.1  on 27-d (existing config)
#   scene   lewm_scene_obj_s3072             obj 0.1  on 26-d (existing config)
# EVALS: pusht/reacher via stock ray_eval_final.sh (cfg objnat); cube_double/scene
# via ray_eval_ogbmulti.sh (cfg obj) on the pre-registered s101-106 sets. cem+icem.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"],"env_vars":{"RAY_JOB_START_TIMEOUT_SECONDS":"14400"}}'
L=/workspace/le-wm/eval_results/fullq_scale.log
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
"pt_scn|lewm_pusht_scale_native_s${SEED}|experiment=pusht_scale_native|bash scripts/ray_train_qnative.sh pusht experiment=pusht_scale_native seed=${SEED}"
"rc_scn|lewm_reacher_scale_native_s${SEED}|experiment=reacher_scale_native|bash scripts/ray_train_qnative.sh reacher experiment=reacher_scale_native seed=${SEED}"
"cd_obj|lewm_cubedouble_obj_s${SEED}|experiment=cubedouble_obj|bash scripts/ray_train_qnative.sh cube_double experiment=cubedouble_obj seed=${SEED}"
"sc_obj|lewm_scene_obj_s${SEED}|experiment=scene_obj|bash scripts/ray_train_qnative.sh scene experiment=scene_obj seed=${SEED}"
)
# task|cfg|run|launcher-args(env included)|out-prefix
EVALS=()
for spec in \
  "pusht|objnat|lewm_pusht_scale_native_s${SEED}|ray_eval_final.sh pusht|final_eval" \
  "reacher|objnat|lewm_reacher_scale_native_s${SEED}|ray_eval_final.sh reacher|final_eval" \
  "cube_double|obj|lewm_cubedouble_obj_s${SEED}|ray_eval_ogbmulti.sh cube_double|final_eval_ogbmulti" \
  "scene|obj|lewm_scene_obj_s${SEED}|ray_eval_ogbmulti.sh scene|final_eval_ogbmulti"; do
  IFS='|' read -r task cfg run largs outp <<< "$spec"
  for sol in cem icem; do
    EVALS+=("${task}|${cfg}|${run}|${largs}|${outp}|${sol}|101 102 103")
    EVALS+=("${task}|${cfg}|${run}|${largs}|${outp}|${sol}|104 105 106")
  done
done

log "start: full-native-q SCALE (pusht 8d / reacher 8d / cube_double 27d / scene 26d)"
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
  for cell in "${EVALS[@]}"; do
    IFS='|' read -r task cfg run largs outp sol seeds <<< "$cell"
    miss=0
    for s in $seeds; do
      gcloud storage ls "$BUCKET/$outp/final_${task}_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
    done
    [ "$miss" = 0 ] && continue
    left=1
    gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 || continue
    [ "$submitted" != 0 ] && continue
    [ "$(nrun "$largs $cfg $run $sol ${seeds%% *}")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    key="ev_${task}_${sol}_${seeds%% *}"
    n=${ATT[$key]:-0}
    [ "$n" -ge 4 ] && { log "$key attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub bash scripts/$largs "$cfg" "$run" "$sol" $seeds)
    if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"
    else log "$key submit FAILED"; fi
    submitted=1; break
  done
  [ "$left" = 0 ] && { log "FULL-Q SCALE TRAININGS + EVALS COMPLETE (64 CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
