#!/usr/bin/env bash
# GATED full-q SCALE across four tasks (2026-08-29, user: pusht/reacher/cube_double/
# scene 各用 lambda=0.05 和 0.1 的 q-gate 做 SCALE 训练,已有的跳过).
# PREREQ: reacher's lambda=0.1 Stage-1 gate is missing -> phase 0 runs it.
# TRAINS (seed 3072, ckpt done-checks skip anything already trained):
#   lewm_<task>_scale_gate{05,10}_s3072 via train_qgate2.py, QGATE_GCS = the task's
#   Stage-1 JSON. EVALS: cem+icem x 6 seeds; pusht/reacher stock, cube_double/scene
#   via the ogbmulti sweeper. 8 arms x 12 = 96 CSVs.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"],"env_vars":{"RAY_JOB_START_TIMEOUT_SECONDS":"14400"}}'
L=/workspace/le-wm/eval_results/gatedq_scale.log
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

Q=$BUCKET/qgate
# name|run|gate-json|probe|command
TRAINS=(
"pt_g05|lewm_pusht_scale_gate05_s${SEED}|$Q/qgate_stage1_pusht_lam0.05.json|experiment=pusht_scale_gate05|bash scripts/ray_train_qgate2.sh pusht experiment=pusht_scale_gate05 seed=${SEED}"
"pt_g10|lewm_pusht_scale_gate10_s${SEED}|$Q/qgate_stage1_pusht_lam0.1.json|experiment=pusht_scale_gate10|bash scripts/ray_train_qgate2.sh pusht experiment=pusht_scale_gate10 seed=${SEED}"
"rc_g05|lewm_reacher_scale_gate05_s${SEED}|$Q/qgate_stage1_reacher_lam0.05.json|experiment=reacher_scale_gate05|bash scripts/ray_train_qgate2.sh reacher experiment=reacher_scale_gate05 seed=${SEED}"
"rc_g10|lewm_reacher_scale_gate10_s${SEED}|$Q/qgate_stage1_reacher_lam0.1.json|experiment=reacher_scale_gate10|bash scripts/ray_train_qgate2.sh reacher experiment=reacher_scale_gate10 seed=${SEED}"
"cd_g05|lewm_cubedouble_scale_gate05_s${SEED}|$Q/qgate_stage1_cube_double_lam0.05.json|experiment=cubedouble_scale_gate05|bash scripts/ray_train_qgate2.sh cube_double experiment=cubedouble_scale_gate05 seed=${SEED}"
"cd_g10|lewm_cubedouble_scale_gate10_s${SEED}|$Q/qgate_stage1_cube_double_lam0.1.json|experiment=cubedouble_scale_gate10|bash scripts/ray_train_qgate2.sh cube_double experiment=cubedouble_scale_gate10 seed=${SEED}"
"sc_g05|lewm_scene_scale_gate05_s${SEED}|$Q/qgate_stage1_scene_lam0.05.json|experiment=scene_scale_gate05|bash scripts/ray_train_qgate2.sh scene experiment=scene_scale_gate05 seed=${SEED}"
"sc_g10|lewm_scene_scale_gate10_s${SEED}|$Q/qgate_stage1_scene_lam0.1.json|experiment=scene_scale_gate10|bash scripts/ray_train_qgate2.sh scene experiment=scene_scale_gate10 seed=${SEED}"
)
# task|cfg|run|launcher-args|out-prefix
EVALS=()
for spec in \
  "pusht|g05|lewm_pusht_scale_gate05_s${SEED}|ray_eval_final.sh pusht|final_eval" \
  "pusht|g10|lewm_pusht_scale_gate10_s${SEED}|ray_eval_final.sh pusht|final_eval" \
  "reacher|g05|lewm_reacher_scale_gate05_s${SEED}|ray_eval_final.sh reacher|final_eval" \
  "reacher|g10|lewm_reacher_scale_gate10_s${SEED}|ray_eval_final.sh reacher|final_eval" \
  "cube_double|g05|lewm_cubedouble_scale_gate05_s${SEED}|ray_eval_ogbmulti.sh cube_double|final_eval_ogbmulti" \
  "cube_double|g10|lewm_cubedouble_scale_gate10_s${SEED}|ray_eval_ogbmulti.sh cube_double|final_eval_ogbmulti" \
  "scene|g05|lewm_scene_scale_gate05_s${SEED}|ray_eval_ogbmulti.sh scene|final_eval_ogbmulti" \
  "scene|g10|lewm_scene_scale_gate10_s${SEED}|ray_eval_ogbmulti.sh scene|final_eval_ogbmulti"; do
  IFS='|' read -r task cfg run largs outp <<< "$spec"
  for sol in cem icem; do
    EVALS+=("${task}|${cfg}|${run}|${largs}|${outp}|${sol}|101 102 103")
    EVALS+=("${task}|${cfg}|${run}|${largs}|${outp}|${sol}|104 105 106")
  done
done

log "start: gated full-q SCALE, 4 tasks x lambda {0.05, 0.1}"
for round in $(seq 1 8000); do
  # phase 0: reacher lambda=0.1 Stage-1 gate
  if ! gcloud storage ls "$Q/qgate_stage1_reacher_lam0.1.json" >/dev/null 2>&1; then
    if [ "$(nrun "ray_qgate_stage1.sh reacher 0.1")" = 0 ] && [ "$(free)" -ge 1 ]; then
      n=${ATT[rc_gate10]:-0}
      if [ "$n" -lt 4 ]; then
        id=$(sub bash scripts/ray_qgate_stage1.sh reacher 0.1)
        [ -n "$id" ] && { ATT[rc_gate10]=$((n+1)); log "reacher lam0.1 gate attempt $((n+1)) -> $id"; }
      else log "rc_gate10 attempt cap"; fi
    fi
  fi
  left=0; submitted=0
  for spec in "${TRAINS[@]}"; do
    IFS='|' read -r name run gate probe cmd <<< "$spec"
    gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 && continue
    left=1
    gcloud storage ls "$gate" >/dev/null 2>&1 || continue
    [ "$(nrun "$probe")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    n=${ATT[$name]:-0}
    [ "$n" -ge 4 ] && { log "$name attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub env QGATE_GCS=$gate $cmd)
    if [ -n "$id" ]; then ATT[$name]=$((n+1)); log "$name attempt $((n+1)) -> $id"
    else log "$name submit FAILED"; fi
    submitted=1
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
    key="ev_${task}_${cfg}_${sol}_${seeds%% *}"
    n=${ATT[$key]:-0}
    [ "$n" -ge 4 ] && { log "$key attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub bash scripts/$largs "$cfg" "$run" "$sol" $seeds)
    if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"
    else log "$key submit FAILED"; fi
    submitted=1; break
  done
  [ "$left" = 0 ] && { log "GATED FULL-Q SCALE COMPLETE (8 arms, 96 CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
