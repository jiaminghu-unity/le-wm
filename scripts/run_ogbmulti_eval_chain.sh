#!/usr/bin/env bash
# Planning evals for the OGB multi-object q-only models (2026-08-29, user: q-only
# 新的几个接测). Phase 1 pre-registers episode sets episodes_{cube_double,scene}_s101-106
# (ray_gen_ogbmulti_sets.sh); phase 2 runs lewm_qinput_cubedouble_s3072 (27-d, cfg q1d)
# and lewm_qinput_scene_s3072 (26-d, cfg q1s) x cem/icem x 6 seeds via
# ray_eval_qinput_ogbmulti.sh -> final_eval_ogbmulti/. nohup babysitter conventions.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/ogbmulti_eval.log
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

sets_done(){ local t=$1 ok=1
  for s in 101 102 103 104 105 106; do
    gcloud storage ls "$BUCKET/eval_sets/episodes_${t}_s${s}_100.json" >/dev/null 2>&1 || ok=0
  done; echo $ok
}

# task|cfg|ckpt
MODELS=(
"cube_double|q1d|lewm_qinput_cubedouble_s3072"
"scene|q1s|lewm_qinput_scene_s3072"
)
CELLS=()
for sol in cem icem; do
  for m in "${MODELS[@]}"; do
    CELLS+=("$m|$sol|101 102 103")
    CELLS+=("$m|$sol|104 105 106")
  done
done

log "start: ogbmulti q-only evals (cube_double 27d + scene 26d, cem/icem x 6 seeds)"
for round in $(seq 1 6000); do
  left=0; submitted=0
  for t in cube_double scene; do
    [ "$(sets_done "$t")" = 1 ] && continue
    left=1
    [ "$(nrun "ray_gen_ogbmulti_sets.sh $t")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    [ "$submitted" != 0 ] && continue
    n=${ATT[gen_$t]:-0}
    [ "$n" -ge 4 ] && { log "gen_$t attempt cap"; continue; }
    id=$(sub bash scripts/ray_gen_ogbmulti_sets.sh "$t")
    if [ -n "$id" ]; then ATT[gen_$t]=$((n+1)); log "gen_$t attempt $((n+1)) -> $id"
    else log "gen_$t submit FAILED"; fi
    submitted=1
  done
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r task cfg ckpt sol seeds <<< "$cell"
    miss=0
    for s in $seeds; do
      gcloud storage ls "$BUCKET/final_eval_ogbmulti/final_${task}_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
    done
    [ "$miss" = 0 ] && continue
    left=1
    [ "$(sets_done "$task")" = 1 ] || continue
    [ "$submitted" != 0 ] && continue
    [ "$(nrun "ray_eval_qinput_ogbmulti.sh $task $cfg $ckpt $sol ${seeds%% *}")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    key="${task}_${sol}_${seeds%% *}"
    n=${ATT[$key]:-0}
    [ "$n" -ge 4 ] && { log "$key attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub bash scripts/ray_eval_qinput_ogbmulti.sh "$task" "$cfg" "$ckpt" "$sol" $seeds)
    if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"
    else log "$key submit FAILED"; fi
    submitted=1; break
  done
  [ "$left" = 0 ] && { log "OGBMULTI Q-ONLY EVALS COMPLETE (24 CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
