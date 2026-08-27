#!/usr/bin/env bash
# Seed-3073 replication EVAL chain (user-approved 2026-08-20): 22 retrained models
# x {cem, icem, mppi} x 6 episode seeds x 5 tiers. gd excluded (user). mppi at the
# stock T=0.5 -- strictly the same protocol as the s3072 main grid.
# Ordering: all cem cells first (headline earliest), then icem, then mppi.
# RESERVE=1: always leave one GPU free for the user ("给我留一个卡").
# DINO-WM ckpts are copied from ckpts_dinowm/ into each task's eval prefix under a
# NEW dir name (dinowm_<task>_s3073) exactly like the s3072 flow -- nothing existing
# is written. nohup babysitter: GCS done-checks, one submission per round, caps.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/replica_eval_3073.log
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

# ---- one-time: stage DINO-WM 3073 ckpts into each task's eval prefix ----
declare -A DWPFX=([pusht]=ckpts [reacher]=ckpts [cube]=ckpts \
                  [tworoom]=ckpts_tworoom [pointmaze]=ckpts_pointmaze)
for t in pusht reacher cube tworoom pointmaze; do
  dst="$BUCKET/${DWPFX[$t]}/dinowm_${t}_s3073"
  gcloud storage ls "$dst/weights_epoch_10.pt" >/dev/null 2>&1 && continue
  log "staging dinowm_${t}_s3073 -> ${DWPFX[$t]}/"
  gcloud storage cp "$BUCKET/ckpts_dinowm/dinowm_${t}_s3073/weights_epoch_10.pt" \
                    "$BUCKET/ckpts_dinowm/dinowm_${t}_s3073/config.json" "$dst/" \
    || log "WARN: staging dinowm_${t}_s3073 failed"
done

# ---- model roster: launcher-args|out-prefix|csv-head ----
# launcher-args: everything after `bash scripts/` up to solver+seeds
MODELS=(
"ray_eval_final.sh pusht c1r73 lewm_c1_s3073|final_eval|final_pusht_c1r73"
"ray_eval_final.sh pusht c3r73 lewm_c3_sig_obj0.1_s3073|final_eval|final_pusht_c3r73"
"ray_eval_final.sh pusht c5r73 lewm_c5_qhead0.3_s3073|final_eval|final_pusht_c5r73"
"ray_eval_final.sh pusht dwr73 dinowm_pusht_s3073|final_eval|final_pusht_dwr73"
"ray_eval_final.sh reacher r1r73 lewm_r1_reacher_s3073|final_eval|final_reacher_r1r73"
"ray_eval_half.sh reacher hqor73 lewm_hq_obj_reacher_s3073|final_eval_half|final_reacher_hqor73"
"ray_eval_final.sh reacher r5r73 lewm_r5_qhead0.4_s3073|final_eval|final_reacher_r5r73"
"ray_eval_final.sh reacher r2r73 lewm_r2_reacher_paep_l015_s3073|final_eval|final_reacher_r2r73"
"ray_eval_final.sh reacher dwr73 dinowm_reacher_s3073|final_eval|final_reacher_dwr73"
"ray_eval_final.sh cube k1r73 lewm_k1_cube_s3073|final_eval|final_cube_k1r73"
"ray_eval_half.sh cube hqor73 lewm_hq_obj_cube_s3073|final_eval_half|final_cube_hqor73"
"ray_eval_final.sh cube k4r73 lewm_k4_cube_qhead_eff0.1_s3073|final_eval|final_cube_k4r73"
"ray_eval_final.sh cube k2r73 lewm_k2_cube_obj_eff0.1_s3073|final_eval|final_cube_k2r73"
"ray_eval_final.sh cube dwr73 dinowm_cube_s3073|final_eval|final_cube_dwr73"
"ray_eval_tworoom.sh t1r73 lewm_t1_tworoom_s3073|final_eval_tworoom|final_tworoom_t1r73"
"ray_eval_tworoom.sh t2r73 lewm_t2_tworoom_obj0.1_s3073|final_eval_tworoom|final_tworoom_t2r73"
"ray_eval_tworoom.sh t5r73 lewm_t5_tworoom_qhead0.1_s3073|final_eval_tworoom|final_tworoom_t5r73"
"ray_eval_tworoom.sh dwr73 dinowm_tworoom_s3073|final_eval_tworoom|final_tworoom_dwr73"
"ray_eval_pointmaze.sh p1r73 lewm_p1_pointmaze_s3073|final_eval_pointmaze|final_pointmaze_p1r73"
"ray_eval_pointmaze.sh p2r73 lewm_p2_pointmaze_s3073|final_eval_pointmaze|final_pointmaze_p2r73"
"ray_eval_pointmaze.sh p5r73 lewm_p5_pointmaze_s3073|final_eval_pointmaze|final_pointmaze_p5r73"
"ray_eval_pointmaze.sh dwr73 dinowm_pointmaze_s3073|final_eval_pointmaze|final_pointmaze_dwr73"
)

# cells ordered cem -> icem -> mppi, each (model, solver) split into 2 seed-halves
CELLS=()
for sol in cem icem mppi; do
  for m in "${MODELS[@]}"; do
    CELLS+=("$m|$sol|101 102 103")
    CELLS+=("$m|$sol|104 105 106")
  done
done

log "start: 3073 replication eval, ${#MODELS[@]} models x cem/icem/mppi, RESERVE=1"
for round in $(seq 1 8000); do
  left=0
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r largs outp head sol seeds <<< "$cell"
    miss=0
    for s in $seeds; do
      gcloud storage ls "$BUCKET/$outp/${head}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
    done
    [ "$miss" = 0 ] && continue
    left=1
    probe="$largs $sol ${seeds%% *}"
    [ "$(nrun "$probe")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue   # reserve lifted 2026-08-26 (user: 卡都用满)
    key="${head}_${sol}_${seeds%% *}"
    n=${ATT[$key]:-0}
    [ "$n" -ge 4 ] && { log "$key attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub bash scripts/$largs "$sol" $seeds)
    if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"
    else log "$key submit FAILED"; fi
    break
  done
  [ "$left" = 0 ] && { log "ALL 3073 REPLICATION EVALS COMPLETE"; exit 0; }
  sleep 200
done
log "round cap"; exit 1
