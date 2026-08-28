#!/usr/bin/env bash
# FIVE-TASK RETEST on fresh episode seed 1024 (2026-08-28, user: "五个task全部重测
# 不测dino 新种子1024"). Re-evaluates all 34 non-DINO-WM grid models (both training
# seeds of every arm) on ONE newly generated, pre-registered episode set per task
# (episodes_<task>_s1024_100.json; env_seed_base convention 40000+(S-101)*10000,
# no collision with any existing set). One job per model runs cem -> icem -> mppi
# sequentially at seed 1024 so worker setup is paid once. All outputs are NEW
# filenames (*_s1024.csv); nothing existing is touched. nohup babysitter conventions.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/retest1024.log
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

S=1024
# ---- phase 1: one fresh episode set per task ----
# gen-name|probe|set-files(space-sep)|command
GENS=(
"gen_pr|ray_gen_pusht_reacher_sets.sh|episodes_pusht_s${S}_100.json episodes_reacher_s${S}_100.json|SEEDS_OVERRIDE='$S' bash scripts/ray_gen_pusht_reacher_sets.sh"
"gen_cube|ray_gen_cube_sets_any.sh|episodes_cube_s${S}_100.json|SEEDS_OVERRIDE='$S' bash scripts/ray_gen_cube_sets_any.sh"
"gen_tworoom|ray_gen_tworoom_sets.sh|episodes_tworoom_s${S}_100.json|SEEDS_OVERRIDE='$S' GATE_ONLY=1 bash scripts/ray_gen_tworoom_sets.sh"
"gen_pointmaze|ray_gen_pointmaze_sets.sh|episodes_pointmaze_s${S}_100.json|SEEDS_OVERRIDE='$S' GATE_ONLY=1 bash scripts/ray_gen_pointmaze_sets.sh"
)
gen_done(){ for f in $1; do gcloud storage ls "$BUCKET/eval_sets/$f" >/dev/null 2>&1 || { echo 0; return; }; done; echo 1; }

# ---- phase 2 roster: task|cfg|launcher-args(ckpt included)|out-prefix ----
MODELS=(
"pusht|c1|ray_eval_final.sh pusht c1 lewm_c1_s3072|final_eval"
"pusht|c1r73|ray_eval_final.sh pusht c1r73 lewm_c1_s3073|final_eval"
"pusht|c3_l01|ray_eval_final.sh pusht c3_l01 lewm_c3_sig_obj0.1_s3072|final_eval"
"pusht|c3r73|ray_eval_final.sh pusht c3r73 lewm_c3_sig_obj0.1_s3073|final_eval"
"pusht|c5_l03|ray_eval_final.sh pusht c5_l03 lewm_c5_qhead0.3_s3072|final_eval"
"pusht|c5r73|ray_eval_final.sh pusht c5r73 lewm_c5_qhead0.3_s3073|final_eval"
"reacher|r1|ray_eval_final.sh reacher r1 lewm_r1_reacher_s3072|final_eval"
"reacher|r1r73|ray_eval_final.sh reacher r1r73 lewm_r1_reacher_s3073|final_eval"
"reacher|hq_obj|ray_eval_half.sh reacher hq_obj lewm_hq_obj_reacher_s3072|final_eval_half"
"reacher|hqor73|ray_eval_half.sh reacher hqor73 lewm_hq_obj_reacher_s3073|final_eval_half"
"reacher|r2_l015|ray_eval_final.sh reacher r2_l015 lewm_r2_reacher_paep_l015_s3072|final_eval"
"reacher|r2r73|ray_eval_final.sh reacher r2r73 lewm_r2_reacher_paep_l015_s3073|final_eval"
"reacher|r5_l04|ray_eval_final.sh reacher r5_l04 lewm_r5_qhead0.4_s3072|final_eval"
"reacher|r5r73|ray_eval_final.sh reacher r5r73 lewm_r5_qhead0.4_s3073|final_eval"
"cube|k1|ray_eval_final.sh cube k1 lewm_k1_cube_s3072|final_eval"
"cube|k1r73|ray_eval_final.sh cube k1r73 lewm_k1_cube_s3073|final_eval"
"cube|hq_obj|ray_eval_half.sh cube hq_obj lewm_hq_obj_cube_s3072|final_eval_half"
"cube|hqor73|ray_eval_half.sh cube hqor73 lewm_hq_obj_cube_s3073|final_eval_half"
"cube|k2|ray_eval_final.sh cube k2 lewm_k2_cube_obj_eff0.1_s3072|final_eval"
"cube|k2r73|ray_eval_final.sh cube k2r73 lewm_k2_cube_obj_eff0.1_s3073|final_eval"
"cube|k4|ray_eval_final.sh cube k4 lewm_k4_cube_qhead_eff0.1_s3072|final_eval"
"cube|k4r73|ray_eval_final.sh cube k4r73 lewm_k4_cube_qhead_eff0.1_s3073|final_eval"
"tworoom|t1|ray_eval_tworoom.sh t1 lewm_t1_tworoom_s3072|final_eval_tworoom"
"tworoom|t1r73|ray_eval_tworoom.sh t1r73 lewm_t1_tworoom_s3073|final_eval_tworoom"
"tworoom|t2|ray_eval_tworoom.sh t2 lewm_t2_tworoom_obj0.1_s3072|final_eval_tworoom"
"tworoom|t2r73|ray_eval_tworoom.sh t2r73 lewm_t2_tworoom_obj0.1_s3073|final_eval_tworoom"
"tworoom|t5|ray_eval_tworoom.sh t5 lewm_t5_tworoom_qhead0.1_s3072|final_eval_tworoom"
"tworoom|t5r73|ray_eval_tworoom.sh t5r73 lewm_t5_tworoom_qhead0.1_s3073|final_eval_tworoom"
"pointmaze|p1|ray_eval_pointmaze.sh p1 lewm_p1_pointmaze_s3072|final_eval_pointmaze"
"pointmaze|p1r73|ray_eval_pointmaze.sh p1r73 lewm_p1_pointmaze_s3073|final_eval_pointmaze"
"pointmaze|p2|ray_eval_pointmaze.sh p2 lewm_p2_pointmaze_s3072|final_eval_pointmaze"
"pointmaze|p2r73|ray_eval_pointmaze.sh p2r73 lewm_p2_pointmaze_s3073|final_eval_pointmaze"
"pointmaze|p5|ray_eval_pointmaze.sh p5 lewm_p5_pointmaze_s3072|final_eval_pointmaze"
"pointmaze|p5r73|ray_eval_pointmaze.sh p5r73 lewm_p5_pointmaze_s3073|final_eval_pointmaze"
)
SET_OF(){ case "$1" in pusht) echo "episodes_pusht_s${S}_100.json";; reacher) echo "episodes_reacher_s${S}_100.json";;
  cube) echo "episodes_cube_s${S}_100.json";; tworoom) echo "episodes_tworoom_s${S}_100.json";;
  pointmaze) echo "episodes_pointmaze_s${S}_100.json";; esac; }

log "start: five-task retest at fresh seed $S, 34 non-DINO models x cem/icem/mppi"
for round in $(seq 1 8000); do
  left=0; submitted=0
  for spec in "${GENS[@]}"; do
    IFS='|' read -r name probe files cmd <<< "$spec"
    [ "$(gen_done "$files")" = 1 ] && continue
    left=1
    [ "$(nrun "$probe")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    [ "$submitted" != 0 ] && continue
    n=${ATT[$name]:-0}
    [ "$n" -ge 4 ] && { log "$name attempt cap"; continue; }
    id=$(sub bash -c "$cmd")
    if [ -n "$id" ]; then ATT[$name]=$((n+1)); log "$name attempt $((n+1)) -> $id"
    else log "$name submit FAILED"; fi
    submitted=1
  done
  for spec in "${MODELS[@]}"; do
    IFS='|' read -r task cfg largs outp <<< "$spec"
    miss=0
    for sol in cem icem mppi; do
      gcloud storage ls "$BUCKET/$outp/final_${task}_${cfg}_${sol}_s${S}.csv" >/dev/null 2>&1 || miss=1
    done
    [ "$miss" = 0 ] && continue
    left=1
    [ "$(gen_done "$(SET_OF "$task")")" = 1 ] || continue
    [ "$submitted" != 0 ] && continue
    [ "$(nrun "$largs cem $S")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    key="${task}_${cfg}"
    n=${ATT[$key]:-0}
    [ "$n" -ge 4 ] && { log "$key attempt cap"; continue; }
    id=$(sub bash -c "bash scripts/$largs cem $S && bash scripts/$largs icem $S && bash scripts/$largs mppi $S")
    if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"
    else log "$key submit FAILED"; fi
    submitted=1
  done
  [ "$left" = 0 ] && { log "FIVE-TASK RETEST s$S COMPLETE (34 models x 3 solvers)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
