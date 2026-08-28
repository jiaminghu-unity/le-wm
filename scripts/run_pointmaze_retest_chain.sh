#!/usr/bin/env bash
# PointMaze RETEST (2026-08-28, user: "重新测pointmaze 我觉得你测得不对").
# The stored 101-106 results audit clean (500 eps/cell, identical episode hashes,
# correct ckpts, uniform per-seed deltas), so a literal rerun would reproduce them.
# The one untested degree of freedom is the episode sample itself: this chain
# pre-registers SIX FRESH episode sets (seeds 201-206, env_seed_base per the
# 40000+(S-101)*10000 convention -> no collision with any existing set) and
# re-evaluates all 8 pointmaze models (4 arms x both training seeds) on them,
# cem -> icem -> mppi. All outputs are NEW filenames (s201-206); nothing existing
# is touched. nohup babysitter conventions.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/pointmaze_retest.log
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

RETEST_SEEDS="201 202 203 204 205 206"
sets_done(){ local ok=1
  for s in $RETEST_SEEDS; do
    gcloud storage ls "$BUCKET/eval_sets/episodes_pointmaze_s${s}_100.json" >/dev/null 2>&1 || ok=0
  done; echo $ok
}

# cfg|ckpt-dir  (both training seeds of all four arms)
MODELS=(
"p1|lewm_p1_pointmaze_s3072"    "p1r73|lewm_p1_pointmaze_s3073"
"p2|lewm_p2_pointmaze_s3072"    "p2r73|lewm_p2_pointmaze_s3073"
"p5|lewm_p5_pointmaze_s3072"    "p5r73|lewm_p5_pointmaze_s3073"
"dw|dinowm_pointmaze_s3072"     "dwr73|dinowm_pointmaze_s3073"
)
CELLS=()
for sol in cem icem mppi; do
  for m in "${MODELS[@]}"; do
    CELLS+=("$m|$sol|201 202 203")
    CELLS+=("$m|$sol|204 205 206")
  done
done

log "start: pointmaze retest on fresh pre-registered sets s201-206, 8 models x 3 solvers"
for round in $(seq 1 8000); do
  # ---- phase 1: fresh episode sets ----
  if [ "$(sets_done)" != 1 ]; then
    if [ "$(nrun "ray_gen_pointmaze_sets.sh")" = 0 ] && [ "$(free)" -ge 1 ]; then
      n=${ATT[gen]:-0}
      if [ "$n" -lt 4 ]; then
        id=$(sub bash -c "SEEDS_OVERRIDE='$RETEST_SEEDS' GATE_ONLY=1 bash scripts/ray_gen_pointmaze_sets.sh")
        if [ -n "$id" ]; then ATT[gen]=$((n+1)); log "gen sets attempt $((n+1)) -> $id"
        else log "gen sets submit FAILED"; fi
      else log "gen sets attempt cap"; fi
    fi
    sleep 240; continue
  fi
  # ---- phase 2: evals ----
  left=0
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r cfg ckpt sol seeds <<< "$cell"
    miss=0
    for s in $seeds; do
      gcloud storage ls "$BUCKET/final_eval_pointmaze/final_pointmaze_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
    done
    [ "$miss" = 0 ] && continue
    left=1
    [ "$(nrun "ray_eval_pointmaze.sh $cfg $ckpt $sol ${seeds%% *}")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    key="${cfg}_${sol}_${seeds%% *}"
    n=${ATT[$key]:-0}
    [ "$n" -ge 4 ] && { log "$key attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub bash scripts/ray_eval_pointmaze.sh "$cfg" "$ckpt" "$sol" $seeds)
    if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"
    else log "$key submit FAILED"; fi
    break
  done
  [ "$left" = 0 ] && { log "POINTMAZE RETEST COMPLETE (48 half-cells)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
