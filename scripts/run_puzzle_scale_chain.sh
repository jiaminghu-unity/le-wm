#!/usr/bin/env bash
# Puzzle-3x3 Stage-2 (2026-08-31): eval sets -> 6 arms {base, obj0.1, gated 01/03/05/10}
# -> cem/icem x 6 seeds = 72 CSVs. Gates already in GCS; env = swm_ext PuzzleEnv fork.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/puzzle_scale.log
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
log "start: puzzle_3x3 Stage-2, 6 arms"
for round in $(seq 1 9000); do
  left=0
  if ! gcloud storage ls "$BUCKET/eval_sets/episodes_puzzle_3x3_s106_100.json" >/dev/null 2>&1; then
    left=1
    try "sets_puzzle" bash scripts/ray_gen_ogbmulti_sets.sh puzzle_3x3
  else
    for arm in base obj g01 g03 g05 g10; do
      case $arm in
        base) run="lewm_puzzle_base_s${SEED}"; exp="puzzle_base"; gate="" ;;
        obj)  run="lewm_puzzle_obj0.1_s${SEED}"; exp="puzzle_obj"; gate="" ;;
        *)    g=${arm#g}; run="lewm_puzzle_scale_gate${g}_s${SEED}"; exp="puzzle_scale_gate${g}"
              case $g in 01) lam=0.01;; 03) lam=0.03;; 05) lam=0.05;; 10) lam=0.1;; esac
              gate="$Q/qgate_stage1_puzzle_3x3_lam${lam}.json" ;;
      esac
      if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
        left=1
        if [ -z "$gate" ]; then
          try "tr_$arm" bash scripts/ray_train_qnative.sh puzzle_3x3 experiment="$exp" seed=$SEED
        else
          try "tr_$arm" env QGATE_GCS="$gate" bash scripts/ray_train_qgate2.sh puzzle_3x3 experiment="$exp" seed=$SEED
        fi
        continue
      fi
      for sol in cem icem; do
        for seeds in "101 102 103" "104 105 106"; do
          miss=0
          for s in $seeds; do
            gcloud storage ls "$BUCKET/final_eval_ogbmulti/final_puzzle_3x3_${arm}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
          done
          [ "$miss" = 0 ] && continue
          left=1
          # shellcheck disable=SC2086
          try "ev_${arm}_${sol}_${seeds%% *}" bash scripts/ray_eval_ogbmulti.sh puzzle_3x3 "$arm" "$run" "$sol" $seeds
        done
      done
    done
  fi
  [ "$left" = 0 ] && { log "PUZZLE STAGE-2 COMPLETE (6 arms, 72 CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
