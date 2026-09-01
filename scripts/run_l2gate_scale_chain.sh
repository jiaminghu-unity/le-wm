#!/usr/bin/env bash
# L2-gate Auto-SCALE arms (2026-09-01, user: L2 变体的 gate 训 Auto-SCALE 看 SR).
# 4 arms on cube, one per L2 Stage-1 lambda {0.01,0.03,0.05,0.1}, mirroring the L1
# dose-curve arms 1:1; evals cem/icem/mppi x 6 seeds = 72 CSVs (cfg ql2gXX).
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/l2gate_scale.log
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
log "start: L2-gate Auto-SCALE cube, 4 arms"
for round in $(seq 1 9000); do
  left=0
  for g in 01 03 05 10; do
    case $g in 01) lam=0.01;; 03) lam=0.03;; 05) lam=0.05;; 10) lam=0.1;; esac
    run="lewm_ql2g${g}_scale_cube_s${SEED}"
    gate="$Q/qgate_stage1_cube_l2_lam${lam}.json"
    if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
      left=1
      try "tr_ql2g${g}" env QGATE_GCS="$gate" bash scripts/ray_train_qgate2.sh cube experiment="ql2g${g}_scale_cube" seed=$SEED
      continue
    fi
    for sol in cem icem mppi; do
      for seeds in "101 102 103" "104 105 106"; do
        miss=0
        for s in $seeds; do
          gcloud storage ls "$BUCKET/final_eval/final_cube_ql2g${g}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
        done
        [ "$miss" = 0 ] && continue
        left=1
        # shellcheck disable=SC2086
        try "ev_ql2g${g}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh cube "ql2g${g}" "$run" "$sol" $seeds
      done
    done
  done
  [ "$left" = 0 ] && { log "L2-GATE AUTO-SCALE COMPLETE (4 arms, 72 CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
