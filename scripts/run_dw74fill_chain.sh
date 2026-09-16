#!/usr/bin/env bash
# DINO-WM seed-3074 fill-in (2026-09-16, user: 没跑的也补上): pusht / reacher /
# tworoom trainings + evals. (scene 3074 was submitted separately; its evals are
# handled by run_seed74_chain.) Eval cfg = dwr74.
#   pusht:   cem/icem x6 (final_eval) + mppi_t T=64 s102-106
#   reacher: cem/icem x6 (final_eval, pinned render venv) + mppi_t T=32 s102-106
#   tworoom: cem/icem x6 (final_eval_tworoom, ckpts_tworoom mirror)
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3074
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/dw74fill.log
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
log "start: DINO-WM 3074 fill-in (pusht/reacher/tworoom)"
for round in $(seq 1 9000); do
  left=0
  for spec in "pusht 64 ckpts final_eval" "reacher 32 ckpts final_eval" "tworoom - ckpts_tworoom final_eval_tworoom"; do
    set -- $spec; task=$1; T=$2; ckdir=$3; evdir=$4
    run="dinowm_${task}_s${SEED}"
    if ! gcloud storage ls "$BUCKET/ckpts_dinowm/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
      left=1; try "tr_dw74_${task}" bash scripts/ray_train_dinowm.sh "$task" $SEED; continue
    fi
    gcloud storage ls "$BUCKET/$ckdir/$run/weights_epoch_10.pt" >/dev/null 2>&1 || \
      gcloud storage cp -r "$BUCKET/ckpts_dinowm/$run" "$BUCKET/$ckdir/"
    for sol in cem icem; do
      for seeds in "101 102 103" "104 105 106"; do
        miss=0
        for s in $seeds; do
          gcloud storage ls "$BUCKET/$evdir/final_${task}_dwr74_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
        done
        [ "$miss" = 0 ] && continue
        left=1
        if [ "$task" = tworoom ]; then
          # shellcheck disable=SC2086
          try "ev_dw74_${task}_${sol}_${seeds%% *}" bash scripts/ray_eval_tworoom.sh dwr74 "$run" "$sol" $seeds
        else
          # shellcheck disable=SC2086
          try "ev_dw74_${task}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh "$task" dwr74 "$run" "$sol" $seeds
        fi
      done
    done
    if [ "$T" != "-" ]; then
      m=0
      for s in 102 103 104 105 106; do
        gcloud storage ls "$BUCKET/final_eval_mppi_t/final_${task}_dwr74_mppiT${T}_s${s}.csv" >/dev/null 2>&1 || m=1
      done
      if [ "$m" = 1 ]; then
        left=1
        try "evm_dw74_${task}" bash scripts/ray_eval_mppi_t.sh "$task" dwr74 ckpts "$run" "$T" 102,103,104,105,106
      fi
    fi
  done
  [ "$left" = 0 ] && { log "DW74 FILL COMPLETE"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
