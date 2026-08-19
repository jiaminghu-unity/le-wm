#!/usr/bin/env bash
# NONLINEAR AutoMetric (residual-MLP embedding metric) on pusht and reacher --
# the two tasks where the linear W hurt. Train phi, then cem x 6 seeds. nohup babysitter: GPU-aware, one submission per round,
# done-checks against GCS, survives session restarts.
#   full q per task: pusht 6-d (=canonical), reacher 6-d joints+finger,
#                    tworoom 2-d (=canonical), pointmaze 2-d (=canonical)
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/wnl_pusht_reacher.log
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
declare -A ATT
free(){ ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" \
  | awk -F'[/ ]' '{print int($2 - $1)}'; }
nrun(){ python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j['status'] in ('RUNNING','PENDING') and sys.argv[1] in (j.get('entrypoint') or '')))
PY
}
sub(){ timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
  --working-dir /workspace/le-wm --runtime-env-json "$EXC" -- "$@" 2>&1 \
  | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }

# task|ckpt_prefix|ckpt_dir|tag|qvariant|eval_cfg
SPECS=(
"pusht|ckpts|lewm_c1_s3072|pusht_c1_NL|-|c1NL"
"reacher|ckpts|lewm_r1_reacher_s3072|reacher_r1_NL|-|r1NL"
)
SEEDS="101,102,103,104,105,106"

log "start: nonlinear metric on pusht+reacher (train + cem x 6 seeds)"
for round in $(seq 1 2000); do
  left=0
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r task ckp ckdir tag qvar cfg <<< "$spec"
    fit_done=0
    gcloud storage ls "$BUCKET/eval/automet_$tag.pt" >/dev/null 2>&1 && fit_done=1
    if [ "$fit_done" = 0 ]; then
      left=1
      [ "$(nrun "ray_automet_train_nl_any.sh $task")" != 0 ] && continue
      [ "$(free)" -lt 1 ] && continue
      n=${ATT[fit_$task]:-0}; [ "$n" -ge 4 ] && { log "fit_$task cap"; continue; }
      id=$(sub bash scripts/ray_automet_train_nl_any.sh "$task" "$ckp" "$ckdir" "$tag")
      [ -n "$id" ] && { ATT[fit_$task]=$((n+1)); log "fit $task -> $id"; }
      break
    fi
    miss=0
    for s in 101 102 103 104 105 106; do
      gcloud storage ls "$BUCKET/final_eval_automet/final_${task}_${cfg}_automet_cem_s${s}.csv" >/dev/null 2>&1 || miss=1
    done
    [ "$miss" = 0 ] && continue
    left=1
    [ "$(nrun "ray_eval_automet_nl_any.sh $task $cfg")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    n=${ATT[ev_$task]:-0}; [ "$n" -ge 4 ] && { log "ev_$task cap"; continue; }
    id=$(sub bash scripts/ray_eval_automet_nl_any.sh "$task" "$cfg" "$ckp" "$ckdir" "automet_$tag.pt" "$SEEDS")
    [ -n "$id" ] && { ATT[ev_$task]=$((n+1)); log "eval $task -> $id"; }
    break
  done
  [ "$left" = 0 ] && { log "NL RUNS COMPLETE"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
