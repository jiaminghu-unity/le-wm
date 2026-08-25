#!/usr/bin/env bash
# Planning eval for the reacher/cube q-only TEACHERS (previously train-side only):
# cem + icem x 6 episode seeds via budget_sweep_qinput_any. Closes the
# "teacher unvalidated" confound in the distillation cross-task table.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/qteacher_eval.log
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

CELLS=()
for spec in "reacher|lewm_qinput_reacher_s3072" "cube|lewm_qinput_cube_s3072"; do
  IFS='|' read -r task run <<< "$spec"
  for sol in cem icem; do
    CELLS+=("${task}|${run}|${sol}|101 102 103")
    CELLS+=("${task}|${run}|${sol}|104 105 106")
  done
done
log "start: q-only teacher planning evals (reacher/cube, cem+icem x 6)"
for round in $(seq 1 2000); do
  left=0
  for cell in "${CELLS[@]}"; do
    IFS='|' read -r task run sol seeds <<< "$cell"
    miss=0
    for s in $seeds; do
      gcloud storage ls "$BUCKET/final_eval/final_${task}_q1_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
    done
    [ "$miss" = 0 ] && continue
    left=1
    [ "$(nrun "ray_eval_qinput_any.sh $task q1 $run $sol ${seeds%% *}")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    key="${task}_${sol}_${seeds%% *}"
    n=${ATT[$key]:-0}
    [ "$n" -ge 4 ] && { log "$key attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub bash scripts/ray_eval_qinput_any.sh "$task" q1 "$run" "$sol" $seeds)
    if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"
    else log "$key submit FAILED"; fi
    break
  done
  [ "$left" = 0 ] && { log "TEACHER EVALS COMPLETE"; exit 0; }
  sleep 200
done
log "round cap"; exit 1
