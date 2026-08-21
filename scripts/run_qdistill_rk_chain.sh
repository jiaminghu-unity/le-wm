#!/usr/bin/env bash
# Metric-distillation extension to Reacher + Cube (2026-08-21, user-requested):
#   stage 1: train q-only-input TEACHERS (reacher 4-d joints q, cube 9-d effector q)
#   stage 2: train pixel DISTILL students (L_obj target = teacher geometry;
#            weights/q_variant paired to the raw-q SCALE full-q arms: 0.15 / 0.1)
#   stage 3: eval students cem + icem x 6 episode seeds (stock launcher; students
#            are plain pixel JEPAs in ckpts/)
# Teachers are training-side only here (their own planning eval needs per-task
# state plumbing; not part of this request). nohup babysitter conventions.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/qdistill_rk.log
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

# name|gate-ckpt(empty=none)|run-dir|probe|command...
TRAINS=(
"teach_r||lewm_qinput_reacher_s${SEED}|experiment=q_qinput_reacher|bash scripts/ray_train_qteacher.sh reacher experiment=q_qinput_reacher seed=${SEED}"
"teach_k||lewm_qinput_cube_s${SEED}|experiment=q_qinput_cube|bash scripts/ray_train_qteacher.sh cube experiment=q_qinput_cube seed=${SEED}"
"dist_r|lewm_qinput_reacher_s${SEED}|lewm_qdistill_reacher0.15_s${SEED}|experiment=qdistill_reacher|env TEACHER_CKPT=lewm_qinput_reacher_s${SEED}/weights_epoch_10.pt bash scripts/ray_train_qdistill.sh reacher experiment=qdistill_reacher seed=${SEED}"
"dist_k|lewm_qinput_cube_s${SEED}|lewm_qdistill_cube0.1_s${SEED}|experiment=qdistill_cube|env TEACHER_CKPT=lewm_qinput_cube_s${SEED}/weights_epoch_10.pt bash scripts/ray_train_qdistill.sh cube experiment=qdistill_cube seed=${SEED}"
)
EVALS=()
for spec in "reacher|lewm_qdistill_reacher0.15_s${SEED}" "cube|lewm_qdistill_cube0.1_s${SEED}"; do
  IFS='|' read -r task run <<< "$spec"
  for sol in cem icem; do
    EVALS+=("${task}|${run}|${sol}|101 102 103")
    EVALS+=("${task}|${run}|${sol}|104 105 106")
  done
done

log "start: qdistill reacher+cube (2 teachers -> 2 students -> cem/icem x 6)"
for round in $(seq 1 4000); do
  left=0; submitted=0
  for spec in "${TRAINS[@]}"; do
    IFS='|' read -r name gate run probe cmd <<< "$spec"
    gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 && continue
    left=1
    if [ -n "$gate" ]; then
      gcloud storage ls "$BUCKET/ckpts/$gate/weights_epoch_10.pt" >/dev/null 2>&1 || continue
    fi
    [ "$(nrun "$probe")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    n=${ATT[$name]:-0}
    [ "$n" -ge 4 ] && { log "$name attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub $cmd)
    if [ -n "$id" ]; then ATT[$name]=$((n+1)); log "$name attempt $((n+1)) -> $id"
    else log "$name submit FAILED"; fi
    submitted=1; break
  done
  if [ "$submitted" = 0 ]; then
    for spec in "${EVALS[@]}"; do
      IFS='|' read -r task run sol seeds <<< "$spec"
      gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 || { left=1; continue; }
      miss=0
      for s in $seeds; do
        gcloud storage ls "$BUCKET/final_eval/final_${task}_qdist_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
      done
      [ "$miss" = 0 ] && continue
      left=1
      [ "$(nrun "ray_eval_final.sh $task qdist $run $sol ${seeds%% *}")" != 0 ] && continue
      [ "$(free)" -lt 1 ] && continue
      key="ev_${task}_${sol}_${seeds%% *}"
      n=${ATT[$key]:-0}
      [ "$n" -ge 4 ] && { log "$key attempt cap"; continue; }
      # shellcheck disable=SC2086
      id=$(sub bash scripts/ray_eval_final.sh "$task" qdist "$run" "$sol" $seeds)
      if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"
      else log "$key submit FAILED"; fi
      break
    done
  fi
  [ "$left" = 0 ] && { log "QDISTILL R+K COMPLETE"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
