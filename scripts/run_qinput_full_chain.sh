#!/usr/bin/env bash
# NATIVE-FULL q-only trainings (2026-08-27, user: q-only 应该用原生完整 q):
# pusht 8-d (incl. velocities), reacher 6-d (joints+finger), cube 22-d (full
# config). Trainings only; planning evals follow once the per-variant eval-side
# q builders are wired. nohup babysitter conventions.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/qinput_full.log
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

# name|gate-ckpt(empty=none)|run-dir|probe|command...
TRAINS=(
"full_cube||lewm_qinput_full_cube_s${SEED}|experiment=q_qinput_full_cube|bash scripts/ray_train_qnative.sh cube experiment=q_qinput_full_cube seed=${SEED}"
"qi_tworoom||lewm_qinput_tworoom_s${SEED}|experiment=q_qinput_tworoom|bash scripts/ray_train_qnative.sh tworoom experiment=q_qinput_tworoom seed=${SEED}"
"qi_pointmaze||lewm_qinput_pointmaze_s${SEED}|experiment=q_qinput_pointmaze|bash scripts/ray_train_qnative.sh pointmaze experiment=q_qinput_pointmaze seed=${SEED}"
)
EVALS=()
# cfg|run|launcher(+隐含 env 参数形式)|solver|seeds
for spec in \
  "cube|q1f|lewm_qinput_full_cube_s${SEED}|ray_eval_qinput_any.sh cube" \
  "tworoom|q1|lewm_qinput_tworoom_s${SEED}|ray_eval_qinput_tworoom.sh" \
  "pointmaze|q1|lewm_qinput_pointmaze_s${SEED}|ray_eval_qinput_pointmaze.sh"; do
  IFS='|' read -r task cfg run largs <<< "$spec"
  for sol in cem icem; do
    EVALS+=("${task}|${cfg}|${run}|${largs}|${sol}|101 102 103")
    EVALS+=("${task}|${cfg}|${run}|${largs}|${sol}|104 105 106")
  done
done
declare -A OUTP=([cube]=final_eval [tworoom]=final_eval_tworoom [pointmaze]=final_eval_pointmaze)

log "start: q-only trainings v2 (cube 22d full-config / tworoom 2d / pointmaze 4d native)"
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
      IFS='|' read -r cfg run sol seeds <<< "$spec"
      gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 || { left=1; continue; }
      miss=0
      for s in $seeds; do
        gcloud storage ls "$BUCKET/final_eval/final_pusht_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
      done
      [ "$miss" = 0 ] && continue
      left=1
      [ "$(nrun "ray_eval_qinput.sh pusht $cfg $run $sol ${seeds%% *}")" != 0 ] && continue
      [ "$(free)" -lt 1 ] && continue
      key="ev_${cfg}_${sol}_${seeds%% *}"
      n=${ATT[$key]:-0}
      [ "$n" -ge 4 ] && { log "$key attempt cap"; continue; }
      # shellcheck disable=SC2086
      id=$(sub bash scripts/ray_eval_qinput.sh pusht "$cfg" "$run" "$sol" $seeds)
      if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"
      else log "$key submit FAILED"; fi
      break
    done
  fi
  if [ "$left" = 0 ]; then
    evleft=0
    for cell in "${EVALS[@]}"; do
      IFS='|' read -r task cfg run largs sol seeds <<< "$cell"
      miss=0
      for sd in $seeds; do
        gcloud storage ls "$BUCKET/${OUTP[$task]}/final_${task}_${cfg}_${sol}_s${sd}.csv" >/dev/null 2>&1 || miss=1
      done
      [ "$miss" = 0 ] && continue
      evleft=1
      [ "$(nrun "${largs%% *} ${largs#* } $cfg $run $sol ${seeds%% *}")" != 0 ] 2>/dev/null && continue
      [ "$(free)" -lt 1 ] && continue
      key="ev_${task}_${sol}_${seeds%% *}"
      n=${ATT[$key]:-0}
      [ "$n" -ge 4 ] && { log "$key attempt cap"; continue; }
      # shellcheck disable=SC2086
      if [ "$task" = cube ]; then
        id=$(sub bash scripts/ray_eval_qinput_any.sh cube "$cfg" "$run" "$sol" $seeds)
      else
        id=$(sub bash scripts/${largs} "$cfg" "$run" "$sol" $seeds)
      fi
      if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"
      else log "$key submit FAILED"; fi
      break
    done
    [ "$evleft" = 0 ] && { log "TRAININGS + EVALS ALL COMPLETE"; exit 0; }
  fi
  sleep 240
done
log "round cap"; exit 1
