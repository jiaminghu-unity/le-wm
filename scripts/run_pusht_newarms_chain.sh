#!/usr/bin/env bash
# PushT new-arms chain (2026-08-20, user-requested, overnight):
#   train  q1  = q-only-input LeWM        (experiment=q1_qinput, seed 3072)
#          c9  = no-SIGReg + aux-only     (experiment=c9_qhead_nosig, seed 3072)
#          c2p = no-SIGReg + L_obj only   (experiment=c2p_obj_projector, seed 3072)
#   then eval each: cem + icem x 6 episode seeds (2 jobs per (cfg,solver), 3 seeds each).
# q1 evaluates through ray_eval_qinput.sh (state-based cost); c9/c2p through the
# stock ray_eval_final.sh. nohup babysitter: GCS done-checks, one submission per
# round, per-key attempt caps. No GPU reserve tonight (user asleep: "train as much
# as you can").
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/pusht_newarms.log
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

# ---- trainings: name|run-dir|command ----
TRAINS=(
"q1|lewm_q1_qinput_s${SEED}|bash scripts/ray_train_replica.sh pusht experiment=q1_qinput data=pusht seed=${SEED}"
"c9|lewm_c9_qhead_nosig0.3_s${SEED}|bash scripts/ray_train_replica.sh pusht experiment=c9_qhead_nosig data=pusht seed=${SEED}"
"c2p|lewm_c2p_obj0.1_s${SEED}|bash scripts/ray_train_replica.sh pusht experiment=c2p_obj_projector data=pusht seed=${SEED}"
)
# ---- evals: cfg|run-dir|launcher|solver|seeds ----
EVALS=()
for spec in "q1|lewm_q1_qinput_s${SEED}|ray_eval_qinput.sh" \
            "c9|lewm_c9_qhead_nosig0.3_s${SEED}|ray_eval_final.sh" \
            "c2p|lewm_c2p_obj0.1_s${SEED}|ray_eval_final.sh"; do
  IFS='|' read -r cfg run launcher <<< "$spec"
  for sol in cem icem; do
    EVALS+=("${cfg}|${run}|${launcher}|${sol}|101 102 103")
    EVALS+=("${cfg}|${run}|${launcher}|${sol}|104 105 106")
  done
done

log "start: pusht new arms (3 trainings -> cem/icem x 6 seeds each)"
for round in $(seq 1 4000); do
  left=0; submitted=0

  for spec in "${TRAINS[@]}"; do
    IFS='|' read -r name run cmd <<< "$spec"
    gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 && continue
    left=1
    [ "$(nrun "$run")" != 0 ] && continue
    # resolve-name check inside launcher uses overrides; nrun keys on run name via seed suffix
    [ "$(nrun "experiment=${name/q1/q1_qinput}")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    n=${ATT[tr_$name]:-0}
    [ "$n" -ge 4 ] && { log "tr_$name attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub $cmd)
    if [ -n "$id" ]; then ATT[tr_$name]=$((n+1)); log "train $name attempt $((n+1)) -> $id"
    else log "train $name submit FAILED"; fi
    submitted=1; break
  done
  if [ "$submitted" = 0 ]; then
    for spec in "${EVALS[@]}"; do
      IFS='|' read -r cfg run launcher sol seeds <<< "$spec"
      gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 || { left=1; continue; }
      miss=0
      for s in $seeds; do
        gcloud storage ls "$BUCKET/final_eval/final_pusht_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
      done
      [ "$miss" = 0 ] && continue
      left=1
      [ "$(nrun "$launcher pusht $cfg $run $sol ${seeds%% *}")" != 0 ] && continue
      [ "$(free)" -lt 1 ] && continue
      key="ev_${cfg}_${sol}_${seeds%% *}"
      n=${ATT[$key]:-0}
      [ "$n" -ge 4 ] && { log "$key attempt cap"; continue; }
      # shellcheck disable=SC2086
      id=$(sub bash scripts/$launcher pusht "$cfg" "$run" "$sol" $seeds)
      if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"
      else log "$key submit FAILED"; fi
      break
    done
  fi
  [ "$left" = 0 ] && { log "ALL PUSHT NEW-ARM RUNS COMPLETE"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
