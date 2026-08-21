#!/usr/bin/env bash
# qdistill chain (2026-08-21): SCALE with the q-only model's geometry as L_obj
# target (train_qdistill.py, teacher frozen) -> cem + icem x 6 episode seeds.
# Single arm; coexists with the 3073 replication chain (this one takes the
# analysis GPU, no reserve check beyond free>=1).
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/qdistill.log
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
"qdist|lewm_c10_qdistill0.1_s${SEED}|bash scripts/ray_train_qdistill.sh pusht experiment=c10_qdistill data=pusht seed=${SEED}"
)
# ---- evals: cfg|run-dir|launcher|solver|seeds ----
EVALS=()
for spec in "qdist|lewm_c10_qdistill0.1_s${SEED}|ray_eval_final.sh"; do
  IFS='|' read -r cfg run launcher <<< "$spec"
  for sol in cem icem; do
    EVALS+=("${cfg}|${run}|${launcher}|${sol}|101 102 103")
    EVALS+=("${cfg}|${run}|${launcher}|${sol}|104 105 106")
  done
done

log "start: qdistill (train -> cem/icem x 6 seeds)"
for round in $(seq 1 4000); do
  left=0; submitted=0

  for spec in "${TRAINS[@]}"; do
    IFS='|' read -r name run cmd <<< "$spec"
    gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 && continue
    left=1
    [ "$(nrun "$run")" != 0 ] && continue
    # resolve-name check inside launcher uses overrides; nrun keys on run name via seed suffix
    [ "$(nrun "experiment=c10_qdistill")" != 0 ] && continue
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
  [ "$left" = 0 ] && { log "QDISTILL COMPLETE"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
