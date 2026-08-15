#!/usr/bin/env bash
# Unified GPU-aware babysitter for ALL remaining DINO-WM evals, with one GPU RESERVED
# for the user's interactive use. Replaces run_dinowm_chain.sh (which submitted without
# throttling) and the two per-task babysitters (which filled every free GPU).
#
# Reservation rule: submit only while free > RESERVE, i.e. after our submission at least
# RESERVE GPUs remain idle. Jobs the user submits themselves count as busy and are never
# touched.
#
# Coverage (trainings are all done; ckpt copies verified present before this was started):
#   pusht/reacher/cube  ray_eval_final.sh    <task> dw dinowm_<task>_s3072  -> final_eval/
#   tworoom             ray_eval_tworoom.sh  dw dinowm_tworoom_s3072       -> final_eval_tworoom/
#   pointmaze           ray_eval_pointmaze.sh dw dinowm_pointmaze_s3072    -> final_eval_pointmaze/
# Skip-if-CSV-exists makes overlap with already-running jobs harmless; per-cell attempt
# caps stop infinite resubmission of a genuinely broken cell.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
RESERVE=0   # was 1; the user released the reserved GPU on 2026-08-12 -- all 8 for dw evals
SEEDS="101 102 103 104 105 106"
L=/workspace/le-wm/eval_results/babysit_dw_reserved.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L"; }
declare -A ATT

# Count actual GPU allocation, not running jobs: CPU-only jobs (viz/probe) would
# otherwise be mistaken for GPU occupants and idle a card for their whole runtime.
free(){ ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" \
  | awk -F'[/ ]' '{print int($2 - $1)}'; }
nrun(){ python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j['status'] in ('RUNNING','PENDING') and sys.argv[1] in (j.get('entrypoint') or '')))
PY
}
submit(){ timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
  --working-dir /workspace/le-wm --runtime-env-json "$EXC" -- "$@" 2>&1 \
  | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }

# task -> "eval-script csv-prefix gcs-result-prefix extra-first-arg"
row(){ case "$1" in
  pusht|reacher|cube) echo "ray_eval_final.sh final_${1}_dw final_eval $1";;
  tworoom)   echo "ray_eval_tworoom.sh final_tworoom_dw final_eval_tworoom -";;
  pointmaze) echo "ray_eval_pointmaze.sh final_pointmaze_dw final_eval_pointmaze -";;
esac; }

log "start: RESERVE=$RESERVE; solver priority icem/mppi > cem > gd (user, 08-14)"
for round in $(seq 1 2000); do
  # gd is gated: with spot capacity scarce and dw-gd seeds costing 10-20 h each,
  # no gd job is submitted while any icem/mppi cell is still missing.
  sampling_left=0
  for t in pointmaze tworoom cube reacher pusht; do
    set -- $(row "$t"); EVS=$1; PFX=$2; EVP=$3; TARG=$4
    for slv in icem mppi; do
      for s in $SEEDS; do
        gcloud storage ls "$BUCKET/$EVP/${PFX}_${slv}_s${s}.csv" >/dev/null 2>&1 || sampling_left=1
      done
    done
  done
  SOLVERS="icem mppi cem"
  [ "$sampling_left" = 0 ] && SOLVERS="icem mppi cem gd"
  left=0
  for t in pointmaze tworoom cube reacher pusht; do
    set -- $(row "$t"); EVS=$1; PFX=$2; EVP=$3; TARG=$4
    CK="dinowm_${t}_s3072"
    for slv in $SOLVERS; do
      ok=1
      for s in $SEEDS; do
        gcloud storage ls "$BUCKET/$EVP/${PFX}_${slv}_s${s}.csv" >/dev/null 2>&1 || ok=0
      done
      [ "$ok" = 1 ] && continue
      left=1
      if [ "$TARG" != "-" ]; then pat="$EVS $TARG dw $CK $slv"; else pat="$EVS dw $CK $slv"; fi
      [ "$(nrun "$pat")" != 0 ] && continue
      [ "$(free)" -le "$RESERVE" ] && continue
      k="${t}_$slv"; n=${ATT[$k]:-0}
      [ "$n" -ge 6 ] && { log "$k attempt cap (6) hit, leaving cell incomplete"; continue; }
      if [ "$TARG" != "-" ]; then id=$(submit bash "scripts/$EVS" "$TARG" dw "$CK" "$slv" $SEEDS)
      else id=$(submit bash "scripts/$EVS" dw "$CK" "$slv" $SEEDS); fi
      if [ -n "$id" ]; then ATT[$k]=$((n+1)); log "$t $slv attempt $((n+1)) -> $id"
      else log "$t $slv submit FAILED"; fi
      # At most ONE submission per round: `ray status` lags a fresh submission by
      # seconds, so a burst in one round oversubmits -- against a spot-shrunk cluster
      # (8 -> 3 GPUs on 08-14) that burned four attempts in 900s queue deaths.
      # left=1 is already set, so breaking cannot trigger the completion exit.
      break 2
    done
  done
  # only a round that scanned ALL solvers (gd included) may declare completion
  [ "$left" = 0 ] && [ "$sampling_left" = 0 ] && { log "ALL DINO-WM EVALS COMPLETE"; exit 0; }
  sleep 240
done
log "round cap hit"; exit 1
