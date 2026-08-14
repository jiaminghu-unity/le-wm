#!/usr/bin/env bash
# Unattended DINO-WM baseline: wait for the pointmaze training smoke, launch the other
# four trainings, then per task: copy the checkpoint into that task's eval prefix (a NEW
# directory -- nothing existing is written), run a one-cell eval smoke, and only if that
# succeeds run the full 4-solver x 6-seed sweep through the task's PROVEN eval script.
# Config label everywhere: dw. CSVs land as final_<task>_dw_<solver>_s<seed>.csv in the
# task's existing final_eval* prefix -- new names, no collisions.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
SEEDS="101 102 103 104 105 106"
LOG=/workspace/le-wm/eval_results/dinowm_chain.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

TASKS="pointmaze tworoom pusht reacher cube"
evprefix(){ case "$1" in pusht|reacher|cube) echo "ckpts|final_eval|ray_eval_final.sh|$1";;
  tworoom) echo "ckpts_tworoom|final_eval_tworoom|ray_eval_tworoom.sh|";;
  pointmaze) echo "ckpts_pointmaze|final_eval_pointmaze|ray_eval_pointmaze.sh|";; esac; }
ckpt_src(){ gcloud storage ls "$BUCKET/ckpts_dinowm/dinowm_$1_s3072/weights_epoch_10.pt" >/dev/null 2>&1; }
nrun(){ python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j.get('type')=='SUBMISSION' and j['status'] in ('RUNNING','PENDING')
          and sys.argv[1] in (j.get('entrypoint') or '')))
PY
}
submit(){ timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait --working-dir /workspace/le-wm \
  --runtime-env-json "$EXC" -- "$@" 2>&1 | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }

# ---- 1. wait for the pointmaze training smoke ----
DEADLINE=$(( $(date +%s) + 6*3600 ))
while ! ckpt_src pointmaze; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "FATAL: pointmaze dinowm training never finished"; exit 1; }
  [ "$(nrun 'ray_train_dinowm.sh pointmaze')" = 0 ] && { n=$(cat /tmp/dwt_pointmaze 2>/dev/null||echo 0)
    [ "$n" -ge 2 ] && { log "FATAL: pointmaze training retry cap"; exit 1; }
    echo $((n+1))>/tmp/dwt_pointmaze; log "pointmaze training attempt $((n+1)) -> $(submit bash scripts/ray_train_dinowm.sh pointmaze)"; }
  log "waiting for dinowm pointmaze checkpoint"; sleep 300
done
log "pointmaze dinowm checkpoint present — launching the other four trainings"
for t in tworoom pusht reacher cube; do
  ckpt_src "$t" || { echo 1 >/tmp/dwt_$t; log "train $t -> $(submit bash scripts/ray_train_dinowm.sh $t)"; }
done

# ---- 2. per-task: copy -> smoke -> sweep, as checkpoints appear ----
declare -A DONE SMOKED; declare -A EATT
for round in $(seq 1 900); do
  alldone=1
  for t in $TASKS; do
    [ "${DONE[$t]:-0}" = 1 ] && continue
    alldone=0
    if ! ckpt_src "$t"; then
      if [ "$(nrun "ray_train_dinowm.sh $t")" = 0 ]; then
        n=$(cat /tmp/dwt_$t 2>/dev/null||echo 0)
        if [ "$n" -lt 3 ]; then echo $((n+1))>/tmp/dwt_$t
          log "train $t retry $((n+1)) -> $(submit bash scripts/ray_train_dinowm.sh $t)"
        else log "train $t: retry cap"; DONE[$t]=fail; fi
      fi
      continue
    fi
    IFS='|' read -r CKP EVP EVS TARG <<< "$(evprefix "$t")"
    CK="dinowm_${t}_s3072"
    gcloud storage ls "$BUCKET/$CKP/$CK/weights_epoch_10.pt" >/dev/null 2>&1 || {
      log "copy $t ckpt -> $CKP/"; gcloud storage cp -r "$BUCKET/ckpts_dinowm/$CK" "$BUCKET/$CKP/"; }
    # smoke: one cell (cem, s101)
    if [ "${SMOKED[$t]:-0}" != 1 ]; then
      if gcloud storage ls "$BUCKET/$EVP/final_${t}_dw_cem_s101.csv" >/dev/null 2>&1; then
        SMOKED[$t]=1; log "$t eval smoke OK"
      elif [ "$(nrun "$EVS $TARG dw")" = 0 ] && [ "$(nrun "$EVS dw")" = 0 ]; then
        n=${EATT[${t}_smoke]:-0}
        if [ "$n" -lt 3 ]; then EATT[${t}_smoke]=$((n+1))
          if [ -n "$TARG" ]; then id=$(submit bash "scripts/$EVS" "$TARG" dw "$CK" cem 101)
          else id=$(submit bash "scripts/$EVS" dw "$CK" cem 101); fi
          log "$t eval smoke attempt $((n+1)) -> $id"
        else log "$t: eval smoke retry cap"; DONE[$t]=fail; fi
      fi
      continue
    fi
    # full sweep: per solver, all seeds (skip-if-present handles s101)
    missing=0
    for slv in cem icem mppi gd; do
      have=1
      for s in $SEEDS; do gcloud storage ls "$BUCKET/$EVP/final_${t}_dw_${slv}_s${s}.csv" >/dev/null 2>&1 || have=0; done
      [ "$have" = 1 ] && continue
      missing=1
      if [ "$(nrun "$EVS $TARG dw $CK $slv")" = 0 ] && [ "$(nrun "$EVS dw $CK $slv")" = 0 ]; then
        n=${EATT[${t}_$slv]:-0}
        if [ "$n" -lt 4 ]; then EATT[${t}_$slv]=$((n+1))
          if [ -n "$TARG" ]; then id=$(submit bash "scripts/$EVS" "$TARG" dw "$CK" "$slv" $SEEDS)
          else id=$(submit bash "scripts/$EVS" dw "$CK" "$slv" $SEEDS); fi
          log "$t $slv sweep attempt $((n+1)) -> $id"
        fi
      fi
    done
    [ "$missing" = 0 ] && { DONE[$t]=1; log "$t COMPLETE (24/24 CSVs)"; }
  done
  [ "$alldone" = 1 ] && { log "DINOWM BASELINE ALL DONE"; exit 0; }
  sleep 300
done
log "chain hit round cap"; exit 1
