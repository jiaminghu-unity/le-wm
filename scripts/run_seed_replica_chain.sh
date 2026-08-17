#!/usr/bin/env bash
# Seed-replication chain: retrain every canonical model at a new training seed.
#   usage: SEED=3073 nohup scripts/run_seed_replica_chain.sh &
#
# 22 trainings: {LeWM, SCALE, Aux} x 5 tasks + DINO-WM x 5. SCALE uses the FINAL
# chosen q per task -- the best-of selection: reacher = half-q (shoulder cos/sin,
# hq_obj), cube = half-q (effector 5-d, hq_obj), pusht/tworoom/pointmaze = full q.
# Aux and LeWM are the canonical full-q arms everywhere. The full-q SCALE arms for
# reacher/cube (r2_l015, k2) are ALSO trained so both reporting frames replicate:
# best-vs-best (SR headline) and same-q pairing (mechanism isolation). Done-check = weights_epoch_10.pt under the family's
# ckpt prefix with the s$SEED name. GPU-aware via real allocation, one submission per
# round, per-run attempt cap. Coexists politely with other babysitters (both submit
# at most one job per free GPU per round).
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED="${SEED:?}"
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/seed_replica_${SEED}.log
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

# name|ckpt-prefix|done-run-name|command...
SPECS=(
"pusht_c1|ckpts|lewm_c1_s${SEED}|bash scripts/ray_train_replica.sh pusht experiment=c1_baseline data=pusht seed=${SEED}"
"pusht_c3|ckpts|lewm_c3_sig_obj0.1_s${SEED}|bash scripts/ray_train_replica.sh pusht experiment=c3_sig_plus_obj data=pusht seed=${SEED}"
"pusht_c5|ckpts|lewm_c5_qhead0.3_s${SEED}|bash scripts/ray_train_replica.sh pusht experiment=c5_qhead data=pusht loss.aux.weight=0.3 seed=${SEED}"
"reacher_r1|ckpts|lewm_r1_reacher_s${SEED}|bash scripts/ray_train_replica.sh reacher experiment=r1_reacher_baseline seed=${SEED}"
"reacher_hq_obj|ckpts_half|lewm_hq_obj_reacher_s${SEED}|bash scripts/ray_train_half_seed.sh reacher obj"
"reacher_r5|ckpts|lewm_r5_qhead0.4_s${SEED}|bash scripts/ray_train_replica.sh reacher experiment=r5_qhead loss.aux.weight=0.4 seed=${SEED}"
"reacher_r2_fullq|ckpts|lewm_r2_reacher_paep_l015_s${SEED}|bash scripts/ray_train_replica.sh reacher experiment=r2_reacher_paep loss.obj.weight=0.15 output_model_name=lewm_r2_reacher_paep_l015_s${SEED} seed=${SEED}"
"cube_k1|ckpts|lewm_k1_cube_s${SEED}|bash scripts/ray_train_replica.sh cube experiment=k1_cube_baseline seed=${SEED}"
"cube_hq_obj|ckpts_half|lewm_hq_obj_cube_s${SEED}|bash scripts/ray_train_half_seed.sh cube obj"
"cube_k4|ckpts|lewm_k4_cube_qhead_eff0.1_s${SEED}|bash scripts/ray_train_replica.sh cube experiment=k4_cube_qhead_eff seed=${SEED}"
"cube_k2_fullq|ckpts|lewm_k2_cube_obj_eff0.1_s${SEED}|bash scripts/ray_train_replica.sh cube experiment=k2_cube_obj_eff seed=${SEED}"
"tworoom_t1|ckpts_tworoom|lewm_t1_tworoom_s${SEED}|bash scripts/ray_train_tworoom_seed.sh base"
"tworoom_t2|ckpts_tworoom|lewm_t2_tworoom_obj0.1_s${SEED}|bash scripts/ray_train_tworoom_seed.sh obj"
"tworoom_t5|ckpts_tworoom|lewm_t5_tworoom_qhead0.1_s${SEED}|bash scripts/ray_train_tworoom_seed.sh aux"
"pointmaze_p1|ckpts_pointmaze|lewm_p1_pointmaze_s${SEED}|bash scripts/ray_train_pointmaze_seed.sh base"
"pointmaze_p2|ckpts_pointmaze|lewm_p2_pointmaze_s${SEED}|bash scripts/ray_train_pointmaze_seed.sh obj"
"pointmaze_p5|ckpts_pointmaze|lewm_p5_pointmaze_s${SEED}|bash scripts/ray_train_pointmaze_seed.sh aux"
"dw_pusht|ckpts_dinowm|dinowm_pusht_s${SEED}|bash scripts/ray_train_dinowm_seed.sh pusht"
"dw_reacher|ckpts_dinowm|dinowm_reacher_s${SEED}|bash scripts/ray_train_dinowm_seed.sh reacher"
"dw_cube|ckpts_dinowm|dinowm_cube_s${SEED}|bash scripts/ray_train_dinowm_seed.sh cube"
"dw_tworoom|ckpts_dinowm|dinowm_tworoom_s${SEED}|bash scripts/ray_train_dinowm_seed.sh tworoom"
"dw_pointmaze|ckpts_dinowm|dinowm_pointmaze_s${SEED}|bash scripts/ray_train_dinowm_seed.sh pointmaze"
)

log "start: seed=$SEED, ${#SPECS[@]} trainings"
for round in $(seq 1 4000); do
  left=0
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r name pfx run cmd <<< "$spec"
    gcloud storage ls "$BUCKET/$pfx/$run/weights_epoch_10.pt" >/dev/null 2>&1 && continue
    left=1
    [ "$(nrun "$cmd")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    n=${ATT[$name]:-0}
    [ "$n" -ge 4 ] && { log "$name attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(SEED=$SEED sub env SEED=$SEED $cmd)
    if [ -n "$id" ]; then ATT[$name]=$((n+1)); log "$name attempt $((n+1)) -> $id"
    else log "$name submit FAILED"; fi
    break   # one submission per round: ray status lags fresh allocations
  done
  [ "$left" = 0 ] && { log "ALL SEED-$SEED TRAININGS COMPLETE"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
