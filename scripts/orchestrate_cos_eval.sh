#!/usr/bin/env bash
# SR sweep with the planner cost switched to cosine distance -- 12 cells.
#   2 tasks x {baseline, L_obj, aux q-head} x {cem, icem} x 6 episode seeds
#
# WHY COSINE, and not simply "another norm". Squared L2 expands to
# ||z_hat||^2 - 2<z_hat,g> + const, so the three variants span the SIGN of the cost's
# dependence on the candidate's own latent magnitude: L2 penalises a large ||z_hat||,
# the dot product rewards it, and cosine is scale-invariant -- the neutral point.
# Neutrality is the one worth testing: MSE training without a stop-gradient shrinks
# predictions toward the mean when the predictor is unsure, which lowers ||z_hat||, and
# under L2 that shrinkage LOWERS the cost -- the planner gets a discount on exactly the
# candidates the model understands least. Cosine is immune to that by construction.
# It is also less entangled with the training objective than the dot product: SIGReg
# constrains the radial distribution, and cosine discards the radial component
# entirely, so it cannot read back what SIGReg shaped.
#
# PRE-REGISTRATION. Seeds s101-s106, the same six every other result uses, fixed before
# any cosine number lands, and all six reported. Solvers cem and icem only (both rank by
# cost alone; mppi rescales through a fixed softmax temperature and gd descends the cost
# gradient at a fixed lr). All five budget tiers are kept: the per-tier breakdown of the
# L1 results showed the largest deviations at T1 but no monotone pattern and SDs as large
# as the means, and restricting to T1 would lose more to per-seed noise than it gains.
#
# ALL THREE TASKS. Reacher was initially excluded because probe_latent_geometry.py
# measured tau(L2, cos) = 0.895-0.906 there with 98% of the same 30 elites kept, against
# tau 0.71-0.82 and 81-89% overlap on Push-T and Cube. That exclusion was withdrawn: the
# same tau/overlap reasoning predicted cosine would perturb outcomes 1.5-2x more than L1
# did, and a direct check found it perturbed FEWER episode outcomes (56 vs 61 of 500 on
# one seed). tau does not predict outcome change well enough to drop a task on, and the
# planner damps any cost change anyway -- CEM executes the MEAN of its 30 elites
# (cem.py:271), so swapping a few of them barely moves the action.
#
# HOW A NULL SHOULD BE READ, stated in advance because the L1 round taught it the hard
# way: L1 perturbed the ranking to tau 0.87-0.92 and returned a null across all three
# tasks, which was underpowered by construction rather than informative. Cosine is a
# 1.5-2x stronger perturbation but still keeps most of the elite set, so a null here is
# evidence about that perturbation size and NOT evidence that the cost function is
# irrelevant.
#
# The comparison is exactly paired against final_eval/: same checkpoints, same episode
# sets, same cem_seed = crc32("episode_id|tier"), same tiers, same code path.
# Results go to final_eval_cost/; nothing existing is written.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
# COST must ride in the runtime env: the job runs on a worker, so an exported shell
# variable here never reaches it -- the first launch failed all six jobs on exactly that.
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"],"env_vars":{"COST":"cos"}}'
SEEDS="101 102 103 104 105 106"
SOLVERS="cem icem"
export COST=cos
LOG=/workspace/le-wm/eval_results/cos_eval.log
mkdir -p "$(dirname "$LOG")"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# task | config label written into the CSV | checkpoint dir under ckpts/
JOBS=(
  "pusht   c1_cos       lewm_c1_s3072"
  "pusht   c3_l01_cos   lewm_c3_sig_obj0.1_s3072"
  "pusht   c5_l03_cos   lewm_c5_qhead0.3_s3072"
  "reacher r1_cos       lewm_r1_reacher_s3072"
  "reacher r2_l015_cos  lewm_r2_reacher_paep_l015_s3072"
  "reacher r5_l04_cos   lewm_r5_qhead0.4_s3072"
  "cube    k1_cos       lewm_k1_cube_s3072"
  "cube    k2_cos       lewm_k2_cube_obj_eff0.1_s3072"
  "cube    k4_cos       lewm_k4_cube_qhead_eff0.1_s3072"
)
declare -A ATTEMPTS

nrunning(){
  python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
pat=sys.argv[1]
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j.get('type')=='SUBMISSION' and j['status'] in ('RUNNING','PENDING')
          and pat in (j.get('entrypoint') or '')))
PY
}

for round in $(seq 1 600); do
  DONECSV=$(gcloud storage ls "$BUCKET/final_eval_cost/" 2>/dev/null | sed 's|.*/||')
  running=$(nrunning "ray_eval_cost.sh")
  todo=(); complete=0; total=0
  for spec in "${JOBS[@]}"; do
    set -- $spec; task=$1; cfg=$2; ck=$3
    for slv in $SOLVERS; do
      total=$((total+1)); missing=0
      for s in $SEEDS; do
        echo "$DONECSV" | grep -qx "final_${task}_${cfg}_${slv}_s${s}.csv" || missing=1
      done
      if [ "$missing" = 0 ]; then complete=$((complete+1))
      else
        key="${task}_${cfg}_${slv}"
        [ "${ATTEMPTS[$key]:-0}" -lt 5 ] && todo+=("$task $cfg $ck $slv")
      fi
    done
  done
  log "round $round: complete $complete/$total, running $running, todo ${#todo[@]}"
  [ "$complete" -ge "$total" ] && { log "COSINE SR SWEEP COMPLETE"; exit 0; }
  if [ ${#todo[@]} -eq 0 ] && [ "$running" -eq 0 ]; then
    log "nothing left to submit but cells missing (attempt cap hit) - stopping"; exit 1
  fi

  # Ray kills a job whose supervisor queues longer than 900s, so submit only as many
  # as there are idle GPUs and come back in three minutes.
  cap=$(ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" | cut -d/ -f2 | cut -d. -f1)
  busy=$(python3 - <<'PY' 2>/dev/null
import json,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j.get('type')=='SUBMISSION' and j['status'] in ('RUNNING','PENDING')))
PY
)
  # leave one GPU for the truth-agreement probe running alongside
  free=$(( ${cap:-8} - ${busy:-0} - 1 )); i=0
  while [ "$free" -gt 0 ] && [ "$i" -lt ${#todo[@]} ]; do
    set -- ${todo[$i]}; task=$1; cfg=$2; ck=$3; slv=$4
    key="${task}_${cfg}_${slv}"
    if [ "$(nrunning "ray_eval_cost.sh $task $cfg $ck $slv")" = "0" ]; then
      id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
            --working-dir /workspace/le-wm --runtime-env-json "$EXC" \
            -- bash scripts/ray_eval_cost.sh "$task" "$cfg" "$ck" "$slv" $SEEDS 2>&1 \
          | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
      if [ -n "$id" ]; then
        ATTEMPTS[$key]=$(( ${ATTEMPTS[$key]:-0} + 1 ))
        log "  submitted $task $cfg $slv -> $id (attempt ${ATTEMPTS[$key]})"
        free=$((free-1))
      else log "  submit FAILED for $task $cfg $slv"; fi
    fi
    i=$((i+1))
  done
  sleep 180
done
log "orchestrator hit round cap"
exit 1
