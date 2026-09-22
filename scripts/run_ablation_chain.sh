#!/usr/bin/env bash
# Paper ablations round (2026-09-22, seed 3072):
#   (A) L_obj / aux weight sweeps on cube + pusht:
#       cube  obj  w in {0.03, 0.3, 1.0}  (0.1 exists as k2)   eval cfg k2_l003/l03/l1
#       cube  aux  w in {0.3, 1.0}        (0.1 exists as k4)   eval cfg k4_l03/l1
#       pusht obj  w in {0.03, 0.3, 1.0}  (0.1 exists as c3)   eval cfg c3_l003/l03/l1
#       pusht aux  w in {0.1, 1.0}        (0.3 exists as c5)   eval cfg c5_l01/l1
#       evals: cem/icem x 6 seeds -> final_eval/
#   (B) mppi temperature sweep: T in {8,32,64,128,256,512} x {baseline, SCALE}
#       x {pusht, reacher, cube, tworoom, pointmaze}; paper-protocol held-out
#       seeds 102-106; existing (cfg,T) cells skip via done-check.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/ablation.log
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
declare -A ATT
# capacity-targeted submission: submit up to TARGET(30) concurrent jobs so the
# queue creates GPU demand and the autoscaler grows toward max_workers=32.
# Occasional 900s pending-timeouts while nodes boot are absorbed by retries.
# dynamic target: current GPU capacity + 4 queued (steady scale-up pressure
# without mass 900s pending-timeouts during a zone stockout), capped at 30.
free(){ python3 - <<'FREEPY' 2>/dev/null
import json, urllib.request, subprocess
try:
    out=subprocess.run(['ray','list','nodes','--format','json'],capture_output=True,text=True,timeout=30).stdout
    cap=int(sum((r.get('resources_total') or {}).get('GPU',0) for r in json.loads(out) if r.get('state')=='ALIVE'))
except Exception:
    cap=8
jobs = json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/', timeout=20))
used = sum(1 for j in jobs if j.get('status') in ('RUNNING','PENDING')
           and ('scripts/ray_' in (j.get('entrypoint') or '') or j.get('entrypoint_num_gpus')))
print(max(min(cap+8,30)-used, 0))
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
try(){ local key=$1; shift
  [ "$(nrun "$*")" != 0 ] && return 1
  [ "$(free)" -lt 1 ] && return 1
  local n=${ATT[$key]:-0}
  [ "$n" -ge 12 ] && { log "$key attempt cap"; return 1; }
  local id; id=$(sub "$@")
  if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"; else log "$key submit FAILED"; fi
}
wtag(){ echo "l$(echo "$1" | tr -d '.' | sed 's/^0*//;s/^$/0/' )"; }  # 0.03->l003? see below
# weight -> label: 0.03 -> l003, 0.1 -> l01, 0.3 -> l03, 1.0 -> l1  (match legacy c3_l01/c5_l03)
lbl(){ case "$1" in 0.03) echo l003;; 0.1) echo l01;; 0.15) echo l015;; 0.3) echo l03;; 0.4) echo l04;; 1.0) echo l1;; *) echo "l$1";; esac; }
log "start: paper ablations (weight sweeps + mppi-T sweep)"
for round in $(seq 1 9000); do
  left=0
  # ---- (A) weight sweeps ----
  for spec in \
    "cube k2_cube_obj_eff loss.obj.weight lewm_k2_cube_obj_eff{W}_s${SEED} k2 0.03 0.15 0.3 1.0" \
    "cube k4_cube_qhead_eff loss.aux.weight lewm_k4_cube_qhead_eff{W}_s${SEED} k4 0.3 0.4 1.0" \
    "pusht c3_sig_plus_obj loss.obj.weight lewm_c3_sig_obj{W}_s${SEED} c3 0.03 0.15 0.3 1.0" \
    "pusht c5_qhead loss.aux.weight lewm_c5_qhead{W}_s${SEED} c5 0.1 0.4 1.0"; do
    set -- $spec; task=$1; exp=$2; key=$3; runpat=$4; short=$5; shift 5
    for W in "$@"; do
      run="${runpat/\{W\}/$W}"
      cfg="${short}_$(lbl $W)"
      if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
        left=1
        try "tr_${cfg}" bash scripts/ray_train_qnative.sh "$task" experiment="$exp" seed=$SEED "$key=$W"
        continue
      fi
      for sol in cem icem; do
        for seeds in "101 102 103" "104 105 106"; do
          miss=0
          for s in $seeds; do
            gcloud storage ls "$BUCKET/final_eval/final_${task}_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
          done
          [ "$miss" = 0 ] && continue
          left=1
          # shellcheck disable=SC2086
          try "ev_${cfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh "$task" "$cfg" "$run" "$sol" $seeds
        done
      done
    done
  done
  # ---- (A2) weight sweeps: reacher / tworoom / pointmaze ----
  # reacher obj (r2, literal name -> explicit output_model_name), paper 0.15 kept as its own row
  for W in 0.03 0.1 0.3 1.0; do
    cfg="r2_$(lbl $W)"; run="lewm_r2_reacher_paep_$(lbl $W)_s${SEED}"
    if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
      left=1; try "tr_${cfg}" bash scripts/ray_train_qnative.sh reacher experiment=r2_reacher_paep seed=$SEED "loss.obj.weight=$W" "output_model_name=$run"
    else
      for sol in cem icem; do for seeds in "101 102 103" "104 105 106"; do
        miss=0; for s2 in $seeds; do gcloud storage ls "$BUCKET/final_eval/final_reacher_${cfg}_${sol}_s${s2}.csv" >/dev/null 2>&1 || miss=1; done
        [ "$miss" = 0 ] && continue; left=1
        # shellcheck disable=SC2086
        try "ev_${cfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh reacher "$cfg" "$run" "$sol" $seeds
      done; done
    fi
  done
  # reacher aux (r5, templated name), paper 0.4 exists
  for W in 0.1 0.3 1.0; do
    cfg="r5_$(lbl $W)"; run="lewm_r5_qhead${W}_s${SEED}"
    if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
      left=1; try "tr_${cfg}" bash scripts/ray_train_qnative.sh reacher experiment=r5_qhead seed=$SEED "loss.aux.weight=$W"
    else
      for sol in cem icem; do for seeds in "101 102 103" "104 105 106"; do
        miss=0; for s2 in $seeds; do gcloud storage ls "$BUCKET/final_eval/final_reacher_${cfg}_${sol}_s${s2}.csv" >/dev/null 2>&1 || miss=1; done
        [ "$miss" = 0 ] && continue; left=1
        # shellcheck disable=SC2086
        try "ev_${cfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh reacher "$cfg" "$run" "$sol" $seeds
      done; done
    fi
  done
  # tworoom obj/aux (templated names; dedicated launcher+eval, ckpts_tworoom)
  for spec in "obj t2_tworoom_obj loss.obj.weight lewm_t2_tworoom_obj{W}_s${SEED} t2 0.03 0.15 0.3 1.0"               "aux t5_tworoom_qhead loss.aux.weight lewm_t5_tworoom_qhead{W}_s${SEED} t5 0.3 0.4 1.0"; do
    set -- $spec; arm=$1; exp=$2; key=$3; runpat=$4; short=$5; shift 5
    for W in "$@"; do
      run="${runpat/\{W\}/$W}"; cfg="${short}_$(lbl $W)"
      if ! gcloud storage ls "$BUCKET/ckpts_tworoom/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
        left=1; try "tr_${cfg}" bash scripts/ray_train_tworoom.sh "$arm" "$key=$W" "seed=$SEED"
      else
        for sol in cem icem; do for seeds in "101 102 103" "104 105 106"; do
          miss=0; for s2 in $seeds; do gcloud storage ls "$BUCKET/final_eval_tworoom/final_tworoom_${cfg}_${sol}_s${s2}.csv" >/dev/null 2>&1 || miss=1; done
          [ "$miss" = 0 ] && continue; left=1
          # shellcheck disable=SC2086
          try "ev_${cfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_tworoom.sh "$cfg" "$run" "$sol" $seeds
        done; done
      fi
    done
  done
  # pointmaze obj/aux (literal names -> explicit output_model_name)
  for spec in "obj p2_pointmaze_obj loss.obj.weight p2 0.03 0.15 0.3 1.0"               "aux p5_pointmaze_qhead loss.aux.weight p5 0.3 0.4 1.0"; do
    set -- $spec; arm=$1; exp=$2; key=$3; short=$4; shift 4
    for W in "$@"; do
      cfg="${short}_$(lbl $W)"; run="lewm_${short}_pointmaze_$(lbl $W)_s${SEED}"
      if ! gcloud storage ls "$BUCKET/ckpts_pointmaze/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
        left=1; try "tr_${cfg}" bash scripts/ray_train_pointmaze.sh "$arm" "$key=$W" "seed=$SEED" "output_model_name=$run"
      else
        for sol in cem icem; do for seeds in "101 102 103" "104 105 106"; do
          miss=0; for s2 in $seeds; do gcloud storage ls "$BUCKET/final_eval_pointmaze/final_pointmaze_${cfg}_${sol}_s${s2}.csv" >/dev/null 2>&1 || miss=1; done
          [ "$miss" = 0 ] && continue; left=1
          # shellcheck disable=SC2086
          try "ev_${cfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_pointmaze.sh "$cfg" "$run" "$sol" $seeds
        done; done
      fi
    done
  done

  # ---- (A3) full-q arms: paper-q vs full/native-q (only where they differ) ----
  # reacher: paper q = joints-only; full = native 8d (joints cos/sin + finger + qvel)
  run="lewm_r2_reacher_nativeq_s${SEED}"; cfg="r2_natq"
  if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
    left=1; try "tr_${cfg}" bash scripts/ray_train_qnative.sh reacher experiment=r2_reacher_paep seed=$SEED "loss.obj.q_variant=reacher_native_full" "output_model_name=$run"
  else
    for sol in cem icem; do for seeds in "101 102 103" "104 105 106"; do
      miss=0; for s2 in $seeds; do gcloud storage ls "$BUCKET/final_eval/final_reacher_${cfg}_${sol}_s${s2}.csv" >/dev/null 2>&1 || miss=1; done
      [ "$miss" = 0 ] && continue; left=1
      # shellcheck disable=SC2086
      try "ev_${cfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh reacher "$cfg" "$run" "$sol" $seeds
    done; done
  fi
  # pointmaze: paper q = pos 2d; full = state native 4d (pos+vel)
  run="lewm_p2_pointmaze_nativeq_s${SEED}"; cfg="p2_natq"
  if ! gcloud storage ls "$BUCKET/ckpts_pointmaze/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
    left=1; try "tr_${cfg}" bash scripts/ray_train_pointmaze.sh obj "loss.obj.q_variant=pointmaze_state_native" "seed=$SEED" "output_model_name=$run"
  else
    for sol in cem icem; do for seeds in "101 102 103" "104 105 106"; do
      miss=0; for s2 in $seeds; do gcloud storage ls "$BUCKET/final_eval_pointmaze/final_pointmaze_${cfg}_${sol}_s${s2}.csv" >/dev/null 2>&1 || miss=1; done
      [ "$miss" = 0 ] && continue; left=1
      # shellcheck disable=SC2086
      try "ev_${cfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_pointmaze.sh "$cfg" "$run" "$sol" $seeds
    done; done
  fi

  # ---- (B) mppi temperature sweep ----
  for spec in \
    "pusht c1 ckpts lewm_c1_s${SEED}" \
    "pusht c3_l01 ckpts lewm_c3_sig_obj0.1_s${SEED}" \
    "reacher r1 ckpts lewm_r1_reacher_s${SEED}" \
    "reacher r2_l015 ckpts lewm_r2_reacher_paep_l015_s${SEED}" \
    "cube k1 ckpts lewm_k1_cube_s${SEED}" \
    "cube k2_l01 ckpts lewm_k2_cube_obj_eff0.1_s${SEED}" \
    "tworoom t1 ckpts_tworoom lewm_t1_tworoom_s${SEED}" \
    "tworoom t2_l01 ckpts_tworoom lewm_t2_tworoom_obj0.1_s${SEED}" \
    "pointmaze p1 ckpts_pointmaze lewm_p1_pointmaze_s${SEED}" \
    "pointmaze p2_l01 ckpts_pointmaze lewm_p2_pointmaze_s${SEED}"; do
    set -- $spec; task=$1; cfg=$2; ckp=$3; run=$4
    for T in 8 32 64 128 256 512; do
      m=0
      for s in 102 103 104 105 106; do
        gcloud storage ls "$BUCKET/final_eval_mppi_t/final_${task}_${cfg}_mppiT${T}_s${s}.csv" >/dev/null 2>&1 || m=1
      done
      [ "$m" = 0 ] && continue
      left=1
      try "mp_${task}_${cfg}_T${T}" bash scripts/ray_eval_mppi_t.sh "$task" "$cfg" "$ckp" "$run" "$T" 102,103,104,105,106
    done
  done
  [ "$left" = 0 ] && { log "ABLATIONS COMPLETE"; exit 0; }
  sleep 60
done
log "round cap"; exit 1
