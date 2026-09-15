#!/usr/bin/env bash
# Seed-3074 round (2026-09-15, user request): (a) finish the parked 3074 evals
# (reacher r1/r2/r5 render-fixed; pusht c3/c5 fill-in), (b) OGBench x 3074 with
# four arms per task — LeWM baseline / LeWM + full-q SCALE / LeWM + Auto-SCALE
# (per-task best gate from the 3072 grid) / DINO-WM.
#
# Best gated per task (mean of cem+icem overall at 3072):
#   cube            qgate03_scale_cube      (L1+hinge   lam 0.03)
#   cube_double     cubedouble_nce_gate03   (L1+InfoNCE lam 0.03)
#   cube_triple     cubetriple_scale_gate10 (L1+hinge   lam 0.1)
#   cube_quadruple  cubequadruple_nce_gate01(L1+InfoNCE lam 0.01)
#   scene           scene_nce_gate05        (L1+InfoNCE lam 0.05)
# Eval cfg suffix r74 keeps 3074 rows distinct. cube: cem/icem x6 (final_eval) +
# mppi_t T32 x5; multi: cem/icem x6 (final_eval_ogbmulti); reacher mppi_t T32,
# pusht mppi_t T64 (paper protocol).
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3074
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/seed74.log
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
declare -A ATT
free(){ python3 - <<'FREEPY' 2>/dev/null
import json, urllib.request
nodes = json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/v0/nodes?limit=100', timeout=20))
rows = nodes.get('data',{}).get('result',{}).get('result',[])
total = sum(n.get('resources_total',{}).get('GPU',0) for n in rows if n.get('state')=='ALIVE')
jobs = json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/', timeout=20))
used = sum(1 for j in jobs if j.get('status') in ('RUNNING','PENDING')
           and ('scripts/ray_' in (j.get('entrypoint') or '') or j.get('entrypoint_num_gpus')))
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
try(){ local key=$1; shift
  [ "$(nrun "$*")" != 0 ] && return 1
  [ "$(free)" -lt 1 ] && return 1
  local n=${ATT[$key]:-0}
  [ "$n" -ge 4 ] && { log "$key attempt cap"; return 1; }
  local id; id=$(sub "$@")
  if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"; else log "$key submit FAILED"; fi
}
Q=$BUCKET/qgate
# want_eval <evdir> <task> <cfg> <sol> <s...>: 0 if all present
missing(){ local dir=$1 task=$2 cfg=$3 sol=$4; shift 4
  for s in "$@"; do
    gcloud storage ls "$BUCKET/$dir/final_${task}_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || return 0
  done; return 1
}
log "start: seed-3074 round"
for round in $(seq 1 9000); do
  left=0

  # ---- (a) parked backfill: reacher + pusht (ckpts exist, evals only) ----
  for spec in \
    "reacher r1r74 lewm_r1_reacher_s3074 32" \
    "reacher r2r74 lewm_r2_reacher_paep_l015_s3074 32" \
    "reacher r5r74 lewm_r5_qhead0.4_s3074 32" \
    "pusht c1r74 lewm_c1_s3074 64" \
    "pusht c3r74 lewm_c3_sig_obj0.1_s3074 64" \
    "pusht c5r74 lewm_c5_qhead0.3_s3074 64"; do
    set -- $spec; task=$1; cfg=$2; run=$3; T=$4
    for sol in cem icem; do
      for seeds in "101 102 103" "104 105 106"; do
        # shellcheck disable=SC2086
        if missing final_eval "$task" "$cfg" "$sol" $seeds; then
          left=1
          # shellcheck disable=SC2086
          try "bk_${cfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh "$task" "$cfg" "$run" "$sol" $seeds
        fi
      done
    done
    m=0
    for s in 102 103 104 105 106; do
      gcloud storage ls "$BUCKET/final_eval_mppi_t/final_${task}_${cfg}_mppiT${T}_s${s}.csv" >/dev/null 2>&1 || m=1
    done
    if [ "$m" = 1 ]; then
      left=1
      try "bkm_${cfg}" bash scripts/ray_eval_mppi_t.sh "$task" "$cfg" ckpts "$run" "$T" 102,103,104,105,106
    fi
  done

  # ---- (b) OGBench x 3074 four arms ----
  #   spec: task | traincfg | eval-cfg | launcher(qn/qg/dw) | gate | evkind
  for spec in \
    "cube|k1_cube_baseline|k1r74|skip||final" \
    "cube|qall_scale_cube|qar74|qn||final" \
    "cube|qgate03_scale_cube|qg03r74|qg|$Q/qgate_stage1_cube_lam0.03.json|final" \
    "cube|dw_cube|dwr74|dw||final" \
    "cube_double|cubedouble_base|baser74|qn||multi" \
    "cube_double|cubedouble_obj|objr74|qn||multi" \
    "cube_double|cubedouble_nce_gate03|agr74|qg|$Q/qgate_stage1_cube_double_nce_lam0.03.json|multi" \
    "cube_double|dw_cube_double|dwr74|dw||multi" \
    "cube_triple|cubetriple_base|baser74|qn||multi" \
    "cube_triple|cubetriple_obj|objr74|qn||multi" \
    "cube_triple|cubetriple_scale_gate10|agr74|qg|$Q/qgate_stage1_cube_triple_lam0.1.json|multi" \
    "cube_triple|dw_cube_triple|dwr74|dw||multi" \
    "cube_quadruple|cubequadruple_base|baser74|qn||multi" \
    "cube_quadruple|cubequadruple_obj|objr74|qn||multi" \
    "cube_quadruple|cubequadruple_nce_gate01|agr74|qg|$Q/qgate_stage1_cube_quadruple_nce_lam0.01.json|multi" \
    "cube_quadruple|dw_cube_quadruple|dwr74|dw||multi" \
    "scene|scene_base|baser74|qn||multi" \
    "scene|scene_obj|objr74|qn||multi" \
    "scene|scene_nce_gate05|agr74|qg|$Q/qgate_stage1_scene_nce_lam0.05.json|multi" \
    "scene|dw_scene|dwr74|dw||multi"; do
    IFS='|' read -r task tcfg ecfg launch gate evkind <<< "$spec"
    if [ "$launch" = dw ]; then
      run="dinowm_${task}_s${SEED}"
      if ! gcloud storage ls "$BUCKET/ckpts_dinowm/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
        left=1; try "tr74_dw_${task}" bash scripts/ray_train_dinowm.sh "$task" $SEED; continue
      fi
      gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 || \
        gcloud storage cp -r "$BUCKET/ckpts_dinowm/$run" "$BUCKET/ckpts/"
    else
      run="$(python3 -c "
import re
t=open('config/train/experiment/${tcfg}.yaml').read()
m=re.search(r'output_model_name: (\S+)',t)
print(m.group(1).replace('\${seed}','${SEED}'))")"
      if [ "$launch" != skip ] && ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
        left=1
        if [ "$launch" = qg ]; then
          try "tr74_${tcfg}" env QGATE_GCS="$gate" bash scripts/ray_train_qgate2.sh "$task" experiment="$tcfg" seed=$SEED
        else
          try "tr74_${tcfg}" bash scripts/ray_train_qnative.sh "$task" experiment="$tcfg" seed=$SEED
        fi
        continue
      fi
      gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 || continue
    fi
    if [ "$evkind" = final ]; then
      for sol in cem icem; do
        for seeds in "101 102 103" "104 105 106"; do
          # shellcheck disable=SC2086
          if missing final_eval "$task" "$ecfg" "$sol" $seeds; then
            left=1
            # shellcheck disable=SC2086
            try "ev74_${task}_${ecfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh "$task" "$ecfg" "$run" "$sol" $seeds
          fi
        done
      done
      m=0
      for s in 102 103 104 105 106; do
        gcloud storage ls "$BUCKET/final_eval_mppi_t/final_${task}_${ecfg}_mppiT32_s${s}.csv" >/dev/null 2>&1 || m=1
      done
      if [ "$m" = 1 ]; then
        left=1
        try "evm74_${task}_${ecfg}" bash scripts/ray_eval_mppi_t.sh "$task" "$ecfg" ckpts "$run" 32 102,103,104,105,106
      fi
    else
      for sol in cem icem; do
        for seeds in "101 102 103" "104 105 106"; do
          # shellcheck disable=SC2086
          if missing final_eval_ogbmulti "$task" "$ecfg" "$sol" $seeds; then
            left=1
            # shellcheck disable=SC2086
            try "ev74_${task}_${ecfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_ogbmulti.sh "$task" "$ecfg" "$run" "$sol" $seeds
          fi
        done
      done
    fi
  done
  [ "$left" = 0 ] && { log "SEED74 ROUND COMPLETE"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
