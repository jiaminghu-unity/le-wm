#!/usr/bin/env bash
# cube 单任务补网格(2026-10-09,user:"都做吧"):把 cube 补成与 4 个多物体任务同构的
# 5 种子满格 —— 12 臂(基线 k1 / 全q qa / hinge λ{.01 qg,.03 qg03,.05 qm,.1 qs,1.0 qg100}
# / NCE λ{.01..1.0 qnceg01..100})× 种子 3073–3076;λ1.0 两臂另补 3072。
# 评测:T1-only,cem/icem × 6 评测种子,落 final_eval/(ray_eval_final.sh)。
# 已有的 k1/qa/qg03 @r73/r74 由 ckpt+CSV 去重自动跳过。
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
Q=$BUCKET/qgate
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/cube_grid.log
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
declare -A ATT
nrun(){ python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j['status'] in ('RUNNING','PENDING') and sys.argv[1] in (j.get('entrypoint') or '')))
PY
}
sub(){ timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
  --working-dir /workspace/le-wm --runtime-env-json "$EXC" -- "$@" 2>&1 \
  | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }
suba(){ timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
  --entrypoint-resources '{"accelerator_type:A100":0.001}' \
  --working-dir /workspace/le-wm --runtime-env-json "$EXC" -- "$@" 2>&1 \
  | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }
try(){ local key=$1; shift
  [ "$(nrun "$*")" != 0 ] && return 1
  local n=${ATT[$key]:-0}; [ "$n" -ge 5 ] && return 1
  local id
  case "$key" in tr_*|s1_*) id=$(suba "$@");; *) id=$(sub "$@");; esac
  [ -n "$id" ] && { ATT[$key]=$((n+1)); log "$key -> $id"; }
}
missing(){ local cfg=$1 sol=$2; shift 2
  for s in "$@"; do
    gcloud storage ls "$BUCKET/final_eval/final_cube_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || return 0
  done; return 1
}
# arm 表:cfg前缀|experiment|json(-=无门)|launcher(qg=qgate2带门 qi=qgate2恒等 rep=replica)
ARMS=(
 "k1|k1_cube_baseline|-|rep"
 "qa|qall_scale_cube|-|qi"
 "qg|qgate_scale_cube|g_star_cube.json|qg"
 "qg03|qgate03_scale_cube|qgate_stage1_cube_lam0.03.json|qg"
 "qm|qgate05_scale_cube|qgate_stage1_cube_lam0.05.json|qg"
 "qs|qgate_sharp_scale_cube|qgate_stage1_cube_lam0.1.json|qg"
 "qg100|qgate100_scale_cube|qgate_stage1_cube_lam1.0.json|qg"
 "qnceg01|qnceg01_scale_cube|qgate_stage1_cube_nce_lam0.01.json|qg"
 "qnceg03|qnceg03_scale_cube|qgate_stage1_cube_nce_lam0.03.json|qg"
 "qnceg05|qnceg05_scale_cube|qgate_stage1_cube_nce_lam0.05.json|qg"
 "qnceg10|qnceg10_scale_cube|qgate_stage1_cube_nce_lam0.1.json|qg"
 "qnceg100|qnceg100_scale_cube|qgate_stage1_cube_nce_lam1.0.json|qg"
)
log "start: cube full-grid reinforcement (12 arms x seeds 3073-3076 + lam1.0@3072)"
for round in $(seq 1 9000); do
  left=0
  for SEED in 3072 3073 3074 3075 3076; do
    sfx="r${SEED#30}"; [ "$SEED" = 3072 ] && sfx=""
    for spec in "${ARMS[@]}"; do
      IFS='|' read -r pre tcfg json launch <<< "$spec"
      # 3072 只补 λ1.0 两臂(其余 3072 臂为原实验,已存在)
      if [ "$SEED" = 3072 ] && [ "$pre" != qg100 ] && [ "$pre" != qnceg100 ]; then continue; fi
      cfg="${pre}${sfx}"
      # λ1.0 NCE 门不存在则自举 stage-1
      if [ "$launch" = qg ] && ! gcloud storage ls "$Q/$json" >/dev/null 2>&1; then
        left=1
        case "$pre" in
          qg100)   try "s1_cube_lam1" bash scripts/ray_qgate_stage1.sh cube 1.0 ;;
          qnceg100) try "s1_cube_nce_lam1" env QGATE_VARIANT=infonce bash scripts/ray_qgate_stage1.sh cube 1.0 ;;
          *) log "WARN 缺门: $json" ;;
        esac
        continue
      fi
      run="lewm_${tcfg}_s${SEED}"
      [ "$pre" = k1 ] && run="lewm_k1_cube_s${SEED}"
      if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
        left=1
        case "$launch" in
          rep) try "tr_${tcfg}_${SEED}" bash scripts/ray_train_replica.sh cube experiment="$tcfg" seed=$SEED ;;
          qi)  try "tr_${tcfg}_${SEED}" bash scripts/ray_train_qgate2.sh cube experiment="$tcfg" seed=$SEED ;;
          qg)  try "tr_${tcfg}_${SEED}" env QGATE_GCS="$Q/$json" bash scripts/ray_train_qgate2.sh cube experiment="$tcfg" seed=$SEED ;;
        esac
        continue
      fi
      for sol in cem icem; do
        for seeds in "101 102 103" "104 105 106"; do
          # shellcheck disable=SC2086
          if missing "$cfg" "$sol" $seeds; then
            left=1
            # shellcheck disable=SC2086
            try "ev_cube_${cfg}_${sol}_${seeds%% *}" env EVAL_TIERS=T1 bash scripts/ray_eval_final.sh cube "$cfg" "$run" "$sol" $seeds
          fi
        done
      done
    done
  done
  [ "$left" = 0 ] && { log "CUBE GRID COMPLETE"; exit 0; }
  sleep 300
done
