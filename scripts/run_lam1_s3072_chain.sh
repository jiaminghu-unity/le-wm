#!/usr/bin/env bash
# λ=1.0 @ 种子3072 补块(2026-10-09):加固波 COMPLETE 后发现 3072 的 g100/nce100
# 16 格(4任务×2臂×2求解器)从未训练/评测——主链只循环 3073–3076,而 3072 的
# λ1.0 臂不在原 3072 实验里。本链复用主链语法,ecfg 不带 r 后缀以匹配 3072 命名。
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
Q=$BUCKET/qgate
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/lam1_s3072.log
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
  local n=${ATT[$key]:-0}; [ "$n" -ge 5 ] && { return 1; }
  local id
  case "$key" in tr_*) id=$(suba "$@");; *) id=$(sub "$@");; esac
  [ -n "$id" ] && { ATT[$key]=$((n+1)); log "$key -> $id"; }
}
missing(){ local task=$1 cfg=$2 sol=$3; shift 3
  for s in "$@"; do
    gcloud storage ls "$BUCKET/final_eval_ogbmulti/final_${task}_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || return 0
  done; return 1
}
SEED=3072
log "start: lam1.0 @3072 fill-in (16 solver-cells)"
for round in $(seq 1 2000); do
  left=0
  for spec in \
    "cube_double|cubedouble" "cube_triple|cubetriple" \
    "cube_quadruple|cubequadruple" "scene|scene"; do
    IFS='|' read -r task stem <<< "$spec"
    for arm in gate100 nce100; do
      case "$arm" in
        gate100) tcfg="${stem}_scale_gate100"; ecfg="g100";  g="$Q/qgate_stage1_${task}_lam1.0.json" ;;
        nce100)  tcfg="${stem}_nce_gate100";  ecfg="nce100"; g="$Q/qgate_stage1_${task}_nce_lam1.0.json" ;;
      esac
      if ! gcloud storage ls "$g" >/dev/null 2>&1; then
        left=1; log "WARN stage1 gate 缺失: $g"; continue
      fi
      run="$(python3 -c "
import re
t=open('config/train/experiment/${tcfg}.yaml').read()
m=re.search(r'output_model_name: (\S+)',t)
print(m.group(1).replace('\${seed}','${SEED}'))")"
      if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
        left=1
        try "tr_${tcfg}_${SEED}" env QGATE_GCS="$g" bash scripts/ray_train_qgate2.sh "$task" experiment="$tcfg" seed=$SEED
        continue
      fi
      for sol in cem icem; do
        for seeds in "101 102 103" "104 105 106"; do
          # shellcheck disable=SC2086
          if missing "$task" "$ecfg" "$sol" $seeds; then
            left=1
            # shellcheck disable=SC2086
            try "ev_${task}_${ecfg}_${sol}_${seeds%% *}" env EVAL_TIERS=T1 bash scripts/ray_eval_ogbmulti.sh "$task" "$ecfg" "$run" "$sol" $seeds
          fi
        done
      done
    done
  done
  [ "$left" = 0 ] && { log "LAM1 S3072 FILLIN COMPLETE"; exit 0; }
  sleep 300
done
