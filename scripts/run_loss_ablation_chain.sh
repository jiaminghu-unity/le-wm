#!/usr/bin/env bash
# 损失部件消融(2026-10-07 user:"loss 三个部件,拆下来看看少了这个部分差多少")。
# 三消融臂 + 两个完整基准(已有):
#   norank = NLL+L1(去排序)       -> 门 qgate_stage1_cube_norank_lam0.03.json -> cfg qabl_norank
#   nonll  = hinge+L1(去 NLL)     -> 门 ..._nonll_lam0.03.json                -> cfg qabl_nonll
#   lam0   = NLL+hinge(去稀疏 λ=0) -> 门 ..._lam0.json                        -> cfg qabl_lam0
# 各臂:stage-1 门 -> stage-2 训练(seed 3072,A100)-> cem/icem × 6 种子(12 CSV)
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/loss_ablation.log
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
  case "$key" in tr_*) id=$(suba "$@");; *) id=$(sub "$@");; esac
  [ -n "$id" ] && { ATT[$key]=$((n+1)); log "$key -> $id"; }
}
# spec: cfg|门文件|stage1 env串与λ
SPECS=(
 "qabl_norank|qgate_stage1_cube_norank_lam0.03.json|QGATE_VARIANT=norank|0.03"
 "qabl_nonll|qgate_stage1_cube_nonll_lam0.03.json|QGATE_VARIANT=nonll_hinge|0.03"
 "qabl_lam0|qgate_stage1_cube_lam0.json||0"
)
log "start: loss-component ablation (3 arms)"
for round in $(seq 1 9000); do
  left=0
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r cfg gate envs lam <<< "$spec"
    if ! gcloud storage ls "$BUCKET/qgate/$gate" >/dev/null 2>&1; then
      left=1
      # shellcheck disable=SC2086
      try "s1_${cfg}" env $envs bash scripts/ray_qgate_stage1.sh cube "$lam"
      continue
    fi
    run="lewm_${cfg}_scale_cube_s${SEED}"
    if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
      left=1
      try "tr_${cfg}" env QGATE_GCS="$BUCKET/qgate/$gate" bash scripts/ray_train_qgate2.sh cube experiment="${cfg}_scale_cube" seed=$SEED
      continue
    fi
    for sol in cem icem; do
      for seeds in "101 102 103" "104 105 106"; do
        miss=0
        for s in $seeds; do gcloud storage ls "$BUCKET/final_eval/final_cube_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1; done
        [ "$miss" = 0 ] && continue
        left=1
        # shellcheck disable=SC2086
        try "ev_${cfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh cube "${cfg}" "$run" "$sol" $seeds
      done
    done
  done
  [ "$left" = 0 ] && { log "LOSS ABLATION COMPLETE (3 arms, 36 CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
