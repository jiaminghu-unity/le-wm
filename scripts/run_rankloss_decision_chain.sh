#!/usr/bin/env bash
# hinge vs InfoNCE 裁决补全(2026-10-06,user: "看看 hinge 和 infonce 最后可不可以选一个")。
# 全部 stage-1(分钟级/臂),四组:
#   1) reacher InfoNCE 安全阀:λ0.03/0.05 —— 盲 gap 是否仍≈0(InfoNCE 的 gap 普遍偏大,阀门必须复验)
#   2) pusht InfoNCE:λ0.03/0.05/0.1 —— 第二任务的 gate 集对比(现只有 hinge)
#   3) gate 集种子稳定性:cube λ0.03,hinge/nce × seed 1,2(seed 0 已有)
#   4) hinge margin 扫描:cube λ0.03,m∈{0.3,3,10}(m=1 已有;τ 扫描的公平对照)
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/rankloss_decision.log
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
try(){ local key=$1; shift
  [ "$(nrun "$*")" != 0 ] && return 1
  local n=${ATT[$key]:-0}; [ "$n" -ge 5 ] && return 1
  local id; id=$(sub "$@")
  [ -n "$id" ] && { ATT[$key]=$((n+1)); log "$key -> $id"; } || log "$key submit FAILED"
}
# spec: key|gate输出文件名|env串|task|lambdas
# 范围修正(2026-10-06 user):论文任务 = cube+double/triple/quadruple+scene;puzzle/pip 剔除,pusht/reacher 已撤。
SPECS=(
 "ch_s1|qgate_stage1_cube_s1_lam0.03.json|QGATE_SEED=1|cube|0.03"
 "ch_s2|qgate_stage1_cube_s2_lam0.03.json|QGATE_SEED=2|cube|0.03"
 "cn_s1|qgate_stage1_cube_nce_s1_lam0.03.json|QGATE_VARIANT=infonce QGATE_SEED=1|cube|0.03"
 "cn_s2|qgate_stage1_cube_nce_s2_lam0.03.json|QGATE_VARIANT=infonce QGATE_SEED=2|cube|0.03"
 "cm0p3|qgate_stage1_cube_m0p3_lam0.03.json|QGATE_MARGIN=0.3|cube|0.03"
 "cm3|qgate_stage1_cube_m3_lam0.03.json|QGATE_MARGIN=3|cube|0.03"
 "cm10|qgate_stage1_cube_m10_lam0.03.json|QGATE_MARGIN=10|cube|0.03"
)
log "start: rank-loss decision battery (12 stage-1 arms)"
for round in $(seq 1 400); do
  left=0
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r key out envs task lam <<< "$spec"
    gcloud storage ls "$BUCKET/qgate/$out" >/dev/null 2>&1 && continue
    left=1
    # shellcheck disable=SC2086
    try "$key" env $envs bash scripts/ray_qgate_stage1.sh "$task" "$lam"
  done
  [ "$left" = 0 ] && { log "RANK-LOSS DECISION BATTERY COMPLETE"; exit 0; }
  sleep 180
done
log "round cap"; exit 1
