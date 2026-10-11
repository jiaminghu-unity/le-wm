#!/usr/bin/env bash
# Stage-1 数据效率曲线(设计 2026-09-01 经用户确认;执行 2026-10-09,loss 已定案 hinge m=1)。
# cube,整轨迹子采样 N∈{25,50,100,250,500,1000,2000} × 子集种子{0,1,2} × λ{0.01,0.1}
# = 42 个 stage-1(每个 ~1 分钟计算);全量参照 = 既有 lam0.01/lam0.1 JSON。
# 判据(后处理):λ0.1 存活集恢复率 / 全维 g* Spearman vs 全量 / 盲差。
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
L=/workspace/le-wm/eval_results/data_eff.log
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
declare -A ATT
nrun(){ python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j['status'] in ('RUNNING','PENDING') and sys.argv[1] in (j.get('entrypoint') or '')))
PY
}
suba(){ timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
  --entrypoint-resources '{"accelerator_type:A100":0.001}' \
  --working-dir /workspace/le-wm \
  --runtime-env-json '{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}' \
  -- "$@" 2>&1 | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }
try(){ local key=$1; shift
  [ "$(nrun "$*")" != 0 ] && return 1
  local n=${ATT[$key]:-0}; [ "$n" -ge 5 ] && return 1
  local id; id=$(suba "$@")
  [ -n "$id" ] && { ATT[$key]=$((n+1)); log "$key -> $id"; }
}
log "start: stage-1 data-efficiency sweep (cube, hinge m=1)"
for round in $(seq 1 2000); do
  left=0
  for N in 25 50 100 250 500 1000 2000; do
    for DS in 0 1 2; do
      need=0
      for LAM in 0.01 0.1; do
        gcloud storage ls "$BUCKET/qgate/qgate_stage1_cube_n${N}_ds${DS}_lam${LAM}.json" >/dev/null 2>&1 || need=1
      done
      [ "$need" = 0 ] && continue
      left=1
      try "s1de_n${N}_ds${DS}" env QGATE_MAXEPS=$N QGATE_DATASEED=$DS bash scripts/ray_qgate_stage1.sh cube 0.01 0.1
    done
  done
  [ "$left" = 0 ] && { log "DATA EFFICIENCY SWEEP COMPLETE (42 jsons)"; exit 0; }
  sleep 240
done
