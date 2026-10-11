#!/usr/bin/env bash
# ContactWorld B3 + Ours 集群训练链(2026-10-09 用户直令"这俩你直接开始跑吧")。
# 各 1 个 A100 任务,nrun 去重 + 5 次重试帽;完成以 GCS 的 ckpt_last + COMPLETE 判定。
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/contactworld/cluster_bundle || exit 1
GCS=gs://prism-training-us/contactworld
L=/workspace/contactworld/cw_cluster.log
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
  --working-dir /workspace/contactworld/cluster_bundle \
  -- "$@" 2>&1 | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }
try(){ local key=$1; shift
  [ "$(nrun "$*")" != 0 ] && return 1
  local n=${ATT[$key]:-0}; [ "$n" -ge 5 ] && return 1
  local id; id=$(suba "$@")
  [ -n "$id" ] && { ATT[$key]=$((n+1)); log "$key -> $id"; }
}
# 数据上传完成门:zarr 体量到位才允许提交
for i in $(seq 1 120); do
  SZ=$(timeout 120 gcloud storage du -s "$GCS/datasets/insertion_usb" 2>/dev/null | awk '{print $1}')
  [ "${SZ:-0}" -gt 6000000000 ] && break
  log "等数据上传(当前 ${SZ:-0} B)"; sleep 60
done
log "start: CW cluster training (b3 + ours)"
for round in $(seq 1 9000); do
  left=0
  for V in b3 ours; do
    if gcloud storage ls "$GCS/outputs/${V}_cluster/COMPLETE" >/dev/null 2>&1; then continue; fi
    left=1
    try "cw_${V}" bash ray_train_cw.sh "$V"
  done
  [ "$left" = 0 ] && { log "CW CLUSTER TRAINING COMPLETE (b3+ours)"; exit 0; }
  sleep 600
done
