#!/usr/bin/env bash
# The dinowm chain burned tworoom's icem/mppi/gd retry attempts on Ray's 900s queue
# timeout (it submits without throttling to free GPUs -- my omission). This babysitter
# re-submits those solvers ONLY when a GPU is actually free, so submissions stop dying
# in the queue. Skip-if-present in ray_eval_tworoom.sh makes overlap with the chain
# harmless; when the CSVs exist the chain marks the task complete on its own.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/babysit_dw_tworoom.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L"; }
declare -A ATT
for round in $(seq 1 500); do
  left=0
  for slv in cem icem mppi gd; do
    ok=1
    for s in 101 102 103 104 105 106; do
      gcloud storage ls "$BUCKET/final_eval_tworoom/final_tworoom_dw_${slv}_s${s}.csv" >/dev/null 2>&1 || ok=0
    done
    [ "$ok" = 1 ] && continue
    left=1
    run=$(python3 - "$slv" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j['status'] in ('RUNNING','PENDING')
          and f'ray_eval_tworoom.sh dw dinowm_tworoom_s3072 {sys.argv[1]}' in (j.get('entrypoint') or '')))
PY
)
    [ "${run:-0}" != 0 ] && continue
    free=$(python3 - <<'PY' 2>/dev/null
import json,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(8-sum(1 for j in d if j['status'] in ('RUNNING','PENDING')))
PY
)
    [ "${free:-0}" -lt 1 ] && continue
    n=${ATT[$slv]:-0}; [ "$n" -ge 3 ] && { log "$slv: babysit cap"; continue; }
    ATT[$slv]=$((n+1))
    id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait --working-dir /workspace/le-wm \
      --runtime-env-json "$EXC" -- bash scripts/ray_eval_tworoom.sh dw dinowm_tworoom_s3072 "$slv" \
      101 102 103 104 105 106 2>&1 | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
    log "tworoom dw $slv attempt $((n+1)) -> ${id:-failed}"
  done
  [ "$left" = 0 ] && { log "TWOROOM DW EVAL COMPLETE"; exit 0; }
  sleep 300
done
