#!/usr/bin/env bash
# Wait for the six reduced-q checkpoints to land in GCS, then run the whole evaluation.
#
# Detached on purpose (nohup): the training finishes unattended, and the hand-off must
# not depend on any interactive session still being alive.
#
# The completion signal is the checkpoint in GCS, not the Ray job status: a job can
# report SUCCEEDED with the upload having failed, and the evaluation needs the file.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
LOG=/workspace/le-wm/eval_results/half_eval.log
mkdir -p "$(dirname "$LOG")"
log(){ echo "[$(date -u +%H:%M:%S)] [wait] $*" | tee -a "$LOG"; }

DEADLINE=$(( $(date +%s) + 14*3600 ))   # training needs ~8h; 14h covers one preemption
while :; do
  have=0; missing=""
  for t in pusht reacher cube; do for a in obj aux; do
    if gcloud storage ls "$BUCKET/ckpts_half/lewm_hq_${a}_${t}_s3072/weights_epoch_10.pt" \
       >/dev/null 2>&1; then have=$((have+1)); else missing="$missing hq_${a}_${t}"; fi
  done; done
  log "checkpoints $have/6;${missing:- none missing}"
  [ "$have" -ge 6 ] && break
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    log "FATAL: deadline reached with $have/6 checkpoints — not starting evaluation"
    exit 1
  fi
  # if every training job has exited and checkpoints are still missing, waiting is futile
  alive=$(python3 - <<'PY' 2>/dev/null
import json,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j.get('type')=='SUBMISSION'
          and 'ray_train_half.sh' in (j.get('entrypoint') or '')
          and j['status'] in ('RUNNING','PENDING')))
PY
)
  if [ "${alive:-0}" = "0" ] && [ "$have" -lt 6 ]; then
    log "no training job in flight and only $have/6 checkpoints — resubmitting the gaps"
    for t in pusht reacher cube; do for a in obj aux; do
      gcloud storage ls "$BUCKET/ckpts_half/lewm_hq_${a}_${t}_s3072/weights_epoch_10.pt" \
        >/dev/null 2>&1 && continue
      n=$(cat "/tmp/half_retry_${a}_${t}" 2>/dev/null || echo 0)
      if [ "$n" -ge 3 ]; then log "  hq_${a}_${t}: retry cap reached, giving up"; continue; fi
      echo $((n+1)) > "/tmp/half_retry_${a}_${t}"
      id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
            --working-dir /workspace/le-wm \
            --runtime-env-json '{"excludes":["ckpts","eval_results","assets",".git","**/__pycache__"]}' \
            -- bash scripts/ray_train_half.sh "$t" "$a" 2>&1 \
          | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
      log "  resubmitted hq_${a}_${t} -> ${id:-FAILED} (retry $((n+1))/3)"
    done; done
    sleep 300
  fi
  sleep 300
done

log "all 6 checkpoints present — starting evaluation"
exec bash scripts/orchestrate_half_eval.sh
