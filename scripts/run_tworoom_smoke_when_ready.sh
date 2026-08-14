#!/usr/bin/env bash
# Wait for the two-room baseline checkpoint, then run one evaluation cell.
#
# Detached (nohup + setsid) so the hand-off does not depend on an interactive session.
# One cell only: the sweep is not started until a human has seen this number. The scene
# reconstruction already passed (0.000 MAE), so what remains unverified is the end-to-end
# eval path -- expert replay, goal handling, and whether baseline SR lands somewhere
# plausible rather than at 0% or 100%, either of which would mean the protocol is wrong
# rather than the model bad.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
LOG=/workspace/le-wm/eval_results/tworoom.log
log(){ echo "[$(date -u +%H:%M:%S)] [smoke] $*" | tee -a "$LOG"; }

DEADLINE=$(( $(date +%s) + 20*3600 ))
while :; do
  if gcloud storage ls "$BUCKET/ckpts_tworoom/lewm_t1_tworoom_s3072/weights_epoch_10.pt" \
     >/dev/null 2>&1; then log "baseline checkpoint present"; break; fi
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "FATAL: deadline waiting for t1"; exit 1; }
  log "waiting for the baseline checkpoint"
  sleep 600
done

if gcloud storage ls "$BUCKET/final_eval_tworoom/final_tworoom_t1_cem_s101.csv" >/dev/null 2>&1; then
  log "smoke cell already present"; exit 0
fi
for try in 1 2 3; do
  id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait --working-dir /workspace/le-wm \
        --runtime-env-json '{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}' \
        -- bash scripts/ray_smoke_tworoom.sh 2>&1 | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
  log "attempt $try -> ${id:-submit failed}"
  [ -z "$id" ] && { sleep 300; continue; }
  while :; do
    s=$(python3 - "$id" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(next((j['status'] for j in d if j.get('submission_id')==sys.argv[1]),''))
PY
)
    case "$s" in
      SUCCEEDED) log "TWOROOM SMOKE CELL DONE"; exit 0;;
      FAILED|STOPPED) log "smoke $s; retrying"; break;;
    esac
    sleep 300
  done
done
log "FATAL: smoke cell never completed"; exit 1
