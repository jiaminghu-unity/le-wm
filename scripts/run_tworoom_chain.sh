#!/usr/bin/env bash
# Wait for the two-room dataset to be staged, then train the three arms.
#
# Detached on purpose (nohup + setsid): the prep job finishes unattended and the hand-off
# must not depend on any interactive session surviving.
#
# The completion signal is the lance dataset in GCS, not the prep job's Ray status: a job
# can report SUCCEEDED with the upload having failed, and training needs the files.
#
# The baseline arm is trained FIRST and alone. It is the only arm that consumes no q, so if
# something is wrong with the q wiring the cheap arm is not what discovers it -- but more
# importantly the first arm to finish publishes the q_stats the other two then read, which
# is how all three end up normalising q with identical numbers. Actually the baseline never
# writes q_stats (weight 0 still builds the normaliser -- train.py:181 does it
# unconditionally), so it publishes them too; obj and aux start after it and reuse.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
LOG=/workspace/le-wm/eval_results/tworoom.log
mkdir -p "$(dirname "$LOG")"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

DEADLINE=$(( $(date +%s) + 3*3600 ))
while :; do
  if gcloud storage ls "$BUCKET/datasets/tworoom.lance/" >/dev/null 2>&1; then
    log "tworoom.lance is staged"; break; fi
  alive=$(python3 - <<'PY' 2>/dev/null
import json,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if 'ray_prep_tworoom' in (j.get('entrypoint') or '')
          and j['status'] in ('RUNNING','PENDING')))
PY
)
  if [ "${alive:-0}" = "0" ]; then log "FATAL: prep job gone and no lance dataset"; exit 1; fi
  [ "$(date +%s)" -gt "$DEADLINE" ] && { log "FATAL: deadline waiting for dataset"; exit 1; }
  log "waiting for tworoom.lance (prep job still running)"
  sleep 180
done

submit(){  # arm
  timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait --working-dir /workspace/le-wm \
    --runtime-env-json "$EXC" -- bash scripts/ray_train_tworoom.sh "$1" 2>&1 \
    | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1
}
ckpt(){ gcloud storage ls "$BUCKET/ckpts_tworoom/lewm_$1_tworoom_s3072/weights_epoch_10.pt" >/dev/null 2>&1; }

# ---- baseline first, alone: it publishes the q_stats the other two read ----
for try in 1 2 3; do
  ckpt t1 && break
  running=$(python3 - <<'PY' 2>/dev/null
import json,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if 'ray_train_tworoom.sh base' in (j.get('entrypoint') or '')
          and j['status'] in ('RUNNING','PENDING')))
PY
)
  if [ "${running:-0}" = "0" ]; then log "baseline attempt $try -> $(submit base)"; fi
  while :; do
    ckpt t1 && { log "baseline checkpoint present"; break 2; }
    r=$(python3 - <<'PY' 2>/dev/null
import json,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if 'ray_train_tworoom.sh base' in (j.get('entrypoint') or '')
          and j['status'] in ('RUNNING','PENDING')))
PY
)
    [ "${r:-0}" = "0" ] && { log "baseline job exited without a checkpoint"; break; }
    sleep 300
  done
done
ckpt t1 || { log "FATAL: baseline never produced a checkpoint"; exit 1; }

# ---- obj and aux in parallel, reusing the published q_stats ----
for round in $(seq 1 200); do
  need=()
  ckpt t2 || need+=(obj)
  ckpt t5 || need+=(aux)
  [ ${#need[@]} -eq 0 ] && { log "TWOROOM TRAINING COMPLETE (3/3)"; exit 0; }
  for a in "${need[@]}"; do
    r=$(python3 - "$a" <<'PY' 2>/dev/null
import json,sys,urllib.request
a=sys.argv[1]
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if f'ray_train_tworoom.sh {a}' in (j.get('entrypoint') or '')
          and j['status'] in ('RUNNING','PENDING')))
PY
)
    n=$(cat "/tmp/tworoom_try_$a" 2>/dev/null || echo 0)
    if [ "${r:-0}" = "0" ]; then
      if [ "$n" -ge 3 ]; then log "$a: retry cap reached"; continue; fi
      echo $((n+1)) > "/tmp/tworoom_try_$a"
      log "$a attempt $((n+1)) -> $(submit "$a")"
    fi
  done
  log "round $round: waiting on ${need[*]}"
  sleep 300
done
log "chain hit round cap"; exit 1
