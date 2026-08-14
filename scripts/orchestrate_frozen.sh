#!/usr/bin/env bash
# Keep the frozen-encoder ablation's 9 runs going until every checkpoint is in GCS.
#
# Ray kills a job that waits more than 900 s for resources, so runs cannot be queued
# ahead of capacity: submit only as many as there are idle GPUs. That is also how the
# first attempt lost reacher/aux (START_TIMEOUT) while reacher/obj died with its
# preempted spot node (SUPERVISOR_ACTOR_DIED) -- both are cluster events, not config
# errors, so a plain retry is the right response.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets",".git","**/__pycache__"]}'
LOG=/workspace/le-wm/eval_results/frozen/orchestrator.log
mkdir -p "$(dirname "$LOG")"
JOBS=("cube base" "cube obj" "cube aux" "pusht base" "pusht obj" "pusht aux"
      "reacher base" "reacher obj" "reacher aux")
declare -A ATTEMPTS
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

for round in $(seq 1 200); do
  DONE=$(gcloud storage ls "$BUCKET/ckpts_frozen/" 2>/dev/null | sed 's|.*ckpts_frozen/||' | tr -d '/')
  INFLIGHT=$(python3 - <<'PY' 2>/dev/null
import json,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j.get('type')=='SUBMISSION' and j['status'] in ('RUNNING','PENDING')
          and 'ray_train_frozen.sh' in (j.get('entrypoint') or '')))
PY
)
  INFLIGHT=${INFLIGHT:-0}
  todo=(); complete=0
  for spec in "${JOBS[@]}"; do
    set -- $spec; t=$1; a=$2
    if echo "$DONE" | grep -qx "lewm_fz_${a}_${t}_s3072"; then
      complete=$((complete+1)); continue
    fi
    key="${t}_${a}"
    [ "${ATTEMPTS[$key]:-0}" -lt 5 ] && todo+=("$t $a")
  done
  log "round $round: complete $complete/9, in-flight $INFLIGHT, todo ${#todo[@]}"
  [ "$complete" -ge 9 ] && { log "ALL FROZEN CKPTS PRESENT"; exit 0; }
  if [ ${#todo[@]} -eq 0 ] && [ "$INFLIGHT" -eq 0 ]; then
    log "nothing left to submit but checkpoints missing (attempt cap) — stopping"; exit 1
  fi

  CAP=$(ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" | cut -d/ -f2 | cut -d. -f1)
  CAP=${CAP:-8}
  FREE=$((CAP - INFLIGHT))
  log "  capacity ${CAP} GPU, in-flight ${INFLIGHT}, free ${FREE}"
  i=0
  while [ "$FREE" -gt 0 ] && [ "$i" -lt ${#todo[@]} ]; do
    set -- ${todo[$i]}; t=$1; a=$2
    running=$(python3 - "$t" "$a" <<'PY' 2>/dev/null
import json,sys,urllib.request
t,a=sys.argv[1:3]
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j.get('type')=='SUBMISSION' and j['status'] in ('RUNNING','PENDING')
          and (j.get('entrypoint') or '').endswith(f'ray_train_frozen.sh {t} {a}')))
PY
)
    if [ "${running:-0}" = "0" ]; then
      id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
            --working-dir /workspace/le-wm --runtime-env-json "$EXC" \
            -- bash scripts/ray_train_frozen.sh "$t" "$a" 2>&1 \
          | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
      if [ -n "$id" ]; then
        ATTEMPTS[$key]=$(( ${ATTEMPTS["${t}_${a}"]:-0} + 1 ))
        ATTEMPTS["${t}_${a}"]=${ATTEMPTS[$key]}
        log "  submitted $t $a -> $id"
        FREE=$((FREE-1))
      else
        log "  submit FAILED for $t $a"
      fi
    fi
    i=$((i+1))
  done
  sleep 180
done
log "orchestrator hit round cap"
