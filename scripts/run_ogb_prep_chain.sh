#!/usr/bin/env bash
# OGBench multi-object data prep chain: cube_double/triple/quadruple + scene.
# One GPU job per task (download -> smoke gate -> replay-render -> lance -> GCS).
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/ogb_prep.log
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
declare -A ATT
free(){ python3 - <<'FREEPY' 2>/dev/null
import ray
ray.init(address='auto', ignore_reinit_error=True, log_to_driver=False)
print(int(ray.available_resources().get('GPU', 0)))
FREEPY
}
nrun(){ python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j['status'] in ('RUNNING','PENDING') and sys.argv[1] in (j.get('entrypoint') or '')))
PY
}
sub(){ timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
  --working-dir /workspace/le-wm --runtime-env-json "$EXC" -- "$@" 2>&1 \
  | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }

log "start: ogbench multi-object prep (4 datasets)"
for round in $(seq 1 3000); do
  left=0
  for task in cube_double scene cube_triple cube_quadruple; do
    gcloud storage ls "$BUCKET/datasets/ogbench/${task}_play.lance/" >/dev/null 2>&1 && continue
    left=1
    [ "$(nrun "ray_ogb_prep.sh $task")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    n=${ATT[$task]:-0}
    [ "$n" -ge 3 ] && { log "$task attempt cap"; continue; }
    id=$(sub bash scripts/ray_ogb_prep.sh "$task")
    if [ -n "$id" ]; then ATT[$task]=$((n+1)); log "$task attempt $((n+1)) -> $id"
    else log "$task submit FAILED"; fi
    break
  done
  [ "$left" = 0 ] && { log "ALL OGB PREP DONE"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
