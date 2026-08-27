#!/usr/bin/env bash
# OGB multi-object three-arm trainings (cube_double + scene, FULL-CONFIG q),
# GATED on the seed-3073 replication being fully complete (mppi 132/132) --
# user's sequencing: 复现做完了就可以搞这个.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/ogbmulti_train.log
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

mppi_done(){ [ "$(gcloud storage ls "$BUCKET"/final_eval*/final_*r73_mppi_s10?.csv 2>/dev/null | wc -l)" -ge 132 ]; }

TRAINS=(
"cd_base|lewm_cubedouble_base_s${SEED}|experiment=cubedouble_base|bash scripts/ray_train_qnative.sh cube_double experiment=cubedouble_base seed=${SEED}"
"cd_obj|lewm_cubedouble_obj0.1_s${SEED}|experiment=cubedouble_obj|bash scripts/ray_train_qnative.sh cube_double experiment=cubedouble_obj seed=${SEED}"
"cd_aux|lewm_cubedouble_aux0.1_s${SEED}|experiment=cubedouble_aux|bash scripts/ray_train_qnative.sh cube_double experiment=cubedouble_aux seed=${SEED}"
"sc_base|lewm_scene_base_s${SEED}|experiment=scene_base|bash scripts/ray_train_qnative.sh scene experiment=scene_base seed=${SEED}"
"sc_obj|lewm_scene_obj0.1_s${SEED}|experiment=scene_obj|bash scripts/ray_train_qnative.sh scene experiment=scene_obj seed=${SEED}"
"sc_aux|lewm_scene_aux0.1_s${SEED}|experiment=scene_aux|bash scripts/ray_train_qnative.sh scene experiment=scene_aux seed=${SEED}"
)
log "start: OGB multi-object 3-arm trainings (gated on replication mppi 132/132)"
until mppi_done; do log "waiting: replication mppi not yet 132/132"; sleep 600; done
log "replication complete -> training phase begins"
for round in $(seq 1 4000); do
  left=0
  for spec in "${TRAINS[@]}"; do
    IFS='|' read -r name run probe cmd <<< "$spec"
    gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 && continue
    left=1
    [ "$(nrun "$probe")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    n=${ATT[$name]:-0}
    [ "$n" -ge 4 ] && { log "$name attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub $cmd)
    if [ -n "$id" ]; then ATT[$name]=$((n+1)); log "$name attempt $((n+1)) -> $id"
    else log "$name submit FAILED"; fi
    break
  done
  [ "$left" = 0 ] && { log "ALL OGB-MULTI TRAININGS COMPLETE"; exit 0; }
  sleep 200
done
log "round cap"; exit 1
