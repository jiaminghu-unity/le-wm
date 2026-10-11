#!/usr/bin/env bash
# 定稿配方的 3073 复种子收尾:等两个训练的 ckpt -> 提交 cem/icem 评测(cfg qg03r73 / qar73)
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/r73_recipe.log
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
  [ -n "$id" ] && { ATT[$key]=$((n+1)); log "$key -> $id"; }
}
log "start: r73 recipe finishing chain"
for round in $(seq 1 600); do
  left=0
  for spec in "qg03r73|lewm_qgate03_scale_cube_s3073" "qar73|lewm_qall_scale_cube_s3073"; do
    cfg="${spec%|*}"; run="${spec#*|}"
    if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then left=1; continue; fi
    for sol in cem icem; do
      for seeds in "101 102 103" "104 105 106"; do
        miss=0
        for s in $seeds; do gcloud storage ls "$BUCKET/final_eval/final_cube_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1; done
        [ "$miss" = 0 ] && continue
        left=1
        # shellcheck disable=SC2086
        try "ev_${cfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh cube "${cfg}" "$run" "$sol" $seeds
      done
    done
  done
  [ "$left" = 0 ] && { log "R73 RECIPE COMPLETE (24 CSVs)"; exit 0; }
  sleep 240
done
