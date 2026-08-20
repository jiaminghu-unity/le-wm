#!/usr/bin/env bash
# Span ablation for trajectory-W on Push-T: FIXED triplet spans 10 / 25 / 40 env
# steps (vs the default U{10..50}). Train each W, then cem x 6 seeds.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/w_span_pusht.log
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
declare -A ATT
free(){ ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" | awk -F'[/ ]' '{print int($2-$1)}'; }
nrun(){ python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j['status'] in ('RUNNING','PENDING') and sys.argv[1] in (j.get('entrypoint') or '')))
PY
}
sub(){ timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
  --working-dir /workspace/le-wm --runtime-env-json "$EXC" -- "$@" 2>&1 \
  | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }
log "start: pusht span ablation 10/25/40"
for round in $(seq 1 1500); do
  left=0
  for SPAN in 10 25 40; do
    tag="pusht_c1_Wd${SPAN}"; cfg="c1Wd${SPAN}"
    if ! gcloud storage ls "$BUCKET/eval/automet_$tag.pt" >/dev/null 2>&1; then
      left=1
      [ "$(nrun "ray_automet_fit_W_any.sh pusht ckpts lewm_c1_s3072 $tag")" != 0 ] && continue
      [ "$(free)" -lt 1 ] && continue
      n=${ATT[f$SPAN]:-0}; [ "$n" -ge 4 ] && continue
      id=$(sub env SPAN=$SPAN bash scripts/ray_automet_fit_W_any.sh pusht ckpts lewm_c1_s3072 "$tag")
      [ -n "$id" ] && { ATT[f$SPAN]=$((n+1)); log "train d$SPAN -> $id"; }
      break
    fi
    miss=0
    for s in 101 102 103 104 105 106; do
      gcloud storage ls "$BUCKET/final_eval_automet/final_pusht_${cfg}_automet_cem_s${s}.csv" >/dev/null 2>&1 || miss=1
    done
    [ "$miss" = 0 ] && continue
    left=1
    [ "$(nrun "ray_eval_automet_any.sh pusht $cfg")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    n=${ATT[e$SPAN]:-0}; [ "$n" -ge 4 ] && continue
    id=$(sub bash scripts/ray_eval_automet_any.sh pusht "$cfg" ckpts lewm_c1_s3072 "automet_$tag.pt" "101,102,103,104,105,106")
    [ -n "$id" ] && { ATT[e$SPAN]=$((n+1)); log "eval d$SPAN -> $id"; }
    break
  done
  [ "$left" = 0 ] && { log "SPAN ABLATION COMPLETE"; exit 0; }
  sleep 240
done
