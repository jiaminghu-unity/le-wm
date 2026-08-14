#!/usr/bin/env bash
# GPU-aware babysitter for the pusht/reacher DINO-WM evals: the chain's submissions die in
# Ray's 900s queue when all GPUs are busy (reacher's smoke burned its cap that way). This
# only submits when a GPU is free. Skip-if-present makes overlap with the chain harmless.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/babysit_dw_big.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L"; }
declare -A ATT
free(){ python3 - <<'PY' 2>/dev/null
import json,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(8-sum(1 for j in d if j['status'] in ('RUNNING','PENDING')))
PY
}
nrun(){ python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j['status'] in ('RUNNING','PENDING') and sys.argv[1] in (j.get('entrypoint') or '')))
PY
}
for round in $(seq 1 900); do
  left=0
  for t in reacher pusht; do
    CK="dinowm_${t}_s3072"
    gcloud storage ls "$BUCKET/ckpts/$CK/weights_epoch_10.pt" >/dev/null 2>&1 || {
      gcloud storage ls "$BUCKET/ckpts_dinowm/$CK/weights_epoch_10.pt" >/dev/null 2>&1 && {
        log "copy $t ckpt"; gcloud storage cp -r "$BUCKET/ckpts_dinowm/$CK" "$BUCKET/ckpts/"; } || { left=1; continue; }; }
    for slv in cem icem mppi gd; do
      ok=1
      for s in 101 102 103 104 105 106; do
        gcloud storage ls "$BUCKET/final_eval/final_${t}_dw_${slv}_s${s}.csv" >/dev/null 2>&1 || ok=0
      done
      [ "$ok" = 1 ] && continue
      left=1
      [ "$(nrun "ray_eval_final.sh $t dw $CK $slv")" != 0 ] && continue
      [ "$(free)" -lt 1 ] && continue
      k="${t}_$slv"; n=${ATT[$k]:-0}; [ "$n" -ge 4 ] && { log "$k babysit cap"; continue; }
      ATT[$k]=$((n+1))
      id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait --working-dir /workspace/le-wm \
        --runtime-env-json "$EXC" -- bash scripts/ray_eval_final.sh "$t" dw "$CK" "$slv" \
        101 102 103 104 105 106 2>&1 | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
      log "$t dw $slv attempt $((n+1)) -> ${id:-failed}"
    done
  done
  [ "$left" = 0 ] && { log "BIG DW EVAL COMPLETE"; exit 0; }
  sleep 300
done
