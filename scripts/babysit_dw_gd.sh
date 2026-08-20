#!/usr/bin/env bash
# Resume the DINO-WM gd cells (user reversed the drop decision once the cluster idled).
# SEED-LEVEL granularity: dw-gd seeds cost 10-20 h each, so one job = one seed and all
# GPUs stay busy instead of one job crawling through six seeds serially.
# GPU-aware (real allocation via ray status), one submission per round, skip-if-CSV-
# present, per-seed attempt cap. Missing set: pusht/reacher/cube x s101-106 + tworoom s106.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/babysit_dw_gd.log
log(){ echo "[$(date -u +%m-%d %H:%M:%S)] $*" | tee -a "$L"; }
declare -A ATT

free(){ ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" \
  | awk -F'[/ ]' '{print int($2 - $1)}'; }
nrun(){ python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j['status'] in ('RUNNING','PENDING') and sys.argv[1] in (j.get('entrypoint') or '')))
PY
}

# task -> "eval-script gcs-prefix csv-prefix extra-first-arg"
row(){ case "$1" in
  pusht|reacher|cube) echo "ray_eval_final.sh final_eval final_${1}_dw $1";;
  tworoom) echo "ray_eval_tworoom.sh final_eval_tworoom final_tworoom_dw -";;
esac; }

log "start: dw gd resume, seed-level jobs"
for round in $(seq 1 3000); do
  left=0
  # cube last: its seeds are the longest; fill GPUs with the cheaper ones first
  for t in tworoom pusht reacher cube; do
    set -- $(row "$t"); EVS=$1; EVP=$2; PFX=$3; TARG=$4
    CK="dinowm_${t}_s3072"
    for s in 101 102 103 104 105 106; do
      gcloud storage ls "$BUCKET/$EVP/${PFX}_gd_s${s}.csv" >/dev/null 2>&1 && continue
      left=1
      if [ "$TARG" != "-" ]; then pat="$EVS $TARG dw $CK gd $s"; else pat="$EVS dw $CK gd $s"; fi
      [ "$(nrun "$pat")" != 0 ] && continue
      [ "$(free)" -lt 2 ] && continue   # RESERVE=1: keep one GPU free for the user's interactive verification (2026-08-18)
      k="${t}_s${s}"; n=${ATT[$k]:-0}
      [ "$n" -ge 5 ] && { log "$k attempt cap"; continue; }
      if [ "$TARG" != "-" ]; then
        id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
          --working-dir /workspace/le-wm --runtime-env-json "$EXC" \
          -- bash "scripts/$EVS" "$TARG" dw "$CK" gd "$s" 2>&1 | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
      else
        id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
          --working-dir /workspace/le-wm --runtime-env-json "$EXC" \
          -- bash "scripts/$EVS" dw "$CK" gd "$s" 2>&1 | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
      fi
      if [ -n "$id" ]; then ATT[$k]=$((n+1)); log "$t gd s$s attempt $((n+1)) -> $id"
      else log "$t gd s$s submit FAILED"; fi
      break 2   # one submission per round; ray status lags fresh submissions
    done
  done
  [ "$left" = 0 ] && { log "ALL DW GD CELLS COMPLETE"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
