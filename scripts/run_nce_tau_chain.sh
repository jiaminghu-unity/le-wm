#!/usr/bin/env bash
# InfoNCE temperature sweep for Auto-SCALE (2026-10-06, user: "infonce 的 temperature
# 从来没调过,先测这个;只看 T1").
# 6 arms on cube, lambda fixed 0.03, tau in {0.3, 3, 10, 30, 100, 300}; tau=1 is the
# existing qnceg03 arm. Per arm: stage-1 gate (tau-patched qgate_stage1.py) -> qgate2
# full train (seed 3072) -> evals cem/icem x 6 seeds = 12 CSVs (cfg qnt<tag>).
# mppi is skipped: the decision metric is T1 under cem/icem only.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/nce_tau.log
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
declare -A ATT
free(){ python3 - <<'FREEPY' 2>/dev/null
import json, urllib.request
nodes = json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/v0/nodes?limit=500', timeout=20))
rows = nodes.get('data',{}).get('result',{}).get('result',[])
total = sum(n.get('resources_total',{}).get('GPU',0) for n in rows if n.get('state')=='ALIVE')
jobs = json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/', timeout=20))
used = sum(1 for j in jobs if j.get('status') in ('RUNNING','PENDING')
           and ('scripts/ray_' in (j.get('entrypoint') or '') or j.get('entrypoint_num_gpus')))
print(max(int(total-used), 0))
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
# 训练必须上 A100(cube 全量训在 L4 上内存/显存不足,2026-10-06 教训)
suba(){ timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
  --entrypoint-resources '{"accelerator_type:A100":0.001}' \
  --working-dir /workspace/le-wm --runtime-env-json "$EXC" -- "$@" 2>&1 \
  | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }
try(){ local key=$1; shift
  [ "$(nrun "$*")" != 0 ] && return 1
  [ "$(free)" -lt 1 ] && return 1
  local n=${ATT[$key]:-0}
  [ "$n" -ge 5 ] && { log "$key attempt cap"; return 1; }
  local id
  case "$key" in tr_*) id=$(suba "$@");; *) id=$(sub "$@");; esac
  if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"; else log "$key submit FAILED"; fi
}
TAUS="0.3 3 10 30 100 300"
log "start: InfoNCE tau sweep, 6 arms, lambda=0.03, T1-decision"
for round in $(seq 1 9000); do
  left=0
  for tau in $TAUS; do
    tag="${tau/./p}"; cfg="qnt${tag}"
    gate="qgate_stage1_cube_nce_tau${tag}_lam0.03.json"
    if ! gcloud storage ls "$BUCKET/qgate/$gate" >/dev/null 2>&1; then
      left=1
      try "s1_${cfg}" env QGATE_VARIANT=infonce_tau QGATE_TAU="$tau" bash scripts/ray_qgate_stage1.sh cube 0.03
      continue
    fi
    run="lewm_${cfg}_scale_cube_s${SEED}"
    if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
      left=1
      try "tr_${cfg}" env QGATE_GCS="$BUCKET/qgate/$gate" bash scripts/ray_train_qgate2.sh cube experiment="${cfg}_scale_cube" seed=$SEED
      continue
    fi
    for sol in cem icem; do
      for seeds in "101 102 103" "104 105 106"; do
        miss=0
        for s in $seeds; do
          gcloud storage ls "$BUCKET/final_eval/final_cube_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
        done
        [ "$miss" = 0 ] && continue
        left=1
        # shellcheck disable=SC2086
        try "ev_${cfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh cube "${cfg}" "$run" "$sol" $seeds
      done
    done
  done
  [ "$left" = 0 ] && { log "NCE TAU SWEEP COMPLETE (6 arms, 72 CSVs)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
