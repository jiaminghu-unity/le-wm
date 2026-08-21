#!/usr/bin/env bash
# Sequential probe runner on the single analysis GPU: pixel-arm P4+zhealth (A),
# then the q-input P4 (B). One job in flight at a time.
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/newarms_probe.log
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
sub(){ timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
  --working-dir /workspace/le-wm --runtime-env-json "$EXC" -- "$@" 2>&1 \
  | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }
waitjob(){ for i in $(seq 1 400); do
  st=$(ray job status "$1" 2>/dev/null | tail -1)
  case "$st" in *SUCCEEDED*) return 0;; *FAILED*|*STOPPED*) return 1;; esac
  sleep 60; done; return 1; }

for step in "A|bash scripts/ray_p4_newarms.sh pusht" "B|bash scripts/ray_p4_qinput.sh pusht"; do
  IFS='|' read -r name cmd <<< "$step"
  ok=1
  for att in 1 2 3; do
    # shellcheck disable=SC2086
    id=$(sub $cmd)
    [ -z "$id" ] && { log "$name submit failed (att $att)"; sleep 120; continue; }
    log "$name attempt $att -> $id"
    if waitjob "$id"; then log "$name SUCCEEDED ($id)"; ok=0; break
    else log "$name FAILED ($id)"; fi
  done
  [ "$ok" != 0 ] && { log "$name gave up"; exit 1; }
done
log "PROBES COMPLETE"
