#!/usr/bin/env bash
# Ray head self-heal (cron every 5 min): if the head does not answer a real
# client ping, force-restart ray + autoscaler, re-request GPUs, and relaunch any
# missing babysitter chains. Root causes on record (2026-08-27): the head VM was
# rebooted twice by GCP host events (it is NOT spot), and a manually started ray
# had no supervisor, so dead subprocesses (dashboard/raylet) never self-healed.
LOG=/workspace/le-wm/eval_results/head_selfheal.log
ts(){ date -u '+%m-%d %H:%M:%S'; }

if timeout 20 curl -sf "http://127.0.0.1:8265/api/v0/nodes?limit=1" >/dev/null 2>&1; then
  exit 0   # healthy (dashboard + state API answering)
fi

echo "[$(ts)] ray unhealthy -> restarting" >> "$LOG"
ray stop --force >> "$LOG" 2>&1
sleep 3
ray start --head --port=6379 --dashboard-host=127.0.0.1 --dashboard-port=8265 \
  --autoscaling-config=/workspace/ray_cluster.resolved.yaml >> "$LOG" 2>&1
sleep 15
python3 -c "
import ray
from ray.autoscaler.sdk import request_resources
ray.init(address='auto', ignore_reinit_error=True, log_to_driver=False)
request_resources(bundles=[{'GPU':1}]*8)" >> "$LOG" 2>&1 && echo "[$(ts)] ray restarted, 8-GPU request placed" >> "$LOG"

# relaunch missing chains (idempotent: GCS done-checks make restarts free)
for chain in run_ogbmulti_eval_chain run_pointmaze_s3074_chain; do
  if ! ps -eo cmd | grep -q "[b]ash scripts/${chain}.sh"; then
    cd /workspace/le-wm && nohup bash "scripts/${chain}.sh" > /dev/null 2>&1 &
    echo "[$(ts)] relaunched $chain" >> "$LOG"
  fi
done
