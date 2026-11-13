#!/bin/bash
# Startup script for extra spot A100-40GB workers (multi-region) attached by hand to the
# jiaming-ray cluster. Mirrors what the autoscaler does for its own workers (user ubuntu,
# system-wide ray 2.56.1, 12 CPU / 1 GPU), then watches for idleness and deletes the VM.
HEAD=10.150.0.3:6379
exec >> /var/log/rb-join.log 2>&1
# Mirror this log to the serial console so a VM that never joins can be diagnosed from the head with
# `gcloud compute instances get-serial-port-output` (no ssh needed).
( tail -n +1 -F /var/log/rb-join.log 2>/dev/null | sed -u 's/^/[rb-join] /' > /dev/ttyS0 ) &
echo "[rb] boot $(date -u)"
md(){ curl -s -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
NAME=$(md name); ZONE=$(md zone | awk -F/ '{print $NF}')
for i in $(seq 1 60); do id ubuntu >/dev/null 2>&1 && break; sleep 5; done
id ubuntu >/dev/null 2>&1 || { useradd -m -s /bin/bash ubuntu; echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-ubuntu; }
grep -qs ubuntu /etc/sudoers.d/* || echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-ubuntu
for i in $(seq 1 90); do nvidia-smi >/dev/null 2>&1 && break; echo "[rb] waiting for nvidia driver ($i)"; sleep 10; done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[rb] driver ok $(date -u +%H:%M:%S)"
export DEBIAN_FRONTEND=noninteractive
# apt can sit behind the boot-time apt-daily / unattended-upgrades run for a long time (2026-11-13: 10 of
# 15 VMs were still at this step 25 min after boot and never joined). Only pip is needed here, so skip apt
# when pip is already present; either way print who holds the apt/dpkg lock (visible on the serial console).
echo "[rb] apt holders: $(pgrep -af 'apt|dpkg|unattended' | cut -c1-100 | tr '\n' ';')"
if python3 -m pip --version >/dev/null 2>&1; then
  echo "[rb] pip present, apt skipped"
else
  timeout 900 apt-get -o DPkg::Lock::Timeout=600 -o Acquire::http::Timeout=30 -o Acquire::Retries=2 -q update 2>&1 | tail -n 4
  timeout 900 apt-get -o DPkg::Lock::Timeout=600 -q install -y python3-pip 2>&1 | tail -n 4
fi
echo "[rb] apt done $(date -u +%H:%M:%S)"
python3 -m pip install -q "ray[default]==2.56.1" google-api-python-client
echo "[rb] pip done $(date -u +%H:%M:%S) ray=$(python3 -c 'import ray;print(ray.__version__)' 2>&1 | tail -1)"
# Custom resource rb=1 (2026-11-13): chains started with ENTRY_RESOURCES='{"rb": 1}' run only on these
# hand-joined VMs. The autoscaler's own workers are being stopped ~35 min after start by the head's
# gpu_reaper cron (it lists Ray nodes with the default 100-row limit and takes the rest for orphans).
# RemoveIPC guard: without lingering, any ssh logout as ubuntu wipes the shared memory of running jobs
loginctl enable-linger ubuntu || true
for i in 1 2 3 4 5; do
  echo "[rb] ray start attempt $i $(date -u +%H:%M:%S)"
  timeout 300 sudo -u ubuntu -H bash -lc "ray stop >/dev/null 2>&1; ray start --address=$HEAD --num-cpus=12 --num-gpus=1 --resources='{\"rb\": 1}' --disable-usage-stats" && break
  echo "[rb] ray start failed ($i), retry"; sleep 30
done
echo "[rb] joined $(date -u)"
# Idle watchdog: after a 25-minute grace, delete this VM once no Ray work has run on it for
# 30 minutes (busy = any Ray worker process that is not ray::IDLE, or a process on the GPU).
(
  sleep 1500; idle=0
  while true; do
    # DashboardAgent / RuntimeEnvAgent are always present and are not work (counting them was a
    # bug: the first version never saw a VM as idle, so VMs did not delete themselves)
    busy=$(ps -eo args | grep -E '^ray::' | grep -v -E 'ray::(IDLE|DashboardAgent|RuntimeEnvAgent)' | grep -c .)
    gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    if [ "$busy" -gt 0 ] || [ "$gpu" -gt 0 ]; then idle=0; else idle=$((idle+1)); fi
    echo "[rb] watchdog busy=$busy gpu=$gpu idle=$idle $(date -u +%H:%M)"
    if [ "$idle" -ge 6 ]; then
      echo "[rb] idle 30 min -> deleting $NAME"
      gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet
    fi
    sleep 300
  done
) &
