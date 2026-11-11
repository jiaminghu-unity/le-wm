#!/bin/bash
# Startup script for extra spot A100-40GB workers (multi-region) attached by hand to the
# jiaming-ray cluster. Mirrors what the autoscaler does for its own workers (user ubuntu,
# system-wide ray 2.56.1, 12 CPU / 1 GPU), then watches for idleness and deletes the VM.
HEAD=10.150.0.3:6379
exec >> /var/log/rb-join.log 2>&1
echo "[rb] boot $(date -u)"
md(){ curl -s -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
NAME=$(md name); ZONE=$(md zone | awk -F/ '{print $NF}')
for i in $(seq 1 60); do id ubuntu >/dev/null 2>&1 && break; sleep 5; done
id ubuntu >/dev/null 2>&1 || { useradd -m -s /bin/bash ubuntu; echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-ubuntu; }
grep -qs ubuntu /etc/sudoers.d/* || echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-ubuntu
for i in $(seq 1 90); do nvidia-smi >/dev/null 2>&1 && break; echo "[rb] waiting for nvidia driver ($i)"; sleep 10; done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export DEBIAN_FRONTEND=noninteractive
apt-get -o DPkg::Lock::Timeout=900 -qq update
apt-get -o DPkg::Lock::Timeout=900 -qq install -y python3-pip
python3 -m pip install -q "ray[default]==2.56.1" google-api-python-client
for i in 1 2 3 4 5; do
  sudo -u ubuntu -H bash -lc "ray stop >/dev/null 2>&1; ray start --address=$HEAD --num-cpus=12 --num-gpus=1 --disable-usage-stats" && break
  echo "[rb] ray start failed ($i), retry"; sleep 30
done
echo "[rb] joined $(date -u)"
# Idle watchdog: after a 25-minute grace, delete this VM once no Ray work has run on it for
# 30 minutes (busy = any Ray worker process that is not ray::IDLE, or a process on the GPU).
(
  sleep 1500; idle=0
  while true; do
    busy=$(ps -eo args | grep -E '^ray::' | grep -vc 'ray::IDLE')
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
