#!/usr/bin/env bash
# Demand-driven multi-region grab of extra SPOT A100-40GB workers for the rebuttal runs.
#   usage: grab_spot40.sh <target>      stop: touch scripts/rebuttal_grab/STOP
# Creates a VM only while Ray has PENDING GPU jobs that booting VMs will not already absorb,
# never holds more than <target> at once, and tries zones round-robin so one stocked-out zone
# does not block the rest. VMs join the cluster via join_startup.sh and delete themselves when
# idle for 30 minutes; a 60 h max run duration is the hard backstop.
export PATH=/usr/local/bin:/snap/bin:/usr/bin:/bin:$PATH
TARGET="${1:?target count}"
D=/home/jiaming.hu/le-wm/scripts/rebuttal_grab
L=$D/grab.log
PROJ=unity-prism-dev
ZONES="us-central1-a us-central1-b us-central1-c us-central1-f us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-east4-a us-east4-b us-east4-c us-west4-b us-west3-b us-east5-a us-east5-b"
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" >> "$L"; }
create(){ # name zone
  timeout 240 gcloud compute instances create "$1" --project=$PROJ --zone="$2" --machine-type=a2-highgpu-1g \
    --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration=60h --maintenance-policy=TERMINATE \
    --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
    --boot-disk-size=150GB --boot-disk-type=pd-balanced --local-ssd=interface=NVME \
    --scopes=cloud-platform --service-account=prism-training-sa@unity-prism-dev.iam.gserviceaccount.com \
    --metadata=enable-oslogin=FALSE --metadata-from-file=startup-script=$D/join_startup.sh,ssh-keys=$D/ssh_keys.txt \
    --labels=owner=jiaming-hu,purpose=scale-rebuttal 2>&1
}
log "start: target $TARGET"
zi=0; read -ra ZA <<< "$ZONES"
while [ ! -f "$D/STOP" ]; do
  VMS=$(timeout 120 gcloud compute instances list --project=$PROJ --filter="labels.purpose=scale-rebuttal AND status:(PROVISIONING STAGING RUNNING)" --format="value(name,networkInterfaces[0].networkIP)" 2>/dev/null) || { sleep 60; continue; }
  cur=$(grep -c . <<< "$VMS")
  read -r pending joined < <(python3 - "$VMS" <<'PY' 2>/dev/null
import json,sys,urllib.request
ips={l.split()[1] for l in sys.argv[1].splitlines() if len(l.split())>1}
jobs=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/',timeout=30))
pend=sum(1 for j in jobs if j.get('status')=='PENDING' and ('scripts/ray_' in (j.get('entrypoint') or '') or j.get('entrypoint_num_gpus')))
rows=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/v0/nodes?limit=500',timeout=30))['data']['result']['result']
alive={n['node_ip'] for n in rows if n.get('state')=='ALIVE'}
print(pend,len(ips&alive))
PY
  )
  [ -n "${pending:-}" ] || { sleep 60; continue; }
  booting=$((cur-joined)); need=$((pending-booting)); room=$((TARGET-cur))
  n=$(( need < room ? need : room ))
  log "have=$cur joined=$joined booting=$booting pending=$pending -> create $([ $n -gt 0 ] && echo $n || echo 0)"
  made=0; fails=0
  while [ "$made" -lt "$n" ] && [ "$fails" -lt "${#ZA[@]}" ] && [ "${#ZA[@]}" -gt 0 ]; do
    z=${ZA[$((zi % ${#ZA[@]}))]}
    name="jm-rb-a40-${z}-$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    out=$(create "$name" "$z")
    if grep -q "RUNNING\|PROVISIONING\|STAGING" <<< "$out" && ! grep -q "ERROR" <<< "$out"; then
      made=$((made+1)); fails=0; log "got $name"
    else
      fails=$((fails+1)); why=$(grep -m1 -oE "ZONE_RESOURCE_POOL_EXHAUSTED|QUOTA_EXCEEDED|Quota|machineType|does not exist|Invalid value|not found|ERROR.{0,120}" <<< "$out" | head -1)
      log "miss $z: ${why:-unknown}"
      if grep -qE "machineType|does not exist in zone|Invalid value for field 'resource.machineType'" <<< "$out"; then
        ZA=("${ZA[@]/$z}"); ZA=(${ZA[@]}); log "drop zone $z (no a2-highgpu)"; continue
      fi
      zi=$((zi+1))
    fi
  done
  sleep 150
done
log "STOP file found, exiting (VMs delete themselves when idle)"
