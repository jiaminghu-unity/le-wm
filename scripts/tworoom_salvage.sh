#!/usr/bin/env bash
# 补传:老快照的 tworoom 训练完成后不上传(RUN 名 bug),从 worker 本地盘捞回。
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
BUCKET=gs://prism-training-us/le-wm
KEY=/home/jiaming.hu/.ssh/ray_worker_key
for round in $(seq 1 400); do
  left=0
  for spec in "obj 0.03 t2_tworoom_obj" "obj 0.3 t2_tworoom_obj" "aux 0.3 t5_tworoom_qhead" "aux 0.4 t5_tworoom_qhead" "aux 1.0 t5_tworoom_qhead"; do
    set -- $spec; arm=$1; w=$2; pre=$3
    [ "$arm" = obj ] && run="lewm_t2_tworoom_obj${w}_s3072" || run="lewm_t5_tworoom_qhead${w}_s3072"
    gcloud storage ls "$BUCKET/ckpts_tworoom/$run/weights_epoch_10.pt" >/dev/null 2>&1 && continue
    left=1
    sid=$(curl -s 'http://127.0.0.1:8265/api/jobs/' | python3 -c "
import json,sys
d=json.load(sys.stdin)
cand=[j for j in d if j['status']=='SUCCEEDED' and 'tworoom.sh $arm' in (j.get('entrypoint') or '') and 'weight=$w ' in ((j.get('entrypoint') or '')+' ')]
cand.sort(key=lambda j:-(j.get('end_time') or 0))
print(cand[0]['submission_id'] if cand else '')")
    [ -z "$sid" ] && continue
    host=$(ray job logs "$sid" 2>/dev/null | grep -oE 'on ray-jiaming-ray-worker-[a-z0-9]+-compute' | head -1 | sed 's/^on //')
    [ -z "$host" ] && continue
    ip=$(gcloud compute instances describe "$host" --zone=us-east4-c --format='value(networkInterfaces[0].networkIP)' 2>/dev/null)
    [ -z "$ip" ] && continue
    timeout 150 ssh -n -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@"$ip" "
      D=/mnt/disks/ssd0/stable-wm/checkpoints/$run
      [ -f \$D/weights_epoch_10.pt ] && gcloud storage cp \$D/weights_epoch_10.pt \$D/config.json $BUCKET/ckpts_tworoom/$run/ && echo SALVAGED $run" 2>/dev/null
  done
  [ "$left" = 0 ] && { echo "TWOROOM SALVAGE COMPLETE"; exit 0; }
  sleep 300
done
