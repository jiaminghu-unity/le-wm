#!/bin/bash
# D2 配对闭环:B3/Ours 以训练口径(软路由)部署,等 D0D2 评测队列让出 GPU
for i in $(seq 1 480); do grep -q "D0D2 EVAL QUEUE COMPLETE" ~/d0d2_status.log 2>/dev/null && break; sleep 30; done
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ~/miniforge3/envs/contactworld
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
cd ~/ContactWorld
for SPEC in "b3soft b3_ckpt_best.pt" "ourssoft ours_ckpt_best.pt"; do
  set -- $SPEC; V=$1; CK=$2
  for OFF in 12 24 36 48; do
    python eval_planner_b1.py --variant $V --ckpt-path $CK \
      --data-root data/demo_data/insertion_usb --isaacgym-cfg-name isaacgym_config_usb \
      --output-dir logs/planning/${V}/step_${OFF} --vision-key front --vision-type image \
      --use-tactile --tactile-key wrist --image-size 224 --history-size 3 \
      --horizon 6 --candidates 100 --topk 8 --iterations 4 --num-envs 100 --num-record 2 \
      --goal-offset-steps ${OFF} --max-steps 15 --stop-on-success --reg-loss-type vc \
      >> ~/soft_eval.log 2>&1 && echo "${V} OFFSET ${OFF} OK" >> ~/d0d2_status.log || echo "${V} OFFSET ${OFF} FAIL" >> ~/d0d2_status.log
  done
  echo "${V} ARM DONE" >> ~/d0d2_status.log
done
gcloud storage rsync -r logs/planning gs://prism-training-us/contactworld/outputs/closedloop_all >/dev/null 2>&1
echo "SOFT EVAL COMPLETE" >> ~/d0d2_status.log
