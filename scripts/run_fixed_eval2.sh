#!/bin/bash
# 全七臂闭环重评(真实动作历史修复版 v2:tracked builder 不再被覆盖,接通失败即硬退出)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ~/miniforge3/envs/contactworld
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
cd ~/ContactWorld
for SPEC in "b1 outputs_b1_ckpt_100k.pt" "ours ours_ckpt_best.pt" "b3 b3_ckpt_best.pt" "dino outputs/d1/d2_best.pt" "duet duet_ckpt_best.pt" "front front_ckpt_best.pt" "wrist wrist_ckpt_best.pt"; do
  set -- $SPEC; V=$1; CK=$2
  for OFF in 12 24 36 48; do
    python eval_planner_b1.py --variant $V --ckpt-path $CK \
      --data-root data/demo_data/insertion_usb --isaacgym-cfg-name isaacgym_config_usb \
      --output-dir logs/planning/${V}fix2/step_${OFF} --vision-key front --vision-type image \
      --use-tactile --tactile-key wrist --image-size 224 --history-size 3 \
      --horizon 6 --candidates 100 --topk 8 --iterations 4 --num-envs 100 --num-record 2 \
      --goal-offset-steps ${OFF} --max-steps 15 --stop-on-success --reg-loss-type vc \
      >> ~/fixed_eval2.log 2>&1 && echo "${V} fix2 OFFSET ${OFF} OK" >> ~/fixedeval2_status.log || echo "${V} fix2 OFFSET ${OFF} FAIL" >> ~/fixedeval2_status.log
  done
  gcloud storage rsync -r logs/planning/${V}fix2 gs://prism-training-us/contactworld/outputs/closedloop_all/${V}fix2 >/dev/null 2>&1
  echo "${V} ARM DONE" >> ~/fixedeval2_status.log
done
echo "FIXED EVAL V2 COMPLETE" >> ~/fixedeval2_status.log
