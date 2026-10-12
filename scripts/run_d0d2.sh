#!/bin/bash
# D0/D1 评测队列:逐环境配对重评(b1/ours/b3,确定性复现 fix2)→ T6 raw(80 态,b1+dino)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ~/miniforge3/envs/contactworld
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
cd ~/ContactWorld
for SPEC in "b1 outputs_b1_ckpt_100k.pt" "ours ours_ckpt_best.pt" "b3 b3_ckpt_best.pt"; do
  set -- $SPEC; V=$1; CK=$2
  for OFF in 12 24 36 48; do
    python eval_planner_b1.py --variant $V --ckpt-path $CK \
      --data-root data/demo_data/insertion_usb --isaacgym-cfg-name isaacgym_config_usb \
      --output-dir logs/planning/${V}fix3pe/step_${OFF} --vision-key front --vision-type image \
      --use-tactile --tactile-key wrist --image-size 224 --history-size 3 \
      --horizon 6 --candidates 100 --topk 8 --iterations 4 --num-envs 100 --num-record 2 \
      --goal-offset-steps ${OFF} --max-steps 15 --stop-on-success --reg-loss-type vc \
      >> ~/d0_pe.log 2>&1 && echo "${V} pe OFFSET ${OFF} OK" >> ~/d0d2_status.log || echo "${V} pe OFFSET ${OFF} FAIL" >> ~/d0d2_status.log
  done
  echo "${V} PE ARM DONE" >> ~/d0d2_status.log
done
for SPEC in "b1 outputs_b1_ckpt_100k.pt" "dino outputs/d1/d2_best.pt"; do
  set -- $SPEC; V=$1; CK=$2
  python action_ranking_diag_T6.py --variant $V --ckpt-path $CK \
    --num-states 80 --k 100 --horizon 6 --goal-offset-steps 12 --seed 0 --save-raw \
    --out outputs/rank_diag_T6raw_${V}.json >> ~/t6raw.log 2>&1 \
    && echo "$V T6RAW OK" >> ~/d0d2_status.log || echo "$V T6RAW FAIL" >> ~/d0d2_status.log
done
gcloud storage rsync -r logs/planning gs://prism-training-us/contactworld/outputs/closedloop_all >/dev/null 2>&1
gcloud storage cp outputs/rank_diag_T6raw_* gs://prism-training-us/contactworld/outputs/rank_diag/ >/dev/null 2>&1
echo "D0D2 EVAL QUEUE COMPLETE" >> ~/d0d2_status.log
