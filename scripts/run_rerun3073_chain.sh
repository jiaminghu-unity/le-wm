#!/usr/bin/env bash
# 3073 双种子复核:A4/A5 每任务最优权重臂 ×10(训练→cem/icem×6种子评测)。
# 评测 cfg 标签 = {short}_{lbl}r73(纯标签,写入 CSV config 列)。
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
EXC_TRAIN='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"],"env_vars":{"RAY_JOB_START_TIMEOUT_SECONDS":"14400"}}'
L=/workspace/le-wm/eval_results/rerun3073.log
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
declare -A ATT
free(){ python3 - <<'FREEPY' 2>/dev/null
import json, urllib.request
nodes=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/v0/nodes?limit=500', timeout=20))
rows=nodes.get('data',{}).get('result',{}).get('result',[])
total=sum(n.get('resources_total',{}).get('GPU',0) for n in rows if n.get('state')=='ALIVE')
jobs=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/', timeout=20))
used=sum(1 for j in jobs if j.get('status') in ('RUNNING','PENDING')
         and ('scripts/ray_' in (j.get('entrypoint') or '') or j.get('entrypoint_num_gpus')))
print(max(int(total+10-used),0))
FREEPY
}
nrun(){ python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j['status'] in ('RUNNING','PENDING') and sys.argv[1] in (j.get('entrypoint') or '')))
PY
}
sub(){ local pin=() env="$EXC"
  case "$1 $2" in
    "bash scripts/ray_train_tworoom.sh"|"bash scripts/ray_train_pointmaze.sh")
      pin=(--entrypoint-resources '{"accelerator_type:L4":0.001}'); env="$EXC_TRAIN";;
    "bash scripts/ray_train_"*) pin=(--entrypoint-resources '{"accelerator_type:A100":0.001}'); env="$EXC_TRAIN";;
    "bash scripts/ray_eval_pointmaze.sh"|"bash scripts/ray_eval_tworoom.sh")
      pin=(--entrypoint-resources '{"accelerator_type:L4":0.001}');;
    "bash scripts/ray_eval_final.sh")
      [ "$3" != cube ] && pin=(--entrypoint-resources '{"accelerator_type:L4":0.001}');;
  esac
  timeout 240 ray job submit --entrypoint-num-gpus=1 "${pin[@]}" --no-wait \
  --working-dir /workspace/le-wm --runtime-env-json "$env" -- "$@" 2>&1 \
  | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }
try(){ local key=$1; shift
  [ "$(nrun "$*")" != 0 ] && return 1
  [ "$(free)" -lt 1 ] && return 1
  local n=${ATT[$key]:-0}
  [ "$n" -ge 99 ] && { log "$key attempt cap"; return 1; }
  local id; id=$(sub "$@")
  if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"; else log "$key submit FAILED"; fi
}
evalcmd(){ # task cfg run sol seeds...
  local task=$1; shift
  case "$task" in
    tworoom)   echo "bash scripts/ray_eval_tworoom.sh $*";;
    pointmaze) echo "bash scripts/ray_eval_pointmaze.sh $*";;
    *)         echo "bash scripts/ray_eval_final.sh $task $*";;
  esac
}
# spec: task | ckpt目录 | 评测目录 | run名 | 评测cfg | 训练命令(不含 bash)
SPECS=(
 "pusht|ckpts|final_eval|lewm_c3_sig_obj0.3_s3073|c3_l03r73|scripts/ray_train_qnative.sh pusht experiment=c3_sig_plus_obj seed=3073 loss.obj.weight=0.3"
 "pusht|ckpts|final_eval|lewm_c5_qhead1.0_s3073|c5_l1r73|scripts/ray_train_qnative.sh pusht experiment=c5_qhead seed=3073 loss.aux.weight=1.0"
 "cube|ckpts|final_eval|lewm_k2_cube_obj_eff1.0_s3073|k2_l1r73|scripts/ray_train_qnative.sh cube experiment=k2_cube_obj_eff seed=3073 loss.obj.weight=1.0"
 "cube|ckpts|final_eval|lewm_k4_cube_qhead_eff0.4_s3073|k4_l04r73|scripts/ray_train_qnative.sh cube experiment=k4_cube_qhead_eff seed=3073 loss.aux.weight=0.4"
 "reacher|ckpts|final_eval|lewm_r2_reacher_paep_l003_s3073|r2_l003r73|scripts/ray_train_qnative.sh reacher experiment=r2_reacher_paep seed=3073 loss.obj.weight=0.03 output_model_name=lewm_r2_reacher_paep_l003_s3073"
 "reacher|ckpts|final_eval|lewm_r5_qhead1.0_s3073|r5_l1r73|scripts/ray_train_qnative.sh reacher experiment=r5_qhead seed=3073 loss.aux.weight=1.0"
 "tworoom|ckpts_tworoom|final_eval_tworoom|lewm_t2_tworoom_obj1.0_s3073|t2_l1r73|scripts/ray_train_tworoom.sh obj loss.obj.weight=1.0 seed=3073"
 "tworoom|ckpts_tworoom|final_eval_tworoom|lewm_t5_tworoom_qhead1.0_s3073|t5_l1r73|scripts/ray_train_tworoom.sh aux loss.aux.weight=1.0 seed=3073"
 "pointmaze|ckpts_pointmaze|final_eval_pointmaze|lewm_p2_pointmaze_l003_s3073|p2_l003r73|scripts/ray_train_pointmaze.sh obj loss.obj.weight=0.03 seed=3073 output_model_name=lewm_p2_pointmaze_l003_s3073"
 "pointmaze|ckpts_pointmaze|final_eval_pointmaze|lewm_p5_pointmaze_l1_s3073|p5_l1r73|scripts/ray_train_pointmaze.sh aux loss.aux.weight=1.0 seed=3073 output_model_name=lewm_p5_pointmaze_l1_s3073"
)
log "start: 3073 rerun (10 best-weight arms)"
for round in $(seq 1 9000); do
  left=0
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r task ckdir evdir run cfg traincmd <<< "$spec"
    if ! gcloud storage ls "$BUCKET/$ckdir/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
      left=1
      # shellcheck disable=SC2086
      try "tr_${cfg}" bash $traincmd
      continue
    fi
    for sol in cem icem; do
      for seeds in "101 102 103" "104 105 106"; do
        miss=0
        for s in $seeds; do
          gcloud storage ls "$BUCKET/$evdir/final_${task}_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
        done
        [ "$miss" = 0 ] && continue
        left=1
        cmd=$(evalcmd "$task" "$cfg" "$run" "$sol" $seeds)
        # shellcheck disable=SC2086
        try "ev_${cfg}_${sol}_${seeds%% *}" $cmd
      done
    done
  done
  [ "$left" = 0 ] && { log "RERUN3073 COMPLETE"; exit 0; }
  sleep 90
done
log "round cap"; exit 1
