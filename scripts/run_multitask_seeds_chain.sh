#!/usr/bin/env bash
# 多任务加固波(2026-10-08 user:"多任务第 2/3 种子的加固当然要补")。
# 统一按定稿配方(hinge 门,λ=0.03)对齐:4 任务 × {基线 base, 不门控 obj, 定稿门控 gate03}
# × 种子 {3073, 3074}。既有的 baser74/objr74 自动跳过;r74 的门控臂此前用的是各任务
# 名义最优门(agr74,非定稿配方),故 gate03 两个种子都是新训。评测 cem/icem × 6 种子。
# 预估新训练 ≈ 16,评测 ≈ 64;训练钉 A100。
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
BUCKET=gs://prism-training-us/le-wm
Q=$BUCKET/qgate
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"]}'
L=/workspace/le-wm/eval_results/multitask_seeds.log
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
declare -A ATT
nrun(){ python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j['status'] in ('RUNNING','PENDING') and sys.argv[1] in (j.get('entrypoint') or '')))
PY
}
sub(){ timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
  --working-dir /workspace/le-wm --runtime-env-json "$EXC" -- "$@" 2>&1 \
  | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }
suba(){ timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
  --entrypoint-resources '{"accelerator_type:A100":0.001}' \
  --working-dir /workspace/le-wm --runtime-env-json "$EXC" -- "$@" 2>&1 \
  | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }
try(){ local key=$1; shift
  [ "$(nrun "$*")" != 0 ] && return 1
  local n=${ATT[$key]:-0}; [ "$n" -ge 5 ] && { return 1; }
  local id
  case "$key" in tr_*) id=$(suba "$@");; *) id=$(sub "$@");; esac
  [ -n "$id" ] && { ATT[$key]=$((n+1)); log "$key -> $id"; }
}
missing(){ local task=$1 cfg=$2 sol=$3; shift 3
  for s in "$@"; do
    gcloud storage ls "$BUCKET/final_eval_ogbmulti/final_${task}_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || return 0
  done; return 1
}
log "start: multi-task seed-reinforcement wave (recipe = hinge gate lam0.03)"
for round in $(seq 1 9000); do
  left=0
  for SEED in 3073 3074 3075 3076; do
    sfx="r${SEED#30}"   # 3073->r73 ... 3076->r76(2026-10-08 user:按3072配置再多跑几个训练种子)
    for spec in \
      "cube_double|cubedouble|qgate_stage1_cube_double_lam0.03.json" \
      "cube_triple|cubetriple|qgate_stage1_cube_triple_lam0.03.json" \
      "cube_quadruple|cubequadruple|qgate_stage1_cube_quadruple_lam0.03.json" \
      "scene|scene|qgate_stage1_scene_lam0.03.json"; do
      IFS='|' read -r task stem gate <<< "$spec"
      ncegate="${gate/_lam0.03/_nce_lam0.03}"
      # 完整 3072 网格复刻:基线/全q + hinge λ×4 + NCE λ×4(2026-10-08 user 提醒)
      arms="base obj gate01 gate03 gate05 gate10 gate100 nce01 nce03 nce05 nce10 nce100"  # λ 网格 {0.01,0.05,0.1,1.0}+定稿0.03(2026-10-08 user)
      for arm in $arms; do
        case "$arm" in
          base)   tcfg="${stem}_base";        ecfg="base${sfx}";  launch=qn; g="$Q/$gate" ;;
          obj)    tcfg="${stem}_obj";         ecfg="obj${sfx}";   launch=qn; g="$Q/$gate" ;;
          gate01) tcfg="${stem}_scale_gate01"; ecfg="g01${sfx}";  launch=qg; g="$Q/${gate/_lam0.03/_lam0.01}" ;;
          gate03) tcfg="${stem}_scale_gate03"; ecfg="g03${sfx}";  launch=qg; g="$Q/$gate" ;;
          gate05) tcfg="${stem}_scale_gate05"; ecfg="g05${sfx}";  launch=qg; g="$Q/${gate/_lam0.03/_lam0.05}" ;;
          gate10) tcfg="${stem}_scale_gate10"; ecfg="g10${sfx}";  launch=qg; g="$Q/${gate/_lam0.03/_lam0.1}" ;;
          nce01)  tcfg="${stem}_nce_gate01";  ecfg="nce01${sfx}"; launch=qg; g="$Q/${gate/_lam0.03/_nce_lam0.01}" ;;
          nce03)  tcfg="${stem}_nce_gate03";  ecfg="nce03${sfx}"; launch=qg; g="$Q/$ncegate" ;;
          nce05)  tcfg="${stem}_nce_gate05";  ecfg="nce05${sfx}"; launch=qg; g="$Q/${gate/_lam0.03/_nce_lam0.05}" ;;
          nce10)  tcfg="${stem}_nce_gate10";  ecfg="nce10${sfx}"; launch=qg; g="$Q/${gate/_lam0.03/_nce_lam0.1}" ;;
          gate100) tcfg="${stem}_scale_gate100"; ecfg="g100${sfx}"; launch=qg; g="$Q/${gate/_lam0.03/_lam1.0}" ;;
          nce100)  tcfg="${stem}_nce_gate100";  ecfg="nce100${sfx}"; launch=qg; g="$Q/${gate/_lam0.03/_nce_lam1.0}" ;;
        esac
        # λ=1.0 的 stage-1 门不存在则先自举
        if [ "$launch" = qg ] && ! gcloud storage ls "$g" >/dev/null 2>&1; then
          left=1
          case "$arm" in
            gate100) try "s1_${task}_lam1" bash scripts/ray_qgate_stage1.sh "$task" 1.0 ;;
            nce100)  try "s1_${task}_nce_lam1" env QGATE_VARIANT=infonce bash scripts/ray_qgate_stage1.sh "$task" 1.0 ;;
          esac
          continue
        fi
        run="$(python3 -c "
import re
t=open('config/train/experiment/${tcfg}.yaml').read()
m=re.search(r'output_model_name: (\S+)',t)
print(m.group(1).replace('\${seed}','${SEED}'))")"
        if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
          left=1
          if [ "$launch" = qg ]; then
            try "tr_${tcfg}_${SEED}" env QGATE_GCS="$g" bash scripts/ray_train_qgate2.sh "$task" experiment="$tcfg" seed=$SEED
          else
            try "tr_${tcfg}_${SEED}" bash scripts/ray_train_qnative.sh "$task" experiment="$tcfg" seed=$SEED
          fi
          continue
        fi
        for sol in cem icem; do
          for seeds in "101 102 103" "104 105 106"; do
            # shellcheck disable=SC2086
            if missing "$task" "$ecfg" "$sol" $seeds; then
              left=1
              # shellcheck disable=SC2086
              try "ev_${task}_${ecfg}_${sol}_${seeds%% *}" env EVAL_TIERS=T1 bash scripts/ray_eval_ogbmulti.sh "$task" "$ecfg" "$run" "$sol" $seeds
            fi
          done
        done
      done
    done
  done
  # cube NCE τ=1 复刻(附录双配方对等)
  for SEED in 3073 3074; do
    sfx="r${SEED#30}"; ecfg="qnceg03${sfx}"; run="lewm_qnceg03_scale_cube_s${SEED}"
    if ! gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1; then
      left=1
      try "tr_qnceg03_${SEED}" env QGATE_GCS="$Q/qgate_stage1_cube_nce_lam0.03.json" bash scripts/ray_train_qgate2.sh cube experiment=qnceg03_scale_cube seed=$SEED
      continue
    fi
    for sol in cem icem; do
      for seeds in "101 102 103" "104 105 106"; do
        m=0
        for s2 in $seeds; do gcloud storage ls "$BUCKET/final_eval/final_cube_${ecfg}_${sol}_s${s2}.csv" >/dev/null 2>&1 || m=1; done
        [ "$m" = 0 ] && continue
        left=1
        # shellcheck disable=SC2086
        try "ev_cube_${ecfg}_${sol}_${seeds%% *}" bash scripts/ray_eval_final.sh cube "${ecfg}" "$run" "$sol" $seeds
      done
    done
  done
  [ "$left" = 0 ] && { log "MULTITASK SEED WAVE COMPLETE"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
