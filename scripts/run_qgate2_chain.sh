#!/usr/bin/env bash
# Stage-2 Auto-SCALE chain (2026-08-29, user: q-gate 出了直接接上 q 做 SCALE 训练).
# GATE: waits for the three qgate_stage1_cube_lam*.json files. Picks lambda by
# rule: prefer 0.01 if its goal-blind verdict is OK, else 0.003 if OK, else 0.01
# with a loud warning (choice + reason logged and copied to qgate/g_star_cube.json).
# TRAINS (seed 3072, comparable to the main grid):
#   qg = Auto-SCALE   lewm_qgate_scale_cube_s3072  (L_obj on sqrt(g*)-scaled 22-d q)
#   qa = SCALE-All    lewm_qall_scale_cube_s3072   (ungated 22-d control)
# EVALS: standard pixel cube eval (ray_eval_final.sh) cem+icem x 6 seeds each
#   -> final_eval/final_cube_{qg,qa}_{cem,icem}_s10x.csv (24 CSVs, new files).
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /workspace/le-wm || exit 1
SEED=3072
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts",".git","**/__pycache__"],"env_vars":{"RAY_JOB_START_TIMEOUT_SECONDS":"14400"}}'
L=/workspace/le-wm/eval_results/qgate2.log
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
declare -A ATT
free(){ python3 - <<'FREEPY' 2>/dev/null
import json, urllib.request
nodes = json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/v0/nodes?limit=100', timeout=20))
rows = nodes.get('data',{}).get('result',{}).get('result',[])
total = sum(n.get('resources_total',{}).get('GPU',0) for n in rows if n.get('state')=='ALIVE')
jobs = json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/', timeout=20))
used = sum(1 for j in jobs if j.get('status')=='RUNNING' and j.get('entrypoint_num_gpus'))
print(max(int(total-used), 0))
FREEPY
}
nrun(){ python3 - "$1" <<'PY' 2>/dev/null
import json,sys,urllib.request
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/'))
print(sum(1 for j in d if j['status'] in ('RUNNING','PENDING') and sys.argv[1] in (j.get('entrypoint') or '')))
PY
}
sub(){ timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait \
  --working-dir /workspace/le-wm --runtime-env-json "$EXC" -- "$@" 2>&1 \
  | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }

GSTAR="$BUCKET/qgate/g_star_cube.json"

pick_gate(){  # 选 λ 并把选中的 json 复制为 g_star_cube.json;stdout 只输出 0/1 成败
  python3 - "$BUCKET" <<'PY'
import json, subprocess, sys
B = sys.argv[1]
def fetch(lam):
    r = subprocess.run(["gcloud","storage","cat",f"{B}/qgate/qgate_stage1_cube_lam{lam}.json"],
                       capture_output=True)
    return json.loads(r.stdout) if r.returncode == 0 else None
cands = {lam: fetch(lam) for lam in ("0.01","0.003","0.03")}
if any(v is None for v in cands.values()):
    print(0); sys.exit()
import math
def usable(p):
    gs = list(p["g_star"].values())
    return p["verdict"] == "OK" and all(isinstance(v, (int, float)) and math.isfinite(v) for v in gs)
choice, reason = None, ""
for lam in ("0.01","0.003","0.03"):
    if usable(cands[lam]):
        choice, reason = lam, f"lambda={lam} verdict OK, gates finite (blind gap {cands[lam]['goal_blind_gap_nats']})"
        break
if choice is None:
    # 绝不拿不可解释/非有限的 gate 去训练:拒绝放行,链子继续等
    print("[pick] REFUSED: no lambda has verdict OK with finite gates", file=__import__("sys").stderr, flush=True)
    print(0); __import__("sys").exit()
payload = cands[choice]; payload["chosen_lambda"] = choice; payload["choice_reason"] = reason
open("/tmp/g_star_cube.json","w").write(json.dumps(payload, ensure_ascii=False, indent=1))
subprocess.run(["gcloud","storage","cp","/tmp/g_star_cube.json",f"{B}/qgate/g_star_cube.json"],
               capture_output=True)
import sys as s2
print(f"[pick] {reason}", file=s2.stderr, flush=True)
print(1)
PY
}

# name|run|probe|command
TRAINS=(
"qg|lewm_qgate_scale_cube_s${SEED}|experiment=qgate_scale_cube|env QGATE_GCS=$GSTAR bash scripts/ray_train_qgate2.sh cube experiment=qgate_scale_cube seed=${SEED}"
"qa|lewm_qall_scale_cube_s${SEED}|experiment=qall_scale_cube|bash scripts/ray_train_qgate2.sh cube experiment=qall_scale_cube seed=${SEED}"
"qs|lewm_qgate_sharp_scale_cube_s${SEED}|experiment=qgate_sharp_scale_cube|env QGATE_GCS=$BUCKET/qgate/qgate_stage1_cube_lam0.1.json bash scripts/ray_train_qgate2.sh cube experiment=qgate_sharp_scale_cube seed=${SEED}"
"qm|lewm_qgate05_scale_cube_s${SEED}|experiment=qgate05_scale_cube|env QGATE_GCS=$BUCKET/qgate/qgate_stage1_cube_lam0.05.json bash scripts/ray_train_qgate2.sh cube experiment=qgate05_scale_cube seed=${SEED}"
"qg03|lewm_qgate03_scale_cube_s${SEED}|experiment=qgate03_scale_cube|env QGATE_GCS=$BUCKET/qgate/qgate_stage1_cube_lam0.03.json bash scripts/ray_train_qgate2.sh cube experiment=qgate03_scale_cube seed=${SEED}"
)
EVALS=()
for spec in "qg|lewm_qgate_scale_cube_s${SEED}" "qa|lewm_qall_scale_cube_s${SEED}" "qs|lewm_qgate_sharp_scale_cube_s${SEED}" "qm|lewm_qgate05_scale_cube_s${SEED}" "qg03|lewm_qgate03_scale_cube_s${SEED}"; do
  IFS='|' read -r cfg run <<< "$spec"
  for sol in cem icem mppi; do
    EVALS+=("$cfg|$run|$sol|101 102 103")
    EVALS+=("$cfg|$run|$sol|104 105 106")
  done
done

log "start: Stage-2 Auto-SCALE cube (qg gated + qa ungated control) at seed $SEED"
for round in $(seq 1 6000); do
  if ! gcloud storage ls "$GSTAR" >/dev/null 2>&1; then
    ok=$(pick_gate 2>>"$L")
    [ "$ok" = 1 ] && log "g_star_cube.json chosen and staged" || true
    [ "$ok" != 1 ] && { sleep 240; continue; }
  fi
  left=0; submitted=0
  for spec in "${TRAINS[@]}"; do
    IFS='|' read -r name run probe cmd <<< "$spec"
    gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 && continue
    left=1
    if [ "$name" = qm ] && ! gcloud storage ls "$BUCKET/qgate/qgate_stage1_cube_lam0.05.json" >/dev/null 2>&1; then continue; fi
    if [ "$name" = qg03 ] && ! gcloud storage ls "$BUCKET/qgate/qgate_stage1_cube_lam0.03.json" >/dev/null 2>&1; then continue; fi
    [ "$(nrun "$probe")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    n=${ATT[$name]:-0}
    [ "$n" -ge 4 ] && { log "$name attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub $cmd)
    if [ -n "$id" ]; then ATT[$name]=$((n+1)); log "$name attempt $((n+1)) -> $id"
    else log "$name submit FAILED"; fi
    submitted=1; break
  done
  for cell in "${EVALS[@]}"; do
    IFS='|' read -r cfg run sol seeds <<< "$cell"
    miss=0
    for s in $seeds; do
      gcloud storage ls "$BUCKET/final_eval/final_cube_${cfg}_${sol}_s${s}.csv" >/dev/null 2>&1 || miss=1
    done
    [ "$miss" = 0 ] && continue
    left=1
    gcloud storage ls "$BUCKET/ckpts/$run/weights_epoch_10.pt" >/dev/null 2>&1 || continue
    [ "$submitted" != 0 ] && continue
    [ "$(nrun "ray_eval_final.sh cube $cfg $run $sol ${seeds%% *}")" != 0 ] && continue
    [ "$(free)" -lt 1 ] && continue
    key="ev_${cfg}_${sol}_${seeds%% *}"
    n=${ATT[$key]:-0}
    [ "$n" -ge 4 ] && { log "$key attempt cap"; continue; }
    # shellcheck disable=SC2086
    id=$(sub bash scripts/ray_eval_final.sh cube "$cfg" "$run" "$sol" $seeds)
    if [ -n "$id" ]; then ATT[$key]=$((n+1)); log "$key attempt $((n+1)) -> $id"
    else log "$key submit FAILED"; fi
    submitted=1; break
  done
  [ "$left" = 0 ] && { log "AUTO-SCALE TRAININGS + EVALS COMPLETE (90 CSVs incl mppi)"; exit 0; }
  sleep 240
done
log "round cap"; exit 1
