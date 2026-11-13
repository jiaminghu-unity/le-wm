#!/usr/bin/env bash
# Rebuttal chain (2026-11-11): train -> cem/icem x 6 episode seeds for every model in a spec file.
#   usage: run_rebuttal_chain.sh <name> <specfile>        (DRY=1 prints what it would submit)
#   spec line: task|ckpt-dir|eval-dir|run|eval-cfg-label|train command
# Same conventions as run_rerun3073_chain.sh / run_replica3074_chain.sh (GCS done-checks make
# restarts free), with three differences forced by the current cluster:
#   * runs from ~/le-wm (the /workspace/le-wm copy is not writable by this user)
#   * nothing is pinned to a GPU type: the L4 pool is max_workers=0 in ray_cluster.resolved.yaml,
#     so an L4-pinned job would pend forever, and an A100 pin blocks autoscaling (see sub()).
#   * done-checks list each bucket prefix once per round instead of one `ls` per file.
export PATH=/usr/local/bin:/snap/bin:/usr/bin:/bin:$PATH
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
NAME="${1:?chain name}"; SPECFILE="${2:?spec file}"
REPO=/home/jiaming.hu/le-wm
cd "$REPO" || exit 1
DONE="$REPO/eval_results/${NAME}.DONE"
[ -f "$DONE" ] && exit 0
BUCKET=gs://prism-training-us/le-wm
EXC='{"excludes":["ckpts","eval_results","assets","artifacts","reports","qgate_jsons",".git","**/__pycache__"],"env_vars":{"RAY_JOB_START_TIMEOUT_SECONDS":"14400"}}'
L="$REPO/eval_results/${NAME}.log"
CAP=40   # pending-timeouts (900 s, set on the head) and boot races are side-effect-free; keep retrying
log(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$L"; }
declare -A ATT
mapfile -t SPECS < <(grep -vE '^\s*(#|$)' "$SPECFILE")
[ "${#SPECS[@]}" -gt 0 ] || { echo "empty spec file" >&2; exit 1; }

# line 1: GPUs we may still ask for (alive + 10 queue pressure - active GPU jobs); then active entrypoints
snapshot(){ python3 - <<'PY' 2>/dev/null
import json, urllib.request
nodes=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/v0/nodes?limit=500', timeout=30))
rows=nodes.get('data',{}).get('result',{}).get('result',[])
total=sum(n.get('resources_total',{}).get('GPU',0) for n in rows if n.get('state')=='ALIVE')
jobs=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/', timeout=30))
act=[j for j in jobs if j.get('status') in ('RUNNING','PENDING')]
used=sum(1 for j in act if 'scripts/ray_' in (j.get('entrypoint') or '') or j.get('entrypoint_num_gpus'))
print(max(int(total+10-used),0))
for j in act: print(j.get('entrypoint') or '')
PY
}
# list one wildcard; "matched no objects" is an empty result, any other failure aborts the round
lsq(){ local out rc; out=$(timeout 180 gcloud storage ls "$1" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then echo "$out"; return 0; fi
  grep -q "matched no objects" <<< "$out" && return 0
  return 1; }
# No accelerator pin at all: ray_cluster.resolved.yaml declares the worker type as {CPU, GPU} only,
# so the autoscaler treats a demand that also asks for accelerator_type:A100 as unsatisfiable and
# never scales up for it (seen 2026-11-11: 12 pinned jobs pending, cluster stuck at 8 nodes). Every
# worker in the pool is an A100 anyway.
# ENTRY_RESOURCES (optional, e.g. '{"rb": 1}'): extra entrypoint resources. rb exists only on the hand-joined
# grab VMs (rebuttal_grab/join_startup.sh), so such jobs never land on autoscaler workers and the
# autoscaler does not scale up for them.
sub(){
  timeout 240 ray job submit --entrypoint-num-gpus=1 ${ENTRY_RESOURCES:+--entrypoint-resources "$ENTRY_RESOURCES"} --no-wait \
    --working-dir "$REPO" --runtime-env-json "$EXC" -- "$@" 2>&1 \
    | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1; }
try(){ local key=$1; shift
  grep -qF -- "$*" <<< "$ACTIVE" && return 1
  [ "$FREE" -lt 1 ] && return 1
  local n=${ATT[$key]:-0}
  [ "$n" -ge "$CAP" ] && { [ "$n" -eq "$CAP" ] && log "$key attempt cap ($CAP) reached"; ATT[$key]=$((n+1)); return 1; }
  if [ -n "${DRY:-}" ]; then echo "DRY $key :: $*"; FREE=$((FREE-1)); return 0; fi
  local id; id=$(sub "$@")
  if [ -n "$id" ]; then ATT[$key]=$((n+1)); FREE=$((FREE-1)); log "$key attempt $((n+1)) -> $id"
  else log "$key submit FAILED"; fi
}

# patterns covering every run / label in the spec (seed suffix of the run, label suffix of the cfg)
SEEDPAT=$(printf '%s\n' "${SPECS[@]}" | awk -F'|' '{n=split($4,a,"_s"); print a[n]}' | sort -u | paste -sd'|')
log "start: $NAME (${#SPECS[@]} models, seeds $SEEDPAT)"
for round in $(seq 1 20000); do
  SNAP=$(snapshot); [ -n "$SNAP" ] || { log "ray API not answering, retry"; sleep 60; continue; }
  FREE=$(head -1 <<< "$SNAP"); ACTIVE=$(tail -n +2 <<< "$SNAP")
  ok=1; CK=""; EV=""
  for d in $(printf '%s\n' "${SPECS[@]}" | cut -d'|' -f2 | sort -u); do
    for s in ${SEEDPAT//|/ }; do x=$(lsq "$BUCKET/$d/*_s${s}/weights_epoch_10.pt") || ok=0; CK+="$x"$'\n'; done
  done
  for d in $(printf '%s\n' "${SPECS[@]}" | cut -d'|' -f3 | sort -u); do
    for s in ${SEEDPAT//|/ }; do x=$(lsq "$BUCKET/$d/final_*r${s}_*.csv") || ok=0; EV+="$x"$'\n'; done
  done
  [ "$ok" = 1 ] || { log "bucket listing failed, retry"; sleep 60; continue; }
  left=0; ntrain=0; nev=0
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r task ckdir evdir run cfg traincmd <<< "$spec"
    if ! grep -qF "/$ckdir/$run/weights_epoch_10.pt" <<< "$CK"; then
      left=1; ntrain=$((ntrain+1))
      # shellcheck disable=SC2086
      try "tr_${task}_${cfg}" $traincmd
      continue
    fi
    for sol in cem icem; do
      for seeds in "101 102 103" "104 105 106"; do
        miss=0
        for s in $seeds; do grep -qF "/$evdir/final_${task}_${cfg}_${sol}_s${s}.csv" <<< "$EV" || miss=1; done
        [ "$miss" = 0 ] && continue
        left=1; nev=$((nev+1))
        # q-input (state-based) models, labels qf*: same launchers, state-key sweeper, and the
        # q-input trainer uploads every task to ckpts/ (not ckpts_tworoom/ or ckpts_pointmaze/)
        pre=""
        case "$cfg" in qf*) pre="env SWEEP_OVERRIDE=scripts/budget_sweep_qinput_any.py CKPT_ROOT_OVERRIDE=ckpts ";; esac
        case "$task" in
          tworoom)   cmd="${pre}bash scripts/ray_eval_tworoom.sh $cfg $run $sol $seeds";;
          pointmaze) cmd="${pre}bash scripts/ray_eval_pointmaze.sh $cfg $run $sol $seeds";;
          # multi-object tasks of the Auto-SCALE grid: same launcher and T1-only protocol as that grid
          cube_double|cube_triple|cube_quadruple|scene)
                     case "$cfg" in
                       qf*) cmd="env EVAL_TIERS=T1 bash scripts/ray_eval_qinput_ogbmulti.sh $task $cfg $run $sol $seeds";;
                       *)   cmd="env EVAL_TIERS=T1 bash scripts/ray_eval_ogbmulti.sh $task $cfg $run $sol $seeds";;
                     esac;;
          *)         cmd="${pre}bash scripts/ray_eval_final.sh $task $cfg $run $sol $seeds";;
        esac
        # shellcheck disable=SC2086
        try "ev_${task}_${cfg}_${sol}_${seeds%% *}" $cmd
      done
    done
  done
  [ -n "${DRY:-}" ] && { echo "DRY: $ntrain trainings and $nev eval jobs outstanding"; exit 0; }
  [ "$left" = 0 ] && { log "$NAME COMPLETE"; touch "$DONE"; exit 0; }
  [ $((round % 15)) -eq 1 ] && log "status: $ntrain trainings, $nev eval jobs outstanding; free=$FREE"
  sleep 120
done
log "round cap"; exit 1
