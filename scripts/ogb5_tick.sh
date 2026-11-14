#!/usr/bin/env bash
# Monitoring pass for chains ogb_base_swim_1024 (LeWM and SWIM at seeds 1024/1025, five tiers) and ogb_t25
# (tiers T2-T5 for the 40 existing Aux / state-based models) on the four multi-object tasks.
#   usage: auxl2_tick.sh [wait-minutes]   waits up to that long first (leaves early when the chain
#          finishes or its process is gone), then prints one status pass and mirrors new eval CSVs.
# Read-only towards the workers (Ray API and job logs only, never ssh).
export PATH=/usr/local/bin:/snap/bin:/usr/bin:/bin:$PATH
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
cd /home/jiaming.hu/le-wm || exit 1
W=${1:-0}; CHAINS="ogb_base_swim_1024 ogb_t25"
alldone(){ for c in $CHAINS; do [ -f eval_results/$c.DONE ] || return 1; done; }
anydead(){ for c in $CHAINS; do [ -f eval_results/$c.DONE ] && continue; pgrep -f "^flock -n -o /home/jiaming.hu/.rebuttal_$c.lock" >/dev/null || return 0; done; return 1; }
end=$(( $(date +%s) + W * 60 ))
until alldone || anydead || [ "$(date +%s)" -ge "$end" ]; do sleep 60; done
date -u '+== %m-%d %H:%M UTC =='
alldone && echo "ALL CHAINS DONE"
for c in $CHAINS; do
  if [ -f eval_results/$c.DONE ]; then st=DONE; elif pgrep -f "^flock -n -o /home/jiaming.hu/.rebuttal_$c.lock" >/dev/null; then st=alive; else st="PROCESS NOT RUNNING"; fi
  echo "chain $c: $st | $(grep -E 'status:|COMPLETE|cap' eval_results/$c.log | tail -1 | cut -c1-100)"
done
pgrep -f "^bash grab_spot40.sh" >/dev/null && echo "grab daemon alive" || echo "GRAB DAEMON NOT RUNNING"
B=gs://prism-training-us/le-wm; M=eval_results/rebuttal_mirror/final_eval_ogbmulti; mkdir -p $M
CK=$(timeout 180 gcloud storage ls "$B/ckpts/lewm_*_s102[45]/weights_epoch_10.pt" 2>/dev/null | grep -cE '(cubedouble|cubetriple|cubequadruple|scene)_(base|scale_gate03)_s')
EV=$(timeout 180 gcloud storage ls "$B/final_eval_ogbmulti/final_*r102[45]_*.csv" 2>/dev/null | grep -cE '_(baser|g03r)102[45]_')
T25=$(timeout 180 gcloud storage ls "$B/final_eval_ogbmulti/final_*_t25*r102[45]_*.csv" 2>/dev/null | wc -l)
echo "LeWM / SWIM at new seeds: trained $CK/16, five-tier eval CSVs $EV/192   |   T2-T5 files for the 40 existing models: $T25/480"
python3 - "$((W + 10))" <<'PY'
import json, sys, urllib.request, collections, time, re, subprocess
win = float(sys.argv[1]); now = time.time() * 1000
d = json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/', timeout=30))
def mine(e):
    multi = any(t in e for t in (' cube_double ', ' cube_triple ', ' cube_quadruple ', ' scene '))
    return multi and bool(re.search(r'(_s|seed=)102[45]', e)) and 'scripts/ray_' in e
J = [j for j in d if mine(j.get('entrypoint') or '')]
print("jobs:", dict(collections.Counter(j['status'] for j in J)))
run = [j for j in J if j['status'] == 'RUNNING']
tr = [j for j in run if 'ray_train_' in j['entrypoint']]
print(f"running: {len(tr)} trainings, {len(run) - len(tr)} evals")
f = [j for j in J if j['status'] == 'FAILED' and (j.get('end_time') or 0) > now - win * 60000]
c = collections.Counter(); other = []
for j in f:
    m = j.get('message') or ''; kind = 'train' if 'ray_train_' in j['entrypoint'] else 'eval'
    if 'node was terminated' in m or 'supervisor actor died' in m or 'node has died' in m: k = 'node died'
    elif 'failed to start within' in m: k = 'pending timeout'
    elif 'lock' in m and ('dpkg' in m or 'apt' in m): k = 'apt lock'
    elif 'apparently in use' in m: k = 'nvme busy'
    else:
        k = 'OTHER'; lines = [l for l in m.strip().splitlines() if l.strip()]
        other.append((j['submission_id'], re.sub(r'^.*bash scripts/', '', j['entrypoint'])[:80], lines[-1][:160] if lines else ''))
    c[(kind, k)] += 1
print(f"failures in last {win:.0f} min:", dict(c))
for o in other[:8]: print("  OTHER:", o)
ok = [j for j in J if j['status'] == 'SUCCEEDED']
for kind in ('ray_train_', 'ray_eval_'):
    v = sorted((j['end_time'] - j['start_time']) / 3.6e6 for j in ok if kind in j['entrypoint'])
    if v: print(f"finished {kind[4:-1]} jobs: n={len(v)} median {v[len(v)//2]:.2f} h (min {v[0]:.2f}, max {v[-1]:.2f})")
print("running trainings (age, run, last progress, last aux diagnostic):")
for j in sorted(tr, key=lambda j: j['start_time'])[:8]:   # oldest 8 only: `ray job logs` takes ~1 min per job
    e = j['entrypoint']
    try:
        out = subprocess.run(["ray", "job", "logs", j["submission_id"]], capture_output=True, text=True, timeout=75).stdout.replace("\r", "\n")
    except Exception:
        out = ""
    prog = [l for l in out.splitlines() if re.search(r"\[Epoch \d+/10\] step", l)]
    diag = [l for l in out.splitlines() if "[auxdiag]" in l]
    shm = "unable to open shared memory" in out or "sem_unlink" in out
    m = re.search(r"output_model_name=(\S+)", e)
    name = m.group(1) if m else re.sub(r"^.*ray_train_qnative\.sh ", "", e)[:44]
    age = (now - j["start_time"]) / 3.6e6
    flag = " <-- HUNG? (shared-memory error)" if shm else (" <-- no progress line" if not prog and age > 1.0 else "")
    p = re.sub(r"^.*(\[Epoch \d+/10\] step\S*\s*\S*).*$", r"\1", prog[-1]).strip()[:34] if prog else "(setting up)"
    g = re.sub(r"^.*\[auxdiag\] ", "", diag[-1]).strip()[:60] if diag else ""
    print(f"  {age:4.1f}h {name:44s} {p:34s} {g}{flag}")
PY
for pat in 'final_*_baser102[45]_*.csv' 'final_*_g03r102[45]_*.csv' 'final_*_t25*r102[45]_*.csv'; do timeout 300 gcloud storage cp -n "$B/final_eval_ogbmulti/$pat" $M/ >/dev/null 2>&1; done
echo "mirrored new CSVs: $(ls $M 2>/dev/null | grep -cE '_(baser|g03r|t25[a-z0-9.]*r)102[45]_')"
echo "grab VMs running: $(timeout 90 gcloud compute instances list --filter='labels.purpose=scale-rebuttal AND status=RUNNING' --format='value(name)' 2>/dev/null | wc -l)"
since=$(date -u -d "-$((W + 10)) min" '+%Y-%m-%dT%H:%M:%SZ')
echo "preemptions of grab VMs, last $((W + 10)) min: $(timeout 120 gcloud compute operations list --filter="operationType=compute.instances.preempted AND insertTime>\"$since\" AND targetLink~jm-rb-a40" --format='value(targetLink)' 2>/dev/null | wc -l)"
