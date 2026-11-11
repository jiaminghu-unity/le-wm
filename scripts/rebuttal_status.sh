#!/usr/bin/env bash
# One-shot status of the rebuttal round (seeds 1024/1025). Read-only.
#   usage: rebuttal_status.sh [minutes]   -- failure breakdown covers the last <minutes> (default 40)
export PATH=/usr/local/bin:/snap/bin:/usr/bin:/bin:$PATH
WIN="${1:-40}"
REPO=/home/jiaming.hu/le-wm; B=gs://prism-training-us/le-wm
date -u '+== %m-%d %H:%M UTC =='
echo "chain: $(pgrep -f '^flock -n -o /home/jiaming.hu/.rebuttal_' >/dev/null && echo alive || echo DEAD)  grab: $(pgrep -f '^bash grab_spot40.sh' >/dev/null && echo alive || echo DEAD)  done-file: $([ -f $REPO/eval_results/all_seeds1024.DONE ] && echo yes || echo no)"
CK=$(for d in ckpts ckpts_tworoom ckpts_pointmaze; do timeout 120 gcloud storage ls "$B/$d/*_s102[45]/weights_epoch_10.pt" 2>/dev/null; done | sed "s#.*/\([^/]*\)/weights_epoch_10.pt#\1#" | sort)
EV=$(for d in final_eval final_eval_tworoom final_eval_pointmaze; do timeout 120 gcloud storage ls "$B/$d/final_*r102[45]_*.csv" 2>/dev/null; done | wc -l)
echo "trained: $(grep -c . <<< "$CK")/52   eval CSVs: $EV/624"
python3 - "$CK" <<'PY'
import sys,collections
c=collections.Counter()
for r in sys.argv[1].split():
    t=[t for t in ('pusht','reacher','cube','tworoom','pointmaze') if t in r]
    c[t[0] if t else ('pusht' if '_c' in r else '?')]+=1
print("trained by task:",dict(c))
PY
python3 - "$WIN" <<'PY'
import json,sys,urllib.request,collections,time
win=float(sys.argv[1]); now=time.time()*1000
d=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/',timeout=30))
mine=[j for j in d if '102' in (j.get('entrypoint') or '') and 'scripts/ray_' in (j.get('entrypoint') or '')]
def task(e): return next((t for t in ('pusht','reacher','cube','tworoom','pointmaze') if t in e),'?')
print("jobs:",dict(collections.Counter(j['status'] for j in mine)))
run=[j for j in mine if j['status']=='RUNNING']
tr=[j for j in run if 'ray_train_' in j['entrypoint']]
print(f"running: {len(tr)} trainings, {len(run)-len(tr)} evals")
ages=collections.defaultdict(list)
for j in tr: ages[task(j['entrypoint'])].append(round((now-j['start_time'])/3.6e6,1))
print("training ages (h):",{k:sorted(v,reverse=True) for k,v in ages.items()})
f=[j for j in mine if j['status']=='FAILED' and (j.get('end_time') or 0)>now-win*60000]
c=collections.Counter(); other=[]
for j in f:
    m=j.get('message') or ''; kind='train' if 'ray_train_' in j['entrypoint'] else 'eval'
    if 'node was terminated' in m or 'supervisor actor died' in m: k='node died'
    elif 'failed to start within' in m: k='pending timeout'
    elif 'lock' in m and ('dpkg' in m or 'apt' in m): k='apt lock'
    elif 'apparently in use' in m: k='nvme busy'
    else:
        k='OTHER'; lines=[l for l in m.strip().splitlines() if l.strip()]
        other.append((j['submission_id'],j['entrypoint'][:70],lines[-1][:140] if lines else ''))
    c[(kind,k)]+=1
print(f"failures in last {win:.0f} min:",dict(c))
for o in other: print("  OTHER:",o)
ok=[j for j in mine if j['status']=='SUCCEEDED' and 'ray_train_' in j['entrypoint']]
dur=collections.defaultdict(list)
for j in ok: dur[task(j['entrypoint'])].append(round((j['end_time']-j['start_time'])/3.6e6,1))
print("finished training durations (h):",{k:(min(v),max(v),len(v)) for k,v in dur.items()})
rows=json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/v0/nodes?limit=500',timeout=30))['data']['result']['result']
g=[n['node_ip'] for n in rows if n['state']=='ALIVE' and (n.get('resources_total') or {}).get('GPU')]
reg={'10.150.':'us-east4(80GB)','10.128.':'us-central1','10.142.':'us-east1','10.138.':'us-west1'}
print("GPU nodes:",len(g),dict(collections.Counter(next((v for k,v in reg.items() if i.startswith(k)),'other') for i in g)))
PY
echo "chain log:"; tail -3 $REPO/eval_results/all_seeds1024.log | cut -c1-130
echo "grab log:";  tail -2 $REPO/scripts/rebuttal_grab/grab.log | cut -c1-150
