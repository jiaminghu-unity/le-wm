#!/usr/bin/env bash
# Submit representation diagnostics (ray_rebuttal_diag.sh) for every rebuttal checkpoint
# (seeds 1024/1025) that has no diagnostic JSON yet. One job per task; safe to re-run.
export PATH=/usr/local/bin:/snap/bin:/usr/bin:/bin:$PATH
export RAY_API_SERVER_ADDRESS='http://127.0.0.1:8265'
REPO=/home/jiaming.hu/le-wm; B=gs://prism-training-us/le-wm
cd "$REPO" || exit 1
EXC='{"excludes":["ckpts","eval_results","assets","artifacts","reports","qgate_jsons",".git","**/__pycache__"]}'
# lewm_qinput_* (state-input models) have no image encoder: the pixel diagnostics do not apply to them
CK=$(for d in ckpts ckpts_tworoom ckpts_pointmaze; do timeout 120 gcloud storage ls "$B/$d/*_s102[45]/weights_epoch_10.pt" 2>/dev/null; done | grep -v "/lewm_qinput_")
DONE=$(timeout 120 gcloud storage ls "$B/eval/rebuttal_diag/" 2>/dev/null)
ACTIVE=$(python3 -c "
import json,urllib.request
for j in json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/',timeout=30)):
    if j.get('status') in ('RUNNING','PENDING') and 'ray_rebuttal_diag.sh' in (j.get('entrypoint') or ''): print(j['entrypoint'].split('ray_rebuttal_diag.sh')[1].split()[0])")
python3 - "$CK" "$DONE" <<'PY' > /tmp/rebuttal_diag_todo.txt
import sys,collections
done={l.rsplit('/',1)[-1][:-5] for l in sys.argv[2].split() if l.endswith('.json')}
todo=collections.defaultdict(list)
for l in sys.argv[1].split():
    pfx,run=l.split('/')[-3],l.split('/')[-2]
    task=('pointmaze' if 'pointmaze' in run else 'tworoom' if 'tworoom' in run else 'cube' if 'cube' in run
          else 'reacher' if ('reacher' in run or run.startswith('lewm_r5_')) else 'pusht')
    if f"{task}__{run}" not in done: todo[task].append(f"{pfx}:{run}")
for t,v in todo.items(): print(t,' '.join(sorted(v)))
PY
while read -r task rest; do
  [ -n "$rest" ] || continue
  if grep -qx "$task" <<< "$ACTIVE"; then echo "$task: diag job already active"; continue; fi
  # shellcheck disable=SC2086
  id=$(timeout 240 ray job submit --entrypoint-num-gpus=1 --no-wait --working-dir "$REPO" --runtime-env-json "$EXC" -- bash scripts/ray_rebuttal_diag.sh $task $rest 2>&1 | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
  echo "$task: $(wc -w <<< "$rest") runs -> $id"
done < /tmp/rebuttal_diag_todo.txt
