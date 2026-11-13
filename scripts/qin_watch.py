#!/usr/bin/env python3
"""Compact view of the state-model (q-input) jobs: status counts, the lines that prove the right
state variant / builder was used, last training progress, and any real error. Read-only.
Prints HIT as the last line when something needs a look (progress, eval finished, real error)."""
import json, re, subprocess, sys, urllib.request
from collections import Counter
SINCE = 1794543000000  # 2026-11-13 04:10 UTC
j = json.load(urllib.request.urlopen('http://127.0.0.1:8265/api/jobs/', timeout=30))
mine = [x for x in j if ('ray_train_qnative' in (x.get('entrypoint') or '') or 'budget_sweep_qinput_any' in (x.get('entrypoint') or ''))
        and (x.get('start_time') or 0) > SINCE]
print('STATUS', dict(Counter(x['status'] for x in mine)))
want = sys.argv[1:]  # optional: only jobs whose entrypoint contains one of these strings trigger HIT
seen_ok = set(open('/tmp/qin_watch_seen.txt').read().split()) if __import__('os').path.exists('/tmp/qin_watch_seen.txt') else set()
hit = False
KEY = r'\[qinput\]|\[qnative\]|\[stats\]|\[replica\]|\[qinput-any\]|\[gl\]|fidelity|MAE|\[done\]|REPLICA TRAIN DONE|\[skip\]|\[run\]'
ERR = r'Traceback|FATAL|AssertionError|KeyError|RuntimeError|ValueError|No such file|CUDA out of memory'
for x in sorted(mine, key=lambda x: x.get('start_time') or 0):
    if x['status'] not in ('RUNNING', 'FAILED', 'SUCCEEDED'):
        continue
    out = subprocess.run(['ray', 'job', 'logs', x['submission_id']], capture_output=True, text=True, timeout=240).stdout.replace('\r', '\n')
    lines = out.splitlines()
    key = list(dict.fromkeys(l for l in lines if re.search(KEY, l) and 'variants registered' not in l))
    err = [l for l in lines if re.search(ERR, l)]
    prog = [l for l in lines if re.search(r'\[Epoch \d+/10\] step', l)]
    msg = (x.get('message') or '')
    node = any(k in msg for k in ('node has died', 'SIGTERM', 'node died', 'timed out', 'Timed out')) or 'node has died' in out
    e = re.sub(r'^.*bash scripts/', '', x.get('entrypoint') or '')
    print('=====', x['status'], x['submission_id'], e[:100], '(node died / pending timeout)' if node and x['status'] == 'FAILED' else '')
    for l in key[-8:]:
        print('   ', l.strip()[:300])
    for l in err[-6:]:
        print('    ERR', l.strip()[:300])
    if prog:
        print('    PROG', prog[-1].strip()[:140])
    if x['status'] == 'FAILED' and not node and not err:
        print('    TAIL', ' | '.join(l.strip()[:140] for l in lines[-5:]))
    interesting = bool(prog) or bool(err) or x['status'] == 'SUCCEEDED' or (x['status'] == 'FAILED' and not node)
    if interesting and x['submission_id'] not in seen_ok and (not want or any(w in (x.get('entrypoint') or '') for w in want)):
        hit = True
print('HIT' if hit else 'WAIT')
