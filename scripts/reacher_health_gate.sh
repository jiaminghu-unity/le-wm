#!/bin/bash
# Epoch-0 health gate for the Reacher runs (spec Step 2): when a run enters
# epoch 1, check its telemetry; on violation, SIGSTOP the run (resumable with
# kill -CONT) and write a loud ALARM file. Detached-safe.
set -u
L=/mnt/data/stable-wm/train_logs
PY=/mnt/data/code/le-wm/.venv/bin/python

check_run() {
    local log=$1 pattern=$2 metrics=$3 has_obj=$4
    until grep -q "\[Epoch 1/" "$L/$log" 2>/dev/null; do sleep 60; done
    verdict=$("$PY" - "$metrics" "$has_obj" <<'PYEOF'
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1])
has_obj = sys.argv[2] == "1"
ep0 = df[df["epoch"] == 0] if "epoch" in df else df
bad = []
zn = ep0["fit/z_norm_mean"].dropna()
if len(zn) and not (12.0 < zn.iloc[-3:].mean() < 15.5):
    bad.append(f"z_norm={zn.iloc[-3:].mean():.2f} not near sqrt(192)")
er = ep0["fit/eff_rank"].dropna()
if len(er) >= 6:
    tail = er.iloc[-6:].to_numpy()
    if all(tail[i] > tail[i+1] for i in range(5)) and tail[-1] < 15:
        bad.append(f"eff_rank monotone collapse tail={tail.round(1).tolist()}")
if len(er) and er.iloc[-3:].mean() < 10:
    bad.append(f"eff_rank={er.iloc[-3:].mean():.1f} < 10")
if has_obj:
    gr = ep0["fit/grad_ratio_obj_pred"].dropna()
    if len(gr) and gr.iloc[-3:].mean() >= 10:
        bad.append(f"grad_ratio={gr.iloc[-3:].mean():.1f} not single-digit")
    sk = ep0["fit/obj_skipped"].dropna()
    if len(sk) and sk.iloc[-1] > 0:
        bad.append(f"obj_skipped={sk.iloc[-1]:.0f}")
print("|".join(bad) if bad else "OK")
PYEOF
)
    if [ "$verdict" = "OK" ]; then
        echo "[$(date)] $log passed epoch-0 health gates" >> "$L/health_gate.log"
    else
        pid=$(pgrep -f "$pattern" | head -1)
        [ -n "$pid" ] && kill -STOP "$pid"
        echo "[$(date)] ALARM $log GATE VIOLATION: $verdict — paused pid $pid (kill -CONT to resume)" \
            | tee "$L/ALARM_$log.txt" >> "$L/health_gate.log"
    fi
}

check_run r1_reacher.log "experiment=r1_reacher_baseline" \
    /mnt/data/cache/stable-pretraining/runs/20260722/083337/71a625df8e5d/metrics.csv 0 &
check_run r2_reacher.log "experiment=r2_reacher_paep" \
    /mnt/data/cache/stable-pretraining/runs/20260722/083337/3d5ca603f3f5/metrics.csv 1 &
wait
