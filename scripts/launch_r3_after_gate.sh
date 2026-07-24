#!/bin/bash
# Spec Step 2: R3 (joints_plus_finger ablation) launches ONLY after R2 passes
# its epoch-0 health gates. Waits on the gate verdict, launches R3 on GPU2,
# then applies the same epoch-0 gate to R3 itself. Detached-safe.
set -u
L=/mnt/data/stable-wm/train_logs
cd /mnt/data/code/le-wm
source .venv/bin/activate
export STABLEWM_HOME=/mnt/data/stable-wm

# wait for R2's gate verdict
until grep -q "r2_reacher.log" "$L/health_gate.log" 2>/dev/null; do sleep 60; done
if [ -f "$L/ALARM_r2_reacher.log.txt" ]; then
    echo "[$(date)] R2 failed health gates — R3 NOT launched" >> "$L/health_gate.log"
    exit 0
fi

echo "[$(date)] R2 passed — launching R3 on GPU2" >> "$L/health_gate.log"
CUDA_VISIBLE_DEVICES=2 setsid nohup python train.py experiment=r3_reacher_paep_finger \
    > "$L/r3_reacher.log" 2>&1 < /dev/null &

# discover R3's metrics dir, then run the same epoch-0 gate
until d=$(grep -oE "runs/20260722/[0-9]+/[a-f0-9]+" "$L/r3_reacher.log" 2>/dev/null | head -1); [ -n "${d:-}" ]; do sleep 30; done
METRICS="/mnt/data/cache/stable-pretraining/$d/metrics.csv"
until grep -q "\[Epoch 1/" "$L/r3_reacher.log" 2>/dev/null; do sleep 60; done

verdict=$(.venv/bin/python - "$METRICS" <<'PYEOF'
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1])
ep0 = df[df["epoch"] == 0] if "epoch" in df else df
bad = []
zn = ep0["fit/z_norm_mean"].dropna()
if len(zn) and not (12.0 < zn.iloc[-3:].mean() < 15.5):
    bad.append(f"z_norm={zn.iloc[-3:].mean():.2f}")
er = ep0["fit/eff_rank"].dropna()
if len(er) >= 6:
    tail = er.iloc[-6:].to_numpy()
    if all(tail[i] > tail[i+1] for i in range(5)) and tail[-1] < 15:
        bad.append(f"eff_rank collapsing tail={tail.round(1).tolist()}")
if len(er) and er.iloc[-3:].mean() < 10:
    bad.append(f"eff_rank={er.iloc[-3:].mean():.1f}<10")
gr = ep0["fit/grad_ratio_obj_pred"].dropna()
if len(gr) and gr.iloc[-3:].mean() >= 10:
    bad.append(f"grad_ratio={gr.iloc[-3:].mean():.1f}")
sk = ep0["fit/obj_skipped"].dropna()
if len(sk) and sk.iloc[-1] > 0:
    bad.append(f"obj_skipped={sk.iloc[-1]:.0f}")
print("|".join(bad) if bad else "OK")
PYEOF
)
if [ "$verdict" = "OK" ]; then
    echo "[$(date)] r3_reacher.log passed epoch-0 health gates" >> "$L/health_gate.log"
else
    pid=$(pgrep -f "experiment=r3_reacher_paep_finger" | head -1)
    [ -n "$pid" ] && kill -STOP "$pid"
    echo "[$(date)] ALARM r3 GATE VIOLATION: $verdict — paused pid $pid" \
        | tee "$L/ALARM_r3_reacher.log.txt" >> "$L/health_gate.log"
fi
