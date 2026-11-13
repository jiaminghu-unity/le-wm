#!/usr/bin/env python3
"""Eval-path regression check: the current launchers + state-key sweeper re-run one file
(CEM, episode set 101) of each existing seed-3072 state model under label `qchk`; this compares
it tier by tier with the August result. Read-only apart from a scratch download dir."""
import csv, collections, os, subprocess, sys, tempfile
B = "gs://prism-training-us/le-wm"
PAIRS = {"pusht": ("final_eval", "q1"), "reacher": ("final_eval", "q1"), "cube": ("final_eval", "q1f"),
         "tworoom": ("final_eval_tworoom", "q1"), "pointmaze": ("final_eval_pointmaze", "q1")}
d = tempfile.mkdtemp(prefix="qchk_")


def load(task, sub, label):
    name = f"final_{task}_{label}_cem_s101.csv"
    r = subprocess.run(["gcloud", "storage", "cp", f"{B}/{sub}/{name}", d + "/"], capture_output=True, text=True)
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    t = collections.defaultdict(list)
    for x in rows:
        t[x["tier"]].append(int(x["success"]))
    same = None
    return {k: 100 * sum(v) / len(v) for k, v in sorted(t.items())}, rows


for task, (sub, old) in PAIRS.items():
    a, b = load(task, sub, old), load(task, sub, "qchk")
    if b is None:
        print(f"{task:10s} check not finished yet")
        continue
    ta, ra = a
    tb, rb = b
    agree = sum(1 for x, y in zip(ra, rb) if x["tier"] == y["tier"] and x["episode_id"] == y["episode_id"] and x["success"] == y["success"])
    print(f"{task:10s} old  " + "  ".join(f"{k} {v:5.1f}" for k, v in ta.items()) + f"   mean {sum(ta.values()) / len(ta):.1f}")
    print(f"{'':10s} new  " + "  ".join(f"{k} {v:5.1f}" for k, v in tb.items()) + f"   mean {sum(tb.values()) / len(tb):.1f}"
          f"   | identical episode outcomes: {agree}/{min(len(ra), len(rb))} (rows {len(ra)} vs {len(rb)})")
