#!/usr/bin/env python3
"""Summarise the rebuttal round (training seeds 1024/1025) from eval_results/rebuttal_mirror.

Overall success = mean over tiers T1-T5 of the per-tier success rate, averaged over the episode
seeds (101-106) that are complete (500 rows = 5 tiers x 100 episodes). A cell is reported only
with all 6 episode seeds unless --partial is given. Writes SUMMARY.md next to the CSVs.
"""
import collections
import csv
import glob
import os
import re
import statistics as st
import sys

M = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results", "rebuttal_mirror")
PARTIAL = "--partial" in sys.argv
ARMS = {  # task -> ordered (label, display name, baseline label)
    "pusht": [("c1", "LeWM", None), ("c3", "SCALE", "c1"), ("c5", "Aux", "c1"), ("lpb", "LpWM", None), ("lps", "LpWM+SCALE", "lpb")],
    "reacher": [("r1", "LeWM", None), ("r2", "SCALE (0.15)", "r1"), ("r2_l01", "SCALE (0.1)", "r1"), ("r5", "Aux", "r1"),
                ("lpb", "LpWM", None), ("lps", "LpWM+SCALE", "lpb")],
    "cube": [("k1", "LeWM", None), ("k2", "SCALE", "k1"), ("k4", "Aux", "k1"), ("lpb", "LpWM", None), ("lps", "LpWM+SCALE", "lpb")],
    "tworoom": [("t1", "LeWM", None), ("t2", "SCALE", "t1"), ("t5", "Aux", "t1"), ("lpb", "LpWM", None), ("lps", "LpWM+SCALE", "lpb")],
    "pointmaze": [("p1", "LeWM", None), ("p2", "SCALE", "p1"), ("p5", "Aux", "p1"), ("lpb", "LpWM", None), ("lps", "LpWM+SCALE", "lpb")],
}
NAMES = {"pusht": "Push-T", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room", "pointmaze": "PointMaze"}

acc = collections.defaultdict(dict)  # (task, arm, seed, solver) -> {episode seed: overall %}
for f in glob.glob(os.path.join(M, "*", "final_*.csv")):
    m = re.match(r"final_(pusht|reacher|cube|tworoom|pointmaze)_(.+)r(102[45])_(cem|icem)_s(\d+)\.csv", os.path.basename(f))
    if not m:
        continue
    task, arm, seed, sol, es = m.groups()
    tiers = collections.defaultdict(list)
    for r in csv.DictReader(open(f)):
        tiers[r["tier"]].append(int(r["success"]))
    if len(tiers) != 5 or any(len(v) < 100 for v in tiers.values()):
        continue
    acc[(task, arm, seed, sol)][es] = st.mean(100 * sum(v) / len(v) for v in tiers.values())


def cell(task, arm, seed, sol):
    v = acc.get((task, arm, seed, sol), {})
    if len(v) == 6 or (PARTIAL and v):
        return st.mean(v.values()), len(v)
    return None, len(v)


out = ["# Rebuttal round: training seeds 1024 / 1025", "",
       "Overall success (%), mean over T1-T5 and 6 episode seeds x 100 episodes. Each cell: seed 1024 / seed 1025.",
       "Delta = arm minus its baseline (LeWM for SCALE and Aux, LpWM for LpWM+SCALE), same seed and solver.",
       "`...` = not finished yet.", ""]
done = total = 0
for task, arms in ARMS.items():
    out += [f"## {NAMES[task]}", "", "| Arm | CEM | iCEM | Delta CEM | Delta iCEM |", "|---|---|---|---|---|"]
    for arm, name, base in arms:
        row = [name]
        vals = {}
        for sol in ("cem", "icem"):
            parts = []
            for seed in ("1024", "1025"):
                v, n = cell(task, arm, seed, sol)
                total += 1
                done += v is not None
                vals[(sol, seed)] = v
                parts.append("..." if v is None else f"{v:.1f}")
            row.append(" / ".join(parts))
        for sol in ("cem", "icem"):
            parts = []
            for seed in ("1024", "1025"):
                b = cell(task, base, seed, sol)[0] if base else None
                v = vals[(sol, seed)]
                parts.append("" if base is None else ("..." if v is None or b is None else f"{v - b:+.1f}"))
            row.append(" / ".join(parts) if base else "")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
out.insert(5, f"Cells complete: {done} / {total}.")
text = "\n".join(out)
open(os.path.join(M, "SUMMARY.md"), "w").write(text + "\n")
print(text)
