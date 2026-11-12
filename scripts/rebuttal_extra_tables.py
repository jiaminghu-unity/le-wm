#!/usr/bin/env python3
"""Rebuttal tables for the two reviewer-9CWa experiments (training seeds 1024 / 1025):
component-split SCALE (Push-T, Cube) and the L2-regularised Aux readout sweep (five tasks).
Per arm: overall success CEM / iCEM (mean over T1-T5 and the 6 episode seeds; `(n=k)` marks a
cell with only k of 6 episode seeds done) and held-out Spearman distance alignment
(full q; on Push-T / Cube also against the high / low subsets). Writes EXTRA.md."""
import csv, glob, collections, statistics as st, json, os
M = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results", "rebuttal_mirror")
EV = {"tworoom": "final_eval_tworoom", "pointmaze": "final_eval_pointmaze"}


def overall(task, label, sol):
    v = []
    for f in glob.glob(f"{M}/{EV.get(task, 'final_eval')}/final_{task}_{label}_{sol}_s*.csv"):
        t = collections.defaultdict(list)
        for r in csv.DictReader(open(f)):
            t[r["tier"]].append(int(r["success"]))
        if len(t) == 5 and all(len(x) >= 100 for x in t.values()):
            v.append(st.mean(100 * sum(x) / len(x) for x in t.values()))
    return (st.mean(v), len(v)) if v else (None, 0)


def fmt(x):
    return "..." if x[0] is None else f"{x[0]:.1f}" + ("" if x[1] == 6 else f" (n={x[1]})")


def align(task, run):
    f = f"{M}/diag/{task}__{run}.json"
    if not os.path.exists(f):
        return "..."
    d = json.load(open(f))["rho"]
    return " / ".join(f"{d[k]['spearman']:.2f}" for k in ("full_q", "high_q", "low_q") if k in d)


AUX = {"pusht": ("c5", "lewm_c5_qhead0.3"), "reacher": ("r5", "lewm_r5_qhead0.4"), "cube": ("k4", "lewm_k4_cube_qhead_eff0.1"),
       "tworoom": ("t5", "lewm_t5_tworoom_qhead0.1"), "pointmaze": ("p5", "lewm_p5_pointmaze")}
BASE = {"pusht": [("LeWM", "c1", "lewm_c1"), ("SCALE", "c3", "lewm_c3_sig_obj0.1")],
        "reacher": [("LeWM", "r1", "lewm_r1_reacher"), ("SCALE (0.15)", "r2", "lewm_r2_reacher_paep_l015")],
        "cube": [("LeWM", "k1", "lewm_k1_cube"), ("SCALE", "k2", "lewm_k2_cube_obj_eff0.1")],
        "tworoom": [("LeWM", "t1", "lewm_t1_tworoom"), ("SCALE", "t2", "lewm_t2_tworoom_obj0.1")],
        "pointmaze": [("LeWM", "p1", "lewm_p1_pointmaze"), ("SCALE", "p2", "lewm_p2_pointmaze")]}
SPLIT = {"pusht": [("SCALE, high-recov. q only", "c3rh", "lewm_rq_pusht_high"), ("SCALE, low-recov. q only", "c3rl", "lewm_rq_pusht_low")],
         "cube": [("SCALE, high-recov. q only", "k2rh", "lewm_rq_cube_high"), ("SCALE, low-recov. q only", "k2rl", "lewm_rq_cube_low")]}
TITLE = {"pusht": "Push-T", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room", "pointmaze": "PointMaze"}
out = ["# Rebuttal: component-split SCALE and L2-regularised Aux (training seeds 1024 / 1025)", "",
       "Success: CEM / iCEM overall %. Alignment: held-out Spearman (full q; on Push-T and Cube: full / high / low).", ""]
for task in ("pusht", "reacher", "cube", "tworoom", "pointmaze"):
    lab, run = AUX[task]
    rows = BASE[task] + SPLIT.get(task, []) + [("Aux", lab, run)]
    for b in ("0.01", "0.1", "1.0"):
        rows.append((f"Aux + L2 {b}", f"{lab}h{b}", None if task == "pointmaze" else f"{run}_hl2_{b}"))
    out += [f"## {TITLE[task]}", "", "| Arm | Seed 1024 | align | Seed 1025 | align |", "|---|---|---|---|---|"]
    for name, l, r in rows:
        cells = []
        for s in ("1024", "1025"):
            rr = r
            if rr is None:  # PointMaze: seed 1024 comes from the pilot runs
                b = name.split()[-1]
                rr = f"lewm_p5_pointmaze_hl2pilot_{b}" if s == "1024" else f"lewm_p5_pointmaze_hl2_{b}"
            cells += [f"{fmt(overall(task, l + 'r' + s, 'cem'))} / {fmt(overall(task, l + 'r' + s, 'icem'))}", align(task, f"{rr}_s{s}")]
        out.append("| " + " | ".join([name] + cells) + " |")
    out.append("")
text = "\n".join(out)
open(os.path.join(M, "EXTRA.md"), "w").write(text + "\n")
print(text)
