#!/usr/bin/env python3
"""Seed-level results and paired differences for the paper's three arms over ALL training seeds
(3072, 3073, 3074, 1024, 1025; Two-Room has no 3074), CEM and iCEM. Answers reviewer 9CWa W2/Q3.

Unit of analysis = one training seed. For each seed and arm, success is averaged over T1-T5 and
the six paired 100-episode sets; the difference between two arms is taken within the seed (same
episodes, same planner noise). Reported: per-seed differences, their mean, a 95% t-interval over
seeds, how many seeds are positive, and the two sources of spread kept apart (SD across the six
evaluation sets within a seed vs SD across training seeds). Writes POOLED.md.

File selection: canonical result directories only (cost-function variants, filtered-episode and
broken-render reruns are excluded); verified to reproduce the paper's pooled Tables 13-17.
"""
import collections, csv, glob, math, os, re, statistics as st

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results")
ARMS = {"pusht": {"LeWM": "lewm_c1", "SCALE": "lewm_c3_sig_obj0.1", "Aux": "lewm_c5_qhead0.3"},
        "reacher": {"LeWM": "lewm_r1_reacher", "SCALE": "lewm_r2_reacher_paep_l015", "Aux": "lewm_r5_qhead0.4"},
        "cube": {"LeWM": "lewm_k1_cube", "SCALE": "lewm_k2_cube_obj_eff0.1", "Aux": "lewm_k4_cube_qhead_eff0.1"},
        "tworoom": {"LeWM": "lewm_t1_tworoom", "SCALE": "lewm_t2_tworoom_obj0.1", "Aux": "lewm_t5_tworoom_qhead0.1"},
        "pointmaze": {"LeWM": "lewm_p1_pointmaze", "SCALE": "lewm_p2_pointmaze", "Aux": "lewm_p5_pointmaze"}}
TITLE = {"pusht": "Push-T", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room", "pointmaze": "PointMaze"}
SEEDS = ["3072", "3073", "3074", "1024", "1025"]
ES = ["101", "102", "103", "104", "105", "106"]
BAD = ("l1", "cos", "cost", "FILTERED", "broken", "mppi", "frozen", "half", "automet")
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
stems = {v: (t, a) for t, d in ARMS.items() for a, v in d.items()}

vals = collections.defaultdict(dict)  # cell -> {directory: value}
for f in glob.glob(os.path.join(R, "**", "final_*.csv"), recursive=True):
    d = os.path.relpath(os.path.dirname(f), R)
    if d == "." or any(b in d for b in BAD):
        continue
    m = re.match(r"final_(pusht|reacher|cube|tworoom|pointmaze)_(.+)_(cem|icem)_s(\d+)\.csv$", os.path.basename(f))
    if not m or m.group(4) not in ES:
        continue
    task, _, sol, es = m.groups()
    rows = list(csv.DictReader(open(f)))
    if not rows:
        continue
    mm = re.match(r"(.+)_s(\d{4})/", rows[0]["checkpoint_path"])
    if not mm or mm.group(1) not in stems or stems[mm.group(1)][0] != task:
        continue
    t = collections.defaultdict(list)
    for r in rows:
        t[r["tier"]].append(int(r["success"]))
    if len(t) != 5 or any(len(v) < 100 for v in t.values()):
        continue
    vals[(task, stems[mm.group(1)][1], mm.group(2), sol, es)][d] = round(st.mean(100 * sum(v) / len(v) for v in t.values()), 6)
conflicts = [k for k, v in vals.items() if len(set(v.values())) > 1]
PRIO = ("rebuttal_mirror", "sd_mirror/final_eval", "wsweep_mirror", "r73_mirror")  # GCS mirrors first


def pick(cell):
    """Deterministic choice when a cell exists in several directories: the GCS mirror wins."""
    d = vals[cell]
    for p in PRIO:
        for k in sorted(d):
            if k.startswith(p):
                return d[k]
    return d[sorted(d)[0]]


def seed_mean(task, arm, seed, sol):
    v = [pick((task, arm, seed, sol, e)) for e in ES if (task, arm, seed, sol, e) in vals]
    return (st.mean(v), st.stdev(v)) if len(v) == 6 else None


out = ["# Paper arms over all training seeds: seed-level results and paired differences", "",
       "Success % = mean over T1-T5 and six 100-episode sets, per training seed. Differences are within-seed",
       "(same episodes and planner noise). CI = 95% t-interval over training seeds. `n/a` = seed not trained.",
       f"Cells present in two versions that differ (all Cube seed 3072, <= 0.6 points per set; the GCS mirror is used): {len(conflicts)}.", ""]
for task in ARMS:
    out += [f"## {TITLE[task]}", ""]
    for sol in ("cem", "icem"):
        out += [f"**{sol.upper() if sol == 'cem' else 'iCEM'}**", "",
                "| | " + " | ".join(SEEDS) + " | mean | SD across seeds | mean SD across eval sets |", "|---|" + "---|" * (len(SEEDS) + 3)]
        sm = {a: {s: seed_mean(task, a, s, sol) for s in SEEDS} for a in ("LeWM", "SCALE", "Aux")}
        for a in ("LeWM", "SCALE", "Aux"):
            have = [sm[a][s] for s in SEEDS if sm[a][s]]
            cells = [f"{sm[a][s][0]:.1f}" if sm[a][s] else "n/a" for s in SEEDS]
            out.append(f"| {a} | " + " | ".join(cells) + f" | {st.mean(x[0] for x in have):.1f} | {st.stdev([x[0] for x in have]):.1f} | {st.mean(x[1] for x in have):.1f} |")
        for a, b in (("SCALE", "LeWM"), ("Aux", "LeWM"), ("SCALE", "Aux")):
            ds = {s: sm[a][s][0] - sm[b][s][0] for s in SEEDS if sm[a][s] and sm[b][s]}
            v = list(ds.values()); n = len(v); mean = st.mean(v); half = T95[n] * st.stdev(v) / math.sqrt(n)
            cells = [f"{ds[s]:+.1f}" if s in ds else "n/a" for s in SEEDS]
            out.append(f"| {a} − {b} | " + " | ".join(cells) + f" | {mean:+.1f} [{mean - half:+.1f}, {mean + half:+.1f}] | {sum(x > 0 for x in v)}/{n} seeds > 0 | |")
        out.append("")
text = "\n".join(out)
open(os.path.join(R, "rebuttal_mirror", "POOLED.md"), "w").write(text + "\n")
print(text)
