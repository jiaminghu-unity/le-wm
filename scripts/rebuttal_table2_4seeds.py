#!/usr/bin/env python3
"""The paper's Table 2 / Tables 13-17 recomputed over FOUR training runs (the two pooled in the paper
plus seeds 1024 and 1025) for LeWM, SCALE and Aux; DINO-WM keeps the paper's two runs.
Writes eval_results/rebuttal_mirror/TABLE2_4SEEDS.md (Markdown, then LaTeX).

Same statistic as the paper: each entry = mean +- SD over set-level measurements (one measurement
= success % on one 100-episode set for one trained run), i.e. 24 measurements for LeWM / SCALE / Aux
and 12 for DINO-WM; Overall averages T1-T5 within each measurement. Bold = best method per column
within each task. The script first re-derives the paper's two-run table and checks it against the
numbers printed in the paper (which also fixes the SD convention).
"""
import collections, csv, glob, os, re, statistics as st
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results")
ARMS = {"pusht": {"LeWM": "lewm_c1", "SCALE": "lewm_c3_sig_obj0.1", "Aux": "lewm_c5_qhead0.3", "DINO-WM": "dinowm_pusht"},
        "reacher": {"LeWM": "lewm_r1_reacher", "SCALE": "lewm_r2_reacher_paep_l015", "Aux": "lewm_r5_qhead0.4",
                    "DINO-WM": "dinowm_reacher"},
        "cube": {"LeWM": "lewm_k1_cube", "SCALE": "lewm_k2_cube_obj_eff0.1", "Aux": "lewm_k4_cube_qhead_eff0.1",
                 "DINO-WM": "dinowm_cube"},
        "tworoom": {"LeWM": "lewm_t1_tworoom", "SCALE": "lewm_t2_tworoom_obj0.1", "Aux": "lewm_t5_tworoom_qhead0.1",
                    "DINO-WM": "dinowm_tworoom"},
        "pointmaze": {"LeWM": "lewm_p1_pointmaze", "SCALE": "lewm_p2_pointmaze", "Aux": "lewm_p5_pointmaze",
                      "DINO-WM": "dinowm_pointmaze"}}
TITLE = {"pusht": "Push-T", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room", "pointmaze": "PointMaze"}
METHODS = ("LeWM", "SCALE", "Aux", "DINO-WM")
PAPER_SEEDS = {"pusht": ("3072", "3073"), "reacher": ("3072", "3073"), "cube": ("3072", "3073"),
               "tworoom": ("3072", "3073"), "pointmaze": ("3072", "3074")}
DINO_SEEDS = ("3072", "3073")
NEW_SEEDS = tuple(os.environ.get("NEW_SEEDS", "1024,1025").split(","))  # e.g. NEW_SEEDS=1025 for a three-run table
NRUN = 2 + len(NEW_SEEDS)
WORD = {3: "three", 4: "four"}[NRUN]
NEWTXT = {1: "one new run", 2: "two new runs"}[len(NEW_SEEDS)]
TAG = "4SEEDS" if NEW_SEEDS == ("1024", "1025") else "PAPER_PLUS_" + "_".join(NEW_SEEDS)
ALL_SEEDS = ("3072", "3073", "3074", "1024", "1025")
ES = ["101", "102", "103", "104", "105", "106"]
TIERS = ["T1", "T2", "T3", "T4", "T5"]
COLS = TIERS + ["Overall"]
BAD = ("l1", "cos", "cost", "FILTERED", "broken", "mppi", "frozen", "half", "automet")
PRIO = ("rebuttal_mirror", "sd_mirror/final_eval", "wsweep_mirror", "r73_mirror")
# a few cells printed in the paper (iCEM, Table 2) to validate the reconstruction: (mean, SD)
PAPER_CHECK = {("pusht", "icem", "LeWM", "T1"): (86.0, 2.4), ("pusht", "icem", "LeWM", "Overall"): (63.9, 3.0),
               ("pusht", "icem", "Aux", "T5"): (27.6, 5.2), ("reacher", "icem", "SCALE", "T1"): (86.7, 3.6),
               ("reacher", "icem", "SCALE", "Overall"): (81.4, 2.0), ("cube", "icem", "SCALE", "T3"): (75.2, 3.0),
               ("cube", "icem", "DINO-WM", "Overall"): (62.5, 1.3), ("tworoom", "icem", "LeWM", "T4"): (77.3, 2.2),
               ("tworoom", "icem", "DINO-WM", "T5"): (92.5, 2.1), ("pointmaze", "icem", "LeWM", "T1"): (84.0, 4.1),
               ("pointmaze", "icem", "Aux", "Overall"): (78.7, 4.3), ("pointmaze", "icem", "DINO-WM", "T3"): (90.8, 3.4),
               ("pusht", "cem", "SCALE", "T1"): (94.8, 3.2), ("cube", "cem", "LeWM", "Overall"): (60.9, 3.4),
               ("pointmaze", "cem", "SCALE", "Overall"): (84.9, 2.7)}
stems = {v: (t, a) for t, d in ARMS.items() for a, v in d.items()}

paths = collections.defaultdict(dict)
for f in glob.glob(os.path.join(R, "**", "final_*.csv"), recursive=True):
    d = os.path.relpath(os.path.dirname(f), R)
    if d == "." or any(b in d for b in BAD):
        continue
    m = re.match(r"final_(pusht|reacher|cube|tworoom|pointmaze)_(.+)_(cem|icem)_s(\d+)\.csv$", os.path.basename(f))
    if not m or m.group(4) not in ES:
        continue
    with open(f) as fh:
        first = next(csv.DictReader(fh), None)
    if not first:
        continue
    mm = re.match(r"(.+)_s(\d{4})/", first["checkpoint_path"])
    if not mm or mm.group(1) not in stems or stems[mm.group(1)][0] != m.group(1) or mm.group(2) not in ALL_SEEDS:
        continue
    paths[(m.group(1), stems[mm.group(1)][1], mm.group(2), m.group(3), m.group(4))][d] = f


def pick(cell):
    d = paths[cell]
    for p in PRIO:
        for k in sorted(d):
            if k.startswith(p):
                return d[k]
    return d[sorted(d)[0]]


_cache = {}


def setlevel(task, arm, seed, sol):
    """-> list of six dicts {tier or 'Overall': success %}, one per evaluation set"""
    key = (task, arm, seed, sol)
    if key not in _cache:
        out = []
        for es in ES:
            t = collections.defaultdict(list)
            for r in csv.DictReader(open(pick((task, arm, seed, sol, es)))):
                t[r["tier"]].append(int(r["success"]))
            assert sorted(t) == TIERS and all(len(v) == 100 for v in t.values()), key + (es,)
            m = {k: Fraction(100 * sum(v), len(v)) for k, v in t.items()}  # exact
            m["Overall"] = sum(m[k] for k in TIERS) / 5
            out.append(m)
        _cache[key] = out
    return _cache[key]


def cell(task, arm, seeds, sol, col):
    v = [m[col] for s in seeds for m in setlevel(task, arm, s, sol)]
    f = [float(x) for x in v]
    return sum(v) / len(v), st.stdev(f), st.pstdev(f), len(v)


# 1) the paper's two-run table reproduces?
bad = []
for (task, sol, arm, col), (pm, psd) in PAPER_CHECK.items():
    seeds = DINO_SEEDS if arm == "DINO-WM" else PAPER_SEEDS[task]
    m, sd1, sd0, n = cell(task, arm, seeds, sol, col)
    m = float(m)
    ok1 = abs(round(m, 1) - pm) < 0.051 and abs(round(sd1, 1) - psd) < 0.051
    ok0 = abs(round(m, 1) - pm) < 0.051 and abs(round(sd0, 1) - psd) < 0.051
    bad.append((task, sol, arm, col, pm, psd, round(m, 2), round(sd1, 2), round(sd0, 2), ok1, ok0))
n1, n0 = sum(b[-2] for b in bad), sum(b[-1] for b in bad)
print(f"paper check: {len(bad)} cells; match with sample SD: {n1}; with population SD: {n0}")
for b in bad:
    if not (b[-2] or b[-1]):
        print("  MISMATCH", b)
SAMPLE = n1 >= n0  # SD convention that reproduces the paper


def r1(fr):
    """one decimal, exact round-half-up"""
    return (Decimal(fr.numerator) / Decimal(fr.denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def fmt(m, sd):
    return f"{r1(m)}±{sd:.1f}"


def build(sol):
    rows = []
    for task in ARMS:
        seeds = PAPER_SEEDS[task] + NEW_SEEDS
        vals = {}
        for arm in METHODS:
            ss = DINO_SEEDS if arm == "DINO-WM" else seeds
            for col in COLS:
                m, sd1, sd0, n = cell(task, arm, ss, sol, col)
                vals[(arm, col)] = (m, sd1 if SAMPLE else sd0, n)
        best = {col: max(r1(vals[(a, col)][0]) for a in METHODS) for col in COLS}  # ties on the printed value are all bold
        for arm in METHODS:
            rows.append((task, arm, [(fmt(*vals[(arm, col)][:2]), r1(vals[(arm, col)][0]) == best[col]) for col in COLS],
                         vals[(arm, "Overall")][2]))
    return rows


out = []
for sol, sname in (("icem", "iCEM"), ("cem", "CEM")):
    rows = build(sol)
    out += [f"**Pooled {sname} success rates (%) across tasks and compute tiers, {WORD} training runs.** Each entry "
            f"reports mean ± SD over set-level measurements. LeWM, SCALE and Aux: {6 * NRUN} measurements ({WORD} independently "
            f"trained runs — the two runs of the original submission and {NEWTXT} — each evaluated on six held-out "
            "sets of 100 episodes). DINO-WM: 12 measurements (the two runs of the original submission). Overall "
            "averages T1–T5 within each measurement. Bold marks the best method per column within each task.", "",
            "| Task | Method | T1 | T2 | T3 | T4 | T5 | Overall |", "|---|---|---|---|---|---|---|---|"]
    for task, arm, cells, n in rows:
        out.append(f"| {TITLE[task]} | {arm} | " + " | ".join(f"**{c}**" if b else c for c, b in cells) + " |")
    out.append("")
out += ["LaTeX (same numbers):", "", "```latex"]
for sol, sname in (("icem", "iCEM"), ("cem", "CEM")):
    rows = build(sol)
    out += [f"% {sname}", r"\begin{tabular}{llcccccc}", r"\toprule",
            r"Task & Method & T1 & T2 & T3 & T4 & T5 & Overall \\", r"\midrule"]
    for i, (task, arm, cells, n) in enumerate(rows):
        lead = rf"\multirow{{4}}{{*}}{{{TITLE[task]}}}" if arm == "LeWM" else ""
        tex = [c.replace("±", r"$\pm$") for c, _ in cells]
        tex = [rf"\textbf{{{c}}}" if b else c for c, (_, b) in zip(tex, cells)]
        out.append(f"{lead} & {arm} & " + " & ".join(tex) + r" \\")
        if arm == "DINO-WM" and i < len(rows) - 1:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", ""]
out.append("```")
p = os.path.join(R, "rebuttal_mirror", f"TABLE2_{TAG}.md")
open(p, "w").write("\n".join(out) + "\n")
print("SD convention:", "sample (n-1)" if SAMPLE else "population (n)")
print("\n".join(out[: out.index("LaTeX (same numbers):")]))
