#!/usr/bin/env python3
"""Tabulate eval_results/rebuttal_mirror/diag/*.json (rebuttal_diag.py output) -> DIAG.md.

Per task and arm, for training seeds 1024 / 1025: Spearman between squared latent and squared
task-state distances (paper Table 12), mean linear-probe R^2 over the components of q (paper
Tables 6-10), and the share of variance in the top 8 principal directions (paper Fig. 2).
Push-T and Cube also get per-component R^2 and alignment against the high / low subsets of q.
"""
import glob, json, os, re, collections

M = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results", "rebuttal_mirror")
NAMES = [  # (regex on run name without seed, display)
    (r"^lewm_(c1|r1_reacher|k1_cube|t1_tworoom|p1_pointmaze)$", "LeWM"),
    (r"^lewm_(c3_sig_obj0\.1|k2_cube_obj_eff0\.1|t2_tworoom_obj0\.1|p2_pointmaze)$", "SCALE"),
    (r"^lewm_r2_reacher_paep_l015$", "SCALE (0.15)"), (r"^lewm_r2_reacher_paep_l01$", "SCALE (0.1)"),
    (r"^lewm_(c5_qhead0\.3|r5_qhead0\.4|k4_cube_qhead_eff0\.1|t5_tworoom_qhead0\.1|p5_pointmaze)$", "Aux"),
    (r"_hl2(pilot)?_([0-9.]+)$", "Aux + L2 {2}"),
    (r"^lewm_rq_\w+_high$", "SCALE, high-recov. q only"), (r"^lewm_rq_\w+_low$", "SCALE, low-recov. q only"),
    (r"^lewm_lpwmr_\w+_base$", "LpWM"), (r"^lewm_lpwmr_\w+_scale$", "LpWM+SCALE"),
]
ORDER = ["LeWM", "SCALE", "SCALE (0.15)", "SCALE (0.1)", "SCALE, high-recov. q only", "SCALE, low-recov. q only", "Aux"]
TASKS = [("pusht", "Push-T"), ("reacher", "Reacher"), ("cube", "Cube"), ("tworoom", "Two-Room"), ("pointmaze", "PointMaze")]


def arm_of(run):
    stem = re.sub(r"_s\d{4}$", "", run)
    if float((re.search(r"_hl2(?:pilot)?_([0-9.]+)$", stem) or [0, "1"])[1]) == 0.0 and "_hl2" in stem:
        return "Aux (pilot replicate)"
    for rx, name in NAMES:
        m = re.search(rx, stem)
        if m:
            return name.format(*([None] + list(m.groups())))
    return stem


data = collections.defaultdict(dict)  # (task, arm) -> seed -> json
for f in glob.glob(os.path.join(M, "diag", "*.json")):
    d = json.load(open(f))
    seed = (re.search(r"_s(\d{4})$", d["run"]) or [0, "?"])[1]
    data[(d["task"], arm_of(d["run"]))][seed] = d


def key(a):
    if a in ORDER:
        return (0, ORDER.index(a), 0.0)
    m = re.match(r"Aux \+ L2 ([0-9.]+)", a)
    if m:
        return (1, 0, float(m.group(1)))
    return (2, 0, 0.0) if a.startswith("LpWM") and "+" not in a else (3, 0, 0.0) if a.startswith("LpWM") else (4, 0, 0.0)


def cell(d, f):
    return "..." if d is None else f(d)


out = ["# Rebuttal diagnostics (held-out, paper protocol)", "",
       "Each cell: training seed 1024 / 1025; seed 3072 where shown is the paper's checkpoint. `...` = not available yet.", ""]
for task, title in TASKS:
    arms = sorted({a for (t, a) in data if t == task}, key=key)
    if not arms:
        continue
    seeds = ["1024", "1025"]
    out += [f"## {title}", "", "| Arm | Distance alignment (Spearman) | Mean probe R^2 | Top-8 variance share |", "|---|---|---|---|"]
    for a in arms:
        ds = [data[(task, a)].get(s) for s in seeds]
        extra = data[(task, a)].get("3072")
        row = [a + (" (+ seed 3072: %.2f)" % extra["rho"]["full_q"]["spearman"] if extra else "")]
        row.append(" / ".join(cell(d, lambda d: f"{d['rho']['full_q']['spearman']:.2f}") for d in ds))
        row.append(" / ".join(cell(d, lambda d: f"{d['full_r2_mean']:.3f}") for d in ds))
        row.append(" / ".join(cell(d, lambda d: f"{d['spectrum_top']['8']:.2f}") for d in ds))
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    if task in ("pusht", "cube"):
        qd = next(iter(data[(task, arms[0])].values()))["qdims"]
        out += [f"Per-component probe R^2 ({title}, seed 1024 / 1025) and alignment against the high / low subsets:", "",
                "| Arm | " + " | ".join(qd) + " | align high | align low |", "|---|" + "---|" * (len(qd) + 2)]
        for a in arms:
            ds = [data[(task, a)].get(s) for s in seeds]
            cols = [" / ".join(cell(d, lambda d, i=i: f"{d['full_r2'][i]:.2f}") for d in ds) for i in range(len(qd))]
            cols += [" / ".join(cell(d, lambda d, g=g: f"{d['rho'][g]['spearman']:.2f}") for d in ds) for g in ("high_q", "low_q")]
            out.append("| " + a + " | " + " | ".join(cols) + " |")
        out.append("")
text = "\n".join(out)
open(os.path.join(M, "DIAG.md"), "w").write(text + "\n")
print(text)
