#!/usr/bin/env python3
"""State-based world model vs the pixel arms (reviewer 3Tuk, W6). Writes STATE.md.

State model = the q-input LeWM (qjepa.QJEPA): the image encoder is replaced by an MLP over the
simulator state, everything else (predictor, SIGReg, latent planning) is LeWM. One variant per
task, the COMPLETE state: Push-T 8-d, Reacher 8-d, Cube 22-d, Two-Room 2-d, PointMaze 4-d.
Training seeds 3072 / 1024 / 1025; success = mean over T1-T5 and six 100-episode sets; the
difference to a pixel arm is taken within the training seed (same episode sets).
Cells are identified by the checkpoint path stored in each CSV, not by the file label.
"""
import collections, csv, glob, math, os, re, statistics as st

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results")
ARMS = {"pusht": {"LeWM": "lewm_c1", "SCALE": "lewm_c3_sig_obj0.1", "Aux": "lewm_c5_qhead0.3",
                  "State": "lewm_qinput_full_pusht"},
        "reacher": {"LeWM": "lewm_r1_reacher", "SCALE": "lewm_r2_reacher_paep_l015", "Aux": "lewm_r5_qhead0.4",
                    "State": "lewm_qinput_native_reacher"},
        "cube": {"LeWM": "lewm_k1_cube", "SCALE": "lewm_k2_cube_obj_eff0.1", "Aux": "lewm_k4_cube_qhead_eff0.1",
                 "State": "lewm_qinput_full_cube"},
        "tworoom": {"LeWM": "lewm_t1_tworoom", "SCALE": "lewm_t2_tworoom_obj0.1", "Aux": "lewm_t5_tworoom_qhead0.1",
                    "State": "lewm_qinput_tworoom"},
        "pointmaze": {"LeWM": "lewm_p1_pointmaze", "SCALE": "lewm_p2_pointmaze", "Aux": "lewm_p5_pointmaze",
                      "State": "lewm_qinput_pointmaze"}}
STATE_DESC = {"pusht": "8-d: agent and block position, block angle (cos/sin), agent velocity",
              "reacher": "8-d: joint angles (cos/sin), fingertip position, joint velocities",
              "cube": "22-d: effector pose, gripper, arm joints, block position and yaw",
              "tworoom": "2-d: agent position (the only moving quantity)",
              "pointmaze": "4-d: position and velocity"}
TITLE = {"pusht": "Push-T", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room", "pointmaze": "PointMaze"}
SEEDS = ["3072", "1024", "1025"]
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
    if not m or m.group(4) not in ES or m.group(2) == "qchk":
        continue
    task, _, sol, es = m.groups()
    rows = list(csv.DictReader(open(f)))
    if not rows:
        continue
    mm = re.match(r"(.+)_s(\d{4})/", rows[0]["checkpoint_path"])
    if not mm or mm.group(1) not in stems or stems[mm.group(1)][0] != task or mm.group(2) not in SEEDS:
        continue
    t = collections.defaultdict(list)
    for r in rows:
        t[r["tier"]].append(int(r["success"]))
    if len(t) != 5 or any(len(v) < 100 for v in t.values()):
        continue
    vals[(task, stems[mm.group(1)][1], mm.group(2), sol, es)][d] = round(st.mean(100 * sum(v) / len(v) for v in t.values()), 6)
PRIO = ("rebuttal_mirror", "sd_mirror/final_eval", "wsweep_mirror", "r73_mirror")  # GCS mirrors first


def pick(cell):
    d = vals[cell]
    for p in PRIO:
        for k in sorted(d):
            if k.startswith(p):
                return d[k]
    return d[sorted(d)[0]]


def seed_mean(task, arm, seed, sol):
    v = [pick((task, arm, seed, sol, e)) for e in ES if (task, arm, seed, sol, e) in vals]
    return st.mean(v) if len(v) == 6 else None


def fmt(x, signed=False):
    return "..." if x is None else (f"{x:+.1f}" if signed else f"{x:.1f}")


def ci(d):
    n = len(d)
    if n < 2:
        return ""
    h = T95[n] * st.stdev(d) / math.sqrt(n)
    return f" [{st.mean(d) - h:+.1f}, {st.mean(d) + h:+.1f}]"


done = sum(1 for t in ARMS for s in SEEDS for sol in ("cem", "icem") if seed_mean(t, "State", s, sol) is not None)
out = ["# State-based world model vs pixel arms (training seeds 3072 / 1024 / 1025)", "",
       "State model = LeWM with the image encoder replaced by an MLP over the complete simulator state;",
       "predictor, regulariser and latent-space planning are unchanged. Success % = mean over T1-T5 and",
       "six 100-episode sets. Each cell: CEM / iCEM. Differences are within the training seed; the last",
       "column is their mean with a 95% t-interval over the three seeds. `...` = not finished yet.",
       f"State-model cells complete: {done} / {len(ARMS) * len(SEEDS) * 2}.", ""]
for task in ARMS:
    out += [f"## {TITLE[task]}", "", f"State fed to the model: {STATE_DESC[task]}.", "",
            "| | " + " | ".join(SEEDS) + " | mean |", "|---|" + "---|" * (len(SEEDS) + 1)]
    sm = {a: {s: {sol: seed_mean(task, a, s, sol) for sol in ("cem", "icem")} for s in SEEDS} for a in ARMS[task]}
    for a in ("LeWM", "SCALE", "Aux", "State"):
        cells = [f"{fmt(sm[a][s]['cem'])} / {fmt(sm[a][s]['icem'])}" for s in SEEDS]
        means = []
        for sol in ("cem", "icem"):
            have = [sm[a][s][sol] for s in SEEDS if sm[a][s][sol] is not None]
            means.append(fmt(st.mean(have)) if len(have) == len(SEEDS) else "...")
        out.append(f"| {a} | " + " | ".join(cells) + f" | {means[0]} / {means[1]} |")
    for ref in ("LeWM", "SCALE"):
        cells, tails = [], []
        for s in SEEDS:
            d = [None if sm["State"][s][sol] is None or sm[ref][s][sol] is None else sm["State"][s][sol] - sm[ref][s][sol]
                 for sol in ("cem", "icem")]
            cells.append(f"{fmt(d[0], True)} / {fmt(d[1], True)}")
        for sol in ("cem", "icem"):
            d = [sm["State"][s][sol] - sm[ref][s][sol] for s in SEEDS
                 if sm["State"][s][sol] is not None and sm[ref][s][sol] is not None]
            tails.append(f"{st.mean(d):+.1f}{ci(d)}" if len(d) == len(SEEDS) else "...")
        out.append(f"| State − {ref} | " + " | ".join(cells) + f" | {tails[0]} / {tails[1]} |")
    out.append("")
p = os.path.join(R, "rebuttal_mirror", "STATE.md")
open(p, "w").write("\n".join(out))
print(f"State-model cells complete: {done} / {len(ARMS) * len(SEEDS) * 2}.  -> {p}")
