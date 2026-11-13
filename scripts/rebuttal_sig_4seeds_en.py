#!/usr/bin/env python3
"""English rebuttal tables: LeWM / SCALE / Aux over FOUR training runs (the two pooled in the paper
plus the two new seeds 1024 and 1025), with DINO-WM (only the paper's two runs exist) for reference.
Writes eval_results/rebuttal_mirror/SIG_4SEEDS_EN.md.

Paper seed pairs: 3072+3073 (PointMaze LeWM-family: 3072+3074; DINO-WM: 3072+3073 on every task).
Table L (wide, one per planner): success by budget tier T1-T5 and their mean.
Tables M / N / O: SCALE vs LeWM / Aux / DINO-WM, tier-averaged success, difference, 95% CI, p-value.
Unit = test episode (600 = 6 sets x 100, same episodes and planner noise for every model);
per-episode difference = difference of the two methods' success averaged over their training runs
and the five tiers; 95% t-interval over episodes; exact two-sided sign-flip permutation test.
"Runs with delta > 0" counts the training seeds at which the within-seed difference is positive.
File selection as in rebuttal_pooled_stats.py (canonical directories, GCS mirrors first). Pure stdlib.
"""
import collections, csv, glob, math, os, re, statistics as st
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
PAPER_SEEDS = {"pusht": ("3072", "3073"), "reacher": ("3072", "3073"), "cube": ("3072", "3073"),
               "tworoom": ("3072", "3073"), "pointmaze": ("3072", "3074")}
DINO_SEEDS = ("3072", "3073")
NEW_SEEDS = tuple(os.environ.get("NEW_SEEDS", "1024,1025").split(","))  # e.g. NEW_SEEDS=1025 for three runs
NRUN = 2 + len(NEW_SEEDS)
WORD = {3: "three", 4: "four"}[NRUN]
NEWTXT = {("1024", "1025"): "two new runs (seeds 1024 and 1025)", ("1025",): "one new run (seed 1025)", ("1024",): "one new run (seed 1024)"}[NEW_SEEDS]
TAG = "4SEEDS" if NEW_SEEDS == ("1024", "1025") else "PAPER_PLUS_" + "_".join(NEW_SEEDS)
# Overall column of the paper's Tables 13-17 for DINO-WM (CEM, iCEM), to confirm the seed pair
PAPER_DINO = {"pusht": (56.5, 51.7), "reacher": (67.0, 73.3), "cube": (60.9, 62.5), "tworoom": (95.6, 98.2),
              "pointmaze": (94.7, 90.4)}
ALL_SEEDS = ("3072", "3073", "3074", "1024", "1025")
ES = ["101", "102", "103", "104", "105", "106"]
TIERS = ["T1", "T2", "T3", "T4", "T5"]
SOLS = (("cem", "CEM"), ("icem", "iCEM"))
BAD = ("l1", "cos", "cost", "FILTERED", "broken", "mppi", "frozen", "half", "automet")
PRIO = ("rebuttal_mirror", "sd_mirror/final_eval", "wsweep_mirror", "r73_mirror")
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


def load(task, arm, seed, sol):
    """-> ({(es, episode_id, tier): 0/1}, pairing keys)"""
    out, keys = {}, []
    for es in ES:
        rows = list(csv.DictReader(open(pick((task, arm, seed, sol, es)))))
        for r in rows:
            out[(es, r["episode_id"], r["tier"])] = int(r["success"])
            keys.append((es, r["tier"], r["episode_id"], r["traj_id"], r["start_idx"], r["cem_seed"]))
    assert len(out) == 3000 and len(keys) == 3000, (task, arm, seed, sol, len(out))
    return out, sorted(keys)


def perm_p(ints):
    obs = abs(sum(ints))
    dist = {0: 1.0}
    for a in ints:
        a = abs(a)
        if a == 0:
            continue
        nd = collections.defaultdict(float)
        for s, p in dist.items():
            nd[s + a] += p / 2
            nd[s - a] += p / 2
        dist = nd
    return min(1.0, sum(p for s, p in dist.items() if abs(s) >= obs - 1e-9))


def fp(p):
    return "<0.0001" if p < 1e-4 else (f"{p:.4f}" if p < 0.01 else f"{p:.3f}")


def pct(vals):
    vals = list(vals)
    return Fraction(100 * sum(vals), len(vals))


def r1(fr, signed=False):
    fr = Fraction(fr).limit_denominator(10 ** 9)
    d = (Decimal(fr.numerator) / Decimal(fr.denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return (f"{d:+}" if signed else f"{d}").replace("-", "−")


def sg(x):
    return f"{x:+.1f}".replace("-", "−")


def t3_p(t):  # two-sided p, Student t with 3 degrees of freedom
    th = math.atan(abs(t) / math.sqrt(3))
    return max(0.0, 1 - (2 / math.pi) * (th + math.sin(th) * math.cos(th)))


tier, avg, comp, seedlvl, dino_check = {}, {}, {}, {}, []
for task in ARMS:
    seeds = PAPER_SEEDS[task] + NEW_SEEDS
    for sol, sname in SOLS:
        data, ref = {}, None
        for arm, ss in (("LeWM", seeds), ("SCALE", seeds), ("Aux", seeds), ("DINO-WM", DINO_SEEDS)):
            for s in ss:
                data[(arm, s)], k = load(task, arm, s, sol)
                assert ref is None or k == ref, (task, sol, arm, s)  # identical episodes / tiers / planner seeds
                ref = k
        keys = sorted(data[("LeWM", seeds[0])])
        runs = {"LeWM": seeds, "SCALE": seeds, "Aux": seeds, "DINO-WM": DINO_SEEDS}
        for arm, ss in runs.items():
            for t in TIERS:
                tier[(task, sname, arm, t)] = pct(data[(arm, s)][k] for s in ss for k in keys if k[2] == t)
            avg[(task, sname, arm)] = pct(data[(arm, s)][k] for s in ss for k in keys)
        dino_check.append((task, sname, float(avg[(task, sname, "DINO-WM")]), PAPER_DINO[task][0 if sol == "cem" else 1]))
        ident = {(k[0], k[2]): (k[3], k[4]) for k in ref}  # (es, episode_id) -> unique start/goal
        eps = sorted({(k[0], k[1]) for k in keys})
        for other in ("LeWM", "Aux", "DINO-WM"):
            na, nb = len(runs["SCALE"]), len(runs[other])
            cl = collections.defaultdict(list)
            for e in eps:
                sa = sum(data[("SCALE", s)][(e[0], e[1], t)] for s in runs["SCALE"] for t in TIERS)
                sb = sum(data[(other, s)][(e[0], e[1], t)] for s in runs[other] for t in TIERS)
                cl[ident[e]].append(Fraction(sa, 5 * na) - Fraction(sb, 5 * nb))
            d = [float(100 * sum(v) / len(v)) for v in cl.values()]
            unit = 2 * 5 * na * nb  # common denominator incl. a duplicated episode
            ints = [int(sum(v) / len(v) * unit) for v in cl.values()]
            assert all(Fraction(i, unit) == sum(v) / len(v) for i, v in zip(ints, cl.values()))
            md, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
            pos = None
            if other != "DINO-WM":
                ds = [float(pct(data[("SCALE", s)][k] for k in keys) - pct(data[(other, s)][k] for k in keys)) for s in seeds]
                pos = sum(x > 0 for x in ds)
                ms, ses = st.mean(ds), st.stdev(ds) / math.sqrt(len(ds))
                tc = {3: 4.303, 4: 3.182}[len(ds)]
                seedlvl[(task, sname, other)] = (ds, ms - tc * ses, ms + tc * ses, t3_p(ms / ses) if len(ds) == 4 else float("nan"))
            comp[(task, sname, other)] = (avg[(task, sname, "SCALE")] - avg[(task, sname, other)],
                                          md - 1.9639 * se, md + 1.9639 * se, perm_p(ints), pos)

runs_txt = (f"LeWM, SCALE and Aux average {WORD} independently trained runs per task: the two runs pooled in the paper "
            f"and {NEWTXT}. DINO-WM averages the two runs from the paper.")
pair_txt = ("All models are evaluated on the same 600 test episodes (6 sets × 100) with identical planner noise, so "
            "outcomes are paired per episode. The per-episode difference is averaged over training runs and the five "
            "budget tiers; the 95% CI is a paired t-interval over episodes and p is from an exact two-sided paired "
            "sign-flip permutation test.")
out = ["**Table L. Planning success rate (%) by planning-budget tier.** " + runs_txt + " Each entry is the mean over "
       "600 test episodes (6 sets × 100) per run. Avg. is the mean over the five tiers.", ""]
for _, sname in SOLS:
    out += [f"**{sname} planner**", "", "| Task | Method | T1 | T2 | T3 | T4 | T5 | Avg. |", "|---|---|---|---|---|---|---|---|"]
    for task in ARMS:
        for arm in ("LeWM", "SCALE", "Aux", "DINO-WM"):
            out.append(f"| {TITLE[task]} | {arm} | " + " | ".join(r1(tier[(task, sname, arm, t)]) for t in TIERS)
                       + f" | {r1(avg[(task, sname, arm)])} |")
    out.append("")
for letter, other, extra in (("M", "LeWM", f"{WORD} runs per method"), ("N", "Aux", f"{WORD} runs per method"),
                             ("O", "DINO-WM", f"SCALE: {WORD} runs; DINO-WM: the two runs from the paper")):
    cols = f"| Task | Planner | {other} | SCALE | Δ | 95% CI | p-value |" + (" Runs with Δ > 0 |" if other != "DINO-WM" else "")
    out += [f"**Table {letter}. SCALE vs. {other}: success rate (%) averaged over the five budget tiers, with paired "
            f"significance tests ({extra}).** Δ = SCALE − {other} in percentage points. " + pair_txt
            + (" The last column counts the training seeds at which SCALE is ahead." if other != "DINO-WM" else ""), "",
            cols, "|---" * (cols.count("|") - 1) + "|"]
    for task in ARMS:
        for _, sname in SOLS:
            d, lo, hi, p, pos = comp[(task, sname, other)]
            row = (f"| {TITLE[task]} | {sname} | {r1(avg[(task, sname, other)])} | {r1(avg[(task, sname, 'SCALE')])} | "
                   f"{r1(d, True)} | [{sg(lo)}, {sg(hi)}] | {fp(p)} |")
            out.append(row + (f" {pos}/{NRUN} |" if pos is not None else ""))
    out.append("")
p = os.path.join(R, "rebuttal_mirror", f"SIG_{TAG}_EN.md")
open(p, "w").write("\n".join(out))
print("\n".join(out))
print("DINO-WM seed pair 3072+3073 vs the paper's Overall:", [(t, s, round(a, 2), b) for t, s, a, b in dino_check])
print(f"seed-level (n = {NRUN}) differences, 95% t-interval and paired t-test p:")
for (task, sname, other), (ds, lo, hi, pv) in seedlvl.items():
    print(f"  SCALE − {other:5s} {TITLE[task]:9s} {sname:4s} " + " ".join(f"{x:+.1f}" for x in ds)
          + f" | mean {st.mean(ds):+.1f} [{lo:+.1f}, {hi:+.1f}] p={fp(pv) if pv == pv else 'n/a'}")
