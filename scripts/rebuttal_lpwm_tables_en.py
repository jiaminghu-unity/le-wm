#!/usr/bin/env python3
"""English rebuttal tables for LpWM vs LpWM+SCALE. Writes eval_results/rebuttal_mirror/LPWM_TABLES_EN.md.

Table A: success rate per task / budget tier / planner for the two methods.
Table B: per task / planner, success averaged over the five tiers, difference, 95% CI and p-value.

Seeds: a training seed is used for a task only if both arms trained normally at that seed, so
Push-T uses seed 1024 only (LpWM+SCALE seed 1025 collapsed); the other tasks average 1024 and 1025.
Statistics (Table B): unit = test episode (600 = 6 sets x 100, same episodes and planner noise for
both models); per-episode difference = mean over the used seeds and the five tiers of (success
with SCALE - success without); 95% t-interval over episodes; exact two-sided sign-flip test.
Pure standard library.
"""
import collections, csv, math, os, statistics as st
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results", "rebuttal_mirror")
D = {"pusht": "final_eval", "reacher": "final_eval", "cube": "final_eval",
     "tworoom": "final_eval_tworoom", "pointmaze": "final_eval_pointmaze"}
TITLE = {"pusht": "Push-T", "reacher": "Reacher", "cube": "Cube", "tworoom": "Two-Room", "pointmaze": "PointMaze"}
SOLS = (("cem", "CEM"), ("icem", "iCEM"))
ALL_SEEDS = ("1024", "1025")
EXCLUDE = {("pusht", "1025")}
ES = [str(e) for e in range(101, 107)]
TIERS = ["T1", "T2", "T3", "T4", "T5"]


def load(task, lab, seed, sol):
    out, keys = {}, []
    for es in ES:
        for r in csv.DictReader(open(f"{R}/{D[task]}/final_{task}_{lab}r{seed}_{sol}_s{es}.csv")):
            out[(es, r["traj_id"], r["start_idx"], r["tier"])] = int(r["success"])
            keys.append((es, r["tier"], r["episode_id"], r["traj_id"], r["start_idx"], r["cem_seed"]))
    assert len(keys) == 3000 and len(out) == 3000, (task, lab, seed, sol)
    return out, keys


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


def sg(x):
    return f"{x:+.1f}".replace("-", "−")


def pct(vals):
    """exact success percentage of a list of 0/1 outcomes"""
    vals = list(vals)
    return Fraction(100 * sum(vals), len(vals))


def r1(fr, signed=False):
    """one decimal, exact round-half-up (no binary-float ties)"""
    d = (Decimal(fr.numerator) / Decimal(fr.denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return (f"{d:+}" if signed else f"{d}").replace("-", "−")


tier, summ = {}, {}
for task in D:
    seeds = [s for s in ALL_SEEDS if (task, s) not in EXCLUDE]
    for sol, sname in SOLS:
        base, scale, ref = {}, {}, None
        for s in seeds:
            base[s], kb = load(task, "lpb", s, sol)
            scale[s], ks = load(task, "lps", s, sol)
            assert kb == ks and (ref is None or kb == ref), (task, sol, s)
            ref = kb
        keys = list(base[seeds[0]])
        for t in TIERS:
            kt = [k for k in keys if k[3] == t]
            tier[(task, t, sname)] = (pct(base[s][k] for s in seeds for k in kt),
                                      pct(scale[s][k] for s in seeds for k in kt))
        cl = collections.defaultdict(list)  # unique test episode -> tier-sum differences, one per (occurrence, seed)
        occ = collections.defaultdict(lambda: collections.defaultdict(int))
        for k in keys:
            for s in seeds:
                occ[(k[1], k[2])][(k[0], s)] += scale[s][k] - base[s][k]
        for e, v in occ.items():
            cl[e] = list(v.values())
        n = len(cl)
        d = [100 * st.mean(v) / 5 for v in cl.values()]
        ints = [round(st.mean(v) * 2 * len(seeds)) for v in cl.values()]
        assert all(abs(i - st.mean(v) * 2 * len(seeds)) < 1e-9 for i, v in zip(ints, cl.values()))
        mb = pct(base[s][k] for s in seeds for k in keys)
        ms = pct(scale[s][k] for s in seeds for k in keys)
        md, se = st.mean(d), st.stdev(d) / math.sqrt(n)
        summ[(task, sname)] = (mb, ms, ms - mb, md - 1.9639 * se, md + 1.9639 * se, perm_p(ints))

seed_note = ("Reacher, Cube, Two-Room and PointMaze average two training seeds (1024, 1025); Push-T uses seed 1024 "
             "only, because the LpWM+SCALE run at seed 1025 collapsed during training and that seed is excluded for "
             "both models.")
out = ["**Table A. LpWM vs. LpWM+SCALE: planning success rate (%) by task, planning-budget tier and planner.** "
       "Each entry is the mean over 600 test episodes (6 sets × 100). Budgets (candidates × iterations): "
       "T1 300×30, T2 150×15, T3 50×10, T4 20×5, T5 10×3. " + seed_note, "",
       "| Task | Tier | Planner | LpWM | LpWM+SCALE |", "|---|---|---|---|---|"]
for task in D:
    for t in TIERS:
        for _, sname in SOLS:
            a, b = tier[(task, t, sname)]
            out.append(f"| {TITLE[task]} | {t} | {sname} | {r1(a)} | {r1(b)} |")
out += ["", "**Table B. LpWM vs. LpWM+SCALE: success rate (%) averaged over the five budget tiers, with paired "
        "significance tests.** Both models are evaluated on the same 600 test episodes with identical planner "
        "noise, so outcomes are paired per episode. Δ = LpWM+SCALE − LpWM in percentage points; the 95% CI is a "
        "paired t-interval over episodes; p is from an exact two-sided paired sign-flip permutation test. "
        + seed_note, "",
        "| Task | Planner | LpWM | LpWM+SCALE | Δ | 95% CI | p-value |", "|---|---|---|---|---|---|---|"]
for task in D:
    for _, sname in SOLS:
        mb, ms, d, lo, hi, p = summ[(task, sname)]
        out.append(f"| {TITLE[task]} | {sname} | {r1(mb)} | {r1(ms)} | {r1(d, True)} | [{sg(lo)}, {sg(hi)}] | {fp(p)} |")
out.append("")
p = os.path.join(R, "LPWM_TABLES_EN.md")
open(p, "w").write("\n".join(out))
print("\n".join(out))
