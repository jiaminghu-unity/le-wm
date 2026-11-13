#!/usr/bin/env python3
"""Seed-level results for the rebuttal (reviewer 9CWa Q3): SCALE vs LeWM and SCALE vs Aux within each
training run, tier-averaged, with a paired 95% CI and exact sign-flip p over the 600 test episodes.
Runs: the two pooled in the paper (A, B) and the two new ones (C = seed 1024, D = seed 1025).
Writes eval_results/rebuttal_mirror/SEEDLEVEL_EN.md. Also prints, per task, the Spearman rank correlation
between distance alignment (Table G) and success (Table E) over the six methods of the constrained-
regression comparison, so that the sentence about it in the rebuttal can be checked.
Reuses the loaders of rebuttal_sig_4seeds_en.py (importing it re-writes SIG_4SEEDS_EN.md unchanged).
"""
import collections, math, os, statistics as st, sys
from fractions import Fraction

# NEW_SEEDS=1025 -> three runs (the two of the submission + seed 1025); default: both new seeds
NEW = tuple(x for x in os.environ.get("NEW_SEEDS", "1024,1025").split(",") if x)
os.environ["NEW_SEEDS"] = ",".join(NEW)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import rebuttal_sig_4seeds_en as S

RUNNAME = {0: "A (submission)", 1: "B (submission)", 2: "C (new)", 3: "D (new)"}
NR = 2 + len(NEW); WORD = {3: "three", 4: "four"}[NR]
NEWRUNS = {1: "run C is new", 2: "runs C and D are new"}[len(NEW)]
TAG = "" if NEW == ("1024", "1025") else "_PAPER_PLUS_" + "_".join(NEW)


def t_p(t, df):  # two-sided p of Student t, df = 2 or 3 (closed forms)
    t = abs(t)
    if df == 2:
        return 1 - t / math.sqrt(t * t + 2)
    th = math.atan(t / math.sqrt(3))
    return max(0.0, 1 - (2 / math.pi) * (th + math.sin(th) * math.cos(th)))


def within_seed(task, sol, a, b, seed):
    da, ka = S.load(task, a, seed, sol)
    db, kb = S.load(task, b, seed, sol)
    assert ka == kb
    # unit = test episode: sum over the five tiers, a start/goal seen in two sets is averaged
    ref = {(k[0], k[2]): (k[3], k[4]) for k in ka}          # (set, episode_id) -> (traj_id, start_idx)
    per = collections.defaultdict(int)
    for (es, ep, tier), v in da.items():
        per[(es, ep)] += v - db[(es, ep, tier)]
    occ = collections.defaultdict(list)
    for k, v in per.items():
        occ[ref[k]].append(v)
    cl = [Fraction(sum(v), len(v)) for v in occ.values()]
    den = 1
    for x in cl:
        den = den * x.denominator // math.gcd(den, x.denominator)
    d = [100 * float(x) / 5 for x in cl]
    md, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    sa, sb = S.pct(da.values()), S.pct(db.values())
    return sa, sb, sa - sb, md - 1.9639 * se, md + 1.9639 * se, S.perm_p([int(x * den) for x in cl])


out = []
for tab, a, b in (("S1", "SCALE", "LeWM"), ("S2", "SCALE", "Aux")):
    out += [f"**Table {tab}. Seed-level results, {a} vs. {b}: success rate (%) averaged over the five budget tiers, "
            "separately for each independently trained run.** Runs A and B are the two runs of the submission; "
            f"{NEWRUNS}. Within a run, {a} and {b} share the training seed and are evaluated on the same 600 test "
            f"episodes (6 sets × 100) with identical planner noise. Δ = {a} − {b} in percentage points; the 95% CI is a "
            "paired t-interval over test episodes within that run and p is from an exact two-sided paired sign-flip "
            "permutation test, so each row reflects evaluation variability only, and the spread across rows of a task "
            "reflects training-seed variability.", "",
            f"| Task | Planner | Run | {b} | {a} | Δ | 95% CI | p-value |", "|---|---|---|---|---|---|---|---|"]
    for task in S.ARMS:
        seeds = S.PAPER_SEEDS[task] + NEW
        for sol, sname in S.SOLS:
            for i, seed in enumerate(seeds):
                sa, sb, d, lo, hi, p = within_seed(task, sol, a, b, seed)
                out.append(f"| {S.TITLE[task]} | {sname} | {RUNNAME[i]} | {S.r1(sb)} | {S.r1(sa)} | {S.r1(d, True)} | "
                           f"[{S.sg(lo)}, {S.sg(hi)}] | {S.fp(p)} |")
    out.append("")
# compact form: one row per task / planner, the four within-run differences, then the two kinds of uncertainty
for tab, a, b in (("S3", "SCALE", "LeWM"), ("S4", "SCALE", "Aux")):
    out += [f"**Table {tab}. {a} − {b} per training run, with training-seed and evaluation uncertainty reported separately.** "
            "Success rate (%) averaged over the five budget tiers; Δ in percentage points. Runs A and B are the two runs "
            f"of the submission; {NEWRUNS}. \"Across runs\": mean of the {WORD} within-run differences, 95% "
            f"t-interval and two-sided paired t-test over the {WORD} training runs (df = {NR - 1}) — training-seed variability. "
            f"\"Across episodes\": the per-episode difference averaged over the {WORD} runs, 95% paired t-interval and "
            "exact two-sided sign-flip permutation test over the 600 test episodes — evaluation variability.", "",
            "| Task | Planner | " + " | ".join(f"Δ run {'ABCD'[i]}" for i in range(NR)) + " | Mean Δ | 95% CI across runs | p across runs | 95% CI across episodes | p across episodes |",
            "|---|---|" + "---|" * (NR + 5)]
    for task in S.ARMS:
        for sol, sname in S.SOLS:
            ds, lo, hi, pv = S.seedlvl[(task, sname, b)]
            pv = t_p(st.mean(ds) / (st.stdev(ds) / math.sqrt(len(ds))), len(ds) - 1)
            d, elo, ehi, ep, pos = S.comp[(task, sname, b)]
            out.append(f"| {S.TITLE[task]} | {sname} | " + " | ".join(S.sg(x) for x in ds) + f" | {S.r1(d, True)} | "
                       f"[{S.sg(lo)}, {S.sg(hi)}] | {S.fp(pv)} | [{S.sg(elo)}, {S.sg(ehi)}] | {S.fp(ep)} |")
    out.append("")
p = os.path.join(S.R, "rebuttal_mirror", f"SEEDLEVEL{TAG}_EN.md")
open(p, "w").write("\n".join(out))
print("\n".join(out))
print("seed order per task:", {t: S.PAPER_SEEDS[t] + NEW for t in S.ARMS})

# ---- rank correlation between alignment (Table G) and success (Table E), six methods per task ----
G = {"Push-T": [0.14, 0.19, 0.55, 0.82, 0.14, 0.77], "Reacher": [0.18, 0.18, 0.42, 0.74, 0.18, 0.52],
     "Cube": [0.02, 0.04, 0.29, 0.67, 0.02, 0.56], "Two-Room": [0.39, 0.41, 0.43, 0.56, 0.72, 0.77],
     "PointMaze": [0.54, 0.53, 0.55, 0.58, 0.73, 0.80]}
E = {"Push-T": ([71.0, 74.1, 75.3, 74.0, 72.0, 74.1], [65.2, 68.8, 69.6, 68.7, 66.1, 68.0]),
     "Reacher": ([71.1, 72.7, 72.0, 71.8, 71.1, 71.3], [79.4, 79.7, 80.0, 80.4, 79.6, 78.4]),
     "Cube": ([61.3, 63.3, 63.6, 63.7, 61.2, 64.6], [64.9, 68.7, 68.8, 69.7, 64.7, 70.6]),
     "Two-Room": ([76.8, 77.9, 80.1, 83.1, 85.8, 89.8], [82.0, 82.8, 85.0, 87.3, 89.6, 93.1]),
     "PointMaze": ([80.3, 87.1, 84.1, 85.8, 81.1, 77.9], [73.4, 85.2, 82.5, 82.4, 78.8, 76.0])}


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y); mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


print("rank correlation alignment vs success over six methods (CEM, iCEM, mean of both):")
for t in G:
    c, i = E[t]; m = [(a + b) / 2 for a, b in zip(c, i)]
    print(f"  {t:9s} {spearman(G[t], c):+.2f} {spearman(G[t], i):+.2f} {spearman(G[t], m):+.2f}")
